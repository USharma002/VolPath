import threading
import time

import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:  # pragma: no cover
    tcnn = None
    HAS_TCNN = False

TORCH_DEVICE = None
if HAS_TORCH:
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from PyQt6.QtCore import QThread, pyqtSignal


def _evaluate_volume_with_network(
    network,
    device,
    resolution,
    scalar_min,
    scalar_scale,
    batch_size,
    bounds=None,
    chunk_slices=False
):
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    if bounds is not None:
        mins, maxs = bounds
    else:
        mins = (0.0, 0.0, 0.0)
        maxs = (1.0, 1.0, 1.0)
    x_lin = torch.linspace(float(mins[0]), float(maxs[0]), resolution, device=device, dtype=dtype)
    y_lin = torch.linspace(float(mins[1]), float(maxs[1]), resolution, device=device, dtype=dtype)
    z_lin = torch.linspace(float(mins[2]), float(maxs[2]), resolution, device=device, dtype=dtype)
    with torch.no_grad():
        if chunk_slices and resolution >= 128:
            xy = torch.stack(torch.meshgrid(x_lin, y_lin, indexing='ij'), dim=-1)
            xy = xy.reshape(-1, 2).to(device=device, dtype=dtype)
            slice_points = xy.shape[0]
            volume_cpu = torch.empty((resolution, resolution, resolution), dtype=torch.float32)
            for k, z_val in enumerate(z_lin):
                z_col = torch.full((slice_points, 1), float(z_val), device=device, dtype=dtype)
                coords = torch.cat([xy, z_col], dim=1)
                slice_chunks = []
                for start in range(0, slice_points, batch_size):
                    end = min(start + batch_size, slice_points)
                    chunk = coords[start:end]
                    pred = network(chunk)
                    slice_chunks.append(pred.float())
                slice_values = torch.cat(slice_chunks, dim=0).view(resolution, resolution)
                volume_cpu[:, :, k] = slice_values.cpu()
            volume = volume_cpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            grid = torch.stack(torch.meshgrid(x_lin, y_lin, z_lin, indexing='ij'), dim=-1)
            flat = grid.view(-1, 3)
            outputs = []
            for start in range(0, flat.shape[0], batch_size):
                end = min(start + batch_size, flat.shape[0])
                chunk = flat[start:end]
                pred = network(chunk)
                outputs.append(pred.float())
            volume = torch.cat(outputs, dim=0)
            volume = volume.view(resolution, resolution, resolution)
    volume = volume * scalar_scale + scalar_min
    return volume.detach().cpu().numpy().astype(np.float32, copy=False)


class NeuralFieldModel:
    """Wrapper around the tinycudann network for inference"""

    def __init__(self, network, device, scalar_min, scalar_scale, batch_size, chunked=False):
        self.network = network
        self.device = device
        self.scalar_min = scalar_min
        self.scalar_scale = scalar_scale
        self.batch_size = batch_size
        self.chunked = chunked
        self.network.eval()

    def generate_volume(self, resolution, bounds=None):
        return _evaluate_volume_with_network(
            self.network,
            self.device,
            resolution,
            self.scalar_min,
            self.scalar_scale,
            self.batch_size,
            bounds,
            chunk_slices=self.chunked
        )


class NeuralFieldTrainer(QThread):
    """Background trainer for neural scalar fields using tinycudann"""

    loss_updated = pyqtSignal(float, int)
    preview_ready = pyqtSignal(object, dict)
    training_complete = pyqtSignal(object, bool)
    status = pyqtSignal(str)

    def __init__(self, coords, scalars, config):
        super().__init__()
        self.coords = coords.astype(np.float32)
        self.scalars = scalars.astype(np.float32)
        self.config = config
        self.dataset_token = None
        raw_bounds = self.config.get('bounds')
        if raw_bounds is not None:
            mins = np.asarray(raw_bounds[0], dtype=np.float32).tolist()
            maxs = np.asarray(raw_bounds[1], dtype=np.float32).tolist()
            self._bounds = (mins, maxs)
        else:
            self._bounds = None
        self.low_memory = bool(self.config.get('low_memory'))
        if HAS_TORCH:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        else:
            self.device = None
            self.dtype = None
        self.scalar_min = float(np.min(self.scalars))
        self.scalar_max = float(np.max(self.scalars))
        self.scalar_scale = max(self.scalar_max - self.scalar_min, 1e-6)
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._paused = False
        self.model = None

    def stop(self):
        self._stop_requested = True
        self._pause_event.set()

    def pause(self):
        if self._paused or self._stop_requested:
            return
        self._paused = True
        self._pause_event.clear()
        self.status.emit("paused")

    def resume(self):
        if not self._paused or self._stop_requested:
            return
        self._paused = False
        self._pause_event.set()
        self.status.emit("resumed")

    @property
    def is_paused(self):
        return self._paused

    def evaluate_live_model(self, resolution, bounds):
        if not HAS_TORCH or self.model is None:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            return _evaluate_volume_with_network(
                self.model,
                self.device,
                resolution,
                self.scalar_min,
                self.scalar_scale,
                self.config['inference_batch'],
                bounds=bounds,
                chunk_slices=self.low_memory and resolution >= 192
            )
        finally:
            if was_training:
                self.model.train()

    def run(self):
        if not HAS_TORCH or not HAS_TCNN:
            self.status.emit("tinycudann/torch not available - install to use neural renderer")
            self.training_complete.emit(None, False)
            return
        if not torch.cuda.is_available():
            self.status.emit("CUDA-capable GPU required for tinycudann training")
            self.training_complete.emit(None, False)
            return

        coords_tensor = torch.from_numpy(self.coords).to(self.device, dtype=self.dtype)
        target = ((self.scalars - self.scalar_min) / self.scalar_scale).astype(np.float32)
        target_tensor = torch.from_numpy(target).unsqueeze(-1).to(self.device, dtype=self.dtype)

        self.model = self._build_model().to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        loss_fn = torch.nn.MSELoss()
        batch_size = self.config['batch_size']
        total_steps = self.config['steps']
        log_interval = max(1, self.config['log_interval'])
        preview_interval = self.config['preview_interval']

        num_samples = coords_tensor.shape[0]

        for step in range(total_steps):
            if self._stop_requested:
                break
            self._pause_event.wait()
            if self._stop_requested:
                break
            sample_idx = torch.randint(0, num_samples, (batch_size,), device=self.device)
            batch_coords = coords_tensor[sample_idx]
            batch_target = target_tensor[sample_idx]

            pred = self.model(batch_coords)
            loss = loss_fn(pred.float(), batch_target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                self.loss_updated.emit(float(loss.item()), int(step))

            if preview_interval > 0 and step % preview_interval == 0:
                preview_res = self.config['preview_resolution']
                try:
                    volume = self._generate_preview(preview_res)
                    self.preview_ready.emit(
                        volume,
                        {
                            'step': step,
                            'loss': float(loss.item()),
                            'bounds': self._bounds
                        }
                    )
                except RuntimeError as err:
                    self.status.emit(f"Preview failed: {err}")

        completed = not self._stop_requested
        model_wrapper = None
        if self.model is not None:
            model_wrapper = NeuralFieldModel(
                self.model,
                self.device,
                self.scalar_min,
                self.scalar_scale,
                self.config['inference_batch'],
                chunked=self.low_memory
            )
        self.training_complete.emit(model_wrapper, completed)

    def _build_model(self):
        if self.low_memory:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 12,
                "n_features_per_level": 2,
                "log2_hashmap_size": 16,
                "base_resolution": 8,
                "per_level_scale": 1.5
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "n_hidden_layers": 2,
                "n_neurons": 32,
                "activation": "ReLU",
                "output_activation": "None"
            }
        else:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 18,
                "base_resolution": 16,
                "per_level_scale": 1.5
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "n_hidden_layers": 2,
                "n_neurons": 64,
                "activation": "ReLU",
                "output_activation": "None"
            }
        return tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=encoding_config,
            network_config=network_config
        )

    def _generate_preview(self, resolution):
        batch = self.config['inference_batch']
        return _evaluate_volume_with_network(
            self.model,
            self.device,
            resolution,
            self.scalar_min,
            self.scalar_scale,
            batch,
            bounds=self._bounds,
            chunk_slices=self.low_memory and resolution >= 192
        )


__all__ = [
    "HAS_TORCH",
    "HAS_TCNN",
    "TORCH_DEVICE",
    "NeuralFieldModel",
    "NeuralFieldTrainer",
    "_evaluate_volume_with_network"
]
