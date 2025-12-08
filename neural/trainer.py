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
    # Create coordinate axes as CPU numpy arrays for low-memory, chunked evaluation
    x_np = np.linspace(float(mins[0]), float(maxs[0]), resolution, dtype=np.float32)
    y_np = np.linspace(float(mins[1]), float(maxs[1]), resolution, dtype=np.float32)
    z_np = np.linspace(float(mins[2]), float(maxs[2]), resolution, dtype=np.float32)
    # Torch-friendly tensors for non-chunked fast path
    x_lin = torch.from_numpy(x_np).to(device=device, dtype=dtype)
    y_lin = torch.from_numpy(y_np).to(device=device, dtype=dtype)
    z_lin = torch.from_numpy(z_np).to(device=device, dtype=dtype)
    with torch.no_grad():
        if chunk_slices and resolution >= 128:
            # Build full XY grid on CPU to avoid holding large tensors on device
            xx, yy = np.meshgrid(x_np, y_np, indexing='ij')
            xy = np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)
            slice_points = xy.shape[0]
            volume_cpu = np.empty((resolution, resolution, resolution), dtype=np.float32)

            # Cap batch size for safety on low-memory GPUs
            safe_batch = min(batch_size, 16384)

            for k, z_val in enumerate(z_np):
                slice_chunks = []
                z_col_value = float(z_val)
                # Process XY in small batches to keep GPU memory low
                for start in range(0, slice_points, safe_batch):
                    end = min(start + safe_batch, slice_points)
                    coords_np = np.empty((end - start, 3), dtype=np.float32)
                    coords_np[:, 0:2] = xy[start:end]
                    coords_np[:, 2] = z_col_value

                    coords_tensor = torch.from_numpy(coords_np).to(device=device, dtype=dtype)
                    pred = network(coords_tensor)
                    # Move predictions to CPU immediately as float32
                    slice_chunks.append(pred.float().cpu().numpy())
                    # free GPU memory used by this batch
                    del coords_tensor, pred
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Concatenate all chunk predictions for this slice and reshape
                slice_values = np.concatenate(slice_chunks, axis=0)
                slice_values = slice_values.reshape((resolution, resolution))
                volume_cpu[:, :, k] = slice_values

            volume = volume_cpu
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
    # Ensure result is on CPU numpy float32
    if isinstance(volume, torch.Tensor):
        result = volume.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        result = np.asarray(volume, dtype=np.float32)
    # Final cleanup
    if device.type == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()
    return result


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

    def save_weights(self, filepath):
        """Save model weights and metadata to a file."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        state = {
            'network_state_dict': self.network.state_dict(),
            'scalar_min': self.scalar_min,
            'scalar_scale': self.scalar_scale,
            'batch_size': self.batch_size,
            'chunked': self.chunked
        }
        torch.save(state, filepath)

    @classmethod
    def load_weights(cls, filepath, low_memory=False):
        """Load model weights from a file and reconstruct the model."""
        if not HAS_TORCH or not HAS_TCNN:
            raise RuntimeError("PyTorch and tinycudann required")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(filepath, map_location=device, weights_only=False)
        
        # Rebuild the network architecture
        if low_memory:
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
        
        network = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config=encoding_config,
            network_config=network_config
        ).to(device)
        
        network.load_state_dict(state['network_state_dict'])
        network.eval()
        
        return cls(
            network=network,
            device=device,
            scalar_min=state['scalar_min'],
            scalar_scale=state['scalar_scale'],
            batch_size=state['batch_size'],
            chunked=state.get('chunked', False)
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
        # Check if custom architecture settings are provided
        arch = self.config.get('architecture', {})
        
        if self.low_memory:
            # Low memory mode: suitable for GPUs with ≤6GB VRAM
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": arch.get('n_levels', 12),
                "n_features_per_level": arch.get('n_features_per_level', 2),
                "log2_hashmap_size": min(arch.get('log2_hashmap_size', 16), 18),  # Cap at 18 for low memory
                "base_resolution": arch.get('base_resolution', 8),
                "per_level_scale": arch.get('per_level_scale', 1.5)
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "n_hidden_layers": min(arch.get('n_hidden_layers', 2), 3),  # Cap at 3 layers
                "n_neurons": min(arch.get('n_neurons', 32), 64),  # Cap at 64 neurons
                "activation": "ReLU",
                "output_activation": "None"
            }
        else:
            # Normal mode: use custom or original defaults
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": arch.get('n_levels', 16),
                "n_features_per_level": arch.get('n_features_per_level', 2),
                "log2_hashmap_size": arch.get('log2_hashmap_size', 18),
                "base_resolution": arch.get('base_resolution', 16),
                "per_level_scale": arch.get('per_level_scale', 1.5)
            }
            network_config = {
                "otype": "FullyFusedMLP",
                "n_hidden_layers": arch.get('n_hidden_layers', 2),
                "n_neurons": arch.get('n_neurons', 64),
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
