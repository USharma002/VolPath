import sys
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QDialog, QScrollArea, QPushButton
)
from vispy import scene
from vispy.app import use_app

from config.optional import HAS_NATURAL_NEIGHBOR
from data_pipeline.loader import DataLoader, load_multi_file_hdf5, load_vtk_polydata
from data_pipeline.processing import DataProcessor
from interpolators.worker import InterpolatorFactory
from neural.trainer import HAS_TORCH, HAS_TCNN, NeuralFieldTrainer
from renderers.camera import CameraController
from renderers.reference import ReferenceCube
from renderers.plenoxel import (
    PlenoxelRenderer,
    build_plenoxel_grid,
    plenoxel_volume_from_nodes,
)
from renderers.scatter import ScatterPlotRenderer
from renderers.volume import VispyVolumeRenderer
from transfer.transfer_function import TransferFunction
from ui.dialogs import BatchConfigDialog, MetadataDialog, ResolutionDialog
from ui.panels import build_control_panel
from ui.subvolume import SubvolumeController
from ui.visualization import VisualizationController
from ui.transfer import TransferFunctionDialog

if HAS_TORCH:
    import torch
else:  # pragma: no cover
    torch = None

use_app('pyqt6')

VOLUME_EXTENT = 1.0

HDF5_EXTENSIONS = {'.h5', '.hdf5'}
VTK_EXTENSIONS = {'.vtk', '.vtp', '.vti'}
SNAPSHOT_FILE_FILTER = (
    "Snapshot Files (*.h5 *.hdf5 *.vtk *.vtp *.vti);;"
    "HDF5 Files (*.h5 *.hdf5);;"
    "VTK Files (*.vtk *.vtp *.vti);;"
    "All Files (*)"
)

# Numba-accelerated nearest neighbor interpolation


class VolumeRenderingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Rendering - Interactive GUI")
        self.resize(1600, 1000)

        # Data caches
        self.data = None
        self.scalars = None
        self.original_coords = None
        self.original_scalars = None
        self.grid_values = None
        self.grid_bounds = None
        self._pending_grid_bounds = None
        self.current_stats = None
        self.snapshot_path = None
        self.loaded_poly = None
        self.coord_min = None
        self.coord_max = None
        self.coord_span = None
        self.snapshot_header = {}
        self.snapshot_files = []
        self.snapshot_metadata = {"headers": {}, "datasets": []}
        self.current_field_name = None
        self.current_transform_name = None
        self.current_subvolume_bounds = None

        # Neural/cache state
        self.neural_trainer = None
        self.neural_model = None
        self.neural_last_preview_meta = None
        self.neural_training_bounds = None
        self.nn_grid_values = None
        self.nn_bounds = None
        self._dataset_token = 0
        self._neural_model_token = None
        self._trainer_dataset_token = None
        self.worker = None
        self.batch_dialog = None

        # Rendering helpers
        self.transfer_function = TransferFunction('magma')
        self.viz_mode = 'volume'
        self.interpolator_factory = InterpolatorFactory()
        self.max_3d_texture_size = None
        self._suppress_camera_event = False

        # UI placeholders populated by panel builders
        self.neural_params = {}
        self.cam_params = {}
        self.log_text = None
        self.progress_bar = None
        self.metadata_btn = None
        self.subvolume_slider_steps = 1000
        self.subvolume_center_sliders = {}
        self.subvolume_center_labels = {}
        self.subvolume_size_slider = None
        self.subvolume_size_label = None
        self.subvolume_group = None
        self.extract_subbox_btn = None
        self.subvolume_box_visual = None
        self.volume_method_combo = None
        self.volume_step_spin = None
        self.volume_threshold_spin = None
        self.plenoxel_show_check = None
        self.plenoxel_min_points_spin = None
        self.plenoxel_min_depth_spin = None
        self.plenoxel_max_depth_spin = None
        self.plenoxel_build_btn = None
        self.plenoxel_stats_label = None

        # Build vispy canvas and camera
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor='#000000',
            size=(1100, 900),
            show=False
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=60,
            azimuth=45,
            elevation=30,
            distance=2.5 * VOLUME_EXTENT,
            center=(VOLUME_EXTENT / 2,) * 3
        )
        self.view.camera.depth_value = 50000
        self.camera_controller = CameraController(self.view.camera, VOLUME_EXTENT)

        # Renderer instances share the same view
        self.volume_renderer = VispyVolumeRenderer(self.view, VOLUME_EXTENT, self.transfer_function)
        self.neural_renderer = VispyVolumeRenderer(self.view, VOLUME_EXTENT, self.transfer_function)
        self.scatter_renderer = ScatterPlotRenderer(self.view, VOLUME_EXTENT, self.transfer_function)
        self.reference_cube = ReferenceCube(self.view, VOLUME_EXTENT)
        self.reference_cube.render(visible=True)
        self.plenoxel_renderer = PlenoxelRenderer(self.view, VOLUME_EXTENT)
        self.plenoxel_nodes = None
        self.plenoxel_stats = None
        self.plenoxel_volume = None
        self.plenoxel_visible = False

        # Subvolume controller needs a view/canvas before wiring the UI
        self.subvolume = SubvolumeController(self, VOLUME_EXTENT)
        self.visualization = VisualizationController(self, VOLUME_EXTENT)

        # Assemble Qt sidebars + canvas
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        control_panel = build_control_panel(self, VOLUME_EXTENT)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(control_panel)
        control_scroll.setMinimumWidth(360)
        main_layout.addWidget(control_scroll, 0)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.canvas.native.setMinimumSize(640, 640)
        right_layout.addWidget(self.canvas.native, 1)

        export_layout = QHBoxLayout()
        screenshot_btn = QPushButton("Save Screenshot…")
        screenshot_btn.clicked.connect(self.save_screenshot)
        export_layout.addWidget(screenshot_btn)

        npy_btn = QPushButton("Save NPY")
        npy_btn.clicked.connect(self.save_npy)
        export_layout.addWidget(npy_btn)

        npz_btn = QPushButton("Save NPZ")
        npz_btn.clicked.connect(self.save_npz)
        export_layout.addWidget(npz_btn)

        batch_btn = QPushButton("Batch Render…")
        batch_btn.clicked.connect(self.batch_render_dialog)
        export_layout.addWidget(batch_btn)
        export_layout.addStretch()
        right_layout.addLayout(export_layout)

        main_layout.addWidget(right_container, 1)

        self.statusBar().showMessage("Load an HDF5 snapshot to begin")
        self.max_3d_texture_size = self._detect_max_3d_texture_size()
        self._connect_camera_events()
        self.visualization.set_mode('volume')
        self.subvolume.update_extract_button_state()

    def _sync_camera_controls(self, state):
        if not state:
            return
        for key, spinbox in self.cam_params.items():
            if key in state:
                spinbox.blockSignals(True)
                spinbox.setValue(state[key])
                spinbox.blockSignals(False)

    def _connect_camera_events(self):
        if not getattr(self, 'view', None) or not getattr(self.view, 'camera', None):
            return
        events = getattr(self.view.camera, 'events', None)
        if events and hasattr(events, 'view_changed'):
            events.view_changed.connect(self._on_camera_view_changed)

    def _detect_max_3d_texture_size(self):
        try:
            value = gl.glGetIntegerv(gl.GL_MAX_3D_TEXTURE_SIZE)
            if isinstance(value, (tuple, list)):
                value = value[0]
            return int(value)
        except Exception:
            return None

    def _on_camera_view_changed(self, event):
        if self._suppress_camera_event:
            return
        if not self.camera_controller:
            return
        state = self.camera_controller.capture_state()
        self._sync_camera_controls(state)

    def _on_transfer_function_live_update(self, *_):
        """Refresh active visuals when transfer function dialog changes"""
        if self.volume_renderer and self.volume_renderer.visual is not None:
            self.volume_renderer.update_transfer_function(self.transfer_function)
        if self.neural_renderer and self.neural_renderer.visual is not None:
            self.neural_renderer.update_transfer_function(self.transfer_function)
        if self.scatter_renderer:
            self.scatter_renderer.update_transfer_function(self.transfer_function)
            if self.scatter_renderer.visual is not None:
                self.visualization.create_scatter_plot(force_visible=(self.viz_mode == 'scatter'))
        self.canvas.update()

    def on_volume_method_changed(self, method: str):
        method = (method or '').strip()
        if not method:
            return
        if self.volume_threshold_spin:
            self.volume_threshold_spin.setEnabled(method == 'iso')
        if self.volume_renderer:
            self.volume_renderer.update_render_settings(method=method)
        if self.grid_values is not None:
            self.visualization.update_volume_visual()

    def on_volume_step_changed(self, value: float):
        if self.volume_renderer:
            self.volume_renderer.update_render_settings(step_size=value)
        if self.volume_renderer and self.volume_renderer.visual is not None:
            self.canvas.update()

    def on_volume_threshold_changed(self, value: float):
        if self.volume_renderer:
            self.volume_renderer.update_render_settings(threshold=value)
        if self.volume_renderer and self.volume_renderer.visual is not None:
            self.canvas.update()

    # --- Plenoxel helpers ------------------------------------------------------

    def _reset_plenoxel_state(self, *, full_reset: bool = False):
        self.plenoxel_nodes = None
        self.plenoxel_stats = None
        self.plenoxel_volume = None
        if self.plenoxel_renderer:
            self.plenoxel_renderer.clear()
        if full_reset:
            self.plenoxel_visible = False
            if self.plenoxel_show_check:
                self.plenoxel_show_check.blockSignals(True)
                self.plenoxel_show_check.setChecked(False)
                self.plenoxel_show_check.blockSignals(False)
        if self.plenoxel_stats_label:
            self.plenoxel_stats_label.setText("Plenoxel grid: not built")

    def _update_plenoxel_controls_state(self):
        data_ready = bool(self.data is not None and self.scalars is not None)
        widgets = [
            self.plenoxel_show_check,
            self.plenoxel_min_points_spin,
            self.plenoxel_min_depth_spin,
            self.plenoxel_max_depth_spin,
            self.plenoxel_build_btn,
        ]
        for widget in widgets:
            if widget:
                widget.setEnabled(data_ready)
        if not data_ready and self.plenoxel_show_check:
            self.plenoxel_show_check.blockSignals(True)
            self.plenoxel_show_check.setChecked(False)
            self.plenoxel_show_check.blockSignals(False)
            self.plenoxel_visible = False

    def _update_plenoxel_stats_label(self):
        if not self.plenoxel_stats_label:
            return
        if not self.plenoxel_stats or not self.plenoxel_stats.get('node_count'):
            self.plenoxel_stats_label.setText("Plenoxel grid: not built")
            return
        stats = self.plenoxel_stats
        self.plenoxel_stats_label.setText(
            f"Cells: {stats['node_count']:,} | Depth ≤ {stats['max_depth_reached']}"
        )

    def on_plenoxel_toggled(self, checked: bool):
        self.plenoxel_visible = bool(checked)
        if self.plenoxel_visible and not self.plenoxel_nodes:
            self.rebuild_plenoxel_grid(auto=True)
        else:
            self.update_plenoxel_visual()

    def on_plenoxel_depth_changed(self):
        if self.plenoxel_min_depth_spin and self.plenoxel_max_depth_spin:
            self.plenoxel_max_depth_spin.setMinimum(self.plenoxel_min_depth_spin.value())
            if self.plenoxel_max_depth_spin.value() < self.plenoxel_min_depth_spin.value():
                self.plenoxel_max_depth_spin.setValue(self.plenoxel_min_depth_spin.value())

    def rebuild_plenoxel_grid(self, auto: bool = False):
        if self.data is None or self.scalars is None:
            if not auto:
                QMessageBox.information(
                    self,
                    "Plenoxel Grid",
                    "Load data and apply a field transform before building the plenoxel grid."
                )
            return
        min_points = self.plenoxel_min_points_spin.value() if self.plenoxel_min_points_spin else 500
        min_depth = self.plenoxel_min_depth_spin.value() if self.plenoxel_min_depth_spin else 1
        max_depth = self.plenoxel_max_depth_spin.value() if self.plenoxel_max_depth_spin else 5
        try:
            nodes, stats = build_plenoxel_grid(
                self.data,
                self.scalars,
                min_points=min_points,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Plenoxel Grid", f"Failed to build plenoxel grid:\n{exc}")
            self.log(f"ERROR building plenoxel grid: {exc}")
            import traceback
            traceback.print_exc()
            return

        self.plenoxel_nodes = nodes
        self.plenoxel_stats = stats
        self._update_plenoxel_stats_label()
        if not nodes:
            self.log("Plenoxel grid empty for current threshold; try lowering minimum particles or depth")
        else:
            self.log(
                f"Plenoxel grid built: {stats['node_count']:,} cells (max depth {stats['max_depth_reached']})"
            )

        try:
            target_depth = self.plenoxel_max_depth_spin.value() if self.plenoxel_max_depth_spin else None
            volume, volume_stats = plenoxel_volume_from_nodes(nodes, target_depth=target_depth)
        except Exception as exc:  # pragma: no cover
            volume = None
            volume_stats = {}
            self.log(f"ERROR generating plenoxel volume: {exc}")

        self.plenoxel_volume = volume
        if volume is not None:
            resolution = volume.shape[0]
            self.grid_values = volume
            self.grid_bounds = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            self.current_stats = volume_stats
            self.log(
                f"Plenoxel volume ready: {resolution}³ voxels from {stats['node_count']:,} cells"
            )
            self.visualization.update_volume_visual()
        else:
            self.log("Plenoxel volume unavailable; volume renderer unchanged")
        self.update_plenoxel_visual()

    def update_plenoxel_visual(self):
        if not self.plenoxel_renderer:
            return
        if not self.plenoxel_visible or not self.plenoxel_nodes:
            self.plenoxel_renderer.clear()
            self.canvas.update()
            return
        self.plenoxel_renderer.render(self.plenoxel_nodes, self.plenoxel_stats, visible=True)
        self.canvas.update()


    def open_transfer_function_dialog(self):
        """Open transfer function editor and apply changes to scatter"""
        if self.original_scalars is None:
            QMessageBox.warning(self, "No Data", "Load data before editing transfer function")
            return

        dialog = TransferFunctionDialog(self.transfer_function, self)
        # Provide data range for editor
        try:
            data_min, data_max = float(self.original_scalars.min()), float(self.original_scalars.max())
        except Exception:
            data_min, data_max = 0.0, 1.0

        dialog.set_data_range(data_min, data_max)
        dialog.transferFunctionChanged.connect(self._on_transfer_function_live_update)

        if dialog.exec():
            # Apply final changes and refresh
            self.transfer_function = dialog.tf
            if self.cmap_combo:
                self.cmap_combo.setCurrentText(self.transfer_function.colormap)
            self._on_transfer_function_live_update()



    

    

    

    def on_resolution_changed(self, value):
        """Update resolution info display"""
        voxel_size = VOLUME_EXTENT / value
        memory_mb = (value ** 3) * 4 / (1024**2)
        
        self.resolution_info.setText(
            f"Volume: {VOLUME_EXTENT:.0f}³ units | Voxel: {voxel_size:.2f} | Memory: {memory_mb:.0f} MB"
        )
        
        # Show warning for high resolutions
        show_warning = False
        warning_text = ""
        
        if value >= 1024:
            show_warning = True
            warning_text = f"⚠ {value}³ >= 4GB memory! May fail or be very slow!"
        elif value >= 768:
            show_warning = True
            warning_text = "⚠ High resolution - ensure you have sufficient GPU memory"
        elif value >= 512:
            show_warning = True
            warning_text = "⚠ Resolution >512 may fail on some GPUs"
        
        # Add GPU-specific warning
        if self.max_3d_texture_size and value > self.max_3d_texture_size:
            show_warning = True
            warning_text = f"⚠⚠ GPU MAX: {self.max_3d_texture_size}³ - THIS WILL FAIL!"
        
        self.resolution_warning.setText(warning_text)
        self.resolution_warning.setVisible(show_warning)
        
        # Change warning color based on severity
        if value > 768:
            self.resolution_warning.setStyleSheet("color: red; font-weight: bold; font-size: 9px;")
        else:
            self.resolution_warning.setStyleSheet("color: orange; font-weight: bold; font-size: 9px;")

    def on_method_changed(self, method):
        """Update UI when interpolation method changes"""
        needs_epsilon = any(kernel in method for kernel in
                          ['gaussian', 'multiquadric', 'inverse_multiquadric', 'inverse_quadratic'])
        
        if needs_epsilon:
            self.epsilon_label.setStyleSheet("color: red; font-weight: bold; font-size: 9px;")
            self.epsilon_label.setText("⚠ This kernel requires epsilon!")
        else:
            self.epsilon_label.setStyleSheet("color: gray; font-size: 9px;")
            self.epsilon_label.setText("(for gaussian, multiquadric, etc.)")

    

    # --- Neural renderer utilities -------------------------------------------------

    def _update_neural_controls_state(self):
        data_ready = bool(self.data is not None and self.scalars is not None)
        trainer = self.neural_trainer
        running = bool(trainer and trainer.isRunning())
        paused = bool(trainer and getattr(trainer, 'is_paused', False))
        if self.neural_train_btn:
            if paused:
                label = "Resume"
            elif running:
                label = "Pause"
            else:
                label = "Train / Resume"
            enable = data_ready or running or paused
            self.neural_train_btn.setEnabled(enable)
            self.neural_train_btn.setText(label)
        if self.neural_render_btn:
            render_ready = bool(
                (self.neural_model is not None) or
                (self.nn_grid_values is not None) or
                (self.grid_values is not None)
            )
            self.neural_render_btn.setEnabled(render_ready)
        if self.neural_reset_btn:
            reset_ready = bool(
                data_ready or running or paused or
                self.neural_model is not None or
                self.nn_grid_values is not None
            )
            self.neural_reset_btn.setEnabled(reset_ready)

    def handle_neural_train_button(self):
        trainer = self.neural_trainer
        if trainer and getattr(trainer, 'is_paused', False):
            trainer.resume()
            if self.neural_status_label:
                self.neural_status_label.setText("Status: training (resumed)")
            self.log("Resumed neural training")
            self._update_neural_controls_state()
            return
        if trainer and trainer.isRunning():
            trainer.pause()
            if self.neural_status_label:
                self.neural_status_label.setText("Status: paused")
            self.log("Paused neural training")
            self._update_neural_controls_state()
            return
        self.start_neural_training()

    def reset_neural_renderer(self):
        self._reset_neural_state()
        self.log("Neural renderer state reset")

    def _reset_neural_state(self):
        if self.neural_trainer and self.neural_trainer.isRunning():
            self.neural_trainer.stop()
        self.neural_trainer = None
        self.neural_model = None
        self.nn_grid_values = None
        self.nn_bounds = None
        self.neural_last_preview_meta = None
        self.neural_training_bounds = None
        self._neural_model_token = None
        self._trainer_dataset_token = None
        if self.neural_renderer:
            self.neural_renderer.clear()
        if self.neural_loss_plot:
            self.neural_loss_plot.clear_plot()
        if self.neural_preview_label:
            self.neural_preview_label.setText("Last preview: —")
        if self.neural_status_label:
            self.neural_status_label.setText("Status: idle (load data)")
        if self.neural_train_btn:
            self.neural_train_btn.setText("Train / Resume")
        self._update_neural_controls_state()

    def _build_neural_config(self):
        if not self.neural_params:
            return None
        config = {
            'steps': self.neural_params['steps'].value(),
            'batch_size': self.neural_params['batch_size'].value(),
            'lr': self.neural_params['lr'].value(),
            'log_interval': max(1, self.neural_params['log_interval'].value()),
            'preview_interval': max(0, self.neural_params['preview_interval'].value()),
            'preview_resolution': self.neural_params['preview_resolution'].value(),
            'inference_batch': self.neural_params['inference_batch'].value(),
        }
        if self.neural_auto_preview_check and not self.neural_auto_preview_check.isChecked():
            config['preview_interval'] = 0
        low_mem = bool(self.neural_low_memory_check and self.neural_low_memory_check.isChecked())
        config['low_memory'] = low_mem
        if low_mem:
            config['batch_size'] = min(config['batch_size'], 8192)
            config['preview_resolution'] = min(config['preview_resolution'], 96)
            config['inference_batch'] = min(config['inference_batch'], 16384)
        return config

    def _normalize_volume_for_render(self, volume):
        if volume is None:
            return None
        arr = np.asarray(volume, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return None
        vmin = float(arr[finite].min())
        vmax = float(arr[finite].max())
        if vmax <= vmin:
            return np.zeros_like(arr, dtype=np.float32)
        normalized = (arr - vmin) / (vmax - vmin)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    def _cache_neural_volume(self, volume, meta=None, bounds=None, dataset_token=None):
        if dataset_token is not None and dataset_token != self._dataset_token:
            self.log("Ignoring neural cache from previous snapshot")
            return
        normalized = self._normalize_volume_for_render(volume)
        if normalized is None:
            self.log("Neural preview contained invalid values")
            return
        if bounds is not None:
            mins = np.asarray(bounds[0], dtype=np.float32)
            maxs = np.asarray(bounds[1], dtype=np.float32)
            bounds = (mins, maxs)
        self.nn_grid_values = normalized
        self.nn_bounds = bounds
        self.neural_last_preview_meta = meta or {}
        if self.neural_preview_label:
            parts = []
            if meta:
                step = meta.get('step')
                if step is not None:
                    parts.append(f"step {step}")
                loss = meta.get('loss')
                if loss is not None:
                    parts.append(f"loss {loss:.3e}")
                res = meta.get('resolution', normalized.shape[0])
                parts.append(f"{res}³")
                src = meta.get('source')
                if src:
                    parts.append(src)
            if not parts:
                parts.append("updated")
            self.neural_preview_label.setText("Last preview: " + " | ".join(parts))
        if self.viz_mode == 'neural':
            self._render_neural_volume()
        else:
            self.log("Neural volume cached – switch to Neural Renderer mode to view it")

    def _get_active_neural_grid(self):
        if self.nn_grid_values is not None:
            return self.nn_grid_values
        return self.grid_values

    def _get_neural_display_resolution(self):
        if self.neural_display_res_spin:
            return self.neural_display_res_spin.value()
        if self.neural_render_res_spin:
            return self.neural_render_res_spin.value()
        return 256

    def _get_training_samples(self):
        if self.original_coords is None or self.original_scalars is None:
            return None, None
        if self.coord_min is None or self.coord_span is None:
            return None, None
        coords = ((self.original_coords - self.coord_min) / self.coord_span).astype(np.float32)
        source_scalars = self.scalars if self.scalars is not None else self.original_scalars
        scalars = np.asarray(source_scalars, dtype=np.float32)
        return coords, scalars

    def _render_neural_volume(self):
        if not self.neural_renderer:
            return
        grid = self._get_active_neural_grid()
        if grid is None:
            self.neural_renderer.clear()
            self.neural_renderer.set_active(False)
            if self.viz_mode == 'neural':
                self.log("No neural volume available yet (train the model or compute a baseline volume)")
            return
        bounds = self.nn_bounds if self.nn_grid_values is not None else self.grid_bounds
        grid_for_render = np.ascontiguousarray(np.transpose(grid, (2, 1, 0)))
        self.neural_renderer.render(
            grid_for_render,
            visible=(self.viz_mode == 'neural'),
            bounds=bounds
        )
        self.canvas.update()

    def start_neural_training(self):
        if self.neural_trainer and self.neural_trainer.isRunning():
            QMessageBox.information(self, "Training", "Neural training is already running")
            return
        if not HAS_TORCH:
            QMessageBox.warning(self, "Neural Renderer", "PyTorch is not installed. Install torch to use neural training.")
            self.log("Neural training requires PyTorch – aborting start")
            return
        if not HAS_TCNN:
            QMessageBox.warning(self, "Neural Renderer", "tiny-cuda-nn (tinycudann) is not installed. Install it to train the neural field.")
            self.log("Neural training requires tinycudann – aborting start")
            return
        coords, scalars = self._get_training_samples()
        if coords is None or scalars is None:
            QMessageBox.warning(
                self,
                "Missing Data",
                "Load data (original particle positions and field) before training"
            )
            return
        config = self._build_neural_config()
        if config is None:
            QMessageBox.warning(self, "Configuration", "Neural controls are not initialized yet")
            return
        if config.get('low_memory'):
            self.log(
                "Low-memory neural mode enabled: limiting hash levels, neuron count, and batch sizes"
            )

        # Discard any previously trained networks so paused renders use the live trainer
        if self.neural_model is not None:
            self.neural_model = None
            self._neural_model_token = None
            self.log("Cleared cached neural model (new training run starting)")

        training_bounds = self.subvolume.get_active_bounds()
        bounds_payload = None
        bounds_desc = "full cube [0,1]^3"
        if training_bounds is not None:
            mins = np.asarray(training_bounds[0], dtype=np.float32)
            maxs = np.asarray(training_bounds[1], dtype=np.float32)
            bounds_payload = (mins.tolist(), maxs.tolist())
            bounds_desc = (
                f"subbox [{mins[0]:.3f}, {mins[1]:.3f}, {mins[2]:.3f}] → "
                f"[{maxs[0]:.3f}, {maxs[1]:.3f}, {maxs[2]:.3f}]"
            )
        config['bounds'] = bounds_payload
        self.neural_training_bounds = bounds_payload

        self.nn_grid_values = None
        self.nn_bounds = None
        if self.neural_loss_plot:
            self.neural_loss_plot.clear_plot()
        if self.neural_preview_label:
            self.neural_preview_label.setText("Last preview: —")
        if self.neural_status_label:
            self.neural_status_label.setText("Status: training...")

        self.neural_trainer = NeuralFieldTrainer(coords, scalars, config)
        self.neural_trainer.dataset_token = self._dataset_token
        self._trainer_dataset_token = self._dataset_token
        self.neural_trainer.loss_updated.connect(self._on_neural_loss)
        self.neural_trainer.preview_ready.connect(self._on_neural_preview)
        self.neural_trainer.training_complete.connect(self._on_neural_training_complete)
        self.neural_trainer.status.connect(self._on_neural_status)
        self.neural_trainer.start()

        self._update_neural_controls_state()
        self.log(
            f"Neural training started with {len(coords):,} original particles | steps={config['steps']:,} "
            f"| batch={config['batch_size']:,} | region={bounds_desc} | mode="
            f"{'low-memory' if config.get('low_memory') else 'standard'}"
        )

    def _on_neural_status(self, text):
        self.log(text)
        if self.neural_status_label:
            self.neural_status_label.setText(f"Status: {text}")

    def _on_neural_loss(self, loss, step):
        if self.neural_loss_plot:
            self.neural_loss_plot.append(step, loss)
        if self.neural_status_label:
            self.neural_status_label.setText(f"Status: step {step:,} | loss {loss:.3e}")

    def _on_neural_preview(self, volume, meta):
        meta = meta or {}
        meta.setdefault('resolution', volume.shape[0] if volume is not None else 0)
        meta.setdefault('source', 'preview')
        bounds = meta.get('bounds')
        if bounds is None and self.neural_training_bounds is not None:
            bounds = self.neural_training_bounds
        if bounds is None and self.subvolume_group and self.subvolume_group.isChecked():
            bounds = self.subvolume.get_active_bounds(require_enabled=True)
        sender = self.sender()
        sender_token = getattr(sender, 'dataset_token', None)
        self._cache_neural_volume(volume, meta, bounds=bounds, dataset_token=sender_token)

    def _on_neural_training_complete(self, model_wrapper, completed):
        sender = self.sender()
        sender_token = getattr(sender, 'dataset_token', None)
        if sender_token is not None and sender_token != self._dataset_token:
            self.log("Ignoring neural model from previous snapshot load")
            if sender is self.neural_trainer:
                self.neural_trainer = None
                self._trainer_dataset_token = None
            self.neural_training_bounds = None
            self._update_neural_controls_state()
            return
        if model_wrapper:
            self.neural_model = model_wrapper
            self._neural_model_token = self._dataset_token
            if completed:
                msg = "Neural training finished – ready to render"
            else:
                msg = "Training stopped early – neural model preserved for rendering"
        else:
            msg = "Neural training stopped before completion"
        self.log(msg)
        if self.neural_status_label:
            self.neural_status_label.setText(f"Status: {msg}")
        self.neural_trainer = None
        self._trainer_dataset_token = None
        self.neural_training_bounds = None
        self._update_neural_controls_state()

    def render_neural_volume(self, resolution=None, source='manual render'):
        bounds = self.current_subvolume_bounds
        trainer = self.neural_trainer
        trainer_paused = bool(trainer and getattr(trainer, 'is_paused', False))
        if isinstance(resolution, bool):  # safeguard for stray signal signatures
            resolution = None
        if self.neural_model is not None and self._neural_model_token != self._dataset_token:
            self.log("Discarding cached neural model from previous snapshot")
            self.neural_model = None
        if self.neural_model is None:
            if trainer and trainer.model is not None:
                if not trainer_paused:
                    QMessageBox.information(
                        self,
                        "Neural Renderer",
                        "Pause training before rendering the in-progress neural field."
                    )
                    self.log("Pause neural training to render the live network")
                    return
                try:
                    label = source or 'manual render'
                    live_resolution = resolution or (self.neural_render_res_spin.value() if self.neural_render_res_spin else 256)
                    self.log(f"Rendering paused neural field at {live_resolution}³ ({label}) ...")
                    volume = trainer.evaluate_live_model(live_resolution, bounds)
                    if volume is None:
                        raise RuntimeError("Trainer model unavailable for rendering")
                    self._cache_neural_volume(
                        volume,
                        {
                            'resolution': live_resolution,
                            'source': f"live trainer ({label})"
                        },
                        bounds=bounds,
                        dataset_token=getattr(trainer, 'dataset_token', self._dataset_token)
                    )
                    self.log("Live neural render ready")
                    return
                except Exception as exc:
                    QMessageBox.critical(self, "Neural Renderer", f"Failed to render paused neural field:\n{exc}")
                    self.log(f"ERROR rendering live neural field: {exc}")
                    return
            if self.nn_grid_values is not None:
                self.log("No trained neural model – showing latest neural preview")
                if self.viz_mode == 'neural':
                    self._render_neural_volume()
                else:
                    self.log("Switch to Neural Renderer mode to view the cached preview")
                return
            if self.grid_values is not None:
                self.log("No trained neural model – displaying interpolated volume instead")
                self.nn_grid_values = None
                self.nn_bounds = None
                if self.viz_mode == 'neural':
                    self._render_neural_volume()
                else:
                    self.log("Switch to Neural Renderer mode to view the interpolated fallback")
            else:
                QMessageBox.warning(self, "Neural Renderer", "Train the neural field or compute a baseline volume first")
            return
        if resolution is None:
            if self.neural_render_res_spin:
                resolution = self.neural_render_res_spin.value()
            else:
                resolution = 256
        label = source or 'manual render'
        try:
            self.log(f"Generating neural volume at {resolution}³ ({label}) ...")
            volume = self.neural_model.generate_volume(resolution, bounds=bounds)
            meta = {
                'resolution': resolution,
                'source': label
            }
            self._cache_neural_volume(
                volume,
                meta,
                bounds=bounds,
                dataset_token=self._dataset_token
            )
            self.log(f"Neural volume ready ({label})")
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:
            QMessageBox.critical(self, "Neural Renderer", f"Failed to render neural volume:\n{exc}")


    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message)

    def show_metadata_dialog(self):
        """Display snapshot metadata in a dialog"""
        has_headers = bool(self.snapshot_metadata.get("headers"))
        has_datasets = bool(self.snapshot_metadata.get("datasets"))
        if not (has_headers or has_datasets):
            QMessageBox.information(self, "Snapshot Metadata", "Load an HDF5 snapshot to view metadata.")
            return

        dialog = MetadataDialog(self.snapshot_metadata, self)
        dialog.exec()

    def load_snapshot_file(self):
        """Load a snapshot file (HDF5 multi-part or VTK mesh)"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Snapshot",
            "",
            SNAPSHOT_FILE_FILTER
        )

        if not filename:
            return

        self.log(f"Loading {filename}...")
        self._reset_snapshot_state()
        suffix = Path(filename).suffix.lower()

        try:
            source_label = "snapshot"
            if suffix in HDF5_EXTENSIONS:
                poly, header, files = self._load_hdf5_snapshot(filename)
                self.snapshot_header = header
                self.snapshot_files = files
                file_count = len(files) or 1
                plural = "s" if file_count != 1 else ""
                source_label = f"HDF5 ({file_count} file{plural})"
            elif suffix in VTK_EXTENSIONS:
                poly, metadata = self._load_vtk_snapshot(filename)
                self.snapshot_header = metadata
                self.snapshot_files = [filename]
                dataset_type = metadata.get('dataset_type', 'VTK dataset') if metadata else 'VTK dataset'
                source_label = dataset_type
            else:
                raise ValueError(f"Unsupported file extension '{suffix}'")

            self.snapshot_path = filename
            self._refresh_metadata(poly)
            total_particles = self._ingest_polydata(poly, source_label=source_label, filename=filename)
            self.log(f"Successfully loaded {total_particles:,} particles from {source_label}")

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{exc}")
            self.log(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    # Backwards-compatible alias (old UI wiring)
    def load_hdf5_file(self):
        self.load_snapshot_file()

    def _reset_snapshot_state(self):
        self._dataset_token += 1
        self._reset_neural_state()
        self._reset_plenoxel_state(full_reset=True)
        self.snapshot_header = {}
        self.snapshot_files = []
        self.snapshot_metadata = {"headers": {}, "datasets": []}
        self.loaded_poly = None
        self.original_coords = None
        self.original_scalars = None
        self.data = None
        self.scalars = None
        self.current_field_name = None
        self.current_transform_name = None
        self.snapshot_path = None
        if hasattr(self, 'metadata_btn'):
            self.metadata_btn.setEnabled(False)
        self.file_label.setText("No file loaded")
        self.data_info_label.setText("Particles: -")
        self.subvolume.remove_visual()
        self.current_subvolume_bounds = None
        self.grid_values = None
        self.grid_bounds = None
        self._pending_grid_bounds = None
        if hasattr(self, 'compute_btn'):
            self.compute_btn.setEnabled(False)
        self.subvolume.update_extract_button_state()
        self._update_plenoxel_controls_state()

    def _load_hdf5_snapshot(self, filename):
        snapshot_data = DataLoader.load_snapshot(filename)
        ptype = self.particle_type_combo.currentText()
        if not snapshot_data or ptype not in snapshot_data:
            raise ValueError(f"No data found for {ptype}")

        poly = DataLoader.create_polydata(snapshot_data[ptype])
        if poly is None or 'Coordinates' not in poly:
            raise ValueError("Failed to create polydata from snapshot")

        header = DataLoader.get_last_header()
        files = DataLoader.get_last_files()
        return poly, header, files

    def _load_vtk_snapshot(self, filename):
        poly, metadata = load_vtk_polydata(filename)
        if poly is None or 'Coordinates' not in poly:
            raise ValueError("VTK file did not include Coordinates data")
        return poly, metadata

    def _ingest_polydata(self, poly, *, source_label, filename):
        if poly is None or 'Coordinates' not in poly:
            raise ValueError("Snapshot did not provide particle coordinates")

        coords = np.asarray(poly['Coordinates'])
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
        if coords.shape[1] != 3:
            raise ValueError("Coordinates array must be N×3")

        self.loaded_poly = poly
        self._reset_neural_state()
        self._reset_plenoxel_state(full_reset=True)

        self.original_coords = coords.copy()
        self._set_normalized_coords(coords)

        display_name = Path(filename).name
        total_particles = len(self.original_coords)
        file_count = len(self.snapshot_files) or 1
        self.file_label.setText(f"Loaded: {display_name}")
        self.data_info_label.setText(
            f"Particles: {total_particles:,} across {file_count} file(s) | Source: {source_label}"
        )
        self.compute_btn.setEnabled(True)
        self.subvolume.update_extract_button_state()
        self.subvolume.update_visual()

        field_names = sorted([k for k in poly.keys() if k != 'Coordinates'])
        if not field_names:
            raise ValueError("No scalar fields available in data")

        preferred_field = self.field_combo.currentText()
        if preferred_field not in field_names:
            preferred_field = field_names[0]
        self._update_field_combo(field_names, preferred_field)

        available_transforms = DataProcessor.get_available_transforms(
            preferred_field,
            poly[preferred_field],
            polydata=poly,
            header=self.snapshot_header
        )
        if not available_transforms:
            available_transforms = ['Linear']
        preferred_transform = self.transform_combo.currentText()
        if preferred_transform not in available_transforms:
            preferred_transform = available_transforms[0]
        self._update_transform_combo(available_transforms, preferred_transform)

        self.apply_field_transform(update_scatter=(self.viz_mode == 'scatter'))
        return total_particles

    def _update_field_combo(self, field_names, selected_field):
        """Populate field combo without emitting change events"""
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItems(field_names)
        if selected_field in field_names:
            index = self.field_combo.findText(selected_field)
            if index >= 0:
                self.field_combo.setCurrentIndex(index)
        self.field_combo.blockSignals(False)

    def _update_transform_combo(self, transforms, selected_transform):
        """Populate transform combo without emitting change events"""
        if not transforms:
            transforms = ['Linear']
        self.transform_combo.blockSignals(True)
        self.transform_combo.clear()
        self.transform_combo.addItems(transforms)
        if selected_transform in transforms:
            index = self.transform_combo.findText(selected_transform)
            if index >= 0:
                self.transform_combo.setCurrentIndex(index)
        self.transform_combo.blockSignals(False)

    def _set_normalized_coords(self, coords):
        self.coord_min = coords.min(axis=0)
        self.coord_max = coords.max(axis=0)
        span = self.coord_max - self.coord_min
        span[span == 0] = 1.0
        self.coord_span = span
        self.data = ((coords - self.coord_min) / self.coord_span).astype(np.float32)

    def _refresh_metadata(self, polydata):
        """Cache header/dataset metadata for info dialog"""
        datasets = []
        if polydata:
            for name, values in polydata.items():
                arr = np.asarray(values)
                length = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
                datasets.append({
                    "name": name,
                    "length": length,
                    "shape": tuple(arr.shape),
                    "dtype": str(arr.dtype)
                })

        self.snapshot_metadata = {
            "headers": dict(self.snapshot_header) if self.snapshot_header else {},
            "datasets": datasets
        }

        if hasattr(self, 'metadata_btn'):
            self.metadata_btn.setEnabled(bool(self.snapshot_metadata["headers"] or datasets))
        self.subvolume.update_extract_button_state()

    def _ensure_scalar_array(self, values):
        """Ensure transformed field collapses to a 1D scalar array"""
        arr = np.asarray(values)
        if arr.ndim == 1:
            return arr
        if arr.ndim >= 2:
            arr2 = arr.reshape(arr.shape[0], -1)
            if arr2.shape[1] == 1:
                return arr2[:, 0]
            return np.linalg.norm(arr2, axis=1)
        return arr.ravel()

    def on_field_changed(self, field_name):
        """Handle scalar field selection"""
        if not self.loaded_poly or not field_name:
            return
        if field_name not in self.loaded_poly:
            return

        transforms = DataProcessor.get_available_transforms(
            field_name,
            self.loaded_poly[field_name],
            polydata=self.loaded_poly,
            header=self.snapshot_header
        )
        if not transforms:
            transforms = ['Linear']

        current_transform = self.transform_combo.currentText()
        if current_transform not in transforms:
            current_transform = transforms[0]
        self._update_transform_combo(transforms, current_transform)

        self.apply_field_transform(update_scatter=(self.viz_mode == 'scatter'))

    def on_transform_changed(self, transform_name):
        """Handle transform selection"""
        if not self.loaded_poly or not transform_name:
            return
        self.apply_field_transform(update_scatter=(self.viz_mode == 'scatter'))

    def apply_field_transform(self, update_scatter=True):
        """Apply current field/transform selection to scalar arrays"""
        if not self.loaded_poly:
            return

        field_name = self.field_combo.currentText()
        if not field_name or field_name not in self.loaded_poly:
            return
        self.current_field_name = field_name

        transform_name = self.transform_combo.currentText() or 'Linear'
        self.current_transform_name = transform_name
        raw_field = self.loaded_poly[field_name]

        try:
            transformed = DataProcessor.apply_transform(
                field_name,
                raw_field,
                transform_name,
                polydata=self.loaded_poly,
                header=self.snapshot_header
            )
        except Exception:
            transformed = raw_field

        scalar_values = self._ensure_scalar_array(transformed)
        self._reset_neural_state()
        self._reset_plenoxel_state()
        self.original_scalars = scalar_values.astype(np.float64, copy=True)
        self.scalars = scalar_values.astype(np.float32, copy=True)
        self._update_neural_controls_state()
        self._update_plenoxel_controls_state()

        # Reset cached visuals so the user recomputes volume for new scalars
        if self.volume_renderer:
            self.volume_renderer.clear()
        self.grid_values = None
        self.grid_bounds = None

        scalar_min = float(np.min(self.original_scalars)) if self.original_scalars.size else 0.0
        scalar_max = float(np.max(self.original_scalars)) if self.original_scalars.size else 0.0
        self.data_info_label.setText(
            f"Particles: {len(self.original_coords):,} | Range: [{scalar_min:.2e}, {scalar_max:.2e}]"
        )
        self.log(f"Field: {field_name}, Transform: {transform_name}")
        self.log(f"Scalar range: [{scalar_min:.3e}, {scalar_max:.3e}]")

        refresh_scatter = update_scatter or (self.scatter_renderer and self.scatter_renderer.visual is not None)
        if refresh_scatter and self.original_coords is not None:
            self.visualization.create_scatter_plot()

    def compute_volume(self):
        """Start volume interpolation"""
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        method = self.method_combo.currentText()
        resolution = self.resolution_spin.value()
        epsilon = self.epsilon_spin.value()

        try:
            interp_points, interp_scalars, bounds = self.subvolume.prepare_interpolation_data()
        except ValueError as exc:
            QMessageBox.warning(self, "Subvolume", str(exc))
            self.log(f"Subvolume aborted: {exc}")
            return
        
        # Check GPU limits
        if self.max_3d_texture_size and resolution > self.max_3d_texture_size:
            reply = QMessageBox.warning(
                self, "GPU Limitation",
                f"Your GPU only supports {self.max_3d_texture_size}³ textures!\n"
                f"Requested: {resolution}³\n\n"
                f"The volume will likely fail to render.\n"
                f"Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Warn about high resolution
        if resolution >= 1024:
            reply = QMessageBox.warning(
                self, "High Resolution Warning",
                f"1024³ requires 4GB of memory and may:\n"
                f"• Crash due to insufficient RAM/VRAM\n"
                f"• Take a very long time (>10 minutes)\n"
                f"• Produce many invalid/NaN values\n\n"
                f"Recommended: Try 512 first.\n\n"
                f"Continue with 1024?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        total_points = len(interp_points)
        self.log(
            f"Starting interpolation: {method}, resolution={resolution}³, epsilon={epsilon}, particles={total_points:,}"
        )
        if bounds:
            mins, maxs = bounds
            self.log(
                "Subvolume bounds: "
                f"X[{mins[0]:.2f}, {maxs[0]:.2f}] "
                f"Y[{mins[1]:.2f}, {maxs[1]:.2f}] "
                f"Z[{mins[2]:.2f}, {maxs[2]:.2f}]"
            )
        self._pending_grid_bounds = bounds
        self.compute_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = self.interpolator_factory.create_worker(
            interp_points,
            interp_scalars,
            resolution,
            method,
            epsilon,
            bounds
        )
        self.worker.progress.connect(self.on_interpolation_progress)
        self.worker.status.connect(self.log)
        self.worker.error.connect(self.on_interpolation_error)
        self.worker.finished.connect(self.on_interpolation_finished)
        self.worker.start()

    def on_interpolation_progress(self, value):
        """Handle progress updates including indeterminate mode"""
        if value == -1:
            # Indeterminate (pulsing) progress during construction
            self.progress_bar.setRange(0, 0)
        else:
            # Normal progress
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)

    def on_interpolation_error(self, error_msg):
        """Handle interpolation errors"""
        self.log(error_msg)
        QMessageBox.critical(self, "Interpolation Failed", error_msg)
        self.compute_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)  # Reset from indeterminate

    def on_interpolation_finished(self, grid_values, elapsed, stats):
        """Handle completed interpolation with detailed statistics"""
        self.grid_values = grid_values
        self.grid_bounds = self._pending_grid_bounds
        self.current_stats = stats
        
        # Reset progress bar from indeterminate if needed
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        
        # Log completion
        accel = stats.get('acceleration', 'CPU')
        self.log(f"✓ Interpolation completed in {elapsed:.2f}s ({accel})")
        self.log(f"Memory: {grid_values.nbytes / (1024**2):.1f} MB")
        
        # Log statistics
        self.log(f"--- Data Quality Report ---")
        self.log(f"Raw range: [{stats['raw_min']:.3e}, {stats['raw_max']:.3e}]")
        self.log(f"Positive values: {stats['positive_count']:,} ({stats['positive_pct']:.1f}%)")
        self.log(f"Valid after log: {stats['valid_count']:,} ({stats['valid_pct']:.1f}%)")
        self.log(f"NaN values: {stats['nan_count']:,}")
        self.log(f"Inf values: {stats['inf_count']:,}")
        self.log(f"Log range: [{stats['log_min']:.2f}, {stats['log_max']:.2f}] (span: {stats['log_range']:.2f})")
        
        # Warn if data quality is poor
        if stats['valid_pct'] < 50:
            self.log("⚠⚠ WARNING: Less than 50% valid data!")
            QMessageBox.warning(
                self, "Poor Data Quality",
                f"Only {stats['valid_pct']:.1f}% of voxels have valid values.\n\n"
                f"This usually means:\n"
                f"• Resolution too high for sparse data\n"
                f"• Interpolation method not suitable\n"
                f"• Need more source particles\n\n"
                f"Consider:\n"
                f"• Lower resolution (256 or 512)\n"
                f"• Try 'nearest' method\n"
                f"• Use more particles"
            )
        
        self.visualization.update_volume_visual()
        if self.viz_mode == 'neural':
            self._render_neural_volume()
        self.compute_btn.setEnabled(True)

    def update_camera(self):
        """Update camera parameters from spinboxes - FIXED"""
        if not self.camera_controller:
            return

        params = {name: spinbox.value() for name, spinbox in self.cam_params.items()}
        self._suppress_camera_event = True
        self.camera_controller.update_from_params(params)
        self.view.camera.view_changed()
        self._suppress_camera_event = False
        self.canvas.update()

    def set_camera_preset(self, azimuth, elevation):
        """Set camera to preset view"""
        self.cam_params['azimuth'].setValue(azimuth)
        self.cam_params['elevation'].setValue(elevation)
        self.update_camera()

    def reset_camera(self):
        """Reset camera to default values"""
        for param, value in CameraController.DEFAULTS.items():
            self.cam_params[param].setValue(value)
        self.update_camera()

    def save_camera_config(self):
        """Save camera configuration"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Camera Config", "", "JSON Files (*.json)"
        )
        
        if not filename:
            return

        config = self.camera_controller.to_config()
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.log(f"Camera config saved")

    def load_camera_config(self):
        """Load camera configuration"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Config", "", "JSON Files (*.json)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            self.camera_controller.apply_config(config)
            self._sync_camera_controls(self.camera_controller.capture_state())
            self.view.camera.view_changed()
            self.canvas.update()
            self.log(f"Camera config loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def save_screenshot(self):
        """Save current view as PNG"""
        scatter_ready = self.scatter_renderer and self.scatter_renderer.visual is not None
        if self.grid_values is None and not scatter_ready:
            QMessageBox.warning(self, "Warning", "No visualization to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Files (*.png)"
        )
        
        if not filename:
            return
        
        native = getattr(self.canvas, 'native', None)
        current_width = native.width() if native else 0
        current_height = native.height() if native else 0
        resolution_dialog = ResolutionDialog(current_width, current_height, self)
        if resolution_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        export_width, export_height = resolution_dialog.get_resolution()
        self.log(f"Exporting screenshot at {export_width}x{export_height}")
        
        try:
            img = self.visualization.render_image_for_export(
                export_width,
                export_height,
                hide_reference_cube=True
            )
            from PIL import Image
            Image.fromarray(img).save(filename)
            self.log(f"Screenshot saved: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def save_npy(self):
        """Save volume data as NPY"""
        if self.grid_values is None:
            QMessageBox.warning(self, "Warning", "No volume to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save NPY", "", "NumPy Files (*.npy)"
        )
        
        if not filename:
            return
        
        try:
            np.save(filename, self.grid_values)
            self.log(f"NPY saved ({self.grid_values.nbytes/(1024**2):.1f} MB)")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def save_npz(self):
        """Save volume data as NPZ with metadata"""
        if self.grid_values is None:
            QMessageBox.warning(self, "Warning", "No volume to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save NPZ", "", "NumPy Files (*.npz)"
        )
        
        if not filename:
            return
        
        try:
            metadata = {
                'method': self.method_combo.currentText(),
                'resolution': self.resolution_spin.value(),
                'epsilon': self.epsilon_spin.value(),
                'particles': len(self.data) if self.data is not None else 0,
                'volume_extent': VOLUME_EXTENT,
                'stats': self.current_stats if self.current_stats else {}
            }
            
            np.savez_compressed(filename,
                              volume=self.grid_values,
                              metadata=np.array([json.dumps(metadata)]))
            
            import os
            size_mb = os.path.getsize(filename) / (1024**2)
            ratio = self.grid_values.nbytes / os.path.getsize(filename)
            self.log(f"NPZ saved ({size_mb:.1f} MB, {ratio:.1f}x compression)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def batch_render_dialog(self):
        """Show batch render configuration dialog"""
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        # Create and show the dialog (non-modal so it stays open)
        self.batch_dialog = BatchConfigDialog(
            self,
            current_resolution=self.resolution_spin.value(),
            current_method=self.method_combo.currentText(),
            current_epsilon=self.epsilon_spin.value(),
            allow_natural_neighbor=HAS_NATURAL_NEIGHBOR,
            batch_runner=self.run_batch_render_with_dialog,
        )
        
        # Show as non-modal
        self.batch_dialog.show()
        self.log("Batch render dialog opened - configure and click 'Start Batch Render'")

    def run_batch_render_with_dialog(self, configs, output_dir, pattern, canvas_size, dialog):
        """Execute batch rendering with FIXED canvas size"""
        QApplication.processEvents()
        
        self.log(f"=== BATCH RENDER STARTED ===")
        self.log(f"Configurations: {len(configs)}")
        self.log(f"Canvas size: {canvas_size[0]}x{canvas_size[1]} (FIXED)")
        self.log(f"Output: {output_dir}")
        
        QApplication.processEvents()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # SAVE ORIGINAL CANVAS SIZE
        original_size = self.canvas.size
        
        # SET FIXED CANVAS SIZE FOR BATCH
        self.canvas.native.resize(canvas_size[0], canvas_size[1])
        QApplication.processEvents()
        self.log(f"Canvas resized to {canvas_size[0]}x{canvas_size[1]}")
        
        # Save camera config from actual view.camera (captures interactive scatter view)
        cam_config = self.camera_controller.to_config()
        cam_config['canvas_size'] = canvas_size
        cam_file = output_path / "camera_config.json"
        with open(cam_file, 'w') as f:
            json.dump(cam_config, f, indent=2)
        self.log(f"Camera config saved to {cam_file}")
        
        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(configs),
            'canvas_size': canvas_size,
            'data_file': self.snapshot_path,
            'particle_count': len(self.data) if self.data is not None else 0,
            'configs': configs
        }
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.log(f"Batch summary saved to {summary_file}")
        
        QApplication.processEvents()
        
        # Process each configuration
        for idx, config in enumerate(configs):
            self.log(f"\n=== Processing config {idx+1}/{len(configs)} ===")
            
            # Update overall progress FIRST
            config_name = f"{config['method']} @ {config['resolution']}³"
            dialog.update_overall_progress(idx, len(configs), config_name)
            
            # Update row status to "Running"
            dialog.update_row_status(
                idx,
                "▶ Running...",
                progress=0,
                style="color: blue; font-size: 10px; font-weight: bold;"
            )
            
            QApplication.processEvents()
            time.sleep(0.1)
            
            self.log(f"Config: {config['method']} @ {config['resolution']}³, ε={config['epsilon']:.2f}")
            
            # Format filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epsilon_str = f"{config['epsilon']:.2f}".replace('.', 'p')
            filename = pattern.format(
                method=config['method'],
                resolution=config['resolution'],
                epsilon=epsilon_str,
                colormap=config['colormap'],
                index=idx,
                timestamp=timestamp
            )
            
            # Update GUI controls to match config
            self.method_combo.setCurrentText(config['method'])
            self.resolution_spin.setValue(config['resolution'])
            self.epsilon_spin.setValue(config['epsilon'])
            self.cmap_combo.setCurrentText(config['colormap'])
            QApplication.processEvents()
            
            # Compute volume (synchronously for batch)
            self.log(f"Computing volume...")
            worker = self.interpolator_factory.create_worker(
                self.data,
                self.scalars,
                config['resolution'],
                config['method'],
                config['epsilon']
            )
            
            # Connect to capture results
            result_holder = {'grid': None, 'stats': None, 'error': None}
            
            def capture_result(grid, elapsed, stats):
                result_holder['grid'] = grid
                result_holder['stats'] = stats
            
            def capture_error(msg):
                result_holder['error'] = msg
            
            # Connect progress to dialog row
            def update_row_progress(progress_value):
                dialog.update_row_status(idx, "▶ Running...", progress=progress_value,
                                        style="color: blue; font-size: 10px; font-weight: bold;")
                QApplication.processEvents()
            
            worker.finished.connect(capture_result)
            worker.error.connect(capture_error)
            worker.progress.connect(update_row_progress)
            worker.status.connect(self.log)
            
            # Run synchronously
            self.log(f"Starting worker.run()...")
            worker.run()
            QApplication.processEvents()
            
            if result_holder['error']:
                self.log(f"ERROR: {result_holder['error']}")
                dialog.update_row_status(
                    idx,
                    "✗ Failed",
                    progress=None,
                    style="color: red; font-size: 10px; font-weight: bold;"
                )
                QApplication.processEvents()
                continue
            
            if result_holder['grid'] is None:
                self.log(f"ERROR: No result from worker")
                dialog.update_row_status(
                    idx,
                    "✗ No Data",
                    progress=None,
                    style="color: red; font-size: 10px; font-weight: bold;"
                )
                QApplication.processEvents()
                continue
            
            # Update volume
            self.grid_values = result_holder['grid']
            self.current_stats = result_holder['stats']
            self.visualization.update_volume_visual()
            QApplication.processEvents()
            
            # ENSURE WE'RE IN VOLUME MODE FOR RENDERING
            if self.viz_mode != 'volume':
                self.visualization.set_mode('volume')
                QApplication.processEvents()
            
            # Save outputs
            if config['save_png']:
                png_file = output_path / f"{filename}.png"
                try:
                    img = self.visualization.render_image_for_export(
                        canvas_size[0],
                        canvas_size[1],
                        hide_reference_cube=True
                    )
                    from PIL import Image
                    Image.fromarray(img).save(png_file)
                    self.log(f"✓ Saved PNG: {png_file.name} ({canvas_size[0]}x{canvas_size[1]})")
                except Exception as e:
                    self.log(f"✗ Failed to save PNG: {e}")
            
            if config['save_npy']:
                npy_file = output_path / f"{filename}.npy"
                try:
                    np.save(npy_file, self.grid_values)
                    self.log(f"✓ Saved NPY: {npy_file.name}")
                except Exception as e:
                    self.log(f"✗ Failed to save NPY: {e}")
            
            # Save metadata for this render
            meta_file = output_path / f"{filename}_metadata.json"
            metadata = {
                'config': config,
                'stats': result_holder['stats'],
                'filename': filename,
                'canvas_size': canvas_size,
                'timestamp': datetime.now().isoformat()
            }
            with open(meta_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.log(f"✓ Saved metadata: {meta_file.name}")
            
            # Mark as complete
            dialog.update_row_status(
                idx,
                "✓ Complete",
                progress=None,
                style="color: green; font-size: 10px; font-weight: bold;"
            )
            
            QApplication.processEvents()
        
        # RESTORE ORIGINAL CANVAS SIZE
        self.canvas.native.resize(original_size[0], original_size[1])
        QApplication.processEvents()
        self.log(f"Canvas restored to {original_size[0]}x{original_size[1]}")
        
        # Update final progress
        dialog.update_overall_progress(len(configs), len(configs), "")
        dialog.on_batch_complete()
        
        self.log(f"\n=== BATCH RENDER COMPLETED ===")
        self.log(f"All outputs saved to: {output_dir}")
        QApplication.processEvents()
        
        QMessageBox.information(
            self, "Batch Render Complete",
            f"Successfully rendered {len(configs)} configurations.\n\n"
            f"Canvas size: {canvas_size[0]}x{canvas_size[1]}\n"
            f"Outputs saved to:\n{output_dir}"
        )


def main():
    app = QApplication(sys.argv)
    window = VolumeRenderingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()