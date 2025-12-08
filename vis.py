import sys
import json
import os
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# WSL / Linux compatibility - MUST be set before any VTK/Qt imports
# =============================================================================
def _setup_linux_rendering():
    """Configure rendering backend for Linux/WSL compatibility."""
    if sys.platform.startswith('linux'):
        # Check if running under WSL
        is_wsl = False
        try:
            with open('/proc/version', 'r') as f:
                is_wsl = 'microsoft' in f.read().lower()
        except Exception:
            pass
        
        # Force software rendering if DISPLAY issues or WSL
        if is_wsl or os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1':
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
            os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
            # Use EGL/OSMesa for offscreen if available
            os.environ.setdefault('PYVISTA_OFF_SCREEN', 'false')
        
        # Ensure XDG_RUNTIME_DIR exists (required by Qt on some systems)
        xdg_runtime = os.environ.get('XDG_RUNTIME_DIR')
        if not xdg_runtime or not os.path.isdir(xdg_runtime):
            fallback = f'/tmp/runtime-{os.getuid()}'
            os.makedirs(fallback, mode=0o700, exist_ok=True)
            os.environ['XDG_RUNTIME_DIR'] = fallback
        
        # Qt platform selection with fallback
        # Try xcb first, but if libxcb-cursor0 is missing, fall back to offscreen
        if 'QT_QPA_PLATFORM' not in os.environ:
            # Check if libxcb-cursor is available
            import subprocess
            try:
                result = subprocess.run(
                    ['ldconfig', '-p'], 
                    capture_output=True, text=True, timeout=5
                )
                has_xcb_cursor = 'libxcb-cursor' in result.stdout
            except Exception:
                has_xcb_cursor = False
            
            if has_xcb_cursor:
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
            else:
                # Fallback: try xcb anyway, Qt may handle it
                # User can override with QT_QPA_PLATFORM=offscreen if needed
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
                print("Note: libxcb-cursor0 may be missing. If the app fails to start, run:")
                print("  sudo apt install libxcb-cursor0")
                print("Or set: export QT_QPA_PLATFORM=offscreen")
        
        # Disable Qt's GPU features that may conflict
        os.environ.setdefault('QT_QUICK_BACKEND', 'software')

_setup_linux_rendering()

# Set Qt API for pyvistaqt before importing it
os.environ["QT_API"] = "pyqt6"

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QDialog, QScrollArea, QPushButton, QFrame
)
from PyQt6.QtGui import QIcon

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
from renderers.volume import PyVistaVolumeRenderer
from renderers.arepovtk import ArepoVTKRenderer
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

# Configure PyVista for Qt
pv.set_plot_theme('dark')
# Disable multisampling which interferes with volume rendering in QtInteractor
pv.global_theme.multi_samples = 1

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
        self.setWindowTitle("Scalar Field Resampler - Volume Rendering")
        self.setWindowIcon(QIcon("icons/app_icon.png"))
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

        # Assemble Qt sidebars + plotter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Build PyVista plotter embedded in Qt using a frame
        self._plotter_frame = QFrame()
        plotter_layout = QVBoxLayout(self._plotter_frame)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self._plotter_frame)
        plotter_layout.addWidget(self.plotter.interactor)
        self.plotter.set_background('black')
        # Disable anti-aliasing - it interferes with volume rendering in QtInteractor
        self.plotter.disable_anti_aliasing()
        
        # Store references for compatibility with old code patterns
        self.canvas = self.plotter  # For backward compatibility
        self.view = self.plotter   # For backward compatibility
        
        # Set up camera
        self.camera_controller = CameraController(self.plotter, VOLUME_EXTENT)
        self.camera_controller.reset()

        # Renderer instances share the same plotter
        self.volume_renderer = PyVistaVolumeRenderer(self.plotter, VOLUME_EXTENT, self.transfer_function)
        self.neural_renderer = PyVistaVolumeRenderer(self.plotter, VOLUME_EXTENT, self.transfer_function)
        self.scatter_renderer = ScatterPlotRenderer(self.plotter, VOLUME_EXTENT, self.transfer_function)
        self.arepovtk_renderer = ArepoVTKRenderer(self.plotter, VOLUME_EXTENT, self.transfer_function)
        self.reference_cube = ReferenceCube(self.plotter, VOLUME_EXTENT)
        self.reference_cube.render(visible=True)
        self.plenoxel_renderer = PlenoxelRenderer(self.plotter, VOLUME_EXTENT)
        self.plenoxel_nodes = None
        self.plenoxel_stats = None
        self.plenoxel_volume = None
        self.plenoxel_visible = False

        # Subvolume controller needs a plotter before wiring the UI
        self.subvolume = SubvolumeController(self, VOLUME_EXTENT)
        self.visualization = VisualizationController(self, VOLUME_EXTENT)

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

        # Add the PyVista plotter frame
        self._plotter_frame.setMinimumSize(640, 640)
        right_layout.addWidget(self._plotter_frame, 1)

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
        
        # Visualization settings button
        vol_settings_btn = QPushButton("Visualization Settings")
        vol_settings_btn.clicked.connect(self.open_volume_render_settings)
        export_layout.addWidget(vol_settings_btn)
        
        # Neural network settings button
        neural_settings_btn = QPushButton("Neural Network Settings")
        neural_settings_btn.clicked.connect(self.open_neural_network_settings)
        export_layout.addWidget(neural_settings_btn)
        
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
        """Connect VTK camera observers for camera change events."""
        try:
            # Get the underlying VTK render window interactor
            iren = self.plotter.iren
            if iren is None:
                return
            
            # Get the actual VTK interactor object
            vtk_iren = iren.interactor if hasattr(iren, 'interactor') else iren
            if vtk_iren is None:
                return
            
            # Add observer for interaction events to sync camera controls
            def on_interaction_end(obj, event):
                if self._suppress_camera_event:
                    return
                if self.camera_controller:
                    state = self.camera_controller.capture_state()
                    self._sync_camera_controls(state)
            
            # Observe end of interaction to update spinboxes
            vtk_iren.AddObserver('EndInteractionEvent', on_interaction_end)
            
            print("[Camera] Connected VTK camera observers")
        except Exception as e:
            print(f"[Camera] Failed to connect camera events: {e}")

    def open_volume_render_settings(self):
        """Open the visualization settings dialog."""
        from ui.dialogs import VolumeRenderSettingsDialog
        
        # Get current settings from volume renderer
        current_settings = {}
        if self.volume_renderer:
            current_settings = {
                'method': getattr(self.volume_renderer, 'render_method', 'mip'),
                'interpolation': getattr(self.volume_renderer, 'interpolation', 'trilinear'),
                'step_size': getattr(self.volume_renderer, 'relative_step_size', 0.5),
                'opacity_scale': getattr(self.volume_renderer, 'opacity_scale', 1.0),
                'shade': getattr(self.volume_renderer, 'shade', False),
            }
        
        # Get current scatter settings
        if self.scatter_renderer:
            current_settings['scatter'] = {
                'point_size': getattr(self.scatter_renderer, 'point_size', 2),
                'render_as_spheres': getattr(self.scatter_renderer, 'render_as_spheres', True),
                'show_scalar_bar': getattr(self.scatter_renderer, 'show_scalar_bar', True),
            }
        
        dialog = VolumeRenderSettingsDialog(self, current_settings)
        dialog.show()

    def open_neural_network_settings(self):
        """Open the neural network settings dialog."""
        from ui.dialogs import NeuralNetworkSettingsDialog
        
        # Get current settings from neural params and stored arch settings
        current_settings = {}
        
        # Training params from panel
        if hasattr(self, 'neural_params'):
            params = self.neural_params
            if 'steps' in params:
                current_settings['steps'] = params['steps'].value()
            if 'batch_size' in params:
                current_settings['batch_size'] = params['batch_size'].value()
            if 'lr' in params:
                current_settings['lr'] = params['lr'].value()
            if 'log_interval' in params:
                current_settings['log_interval'] = params['log_interval'].value()
            if 'preview_interval' in params:
                current_settings['preview_interval'] = params['preview_interval'].value()
            if 'preview_resolution' in params:
                current_settings['preview_resolution'] = params['preview_resolution'].value()
            if 'inference_batch' in params:
                current_settings['inference_batch'] = params['inference_batch'].value()
        
        # Render/display resolution
        if hasattr(self, 'neural_render_res_spin'):
            current_settings['render_resolution'] = self.neural_render_res_spin.value()
        if hasattr(self, 'neural_display_res_spin'):
            current_settings['display_resolution'] = self.neural_display_res_spin.value()
        
        # Architecture settings (from stored settings or original defaults)
        arch_defaults = {
            'n_levels': 16,
            'n_features_per_level': 2,
            'log2_hashmap_size': 18,
            'base_resolution': 16,
            'per_level_scale': 1.5,
            'n_hidden_layers': 2,
            'n_neurons': 64,
        }
        arch_settings = getattr(self, '_neural_arch_settings', arch_defaults)
        current_settings.update(arch_settings)
        
        dialog = NeuralNetworkSettingsDialog(self, current_settings)
        dialog.show()

    def _detect_max_3d_texture_size(self):
        # PyVista/VTK doesn't have this limitation as prominently
        # Return a reasonable default
        try:
            # Try to get OpenGL info through VTK
            import vtk
            return 2048  # Safe default for most GPUs
        except Exception:
            return 2048

    def _on_camera_view_changed(self, event):
        if self._suppress_camera_event:
            return
        if not self.camera_controller:
            return
        state = self.camera_controller.capture_state()
        self._sync_camera_controls(state)

    def _on_transfer_function_live_update(self, *_):
        """Refresh active visuals when transfer function dialog changes"""
        if self.volume_renderer and self.volume_renderer.actor is not None:
            self.volume_renderer.update_transfer_function(self.transfer_function)
        if self.neural_renderer and self.neural_renderer.actor is not None:
            self.neural_renderer.update_transfer_function(self.transfer_function)
        if self.scatter_renderer:
            self.scatter_renderer.update_transfer_function(self.transfer_function)
            # Update scatter plot in real-time (always recreate to apply new colormap)
            if self.scatter_renderer.actor is not None:
                self.visualization.create_scatter_plot(force_visible=(self.viz_mode == 'scatter'))
        self.plotter.render()

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
        if self.volume_renderer and self.volume_renderer.actor is not None:
            self.plotter.render()

    def on_volume_threshold_changed(self, value: float):
        if self.volume_renderer:
            self.volume_renderer.update_render_settings(threshold=value)
        if self.volume_renderer and self.volume_renderer.actor is not None:
            self.plotter.render()

    # --- Scatter plot settings handlers ----------------------------------------

    def on_scatter_point_size_changed(self, value: int):
        """Handle point size change for scatter plot."""
        if self.scatter_renderer:
            self.scatter_renderer.set_point_size(value)
            if self.scatter_renderer.actor is not None:
                # Re-render to apply new point size
                self.visualization.create_scatter_plot(force_visible=(self.viz_mode == 'scatter'))

    def on_scatter_spheres_toggled(self, enabled: bool):
        """Handle render-as-spheres toggle for scatter plot."""
        if self.scatter_renderer:
            self.scatter_renderer.set_render_as_spheres(enabled)
            if self.scatter_renderer.actor is not None:
                # Re-render to apply sphere mode
                self.visualization.create_scatter_plot(force_visible=(self.viz_mode == 'scatter'))

    def on_scatter_scalar_bar_toggled(self, enabled: bool):
        """Handle scalar bar visibility toggle for scatter plot."""
        if self.scatter_renderer:
            self.scatter_renderer.set_show_scalar_bar(enabled)
            if self.scatter_renderer.actor is not None:
                # Re-render to apply scalar bar setting
                self.visualization.create_scatter_plot(force_visible=(self.viz_mode == 'scatter'))

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

    def on_bounding_box_toggled(self, checked: bool):
        """Toggle visibility of the main bounding box (reference cube)."""
        if self.reference_cube and self.reference_cube.actor:
            self.reference_cube.actor.SetVisibility(checked)
            self.plotter.render()

    def on_subvolume_box_toggled(self, checked: bool):
        """Toggle visibility of the subvolume selection box."""
        if self.subvolume_box_visual is not None:
            try:
                self.subvolume_box_visual.SetVisibility(checked)
                self.plotter.render()
            except Exception:
                pass

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
            self.plotter.render()
            return
        self.plenoxel_renderer.render(self.plenoxel_nodes, self.plenoxel_stats, visible=True)
        self.plotter.render()


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
        
        # Set histogram data for background visualization
        dialog.opacity_editor.set_histogram(self.original_scalars)
        
        dialog.transferFunctionChanged.connect(self._on_transfer_function_live_update)

        if dialog.exec():
            # Apply final changes and refresh
            self.transfer_function = dialog.tf
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
        # Enable save button only when a trained model exists
        if hasattr(self, 'neural_save_btn') and self.neural_save_btn:
            self.neural_save_btn.setEnabled(self.neural_model is not None)

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

    def save_neural_weights(self):
        """Save trained neural network weights to a file."""
        if self.neural_model is None:
            QMessageBox.warning(self, "No Model", "No trained neural model to save.")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neural Network Weights",
            "",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        if not filepath:
            return
        
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        
        try:
            self.neural_model.save_weights(filepath)
            self.log(f"Neural weights saved to: {filepath}")
            if self.neural_status_label:
                self.neural_status_label.setText("Status: weights saved ✓")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save weights:\n{e}")
            self.log(f"Failed to save neural weights: {e}")

    def load_neural_weights(self):
        """Load pre-trained neural network weights from a file."""
        from PyQt6.QtWidgets import QFileDialog
        from neural.trainer import NeuralFieldModel, HAS_TORCH, HAS_TCNN
        
        if not HAS_TORCH or not HAS_TCNN:
            QMessageBox.warning(self, "Missing Dependencies", 
                "PyTorch and tinycudann are required to load neural weights.")
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Neural Network Weights",
            "",
            "PyTorch Model (*.pt);;All Files (*)"
        )
        if not filepath:
            return
        
        try:
            low_memory = bool(self.neural_low_memory_check and self.neural_low_memory_check.isChecked())
            self.neural_model = NeuralFieldModel.load_weights(filepath, low_memory=low_memory)
            self.log(f"Neural weights loaded from: {filepath}")
            
            if self.neural_status_label:
                self.neural_status_label.setText("Status: weights loaded ✓")
            
            # Enable render and save buttons
            self._update_neural_controls_state()
            if self.neural_render_btn:
                self.neural_render_btn.setEnabled(True)
            if self.neural_save_btn:
                self.neural_save_btn.setEnabled(True)
            if self.neural_reset_btn:
                self.neural_reset_btn.setEnabled(True)
            
            QMessageBox.information(self, "Weights Loaded", 
                f"Neural network weights loaded successfully.\n\n"
                f"Click 'Render Neural Volume' to generate the volume.")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to load weights:\n{e}")
            self.log(f"Failed to load neural weights: {e}")

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
        
        # Include custom architecture settings if set
        if hasattr(self, '_neural_arch_settings') and self._neural_arch_settings:
            config['architecture'] = self._neural_arch_settings
        
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
        # Don't transpose - keep same axis orientation as scatter plot
        grid_for_render = np.ascontiguousarray(grid)
        self.neural_renderer.render(
            grid_for_render,
            visible=(self.viz_mode == 'neural'),
            bounds=bounds
        )
        self.plotter.render()

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
        # Use faster axis parameter for min/max
        self.coord_min = np.min(coords, axis=0)
        self.coord_max = np.max(coords, axis=0)
        span = self.coord_max - self.coord_min
        span[span == 0] = 1.0
        self.coord_span = span
        # In-place operations where possible
        self.data = (coords - self.coord_min)
        self.data /= self.coord_span
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

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

    def on_particle_type_changed(self, ptype):
        """Handle particle type change - reload data for the new particle type"""
        if not self.snapshot_path:
            return

        # Only HDF5 files support particle types
        suffix = Path(self.snapshot_path).suffix.lower()
        if suffix not in HDF5_EXTENSIONS:
            self.log(f"Particle type selection not supported for {suffix} files")
            return

        self.log(f"Switching to {ptype}...")

        try:
            # Reload the snapshot data and select new particle type
            snapshot_data = DataLoader.load_snapshot(self.snapshot_path)
            if not snapshot_data or ptype not in snapshot_data:
                self.log(f"WARNING: {ptype} not found in snapshot")
                QMessageBox.warning(self, "Warning", f"{ptype} not found in this snapshot")
                return

            poly = DataLoader.create_polydata(snapshot_data[ptype])
            if poly is None or 'Coordinates' not in poly:
                raise ValueError(f"Failed to create polydata for {ptype}")

            # Re-ingest the new particle type data
            self._refresh_metadata(poly)
            total_particles = self._ingest_polydata(
                poly,
                source_label=f"HDF5 ({ptype})",
                filename=self.snapshot_path
            )
            self.log(f"Loaded {total_particles:,} particles for {ptype}")

            # If in scatter mode, immediately update the visualization
            if self.viz_mode == 'scatter' and self.data is not None and self.scalars is not None:
                self.visualization._update_scatter()

        except Exception as exc:
            self.log(f"ERROR switching particle type: {exc}")
            import traceback
            traceback.print_exc()

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
        # Store as float32 to reduce memory usage
        self.original_scalars = scalar_values.astype(np.float32, copy=True)
        self.scalars = self.original_scalars.copy()  # Separate copy for modifications
        self._update_neural_controls_state()
        self._update_plenoxel_controls_state()

        # Reset cached visuals so the user recomputes volume for new scalars
        if self.volume_renderer:
            self.volume_renderer.clear()
        self.grid_values = None
        self.grid_bounds = None

        scalar_min = float(self.original_scalars.min()) if self.original_scalars.size else 0.0
        scalar_max = float(self.original_scalars.max()) if self.original_scalars.size else 0.0
        self.data_info_label.setText(
            f"Particles: {len(self.original_coords):,} | Range: [{scalar_min:.2e}, {scalar_max:.2e}]"
        )
        self.log(f"Field: {field_name}, Transform: {transform_name}")
        self.log(f"Scalar range: [{scalar_min:.3e}, {scalar_max:.3e}]")

        refresh_scatter = update_scatter or (self.scatter_renderer and self.scatter_renderer.actor is not None)
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
        self._suppress_camera_event = False
        self.plotter.render()

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
            self.plotter.render()
            self.log(f"Camera config loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{str(e)}")

    def save_screenshot(self):
        """Save current view as PNG"""
        scatter_ready = self.scatter_renderer and self.scatter_renderer.actor is not None
        if self.grid_values is None and not scatter_ready:
            QMessageBox.warning(self, "Warning", "No visualization to save")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Files (*.png)"
        )
        
        if not filename:
            return
        
        current_width = self.plotter.window_size[0]
        current_height = self.plotter.window_size[1]
        resolution_dialog = ResolutionDialog(current_width, current_height, self)
        if resolution_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        export_width, export_height = resolution_dialog.get_resolution()
        self.log(f"Exporting screenshot at {export_width}x{export_height}")
        
        try:
            img = self.visualization.render_image_for_export(
                export_width,
                export_height,
                hide_reference_cube=True,
                hide_subvolume_box=True
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
        
        # Determine available modes
        has_neural = (self.neural_model is not None or 
                      (self.neural_trainer is not None and self.neural_trainer.model is not None))
        has_plenoxel = (hasattr(self, 'plenoxel_nodes') and 
                        self.plenoxel_nodes is not None)
        
        # Create and show the dialog (non-modal so it stays open)
        self.batch_dialog = BatchConfigDialog(
            self,
            current_resolution=self.resolution_spin.value(),
            current_method=self.method_combo.currentText(),
            current_epsilon=self.epsilon_spin.value(),
            allow_natural_neighbor=HAS_NATURAL_NEIGHBOR,
            batch_runner=self.run_batch_render_with_dialog,
            has_neural_model=has_neural,
            has_plenoxel=has_plenoxel,
        )
        
        # Show as non-modal
        self.batch_dialog.show()
        self.log("Batch render dialog opened - configure and click 'Start Batch Render'")

    def run_batch_render_with_dialog(self, configs, output_dir, pattern, canvas_size, dialog):
        """Execute batch rendering with FIXED canvas size for all render modes.
        
        Supports: volume, scatter, neural, plenoxel modes.
        All renders use the same camera position for metric comparison.
        """
        QApplication.processEvents()
        
        self.log(f"=== BATCH RENDER STARTED ===")
        self.log(f"Configurations: {len(configs)}")
        self.log(f"Canvas size: {canvas_size[0]}x{canvas_size[1]} (FIXED)")
        self.log(f"Output: {output_dir}")
        
        QApplication.processEvents()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # SAVE ORIGINAL STATE
        original_size = self.plotter.window_size
        original_mode = self.viz_mode
        
        # SET FIXED PLOTTER SIZE FOR BATCH
        self.plotter.window_size = [canvas_size[0], canvas_size[1]]
        QApplication.processEvents()
        self.log(f"Plotter resized to {canvas_size[0]}x{canvas_size[1]}")
        
        # Save camera config from plotter camera (for pixel-perfect reproduction)
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
            mode = config.get('mode', 'volume')
            method = config.get('method', 'linear')
            resolution = config.get('resolution', 256)
            label = config.get('label', '')
            
            self.log(f"\n=== Processing config {idx+1}/{len(configs)} ===")
            self.log(f"Mode: {mode}, Method: {method}, Resolution: {resolution}")
            
            # Update overall progress FIRST
            config_name = f"{mode}: {label or method} @ {resolution}³"
            dialog.update_overall_progress(idx, len(configs), config_name)
            
            # Update row status to "Running"
            dialog.update_row_status(
                idx,
                "▶ Running...",
                progress=-1,  # Indeterminate
                style="color: blue; font-size: 10px; font-weight: bold;"
            )
            
            QApplication.processEvents()
            
            # Apply colormap
            self.transfer_function.set_colormap(config['colormap'])
            QApplication.processEvents()
            
            # Format filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epsilon_str = f"{config['epsilon']:.2f}".replace('.', 'p')
            max_depth = config.get('max_depth', 5)
            min_points = config.get('min_points', 500)
            try:
                filename = pattern.format(
                    mode=mode,
                    method=method,
                    resolution=resolution,
                    epsilon=epsilon_str,
                    max_depth=max_depth,
                    min_points=min_points,
                    colormap=config['colormap'],
                    label=label or f"config{idx}",
                    index=idx,
                    timestamp=timestamp
                )
            except KeyError:
                # Fallback if pattern has missing keys
                filename = f"render_{mode}_{method}_{resolution}_{idx}"
            
            success = False
            grid_to_save = None
            stats_to_save = None
            
            try:
                if mode == 'volume':
                    success, grid_to_save, stats_to_save = self._batch_render_volume(
                        config, idx, dialog
                    )
                elif mode == 'scatter':
                    success = self._batch_render_scatter(config, idx, dialog)
                    grid_to_save = None  # Scatter doesn't have grid
                elif mode == 'neural':
                    success, grid_to_save = self._batch_render_neural(
                        config, idx, dialog
                    )
                elif mode == 'plenoxel':
                    success, grid_to_save = self._batch_render_plenoxel(config, idx, dialog)
                elif mode == 'arepovtk':
                    success, grid_to_save = self._batch_render_arepovtk(
                        config, idx, dialog, canvas_size, cam_config
                    )
                else:
                    self.log(f"Unknown mode: {mode}")
                    dialog.update_row_status(
                        idx, "✗ Unknown Mode", progress=None,
                        style="color: red; font-size: 10px; font-weight: bold;"
                    )
                    continue
                    
            except Exception as e:
                self.log(f"ERROR in {mode} render: {e}")
                import traceback
                traceback.print_exc()
                dialog.update_row_status(
                    idx, f"✗ Error: {str(e)[:20]}", progress=None,
                    style="color: red; font-size: 10px; font-weight: bold;"
                )
                QApplication.processEvents()
                continue
            
            if not success:
                dialog.update_row_status(
                    idx, "✗ Failed", progress=None,
                    style="color: red; font-size: 10px; font-weight: bold;"
                )
                QApplication.processEvents()
                continue
            
            # Restore camera position before screenshot (ensures consistency)
            self._restore_camera_from_config(cam_config)
            QApplication.processEvents()
            
            # Save outputs
            if config['save_png']:
                png_file = output_path / f"{filename}.png"
                try:
                    # ArepoVTK renders directly to image, use cached result
                    if mode == 'arepovtk' and hasattr(self, '_arepovtk_rendered_image') and self._arepovtk_rendered_image is not None:
                        img = self._arepovtk_rendered_image
                        # Convert float [0,1] to uint8 [0,255]
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                        self._arepovtk_rendered_image = None  # Clear cache
                    else:
                        img = self.visualization.render_image_for_export(
                            canvas_size[0],
                            canvas_size[1],
                            hide_reference_cube=True,
                            hide_subvolume_box=True
                        )
                    from PIL import Image
                    Image.fromarray(img).save(png_file)
                    self.log(f"✓ Saved PNG: {png_file.name} ({canvas_size[0]}x{canvas_size[1]})")
                except Exception as e:
                    self.log(f"✗ Failed to save PNG: {e}")
                    import traceback
                    traceback.print_exc()
            
            if config['save_npy'] and grid_to_save is not None:
                npy_file = output_path / f"{filename}.npy"
                try:
                    np.save(npy_file, grid_to_save)
                    self.log(f"✓ Saved NPY: {npy_file.name}")
                except Exception as e:
                    self.log(f"✗ Failed to save NPY: {e}")
            
            # Save metadata for this render
            meta_file = output_path / f"{filename}_metadata.json"
            metadata = {
                'config': config,
                'stats': stats_to_save,
                'filename': filename,
                'canvas_size': canvas_size,
                'timestamp': datetime.now().isoformat(),
                'camera': cam_config
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
        
        # RESTORE ORIGINAL STATE
        self.plotter.window_size = [original_size[0], original_size[1]]
        self.visualization.set_mode(original_mode)
        QApplication.processEvents()
        self.log(f"Plotter restored to {original_size[0]}x{original_size[1]}")
        
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

    def _restore_camera_from_config(self, cam_config):
        """Restore camera position from saved config for pixel-perfect consistency."""
        try:
            camera = self.plotter.camera
            if 'position' in cam_config:
                camera.position = cam_config['position']
            if 'focal_point' in cam_config:
                camera.focal_point = cam_config['focal_point']
            elif 'center' in cam_config:
                camera.focal_point = cam_config['center']
            if 'up' in cam_config:
                camera.up = cam_config['up']
            if 'fov' in cam_config:
                camera.view_angle = cam_config['fov']
            self.plotter.render()
        except Exception as e:
            self.log(f"Warning: Could not restore camera: {e}")

    def _batch_render_volume(self, config, idx, dialog):
        """Render volume mode for batch processing."""
        # Update GUI controls to match config
        self.method_combo.setCurrentText(config['method'])
        self.resolution_spin.setValue(config['resolution'])
        self.epsilon_spin.setValue(config['epsilon'])
        QApplication.processEvents()
        
        # Compute volume (synchronously for batch)
        self.log(f"Computing volume: {config['method']} @ {config['resolution']}³")
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
            dialog.update_row_status(idx, "▶ Computing...", progress=progress_value,
                                    style="color: blue; font-size: 10px; font-weight: bold;")
            QApplication.processEvents()
        
        worker.finished.connect(capture_result)
        worker.error.connect(capture_error)
        worker.progress.connect(update_row_progress)
        worker.status.connect(self.log)
        
        # Run synchronously
        worker.run()
        QApplication.processEvents()
        
        if result_holder['error']:
            self.log(f"ERROR: {result_holder['error']}")
            return False, None, None
        
        if result_holder['grid'] is None:
            self.log(f"ERROR: No result from worker")
            return False, None, None
        
        # CRITICAL: Clear neural renderer completely before showing volume
        if self.neural_renderer:
            self.neural_renderer.clear()
        if self.scatter_renderer:
            self.scatter_renderer.clear()
        if self.plenoxel_renderer:
            self.plenoxel_renderer.clear()
        QApplication.processEvents()
        
        # Update volume
        self.grid_values = result_holder['grid']
        self.current_stats = result_holder['stats']
        self.visualization.update_volume_visual()
        
        # Make sure volume renderer is visible
        if self.volume_renderer:
            self.volume_renderer.set_active(True)
        
        # Update mode state
        self.viz_mode = 'volume'
        QApplication.processEvents()
        
        return True, result_holder['grid'], result_holder['stats']

    def _batch_render_scatter(self, config, idx, dialog):
        """Render scatter mode for batch processing."""
        self.log(f"Rendering scatter plot")
        dialog.update_row_status(idx, "▶ Scatter...", progress=-1,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # CRITICAL: Clear volume and neural renderers completely
        if self.volume_renderer:
            self.volume_renderer.clear()
        if self.neural_renderer:
            self.neural_renderer.clear()
        if self.plenoxel_renderer:
            self.plenoxel_renderer.clear()
        QApplication.processEvents()
        
        # Create/update scatter plot and make visible
        self.visualization.create_scatter_plot(force_visible=True)
        if self.scatter_renderer:
            self.scatter_renderer.set_active(True)
        
        # Update mode state
        self.viz_mode = 'scatter'
        QApplication.processEvents()
        
        return True

    def _batch_render_neural(self, config, idx, dialog):
        """Render neural mode for batch processing.
        
        Supports rendering from:
        1. Cached trained model (after training completes)
        2. Live trainer model (during training)
        3. Currently displayed neural grid (fallback - saves current visualization)
        """
        resolution = config.get('resolution', 256)
        self.log(f"Rendering neural volume @ {resolution}³")
        dialog.update_row_status(idx, "▶ Neural inference...", progress=-1,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # Check if we have a neural model
        trainer = self.neural_trainer
        model = self.neural_model
        
        grid_values = None
        
        if model is not None:
            # Use the cached trained model (after training completes)
            try:
                self.log(f"Using cached neural model for inference @ {resolution}³")
                grid_values = model.generate_volume(resolution, bounds=self.current_subvolume_bounds)
            except Exception as e:
                self.log(f"Failed to generate from cached model: {e}")
                import traceback
                traceback.print_exc()
                # Don't return yet - try fallbacks
                
        if grid_values is None and trainer is not None and trainer.model is not None:
            # Use the live trainer model (during training)
            try:
                self.log(f"Using live trainer model for inference @ {resolution}³")
                grid_values = trainer.evaluate_live_model(resolution, self.current_subvolume_bounds)
            except Exception as e:
                self.log(f"Failed to generate from trainer: {e}")
                import traceback
                traceback.print_exc()
                # Don't return yet - try fallback
        
        # Fallback: use currently displayed neural grid if available
        if grid_values is None and self.nn_grid_values is not None:
            self.log(f"Using currently displayed neural grid (shape={self.nn_grid_values.shape})")
            grid_values = self.nn_grid_values
        
        if grid_values is None:
            self.log("No neural data available - train a model or switch to neural mode first")
            return False, None
        
        self.log(f"Neural inference complete: shape={grid_values.shape}, range=[{grid_values.min():.4f}, {grid_values.max():.4f}]")
        
        # Store the neural grid values
        self.nn_grid_values = grid_values
        
        # Compute bounds for neural grid - use normalized bounds (0-1)
        # The render() method will apply VOLUME_EXTENT internally
        if self.current_subvolume_bounds:
            mins, maxs = self.current_subvolume_bounds
            bounds = (mins, maxs)  # Keep normalized - render() applies extent
        else:
            bounds = None
        self.nn_bounds = bounds
        
        # CRITICAL: Actually CLEAR the volume renderer (not just hide)
        # This removes the actor entirely from the plotter
        if self.volume_renderer:
            self.volume_renderer.clear()
            self.log("Cleared volume renderer")
        
        # Clear scatter renderer too
        if self.scatter_renderer:
            self.scatter_renderer.clear()
        
        # Clear plenoxel renderer
        if self.plenoxel_renderer:
            self.plenoxel_renderer.clear()
        
        QApplication.processEvents()
        
        # Now render the neural volume
        dialog.update_row_status(idx, "▶ Rendering neural...", progress=50,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        grid_for_render = np.ascontiguousarray(grid_values)
        self.neural_renderer.render(
            grid_for_render,
            visible=True,
            bounds=bounds
        )
        self.neural_renderer.set_active(True)
        self.log(f"Neural renderer actor: {self.neural_renderer.actor is not None}")

        # If the actor wasn't created, try a conservative re-render without bounds
        if self.neural_renderer.actor is None:
            self.log("Neural actor missing after first render — retrying with bounds=None")
            try:
                self.neural_renderer.render(grid_for_render, visible=True, bounds=None)
                self.neural_renderer.set_active(True)
                QApplication.processEvents()
                self.plotter.render()
            except Exception as e:
                self.log(f"Exception during neural re-render: {e}")

        # If still missing, abort and report failure so we don't save an empty screenshot
        if self.neural_renderer.actor is None:
            self.log("ERROR: Neural renderer produced no actor — aborting this render (no screenshot will be saved)")
            return False, None
        
        # Update mode state
        self.viz_mode = 'neural'
        
        # Update UI buttons
        if hasattr(self, 'volume_mode_btn') and self.volume_mode_btn:
            self.volume_mode_btn.blockSignals(True)
            self.volume_mode_btn.setChecked(False)
            self.volume_mode_btn.blockSignals(False)
        if hasattr(self, 'scatter_mode_btn') and self.scatter_mode_btn:
            self.scatter_mode_btn.blockSignals(True)
            self.scatter_mode_btn.setChecked(False)
            self.scatter_mode_btn.blockSignals(False)
        if hasattr(self, 'neural_mode_btn') and self.neural_mode_btn:
            self.neural_mode_btn.blockSignals(True)
            self.neural_mode_btn.setChecked(True)
            self.neural_mode_btn.blockSignals(False)
        
        self.plotter.render()
        QApplication.processEvents()
        
        return True, grid_values

    def _batch_render_arepovtk(self, config, idx, dialog, canvas_size, cam_config):
        """Render using ArepoVTK-style CPU ray marcher.
        
        This bypasses the normal volume renderer and directly renders
        to an image using CPU ray marching with IDW or Natural Neighbor interpolation.
        
        Config parameters:
        - step_size: Step size as fraction of volume diagonal (default 0.01)
        - idw_power: IDW power parameter (default 2.0, ArepoVTK standard)
        - k_neighbors: Number of neighbors for KD-tree IDW (default 32)
        - interpolation_method: 'idw' or 'natural_neighbor' (Sibson, matches Illustris TNG)
        - nn_grid_resolution: Grid resolution for Natural Neighbor mode (default 128)
        """
        self.log(f"Starting ArepoVTK CPU ray marcher...")
        dialog.update_row_status(idx, "▶ ArepoVTK Ray Marching...", progress=-1,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # Check if data is available
        if self.data is None or self.scalars is None:
            self.log("No data available for ArepoVTK renderer")
            return False, None
        
        # Set data on the ArepoVTK renderer
        self.arepovtk_renderer.set_data(self.data, self.scalars)
        self.arepovtk_renderer.update_transfer_function(self.transfer_function)
        
        # Configure ArepoVTK renderer parameters from config
        step_size = config.get('step_size', 0.01)  # As fraction of diagonal
        idw_power = config.get('idw_power', 2.0)   # ArepoVTK default is 2.0
        k_neighbors = config.get('k_neighbors', 32)  # KD-tree neighbors
        interpolation_method = config.get('interpolation_method', 'idw')  # 'idw' or 'natural_neighbor'
        nn_grid_resolution = config.get('nn_grid_resolution', 128)  # Grid res for NN
        
        self.arepovtk_renderer.step_size = step_size
        self.arepovtk_renderer.idw_power = idw_power
        self.arepovtk_renderer.interpolation_method = interpolation_method
        self.arepovtk_renderer.nn_grid_resolution = nn_grid_resolution
        
        self.log(f"  Interpolation: {interpolation_method}")
        self.log(f"  Step size: {step_size}, IDW power: {idw_power}, K neighbors: {k_neighbors}")
        if interpolation_method == 'natural_neighbor':
            self.log(f"  NN grid resolution: {nn_grid_resolution}")
        self.log(f"  Canvas size: {canvas_size[0]}x{canvas_size[1]}")
        
        # Get bounds - use subvolume if active
        bounds = self.subvolume.get_active_bounds()
        if bounds:
            bounds = (np.array(bounds[0]), np.array(bounds[1]))
            self.log(f"  Using subvolume bounds: {bounds}")
        
        # Prepare camera config
        camera_position = np.array(cam_config.get('position', [0, 0, 2]))
        camera_focal = np.array(cam_config.get('focal_point', cam_config.get('center', [0.5, 0.5, 0.5])))
        camera_up = np.array(cam_config.get('up', cam_config.get('up_vector', [0, 1, 0])))
        camera_fov = cam_config.get('fov', cam_config.get('view_angle', 45.0))
        
        self.log(f"  Camera position: {camera_position}")
        self.log(f"  Camera focal: {camera_focal}")
        self.log(f"  Camera FOV: {camera_fov}")
        
        def progress_callback(pct):
            dialog.update_row_status(idx, f"▶ Ray Marching {pct}%", progress=pct,
                                    style="color: blue; font-size: 10px; font-weight: bold;")
            QApplication.processEvents()
        
        try:
            # Render to image using CPU ray marcher
            image = self.arepovtk_renderer.render_to_image(
                width=canvas_size[0],
                height=canvas_size[1],
                camera_position=camera_position,
                camera_focal=camera_focal,
                camera_up=camera_up,
                fov=camera_fov,
                bounds=bounds,
                progress_callback=progress_callback,
                k_neighbors=k_neighbors
            )
            
            if image is None or image.size == 0:
                self.log("ArepoVTK render returned empty image")
                return False, None
            
            self.log(f"  Rendered image: {image.shape}, range=[{image.min():.4f}, {image.max():.4f}]")
            
            # Save directly as PNG (don't use PyVista screenshot)
            # Note: we return the image directly and let batch caller save it
            # Also save with alpha channel intact
            
            # Hide all other renderers for the "screenshot" 
            # (ArepoVTK renders directly to image, no PyVista display)
            if self.volume_renderer:
                self.volume_renderer.set_active(False)
            if self.scatter_renderer:
                self.scatter_renderer.set_active(False)
            if self.neural_renderer:
                self.neural_renderer.set_active(False)
            if self.plenoxel_renderer:
                self.plenoxel_renderer.set_active(False)
            QApplication.processEvents()
            
            # For ArepoVTK, we override the normal PNG save since we already have the image
            # Store in a special attribute for batch saver to use
            self._arepovtk_rendered_image = image
            
            # Update mode state
            self.viz_mode = 'arepovtk'
            
            dialog.update_row_status(idx, "▶ Saving...", progress=95,
                                    style="color: blue; font-size: 10px; font-weight: bold;")
            QApplication.processEvents()
            
            # Return success (no grid to save - ArepoVTK works differently)
            return True, None
            
        except Exception as e:
            self.log(f"ArepoVTK render failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _batch_render_plenoxel(self, config, idx, dialog):
        """Render plenoxel volume for batch processing.
        
        Builds a plenoxel grid and converts it to a volume for rendering.
        Does NOT show the wireframe overlay - just the volume data.
        """
        max_depth = config.get('max_depth', 5)
        min_points = config.get('min_points', 500)
        
        self.log(f"Building plenoxel grid (max_depth={max_depth}, min_points={min_points})")
        dialog.update_row_status(idx, "▶ Building Plenoxel...", progress=-1,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # Check if data is available
        if self.data is None or self.scalars is None:
            self.log("No data available for plenoxel grid")
            return False, None
        
        # Build plenoxel grid with specified settings
        try:
            nodes, stats = build_plenoxel_grid(
                self.data,
                self.scalars,
                min_points=min_points,
                min_depth=1,
                max_depth=max_depth,
            )
            self.plenoxel_nodes = nodes
            self.plenoxel_stats = stats
            self.log(f"Plenoxel grid built: {stats['node_count']} cells, depth={stats['max_depth_reached']}")
        except Exception as e:
            self.log(f"Failed to build plenoxel grid: {e}")
            return False, None
        
        if not nodes:
            self.log("No plenoxel nodes generated")
            return False, None
        
        dialog.update_row_status(idx, "▶ Converting to volume...", progress=30,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # Convert plenoxel grid to volume data
        try:
            volume, volume_stats = plenoxel_volume_from_nodes(nodes, target_depth=max_depth)
            if volume is None:
                self.log("Failed to convert plenoxel to volume")
                return False, None
            self.log(f"Plenoxel volume: {volume.shape}, range=[{volume_stats['raw_min']:.4f}, {volume_stats['raw_max']:.4f}]")
        except Exception as e:
            self.log(f"Failed to convert plenoxel to volume: {e}")
            return False, None
        
        dialog.update_row_status(idx, "▶ Rendering volume...", progress=60,
                                style="color: blue; font-size: 10px; font-weight: bold;")
        QApplication.processEvents()
        
        # Hide other renderers
        if self.neural_renderer:
            self.neural_renderer.set_active(False)
        if self.scatter_renderer:
            self.scatter_renderer.set_active(False)
        # Hide plenoxel wireframe overlay
        if self.plenoxel_renderer:
            self.plenoxel_renderer.set_active(False)
        QApplication.processEvents()
        
        # Render the plenoxel volume using volume renderer
        self.grid_values = volume
        grid_for_render = np.ascontiguousarray(volume)
        self.volume_renderer.render(grid_for_render, visible=True, bounds=None)
        self.volume_renderer.set_active(True)
        
        # Update mode state
        self.viz_mode = 'volume'
        
        self.plotter.render()
        QApplication.processEvents()
        
        return True, volume
        
        return True


def main():
    app = QApplication(sys.argv)
    window = VolumeRenderingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()