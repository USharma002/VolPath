"""Visualization/controller utilities for renderer orchestration."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox


class VisualizationController:
    """Encapsulates viz-mode switching plus renderer helpers."""

    def __init__(self, gui: "VolumeRenderingGUI", extent: float) -> None:
        self.gui = gui
        self.extent = float(extent)

    # ------------------------------------------------------------------
    # Mode switching / renderer activation
    # ------------------------------------------------------------------
    def set_mode(self, mode: Optional[str]) -> None:
        gui = self.gui
        mode = (mode or '').lower()
        if mode == gui.viz_mode:
            return
        if mode not in {'volume', 'scatter', 'neural'}:
            gui.log(f"Unknown visualization mode: {mode}")
            return

        cam_state = gui.camera_controller.capture_state() if gui.camera_controller else None

        def _check(button, value):
            if button:
                button.blockSignals(True)
                button.setChecked(value)
                button.blockSignals(False)

        controls_enabled = False

        if mode == 'volume':
            _check(getattr(gui, 'volume_mode_btn', None), True)
            _check(getattr(gui, 'scatter_mode_btn', None), False)
            _check(getattr(gui, 'neural_mode_btn', None), False)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Volume Rendering (interpolated)")
            if gui.volume_renderer:
                if gui.grid_values is not None:
                    self.update_volume_visual()
                gui.volume_renderer.set_active(True)
            if gui.scatter_renderer:
                gui.scatter_renderer.set_active(False)
            if gui.neural_renderer:
                gui.neural_renderer.set_active(False)
            controls_enabled = True
        elif mode == 'scatter':
            _check(getattr(gui, 'volume_mode_btn', None), False)
            _check(getattr(gui, 'scatter_mode_btn', None), True)
            _check(getattr(gui, 'neural_mode_btn', None), False)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Particle Scatter Plot")
            if gui.scatter_renderer:
                self.create_scatter_plot(force_visible=True)
                gui.scatter_renderer.set_active(True)
            if gui.volume_renderer:
                gui.volume_renderer.set_active(False)
            if gui.neural_renderer:
                gui.neural_renderer.set_active(False)
        else:  # neural
            _check(getattr(gui, 'volume_mode_btn', None), False)
            _check(getattr(gui, 'scatter_mode_btn', None), False)
            _check(getattr(gui, 'neural_mode_btn', None), True)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Neural Renderer (tinycudann)")
            if gui.neural_renderer:
                gui._render_neural_volume()
                gui.neural_renderer.set_active(gui.neural_renderer.visual is not None)
            if gui.volume_renderer:
                gui.volume_renderer.set_active(False)
            if gui.scatter_renderer:
                gui.scatter_renderer.set_active(False)

        for widget in (
            getattr(gui, 'method_combo', None),
            getattr(gui, 'epsilon_spin', None),
            getattr(gui, 'resolution_spin', None),
            getattr(gui, 'compute_btn', None)
        ):
            if widget:
                widget.setEnabled(controls_enabled)

        gui.viz_mode = mode

        if cam_state and gui.camera_controller:
            gui._suppress_camera_event = True
            gui.camera_controller.restore_state(cam_state)
            gui._suppress_camera_event = False
            gui._sync_camera_controls(cam_state)
            if gui.view and gui.view.camera:
                gui.view.camera.view_changed()

        if gui.reference_cube:
            gui.reference_cube.set_active(True)
        if gui.canvas:
            gui.canvas.update()
        gui.log(f"Switched to {mode} mode (camera preserved)")

    # ------------------------------------------------------------------
    # Visual creation helpers
    # ------------------------------------------------------------------
    def create_scatter_plot(self, force_visible: bool = False) -> None:
        gui = self.gui
        if gui.data is None or gui.original_scalars is None:
            gui.log("ERROR: No original data for scatter plot")
            return

        try:
            resolution = gui.grid_values.shape[0] if gui.grid_values is not None else gui.resolution_spin.value()
            visible = (gui.viz_mode == 'scatter') or force_visible
            coords = gui.data
            scalars = gui.original_scalars
            bounds = gui.subvolume.get_active_bounds()
            if bounds:
                mins, maxs = bounds
                mask = np.all((gui.data >= mins) & (gui.data <= maxs), axis=1)
                if mask.any():
                    coords = gui.data[mask]
                    scalars = gui.original_scalars[mask]
                else:
                    coords = np.empty((0, 3), dtype=np.float32)
                    scalars = np.empty((0,), dtype=np.float32)
            result = gui.scatter_renderer.render(coords, scalars, resolution, visible=visible)
            gui.canvas.update()
            if result:
                gui.log(f"✓ Scatter plot created ({result['points']:,} points) in unit cube")
        except Exception as exc:  # pragma: no cover
            gui.log(f"ERROR creating scatter plot: {exc}")
            import traceback
            traceback.print_exc()

    def update_volume_visual(self) -> None:
        gui = self.gui
        if gui.grid_values is None or not gui.volume_renderer:
            return

        try:
            resolution = gui.grid_values.shape[0]
            bounds = gui.grid_bounds
            if bounds:
                mins, maxs = bounds
                span = np.asarray(maxs) - np.asarray(mins)
            else:
                span = np.ones(3, dtype=np.float32)
            voxel_size = (self.extent * span) / max(resolution, 1)
            gui.log(
                f"Creating volume visual: {resolution}³ voxels covering span {span[0]:.2f}×{span[1]:.2f}×{span[2]:.2f}"
            )

            grid_for_render = np.ascontiguousarray(np.transpose(gui.grid_values, (2, 1, 0)))

            gui.volume_renderer.render(
                grid_for_render,
                visible=(gui.viz_mode == 'volume'),
                bounds=bounds
            )

            if resolution >= 512 and hasattr(gui.view.camera, 'depth_value'):
                gui.view.camera.depth_value = 50000

            gui.camera_controller.ensure_center()
            gui.view.camera.view_changed()
            gui.canvas.update()

            center = self.extent / 2.0
            gui.log("✓ Volume rendered successfully!")
            gui.log(
                f"Extent: {self.extent:.1f}³, Voxels: {resolution}³, Voxel size: "
                f"({voxel_size[0]:.4f}, {voxel_size[1]:.4f}, {voxel_size[2]:.4f})"
            )
            gui.log(f"Camera center: ({center:.1f}, {center:.1f}, {center:.1f})")

        except Exception as exc:  # pragma: no cover
            gui.log(f"ERROR creating volume visual: {exc}")
            QMessageBox.critical(
                gui,
                "Rendering Failed",
                f"Failed to create volume visualization:\n{exc}\n\n"
                "This usually means:\n"
                "• GPU texture size limit exceeded\n"
                "• Insufficient GPU memory\n\n"
                "Try a lower resolution (512 or less)."
            )
            import traceback
            traceback.print_exc()

    def update_colormap(self) -> None:
        gui = self.gui
        try:
            gui.transfer_function.set_colormap(gui.cmap_combo.currentText())
        except Exception:
            pass
        if gui.volume_renderer and gui.volume_renderer.visual is not None:
            gui.volume_renderer.update_transfer_function(gui.transfer_function)
            gui.canvas.update()
        if gui.neural_renderer and gui.neural_renderer.visual is not None:
            gui.neural_renderer.update_transfer_function(gui.transfer_function)
            gui.canvas.update()
        if gui.scatter_renderer:
            gui.scatter_renderer.update_transfer_function(gui.transfer_function)
        if gui.scatter_renderer and gui.scatter_renderer.visual is not None:
            self.create_scatter_plot()

    def render_image_for_export(self, width=None, height=None, hide_reference_cube=True):
        gui = self.gui
        native = getattr(gui.canvas, 'native', None)
        if native is None:
            raise RuntimeError("Canvas not initialized")

        original_size = (native.width(), native.height())
        resize_needed = (
            width is not None and height is not None and
            (width != original_size[0] or height != original_size[1])
        )

        if resize_needed:
            native.resize(int(width), int(height))
            QApplication.processEvents()

        cube_renderer = getattr(gui, 'reference_cube', None)
        cube_was_active = bool(
            hide_reference_cube and cube_renderer and
            cube_renderer.visual is not None and getattr(cube_renderer, '_active', False)
        )

        if cube_was_active:
            cube_renderer.set_active(False)
            QApplication.processEvents()

        try:
            img = gui.canvas.render()
        finally:
            if cube_was_active:
                cube_renderer.set_active(True)
                QApplication.processEvents()

            if resize_needed:
                native.resize(original_size[0], original_size[1])
                QApplication.processEvents()

        return img