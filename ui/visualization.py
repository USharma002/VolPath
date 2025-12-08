"""Visualization/controller utilities for renderer orchestration."""

from __future__ import annotations

import gc
from typing import Optional
import weakref

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox


class VisualizationController:
    """Encapsulates viz-mode switching plus renderer helpers.
    
    Memory optimized with weakref to gui and explicit cleanup.
    """
    
    __slots__ = ('_gui_ref', 'extent', '_render_pending')

    def __init__(self, gui: "VolumeRenderingGUI", extent: float) -> None:
        # Use weakref to avoid circular references
        self._gui_ref = weakref.ref(gui)
        self.extent = float(extent)
        self._render_pending = False  # Flag to batch render calls
    
    @property
    def gui(self):
        """Get gui from weakref."""
        return self._gui_ref()

    def _schedule_render(self):
        """Mark that a render is needed, but defer until end of operation."""
        self._render_pending = True

    def _flush_render(self):
        """Execute pending render if scheduled."""
        gui = self.gui
        if self._render_pending and gui and gui.plotter:
            gui.plotter.render()
            self._render_pending = False

    # ------------------------------------------------------------------
    # Mode switching / renderer activation
    # ------------------------------------------------------------------
    def set_mode(self, mode: Optional[str]) -> None:
        gui = self.gui
        mode = (mode or '').lower()
        if mode == gui.viz_mode:
            return
        if mode not in {'volume', 'scatter', 'neural', 'plenoxel'}:
            gui.log(f"Unknown visualization mode: {mode}")
            return

        # Capture camera state BEFORE any changes
        cam_state = gui.camera_controller.capture_state() if gui.camera_controller else None

        def _check(button, value):
            if button:
                button.blockSignals(True)
                button.setChecked(value)
                button.blockSignals(False)

        controls_enabled = False
        plenoxel_renderer = getattr(gui, 'plenoxel_renderer', None)

        if mode == 'volume':
            _check(getattr(gui, 'volume_mode_btn', None), True)
            _check(getattr(gui, 'scatter_mode_btn', None), False)
            _check(getattr(gui, 'neural_mode_btn', None), False)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Volume Rendering (interpolated)")
            
            # Only create volume if it doesn't exist yet
            if gui.volume_renderer:
                if gui.volume_renderer.actor is None and gui.grid_values is not None:
                    self.update_volume_visual()
                gui.volume_renderer.set_active(True)
            if gui.scatter_renderer:
                gui.scatter_renderer.set_active(False)
            if gui.neural_renderer:
                gui.neural_renderer.set_active(False)
            if plenoxel_renderer:
                plenoxel_renderer.set_active(False)
            controls_enabled = True
            
        elif mode == 'scatter':
            _check(getattr(gui, 'volume_mode_btn', None), False)
            _check(getattr(gui, 'scatter_mode_btn', None), True)
            _check(getattr(gui, 'neural_mode_btn', None), False)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Particle Scatter Plot")
            
            # Only create scatter if it doesn't exist yet
            if gui.scatter_renderer:
                if gui.scatter_renderer.actor is None and gui.data is not None:
                    self.create_scatter_plot(force_visible=True)
                gui.scatter_renderer.set_active(True)
            if gui.volume_renderer:
                gui.volume_renderer.set_active(False)
            if gui.neural_renderer:
                gui.neural_renderer.set_active(False)
            if plenoxel_renderer:
                plenoxel_renderer.set_active(False)
                
        elif mode == 'neural':
            _check(getattr(gui, 'volume_mode_btn', None), False)
            _check(getattr(gui, 'scatter_mode_btn', None), False)
            _check(getattr(gui, 'neural_mode_btn', None), True)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Neural Renderer (tinycudann)")
            
            # Only render neural if it doesn't exist yet
            if gui.neural_renderer:
                if gui.neural_renderer.actor is None:
                    gui._render_neural_volume()
                gui.neural_renderer.set_active(gui.neural_renderer.actor is not None)
            if gui.volume_renderer:
                gui.volume_renderer.set_active(False)
            if gui.scatter_renderer:
                gui.scatter_renderer.set_active(False)
            if plenoxel_renderer:
                plenoxel_renderer.set_active(False)
                
        elif mode == 'plenoxel':
            _check(getattr(gui, 'volume_mode_btn', None), False)
            _check(getattr(gui, 'scatter_mode_btn', None), False)
            _check(getattr(gui, 'neural_mode_btn', None), False)
            if getattr(gui, 'viz_info_label', None):
                gui.viz_info_label.setText("Mode: Plenoxel (Adaptive Sparse Voxel Grid)")
            
            if plenoxel_renderer:
                plenoxel_renderer.set_active(plenoxel_renderer.actor is not None)
            if gui.volume_renderer:
                gui.volume_renderer.set_active(False)
            if gui.scatter_renderer:
                gui.scatter_renderer.set_active(False)
            if gui.neural_renderer:
                gui.neural_renderer.set_active(False)

        for widget in (
            getattr(gui, 'method_combo', None),
            getattr(gui, 'epsilon_spin', None),
            getattr(gui, 'resolution_spin', None),
            getattr(gui, 'compute_btn', None)
        ):
            if widget:
                widget.setEnabled(controls_enabled)

        # Show/hide panel sections based on mode
        interpolation_group = getattr(gui, 'interpolation_group', None)
        neural_settings_group = getattr(gui, 'neural_settings_group', None)
        
        if interpolation_group:
            # Interpolation settings only visible in volume mode
            interpolation_group.setVisible(mode == 'volume')
        
        if neural_settings_group:
            # Neural settings only visible in neural mode
            neural_settings_group.setVisible(mode == 'neural')
        
        gui.viz_mode = mode

        # Respect bounding box checkbox state
        show_bbox = getattr(gui, 'show_bounding_box_check', None)
        if gui.reference_cube:
            bbox_visible = show_bbox.isChecked() if show_bbox else True
            gui.reference_cube.set_active(bbox_visible)
        
        # Ensure subvolume box is visible in all modes if enabled
        # First, ensure the visual exists if subvolume is enabled
        subvolume_group = getattr(gui, 'subvolume_group', None)
        if subvolume_group and subvolume_group.isChecked() and gui.subvolume_box_visual is None:
            # Re-create subvolume visual if it was lost
            if hasattr(gui, 'subvolume') and gui.subvolume:
                gui.subvolume.update_visual()
        
        # Respect subvolume box checkbox state
        show_subvol = getattr(gui, 'show_subvolume_box_check', None)
        if gui.subvolume_box_visual is not None:
            subvol_visible = show_subvol.isChecked() if show_subvol else True
            try:
                gui.subvolume_box_visual.SetVisibility(subvol_visible)
            except Exception:
                pass
        
        # Restore camera state as the FINAL step before render
        # This ensures no other operation can reset the camera after this
        if cam_state and gui.camera_controller:
            gui._suppress_camera_event = True
            camera = gui.plotter.camera
            # Directly set VTK camera properties to ensure they stick
            if 'position' in cam_state:
                camera.position = cam_state['position']
            if 'center' in cam_state:
                camera.focal_point = cam_state['center']
            if 'up' in cam_state:
                camera.up = cam_state['up']
            if 'fov' in cam_state and hasattr(camera, 'view_angle'):
                camera.view_angle = cam_state['fov']
            gui._suppress_camera_event = False
            gui._sync_camera_controls(cam_state)
            
        # Final render
        if gui.plotter:
            gui.plotter.render()
            
        gui.log(f"Switched to {mode} mode")

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
                mins, maxs = np.asarray(bounds[0]), np.asarray(bounds[1])
                # Vectorized bounds check
                mask = np.all((gui.data >= mins) & (gui.data <= maxs), axis=1)
                if mask.any():
                    coords = gui.data[mask]
                    scalars = gui.original_scalars[mask]
                else:
                    coords = np.empty((0, 3), dtype=np.float32)
                    scalars = np.empty((0,), dtype=np.float32)
            
            # Set scalar bar title from current field and transform (matching reference config)
            field_name = getattr(gui, 'current_field_name', None) or 'Scalar'
            transform_name = getattr(gui, 'current_transform_name', None)
            if transform_name:
                scalar_bar_title = f"{field_name}\n({transform_name})"
            else:
                scalar_bar_title = field_name
            gui.scatter_renderer.set_scalar_bar_title(scalar_bar_title)
            
            result = gui.scatter_renderer.render(coords, scalars, resolution, visible=visible)
            # Don't render here - let caller handle it to preserve camera
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

            # Don't transpose - use grid_values directly
            # VTK's Fortran-order raveling in volume.py handles the axis ordering
            # Transposing here would swap X and Z axes, misaligning with scatter coordinates
            grid_for_render = np.ascontiguousarray(gui.grid_values)

            gui.volume_renderer.render(
                grid_for_render,
                visible=(gui.viz_mode == 'volume'),
                bounds=bounds
            )

            # Don't call ensure_center() or render() here
            # The camera should be preserved across mode switches and volume updates
            # Let the caller handle the final render

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

    def update_colormap(self, colormap_name: str = None) -> None:
        """Update colormap from transfer function or explicit name."""
        gui = self.gui
        if colormap_name:
            gui.transfer_function.set_colormap(colormap_name)
        
        # Batch updates - don't render after each one
        if gui.volume_renderer and gui.volume_renderer.actor is not None:
            gui.volume_renderer.update_transfer_function(gui.transfer_function)
        if gui.neural_renderer and gui.neural_renderer.actor is not None:
            gui.neural_renderer.update_transfer_function(gui.transfer_function)
        if gui.scatter_renderer:
            gui.scatter_renderer.update_transfer_function(gui.transfer_function)
            if gui.scatter_renderer.actor is not None:
                self.create_scatter_plot()
        
        # Single render at the end
        if gui.plotter:
            gui.plotter.render()

    def render_image_for_export(self, width=None, height=None, hide_reference_cube=True, hide_subvolume_box=True):
        """Render current scene to image for export/screenshot.
        
        Args:
            width: Output image width (None = current window width)
            height: Output image height (None = current window height)
            hide_reference_cube: If True, temporarily hide the reference cube (white box)
            hide_subvolume_box: If True, temporarily hide the subvolume box (yellow box)
        """
        gui = self.gui
        if gui.plotter is None:
            raise RuntimeError("Plotter not initialized")

        # Store original window size
        original_size = gui.plotter.window_size
        resize_needed = (
            width is not None and height is not None and
            (width != original_size[0] or height != original_size[1])
        )

        if resize_needed:
            gui.plotter.window_size = (int(width), int(height))
            QApplication.processEvents()

        # Hide reference cube (white bounding box) - always hide if actor exists
        cube_renderer = getattr(gui, 'reference_cube', None)
        cube_was_visible = False
        if hide_reference_cube and cube_renderer and cube_renderer.actor is not None:
            try:
                cube_was_visible = bool(cube_renderer.actor.GetVisibility())
                if cube_was_visible:
                    cube_renderer.actor.SetVisibility(False)
                    QApplication.processEvents()
            except Exception:
                cube_was_visible = False

        # Hide subvolume box (yellow selection box)
        subvol_box = getattr(gui, 'subvolume_box_visual', None)
        subvol_was_visible = False
        if hide_subvolume_box and subvol_box is not None:
            try:
                subvol_was_visible = bool(subvol_box.GetVisibility())
                if subvol_was_visible:
                    subvol_box.SetVisibility(False)
                    QApplication.processEvents()
            except Exception:
                subvol_was_visible = False

        try:
            # PyVista screenshot
            img = gui.plotter.screenshot(return_img=True)
        finally:
            if cube_was_visible:
                try:
                    cube_renderer.actor.SetVisibility(True)
                    QApplication.processEvents()
                except Exception:
                    pass
            
            if subvol_was_visible:
                try:
                    subvol_box.SetVisibility(True)
                    QApplication.processEvents()
                except Exception:
                    pass

            if resize_needed:
                gui.plotter.window_size = original_size
                QApplication.processEvents()

        return img