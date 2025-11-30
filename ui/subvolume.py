"""Controller for subvolume widgets and interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from vispy import scene

Bounds = Optional[Tuple[np.ndarray, np.ndarray]]


@dataclass
class SubvolumeController:
    gui: "VolumeRenderingGUI"
    volume_extent: float

    def on_toggled(self, checked: bool) -> None:
        self._clamp_centers_for_size()
        state = "enabled" if checked else "disabled"
        gui = self.gui
        gui.current_subvolume_bounds = None if not checked else self.get_active_bounds()
        if hasattr(gui, "log_text"):
            gui.log(f"Subvolume zoom {state}.")
        self.update_visual()
        self.update_extract_button_state()

    def on_slider_changed(self, axis: str) -> None:
        sliders = self.gui.subvolume_center_sliders
        labels = self.gui.subvolume_center_labels
        if axis not in sliders:
            return
        slider = sliders[axis]
        value = self._slider_to_float(slider)
        labels[axis].setText(f"{value:.3f}")
        self._clamp_centers_for_size()
        self.update_visual()
        self._sync_current_bounds()

    def on_size_changed(self, _value: Optional[int] = None) -> None:
        size_val = self._get_size_value()
        if self.gui.subvolume_size_label:
            self.gui.subvolume_size_label.setText(f"{size_val:.3f}")
        self._clamp_centers_for_size()
        self.update_visual()
        self._sync_current_bounds()

    def prepare_interpolation_data(self):
        gui = self.gui
        if gui.data is None or gui.scalars is None:
            raise ValueError("Load particle data before interpolating")

        bounds = self.get_active_bounds()
        data = gui.data.astype(np.float32, copy=False)
        scalars = gui.scalars.astype(np.float32, copy=False)

        if not bounds:
            gui.current_subvolume_bounds = None
            return data, scalars, None

        mins, maxs = bounds
        inside_mask = np.all((gui.data >= mins) & (gui.data <= maxs), axis=1)
        inside_count = int(inside_mask.sum())
        if inside_count < 64:
            gui.log(
                f"⚠ Only {inside_count} particles fall inside the selected subbox; interpolation may be noisy."
            )

        gui.current_subvolume_bounds = (mins, maxs)
        return data, scalars, (mins, maxs)

    def update_extract_button_state(self) -> None:
        gui = self.gui
        if not getattr(gui, "extract_subbox_btn", None):
            return
        enabled = bool(
            gui.subvolume_group and gui.subvolume_group.isChecked() and gui.data is not None
        )
        gui.extract_subbox_btn.setEnabled(enabled)

    def update_visual(self) -> None:
        gui = self.gui
        if not gui.view or not gui.canvas:
            return
        bounds = self.get_active_bounds(require_enabled=False)
        if not bounds or not (gui.subvolume_group and gui.subvolume_group.isChecked()):
            self.remove_visual()
            return

        mins, maxs = bounds
        min_corner = np.array(mins, dtype=np.float32) * self.volume_extent
        max_corner = np.array(maxs, dtype=np.float32) * self.volume_extent
        span = np.maximum(max_corner - min_corner, 1e-4)
        max_corner = min_corner + span

        corners = np.array([
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
        ], dtype=np.float32)

        edges = np.array([
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.uint32)

        if gui.subvolume_box_visual:
            gui.subvolume_box_visual.parent = None
            gui.subvolume_box_visual = None

        line = scene.visuals.Line(
            pos=corners,
            connect=edges,
            color=(1.0, 0.8, 0.2, 1.0),
            width=2.0,
            method='gl',
            parent=gui.view.scene
        )
        line.set_gl_state(depth_test=True, blend=False)
        gui.subvolume_box_visual = line
        gui.canvas.update()

    def remove_visual(self) -> None:
        gui = self.gui
        if gui.subvolume_box_visual:
            gui.subvolume_box_visual.parent = None
            gui.subvolume_box_visual = None
            if gui.canvas:
                gui.canvas.update()

    def get_active_bounds(self, require_enabled: bool = True) -> Bounds:
        gui = self.gui
        group = getattr(gui, 'subvolume_group', None)
        if require_enabled and (not group or not group.isChecked()):
            return None
        if not gui.subvolume_center_sliders or not gui.subvolume_size_slider:
            return None
        centers = np.array([
            self._slider_to_float(gui.subvolume_center_sliders[axis])
            for axis in 'xyz'
        ], dtype=np.float32)
        size = self._get_size_value()
        half = size / 2.0
        mins = np.clip(centers - half, 0.0, 1.0)
        maxs = np.clip(centers + half, 0.0, 1.0)
        if np.any(maxs - mins < 0.005):
            return None
        return mins, maxs

    def sync_current_bounds(self) -> None:
        self._sync_current_bounds()

    def extract_subbox(self) -> None:
        gui = self.gui
        bounds = self.get_active_bounds()
        if not bounds:
            QMessageBox.information(gui, "Subbox", "Enable subvolume zoom before extracting.")
            return
        if gui.data is None or gui.original_scalars is None or gui.original_coords is None:
            QMessageBox.warning(gui, "Subbox", "Load data before extracting a subbox.")
            return

        mins, maxs = bounds
        mask = np.all((gui.data >= mins) & (gui.data <= maxs), axis=1)
        if not mask.any():
            QMessageBox.warning(gui, "Subbox", "No particles fall inside the selected box.")
            return

        coords_world = gui.original_coords[mask]
        scalars = gui.original_scalars[mask]
        world_bounds = None
        if gui.coord_min is not None and gui.coord_span is not None:
            world_bounds = np.stack([
                gui.coord_min + mins * gui.coord_span,
                gui.coord_min + maxs * gui.coord_span
            ])

        filename, _ = QFileDialog.getSaveFileName(
            gui,
            "Save Subbox",
            "subbox.npz",
            "NumPy Archive (*.npz)"
        )
        if not filename:
            return

        payload = {
            'coords': coords_world,
            'scalars': scalars,
            'bounds_normalized': np.stack([mins, maxs]),
            'field': gui.current_field_name,
            'transform': gui.current_transform_name
        }
        if world_bounds is not None:
            payload['bounds_world'] = world_bounds
        np.savez(filename, **payload)
        gui.log(f"✓ Saved subbox with {coords_world.shape[0]:,} particles → {filename}")
        QMessageBox.information(
            gui,
            "Subbox Saved",
            f"Saved {coords_world.shape[0]:,} particles to\n{filename}"
        )

    def _slider_to_float(self, slider) -> float:
        maximum = max(slider.maximum(), 1)
        return slider.value() / maximum

    def _get_size_value(self) -> float:
        slider = self.gui.subvolume_size_slider
        if not slider:
            return 1.0
        maximum = max(slider.maximum(), 1)
        return max(slider.value() / maximum, 0.01)

    def _clamp_centers_for_size(self) -> None:
        gui = self.gui
        if not gui.subvolume_center_sliders or not gui.subvolume_size_slider:
            return
        half = self._get_size_value() / 2.0
        for axis, slider in gui.subvolume_center_sliders.items():
            min_allowed = half
            max_allowed = 1.0 - half
            value = self._slider_to_float(slider)
            clamped = np.clip(value, min_allowed, max_allowed)
            if not np.isclose(clamped, value):
                slider.blockSignals(True)
                slider.setValue(int(clamped * slider.maximum()))
                slider.blockSignals(False)
                value = clamped
            gui.subvolume_center_labels[axis].setText(f"{value:.3f}")

    def _sync_current_bounds(self) -> None:
        gui = self.gui
        group = getattr(gui, 'subvolume_group', None)
        if group and group.isChecked():
            gui.current_subvolume_bounds = self.get_active_bounds()
        else:
            gui.current_subvolume_bounds = None
