import numpy as np
from vispy import scene

from renderers.base import RendererBase


class ScatterPlotRenderer(RendererBase):
    """Particle scatter plot renderer sharing transfer function state"""

    def __init__(self, view, extent, transfer_function):
        super().__init__(view, extent)
        self.transfer_function = transfer_function
        self.point_budget = None  # type: int | None
        self._sampling_seed = 1337

    def update_transfer_function(self, transfer_function):
        self.transfer_function = transfer_function

    def set_point_budget(self, max_points: int | None):
        """Limit the maximum number of rendered particles (None disables)."""
        if max_points is None:
            self.point_budget = None
        elif max_points <= 0:
            self.point_budget = None
        else:
            self.point_budget = int(max_points)

        # force re-render with the new sampling constraints on next call
        self.clear()

    def set_sampling_seed(self, seed: int):
        """Control random sampling deterministically for reproducible clouds."""
        self._sampling_seed = int(seed)

    def _ensure_visual(self):
        if self.visual is not None:
            return
        self.visual = scene.visuals.Markers(parent=None)
        self.visual.transform = scene.STTransform()
        self.visual.set_gl_state(
            'translucent',
            depth_test=True,
            blend=True,
            blend_func=('src_alpha', 'one_minus_src_alpha')
        )

    def _downsample_by_budget(self, coords: np.ndarray, scalars: np.ndarray):
        if self.point_budget is None:
            return coords, scalars

        total_points = len(coords)
        if total_points <= self.point_budget:
            return coords, scalars

        rng = np.random.default_rng(self._sampling_seed)
        selection = rng.choice(total_points, size=self.point_budget, replace=False)
        return coords[selection], scalars[selection]

    def render(self, coords_unit, scalars, resolution, visible=True):
        if coords_unit is None or scalars is None:
            return None

        coords_unit = np.ascontiguousarray(coords_unit, dtype=np.float32)
        scalars_arr = np.ascontiguousarray(scalars, dtype=np.float32)
        coords_unit, scalars_arr = self._downsample_by_budget(coords_unit, scalars_arr)
        if coords_unit.size == 0 or scalars_arr.size == 0:
            self.clear()
            return {'points': 0, 'scale': 1.0, 'resolution': resolution}

        finite_mask = np.isfinite(scalars_arr)
        if not finite_mask.any():
            scalars_arr = np.zeros_like(scalars_arr)
            finite_mask = np.ones_like(scalars_arr, dtype=bool)
        valid_values = scalars_arr[finite_mask]
        scalars_min = float(valid_values.min())
        scalars_max = float(valid_values.max())
        if scalars_max > scalars_min:
            scalars_norm = (scalars_arr - scalars_min) / (scalars_max - scalars_min)
        else:
            scalars_norm = np.ones_like(scalars_arr) * 0.5
        scalars_norm = np.clip(scalars_norm, 0.0, 1.0)

        indices = np.clip((scalars_norm * 255).astype(np.int32), 0, 255)
        color_table = self.transfer_function.get_color_table()
        colors = color_table[indices]

        self._ensure_visual()
        self.visual.set_data(
            coords_unit,
            edge_width=0,
            face_color=colors,
            size=5,
            edge_color=None
        )

        self.set_active(visible)
        return {
            'points': len(coords_unit),
            'scale': 1.0,
            'resolution': resolution
        }


__all__ = ["ScatterPlotRenderer"]