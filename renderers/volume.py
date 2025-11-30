import numpy as np
from vispy import scene

from renderers.base import RendererBase


class VispyVolumeRenderer(RendererBase):
    """Default volume renderer backed by vispy Volume visual"""

    def __init__(self, view, extent, transfer_function):
        super().__init__(view, extent)
        self.transfer_function = transfer_function
        self.render_method = 'mip'
        self.relative_step_size = 1.0
        self.iso_threshold = 0.5

    def update_transfer_function(self, transfer_function):
        self.transfer_function = transfer_function
        if self.visual is not None:
            self.visual.cmap = self.transfer_function.get_vispy_colormap()

    def update_render_settings(self, *, method=None, step_size=None, threshold=None):
        if method:
            self.render_method = method
        if step_size is not None:
            self.relative_step_size = max(0.01, float(step_size))
        if threshold is not None:
            self.iso_threshold = float(threshold)
        self._apply_visual_settings()

    def render(self, grid_values, visible=True, bounds=None):
        if grid_values is None:
            self.clear()
            return None

        resolution = grid_values.shape[0]
        cmap = self.transfer_function.get_vispy_colormap()
        parent = self.view.scene if visible else None

        if self.visual is None:
            self.visual = scene.visuals.Volume(
                grid_values,
                parent=parent,
                cmap=cmap,
                clim=(0, 1),
                method=self.render_method
            )
        else:
            self.visual.parent = parent
            self.visual.set_data(grid_values)
            self.visual.cmap = cmap
            self.visual.clim = (0, 1)
        self._apply_visual_settings()

        if bounds is not None:
            mins, maxs = bounds
            mins = np.asarray(mins, dtype=np.float32)
            maxs = np.asarray(maxs, dtype=np.float32)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.ones(3, dtype=np.float32)

        scale_norm = maxs - mins
        scale_norm[scale_norm <= 1e-6] = 1e-6
        voxel_scale = self.extent * scale_norm / resolution
        half_voxel = voxel_scale / 2.0
        translate = self.extent * mins + half_voxel

        transform = scene.STTransform(
            scale=tuple(voxel_scale.tolist()),
            translate=tuple(translate.tolist())
        )
        self.visual.transform = transform

        self.set_active(visible)
        return self.visual

    def _apply_visual_settings(self):
        if not self.visual:
            return
        try:
            if hasattr(self.visual, 'method'):
                self.visual.method = self.render_method
        except Exception:
            pass
        try:
            if hasattr(self.visual, 'relative_step_size'):
                self.visual.relative_step_size = float(self.relative_step_size)
        except Exception:
            pass
        try:
            if hasattr(self.visual, 'threshold'):
                self.visual.threshold = float(self.iso_threshold)
        except Exception:
            pass


__all__ = ["VispyVolumeRenderer"]
