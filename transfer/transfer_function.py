import numpy as np
from vispy.color import Colormap


class TransferFunction:
    """Manages color and opacity mapping for data visualization"""

    def __init__(self, colormap='viridis'):
        self.opacity_points = [[0.0, 0.0], [1.0, 1.0]]  # [x, opacity] pairs
        self._opacity_array = None
        self._colormap_cache = None
        self._rgba_table_cache = None
        self.set_colormap(colormap)

    def _update_opacity_array(self):
        sorted_points = sorted(self.opacity_points, key=lambda p: p[0])
        x = np.array([p[0] for p in sorted_points])
        y = np.array([p[1] for p in sorted_points])
        x_new = np.linspace(0, 1, 256)
        self._opacity_array = np.clip(np.interp(x_new, x, y), 0, 1)
        self._colormap_cache = None
        self._rgba_table_cache = None

    def get_opacity_array(self):
        if self._opacity_array is None:
            self._update_opacity_array()
        return self._opacity_array

    def set_colormap(self, colormap_name):
        if getattr(self, 'colormap', None) == colormap_name:
            return
        self.colormap = colormap_name
        self._colormap_cache = None
        self._rgba_table_cache = None

    def _build_rgba_table(self):
        import matplotlib.pyplot as plt

        x = np.linspace(0, 1, 256)
        rgba = plt.get_cmap(self.colormap)(x)
        rgba = np.array(rgba, dtype=np.float32)
        rgba[:, 3] = self.get_opacity_array()
        return rgba

    def get_vispy_colormap(self):
        if self._colormap_cache is None:
            rgba = self._build_rgba_table()
            self._colormap_cache = Colormap(rgba)
            self._rgba_table_cache = rgba
        return self._colormap_cache

    def get_color_table(self):
        if self._rgba_table_cache is None:
            self.get_vispy_colormap()
        return self._rgba_table_cache

    def get_opacity_for_scalar(self, scalar_value, min_val, max_val):
        if max_val <= min_val:
            return 1.0
        normalized = (scalar_value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        index = int(normalized * 255)
        return self._opacity_array[index]

    def add_point(self, x_val, y_val):
        self.opacity_points.append([x_val, y_val])
        self.opacity_points.sort(key=lambda p: p[0])
        self._update_opacity_array()

    def remove_point(self, index):
        if 0 < index < len(self.opacity_points) - 1:
            self.opacity_points.pop(index)
            self._update_opacity_array()

    def update_point(self, index, x_val, y_val):
        if index == 0:
            x_val = 0.0
        elif index == len(self.opacity_points) - 1:
            x_val = 1.0
        self.opacity_points[index] = [x_val, y_val]
        self.opacity_points.sort(key=lambda p: p[0])
        self._update_opacity_array()

    def reset(self):
        self.opacity_points = [[0.0, 0.0], [1.0, 1.0]]
        self._update_opacity_array()


__all__ = ["TransferFunction"]
