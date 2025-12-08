import numpy as np

# Global colormap cache to avoid repeated matplotlib imports and lookups
_COLORMAP_CACHE = {}
_LINSPACE_256 = np.linspace(0, 1, 256, dtype=np.float32)  # Pre-computed linspace


def get_cached_colormap(name):
    """Get a cached matplotlib colormap by name."""
    if name not in _COLORMAP_CACHE:
        import matplotlib.pyplot as plt
        _COLORMAP_CACHE[name] = plt.get_cmap(name)
    return _COLORMAP_CACHE[name]


def clear_colormap_cache():
    """Clear the colormap cache to free memory."""
    _COLORMAP_CACHE.clear()


class TransferFunction:
    """Manages color and opacity mapping for data visualization.
    
    Uses matplotlib colormaps directly (PyVista also uses matplotlib internally).
    Memory optimized with __slots__.
    """
    
    __slots__ = ('colormap', 'opacity_points', '_opacity_array', 
                 '_colormap_cache', '_rgba_table_cache', '_cmap_obj')

    def __init__(self, colormap='viridis'):
        # Default: linear opacity ramp from 0 to 1
        self.opacity_points = [[0.0, 0.0], [1.0, 1.0]]  # [x, opacity] pairs
        self._opacity_array = None
        self._colormap_cache = None
        self._rgba_table_cache = None
        self._cmap_obj = None  # Cached colormap object
        self.colormap = None  # Initialize before set_colormap
        self.set_colormap(colormap)

    def _update_opacity_array(self):
        sorted_points = sorted(self.opacity_points, key=lambda p: p[0])
        x = np.array([p[0] for p in sorted_points], dtype=np.float32)
        y = np.array([p[1] for p in sorted_points], dtype=np.float32)
        self._opacity_array = np.clip(np.interp(_LINSPACE_256, x, y), 0, 1).astype(np.float32)
        self._rgba_table_cache = None

    def get_opacity_array(self):
        if self._opacity_array is None:
            self._update_opacity_array()
        return self._opacity_array

    def set_colormap(self, colormap_name):
        if getattr(self, 'colormap', None) == colormap_name:
            return
        self.colormap = colormap_name
        self._cmap_obj = get_cached_colormap(colormap_name)
        self._rgba_table_cache = None

    def _build_rgba_table(self):
        if self._cmap_obj is None:
            self._cmap_obj = get_cached_colormap(self.colormap)
        rgba = np.asarray(self._cmap_obj(_LINSPACE_256), dtype=np.float32)
        rgba[:, 3] = self.get_opacity_array()
        return rgba

    def get_pyvista_colormap(self):
        """Return the colormap name for PyVista (uses matplotlib colormaps)."""
        return self.colormap

    def get_color_table(self):
        if self._rgba_table_cache is None:
            self._rgba_table_cache = self._build_rgba_table()
        return self._rgba_table_cache

    def get_opacity_transfer_function(self):
        """Return opacity as a list of [value, opacity] pairs for VTK/PyVista."""
        opacity = self.get_opacity_array()
        # Sample at regular intervals for performance (32 samples instead of 256)
        return [[i / 255.0, float(opacity[i])] for i in range(0, 256, 8)]

    def get_opacity_for_scalar(self, scalar_value, min_val, max_val):
        if max_val <= min_val:
            return 1.0
        normalized = np.clip((scalar_value - min_val) / (max_val - min_val), 0, 1)
        index = int(normalized * 255)
        return float(self._opacity_array[index]) if self._opacity_array is not None else 1.0

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
        # Reset to linear opacity ramp
        self.opacity_points = [[0.0, 0.0], [1.0, 1.0]]
        self._update_opacity_array()
    
    # Legacy compatibility aliases
    def get_vispy_colormap(self):
        """Legacy method - returns a colormap object for backward compatibility."""
        # Return the RGBA table which can be used similarly
        return self.get_color_table()


__all__ = ["TransferFunction"]
