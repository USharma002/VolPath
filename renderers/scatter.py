from typing import Optional

import numpy as np
import pyvista as pv

from renderers.base import RendererBase


class ScatterPlotRenderer(RendererBase):
    """Particle scatter plot renderer using PyVista point clouds.
    
    Memory optimized with cached buffers and proper cleanup.
    """

    def __init__(self, plotter, extent, transfer_function):
        super().__init__(plotter, extent)
        self.transfer_function = transfer_function
        self.point_budget = None  # type: int | None
        self._sampling_seed = 1337
        self._point_cloud = None
        # Display settings (using reference config: point_size=4, spheres=True)
        self.point_size = 4
        self.render_as_spheres = True
        self.show_scalar_bar = False  # Off by default
        self._scalar_bar_actor = None
        # Scalar bar title for context
        self.scalar_bar_title = "Scalar Value"
        # Cached arrays to avoid reallocation
        self._cached_opacity_indices = None
        self._last_scalars_size = 0
        # Cached RNG for deterministic downsampling
        self._rng = None

    def update_transfer_function(self, transfer_function):
        self.transfer_function = transfer_function

    def update_opacity_from_transfer_function(self):
        """Update opacity values in-place without recreating the scatter plot."""
        if self._point_cloud is None or 'scalars' not in self._point_cloud.array_names:
            return
        
        try:
            # Get current scalars and compute new opacity
            scalars_norm = self._point_cloud['scalars']
            opacity_array = self.transfer_function.get_opacity_array()
            
            # Reuse cached index array if size matches
            if self._cached_opacity_indices is not None and len(scalars_norm) == self._last_scalars_size:
                np.multiply(scalars_norm, 255, out=self._cached_opacity_indices)
                np.clip(self._cached_opacity_indices, 0, 255, out=self._cached_opacity_indices)
                scalar_indices = self._cached_opacity_indices.astype(np.int32)
            else:
                scalar_indices = np.clip((scalars_norm * 255).astype(np.int32), 0, 255)
            
            point_opacities = opacity_array[scalar_indices]
            
            # Update opacity in place
            self._point_cloud['opacity'] = point_opacities
            
            # Force mesh update
            if self.actor is not None:
                self.actor.mapper.Update()
        except Exception as e:
            print(f"Scatter opacity update error: {e}")

    def set_point_size(self, size: int):
        """Set the point size for rendering."""
        self.point_size = max(1, min(20, int(size)))

    def set_render_as_spheres(self, enabled: bool):
        """Enable/disable rendering points as spheres."""
        self.render_as_spheres = bool(enabled)

    def set_show_scalar_bar(self, enabled: bool):
        """Show/hide the scalar bar."""
        self.show_scalar_bar = bool(enabled)
        if self._scalar_bar_actor:
            self._scalar_bar_actor.SetVisibility(enabled)
            self.plotter.render()

    def set_scalar_bar_title(self, title: str):
        """Set the title for the scalar bar."""
        self.scalar_bar_title = str(title)

    def set_point_budget(self, max_points: Optional[int]):
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

    def clear(self, render: bool = False):
        """Clear the scatter actor and scalar bar, releasing memory."""
        # Remove scalar bar first
        if self._scalar_bar_actor is not None:
            try:
                plotter = self.plotter
                if plotter is not None:
                    plotter.remove_scalar_bar()
            except Exception:
                pass
            self._scalar_bar_actor = None
        
        # Release point cloud memory
        if self._point_cloud is not None:
            self._point_cloud = None
        
        # Call base clear to remove the main actor
        super().clear(render=render)

    def _downsample_by_budget(self, coords: np.ndarray, scalars: np.ndarray):
        if self.point_budget is None:
            return coords, scalars

        total_points = len(coords)
        if total_points <= self.point_budget:
            return coords, scalars

        # Reuse RNG for consistent results and avoid recreation overhead
        if self._rng is None:
            self._rng = np.random.default_rng(self._sampling_seed)
        else:
            # Reset to same seed for deterministic behavior
            self._rng = np.random.default_rng(self._sampling_seed)
        
        selection = self._rng.choice(total_points, size=self.point_budget, replace=False)
        return coords[selection], scalars[selection]

    def render(self, coords_unit, scalars, resolution, visible=True):
        if coords_unit is None or scalars is None:
            return None

        # Ensure contiguous arrays, avoid copy if already contiguous and correct dtype
        if not coords_unit.flags['C_CONTIGUOUS'] or coords_unit.dtype != np.float32:
            coords_unit = np.ascontiguousarray(coords_unit, dtype=np.float32)
        if not scalars.flags['C_CONTIGUOUS'] or scalars.dtype != np.float32:
            scalars_arr = np.ascontiguousarray(scalars, dtype=np.float32)
        else:
            scalars_arr = scalars
            
        coords_unit, scalars_arr = self._downsample_by_budget(coords_unit, scalars_arr)
        if coords_unit.size == 0 or scalars_arr.size == 0:
            self.clear()
            return {'points': 0, 'scale': 1.0, 'resolution': resolution}

        # Fast finite check and normalization
        finite_mask = np.isfinite(scalars_arr)
        if not finite_mask.any():
            scalars_arr = np.zeros_like(scalars_arr)
            finite_mask = np.ones(len(scalars_arr), dtype=bool)
        
        valid_values = scalars_arr[finite_mask]
        scalars_min = float(valid_values.min())
        scalars_max = float(valid_values.max())
        
        if scalars_max > scalars_min:
            # In-place normalization to avoid allocation
            scalars_norm = scalars_arr.copy()  # Need copy to not modify original
            scalars_norm -= scalars_min
            scalars_norm /= (scalars_max - scalars_min)
            np.clip(scalars_norm, 0.0, 1.0, out=scalars_norm)
        else:
            scalars_norm = np.full(len(scalars_arr), 0.5, dtype=np.float32)

        # Scale coordinates to volume extent (in-place multiplication)
        coords_scaled = coords_unit * self.extent
        
        # Create PyVista point cloud
        self._point_cloud = pv.PolyData(coords_scaled)
        self._point_cloud['scalars'] = scalars_norm
        
        # Compute per-point opacity from transfer function efficiently
        opacity_array = self.transfer_function.get_opacity_array()
        
        # Pre-allocate or reuse index buffer
        n_points = len(scalars_norm)
        if self._cached_opacity_indices is None or self._last_scalars_size != n_points:
            self._cached_opacity_indices = np.empty(n_points, dtype=np.float32)
            self._last_scalars_size = n_points
        
        # Compute indices efficiently
        np.multiply(scalars_norm, 255, out=self._cached_opacity_indices)
        scalar_indices = np.clip(self._cached_opacity_indices, 0, 255).astype(np.int32)
        point_opacities = opacity_array[scalar_indices]
        
        # Store point cloud data before clearing
        point_cloud = pv.PolyData(coords_scaled)
        point_cloud['scalars'] = scalars_norm
        point_cloud['opacity'] = point_opacities

        # Clear previous actor (but don't clear _point_cloud yet)
        if self._scalar_bar_actor is not None:
            try:
                if self.plotter is not None:
                    self.plotter.remove_scalar_bar()
            except Exception:
                pass
            self._scalar_bar_actor = None
        super().clear(render=False)
        
        # Now assign the new point cloud
        self._point_cloud = point_cloud

        if not visible:
            return {'points': len(coords_unit), 'scale': 1.0, 'resolution': resolution}

        try:
            # Get colormap from transfer function
            cmap_name = self.transfer_function.colormap
            
            # Scalar bar arguments matching reference config
            scalar_bar_args = {
                'title': self.scalar_bar_title,
                'color': 'white',
                'width': 0.08,
                'height': 0.5,
                'vertical': True,
                'n_labels': 5,
                'fmt': '%.2f',
            }
            
            # Add points with per-point opacity from transfer function
            self.actor = self.plotter.add_points(
                self._point_cloud,
                scalars='scalars',
                cmap=cmap_name,
                clim=[0, 1],
                point_size=self.point_size,
                render_points_as_spheres=self.render_as_spheres,
                opacity='opacity',  # Use per-point opacity from transfer function
                show_scalar_bar=self.show_scalar_bar,
                scalar_bar_args=scalar_bar_args,
            )
            
            # Track scalar bar actor for visibility toggling
            if self.show_scalar_bar:
                self._scalar_bar_actor = self.plotter.scalar_bar
            
            self.set_active(visible)
            
        except Exception as e:
            print(f"Scatter plot error: {e}")
            import traceback
            traceback.print_exc()
            return None

        return {
            'points': len(coords_unit),
            'scale': 1.0,
            'resolution': resolution
        }


__all__ = ["ScatterPlotRenderer"]