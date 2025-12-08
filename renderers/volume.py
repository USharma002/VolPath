import numpy as np
import pyvista as pv
import vtk

from renderers.base import RendererBase
from transfer.transfer_function import get_cached_colormap


class PyVistaVolumeRenderer(RendererBase):
    """Volume renderer backed by PyVista uniform grid volume rendering"""

    def __init__(self, plotter, extent, transfer_function):
        super().__init__(plotter, extent)
        self.transfer_function = transfer_function
        self.render_method = 'mip'  # mip, minip, composite, additive
        self.relative_step_size = 0.5  # Default: 0.5 for good quality
        self.iso_threshold = 0.5
        self.interpolation = 'trilinear'  # trilinear or nearest
        self.opacity_scale = 1.0
        self.shade = False
        self._current_grid = None
        self._volume_name = f"volume_{id(self)}"
        self._base_sample_distance = None
        self._cached_cmap = None
        self._cached_cmap_name = None

    def update_transfer_function(self, transfer_function):
        """Update the transfer function and refresh the volume appearance."""
        self.transfer_function = transfer_function
        if self.actor is not None:
            self._update_volume_properties()
            self.plotter.render()

    def _update_volume_properties(self):
        """Update volume color and opacity from transfer function without re-adding."""
        if self.actor is None:
            return
            
        try:
            vol_prop = self.actor.GetProperty()
            
            # Update color transfer function from colormap using cached cmap
            color_tf = vol_prop.GetRGBTransferFunction(0)
            color_tf.RemoveAllPoints()
            
            # Use cached colormap
            cmap_name = self.transfer_function.colormap
            if self._cached_cmap_name != cmap_name:
                self._cached_cmap = get_cached_colormap(cmap_name)
                self._cached_cmap_name = cmap_name
            
            cmap = self._cached_cmap
            # Sample every 8th point for performance (32 samples instead of 64)
            for i in range(0, 256, 8):
                t = i / 255.0
                r, g, b, _ = cmap(t)
                color_tf.AddRGBPoint(t, r, g, b)
            
            # Update opacity transfer function
            opacity_tf = vol_prop.GetScalarOpacity(0)
            opacity_tf.RemoveAllPoints()
            opacity_array = self.transfer_function.get_opacity_array()
            # Sample every 8th point for performance
            for i in range(0, 256, 8):
                t = i / 255.0
                opacity_tf.AddPoint(t, float(opacity_array[i]))
                
        except Exception as e:
            print(f"[VolumeRenderer] Error updating properties: {e}")

    def update_render_settings(self, *, method=None, step_size=None, threshold=None):
        """Update rendering parameters."""
        needs_rebuild = False
        
        if method is not None and method != self.render_method:
            self.render_method = method
            needs_rebuild = True  # Blending mode change requires rebuild
            
        if step_size is not None:
            self.relative_step_size = max(0.1, min(4.0, float(step_size)))
            # Update sample distance without rebuild
            if self.actor is not None and self._base_sample_distance is not None:
                try:
                    vol_mapper = self.actor.GetMapper()
                    new_dist = self._base_sample_distance * self.relative_step_size
                    vol_mapper.SetSampleDistance(new_dist)
                except Exception:
                    pass
                    
        if threshold is not None:
            self.iso_threshold = float(threshold)
            
        if needs_rebuild and self.actor is not None:
            self._rerender_volume()
        elif self.actor is not None:
            self.plotter.render()

    def _rerender_volume(self):
        """Re-render volume with current settings."""
        if self._current_grid is None:
            return
        was_visible = self.actor is not None
        self.clear()
        if was_visible:
            self._add_volume_actor()

    def render(self, grid_values, visible=True, bounds=None):
        if grid_values is None:
            self.clear()
            return None

        # Check for GPU texture size limits - downsample if too large
        max_texture_dim = self._get_max_texture_size()
        original_shape = grid_values.shape
        
        if any(dim > max_texture_dim for dim in grid_values.shape):
            # Downsample to fit GPU texture limits
            scale_factor = max_texture_dim / max(grid_values.shape)
            new_shape = tuple(max(1, int(dim * scale_factor)) for dim in grid_values.shape)
            print(f"[VolumeRenderer] Downsampling {original_shape} → {new_shape} (GPU texture limit: {max_texture_dim})")
            
            # Use scipy zoom for smooth downsampling
            try:
                from scipy.ndimage import zoom
                zoom_factors = tuple(n / o for n, o in zip(new_shape, original_shape))
                grid_values = zoom(grid_values, zoom_factors, order=1).astype(np.float32)
            except ImportError:
                # Fallback: simple strided slicing
                strides = tuple(max(1, o // n) for o, n in zip(original_shape, new_shape))
                grid_values = grid_values[::strides[0], ::strides[1], ::strides[2]].copy()
            
            print(f"[VolumeRenderer] Downsampled to {grid_values.shape}")

        print(f"[VolumeRenderer] Grid shape: {grid_values.shape}, range: [{grid_values.min():.4f}, {grid_values.max():.4f}]")
        
        # Set spacing and origin based on bounds
        if bounds is not None:
            mins, maxs = bounds
            mins = np.asarray(mins, dtype=np.float32)
            maxs = np.asarray(maxs, dtype=np.float32)
        else:
            mins = np.zeros(3, dtype=np.float32)
            maxs = np.ones(3, dtype=np.float32)

        scale_norm = maxs - mins
        scale_norm[scale_norm <= 1e-6] = 1e-6
        
        # Calculate spacing
        grid_shape = np.array(grid_values.shape, dtype=np.float32)
        world_size = self.extent * scale_norm
        spacing = world_size / (grid_shape - 1)
        origin = self.extent * mins

        # Create uniform grid
        grid = pv.ImageData(
            dimensions=grid_values.shape,
            spacing=tuple(spacing.tolist()),
            origin=tuple(origin.tolist())
        )
        
        # Add scalar data - VTK expects Fortran order
        # Avoid copy if already float32 and contiguous
        if grid_values.dtype == np.float32:
            scalar_data = grid_values.ravel(order='F')
        else:
            scalar_data = grid_values.astype(np.float32).ravel(order='F')
        grid.point_data['values'] = scalar_data
        
        # Calculate base sample distance for quality control
        grid_size = max(grid_values.shape)
        base_sample_distance = 1.0 / grid_size
        
        # Clear previous actor (but don't clear _current_grid yet)
        plotter = self.plotter
        if self.actor is not None and plotter is not None:
            try:
                plotter.remove_actor(self._volume_name, render=False)
            except Exception:
                try:
                    plotter.remove_actor(self.actor, render=False)
                except Exception:
                    pass
        self.actor = None
        
        # Now assign the new grid and sample distance
        self._current_grid = grid
        self._base_sample_distance = base_sample_distance
        
        if not visible:
            return None
        
        return self._add_volume_actor()

    def _get_max_texture_size(self):
        """Get maximum 3D texture dimension supported by GPU."""
        import os
        
        # Allow override via environment variable
        env_limit = os.environ.get('VOLPATH_MAX_TEXTURE_SIZE')
        if env_limit:
            try:
                return int(env_limit)
            except ValueError:
                pass
        
        # Try to query OpenGL limit via VTK
        try:
            import vtk
            # Create a temporary render window to query GL limits
            rw = vtk.vtkRenderWindow()
            rw.SetOffScreenRendering(1)
            rw.SetSize(1, 1)
            rw.Render()
            
            # Query capabilities
            gl_info = rw.ReportCapabilities()
            rw.Finalize()
            del rw
            
            # Parse GL_MAX_3D_TEXTURE_SIZE from capabilities
            for line in gl_info.split('\n'):
                if 'Max 3D Texture Size' in line or 'GL_MAX_3D_TEXTURE_SIZE' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            detected = int(parts[1].strip())
                            print(f"[VolumeRenderer] Detected GPU max 3D texture size: {detected}")
                            return detected
                        except ValueError:
                            pass
        except Exception as e:
            print(f"[VolumeRenderer] Could not detect GPU limits: {e}")
        
        # Default for modern discrete GPUs (RTX/GTX series, AMD RX)
        # 2048³ = 8GB for float32, but we use normalized 0-1 so it's manageable
        # Most discrete GPUs from last 10 years support 2048³ or higher
        # Set VOLPATH_MAX_TEXTURE_SIZE env var if you need a different value
        print("[VolumeRenderer] Using default max texture size: 2048 (set VOLPATH_MAX_TEXTURE_SIZE to override)")
        return 2048

    def _add_volume_actor(self):
        """Add the volume actor to the plotter."""
        if self._current_grid is None:
            return None
            
        try:
            # Map render method to blending mode
            blending = 'composite'
            if self.render_method == 'mip':
                blending = 'maximum'
            elif self.render_method == 'minip':
                blending = 'minimum'
            elif self.render_method == 'additive':
                blending = 'additive'
            
            # Get opacity from transfer function and apply scale
            opacity_array = self.transfer_function.get_opacity_array()
            # Sample to 8 points for PyVista opacity parameter
            opacity = [float(opacity_array[int(i * 255 / 7)] * self.opacity_scale) for i in range(8)]
            opacity = [min(1.0, max(0.0, o)) for o in opacity]  # Clamp
            
            cmap_name = self.transfer_function.colormap
            
            # Add volume with current settings
            self.actor = self.plotter.add_volume(
                self._current_grid,
                scalars='values',
                cmap=cmap_name,
                opacity=opacity,
                clim=[0.0, 1.0],
                show_scalar_bar=False,
                shade=self.shade,
                ambient=0.3 if self.shade else 0.0,
                diffuse=0.6 if self.shade else 1.0,
                specular=0.3 if self.shade else 0.0,
                blending=blending,
                mapper='gpu',
                name=self._volume_name,
            )
            
            if self.actor is not None:
                self.set_active(True)
                self._apply_advanced_settings()
                self.plotter.render()
                print(f"[VolumeRenderer] Volume added: {blending}, interp={self.interpolation}, step={self.relative_step_size}")
            
        except Exception as e:
            print(f"[VolumeRenderer] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        return self.actor

    def _apply_advanced_settings(self):
        """Apply interpolation, step size, and other quality settings."""
        if self.actor is None:
            return
            
        try:
            vol_prop = self.actor.GetProperty()
            
            # Set interpolation type FIRST - this is the key for smooth vs blocky
            if self.interpolation == 'trilinear':
                vol_prop.SetInterpolationTypeToLinear()
            else:
                vol_prop.SetInterpolationTypeToNearest()
            
            # Update opacity with scale - sample every 8th point
            opacity_array = self.transfer_function.get_opacity_array()
            opacity_tf = vol_prop.GetScalarOpacity(0)
            opacity_tf.RemoveAllPoints()
            for i in range(0, 256, 8):
                t = i / 255.0
                o = min(1.0, float(opacity_array[i]) * self.opacity_scale)
                opacity_tf.AddPoint(t, o)
                
        except Exception as e:
            print(f"[VolumeRenderer] Property settings error: {e}")
        
        # Configure mapper separately - some methods may not exist in all VTK versions
        try:
            vol_mapper = self.actor.GetMapper()
            
            # Disable auto-adjust to use our step size
            if hasattr(vol_mapper, 'SetAutoAdjustSampleDistances'):
                vol_mapper.SetAutoAdjustSampleDistances(False)
            
            # Set sample distance based on step size parameter
            if self._base_sample_distance and hasattr(vol_mapper, 'SetSampleDistance'):
                sample_dist = self._base_sample_distance * self.relative_step_size
                vol_mapper.SetSampleDistance(sample_dist)
            
            # This method may not exist in all VTK versions - skip if not available
            if hasattr(vol_mapper, 'SetInteractiveAdjustSampleDistances'):
                vol_mapper.SetInteractiveAdjustSampleDistances(True)
                
        except Exception as e:
            print(f"[VolumeRenderer] Mapper settings error: {e}")

    def clear(self, render: bool = False):
        """Remove the volume from the plotter and release memory."""
        plotter = self.plotter
        if self.actor is not None and plotter is not None:
            try:
                plotter.remove_actor(self._volume_name, render=False)
            except Exception:
                try:
                    plotter.remove_actor(self.actor, render=False)
                except Exception:
                    pass
        self.actor = None
        # Release grid reference to free memory
        self._current_grid = None
        self._base_sample_distance = None
        if render and plotter is not None:
            plotter.render()


# Backward compatibility alias
VispyVolumeRenderer = PyVistaVolumeRenderer


__all__ = ["PyVistaVolumeRenderer", "VispyVolumeRenderer"]
