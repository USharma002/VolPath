import numpy as np


class CameraController:
    """Encapsulates camera state for PyVista plotter."""

    DEFAULTS = {'azimuth': 45, 'elevation': 30, 'roll': 0, 'distance': 2.0, 'fov': 60}

    def __init__(self, plotter, extent):
        self.plotter = plotter
        self.extent = extent
        self._center = None
        self.ensure_center(force_set=True)  # Only set focal point during init

    def ensure_center(self, force_set=False):
        """Update internal center value. Only sets camera focal point if force_set=True."""
        if self.plotter is None:
            return
        center = self.extent / 2.0
        self._center = (center, center, center)
        # Only set focal point explicitly when requested (e.g., during reset)
        # Otherwise just cache the center value without moving the camera
        if force_set:
            try:
                self.plotter.camera.focal_point = self._center
            except Exception:
                pass

    def capture_state(self):
        if self.plotter is None:
            return {}
        try:
            camera = self.plotter.camera
            position = np.array(camera.position)
            focal = np.array(camera.focal_point)
            up = np.array(camera.up)
            
            # Calculate spherical coordinates from camera position
            diff = position - focal
            distance = float(np.linalg.norm(diff))
            
            # Calculate azimuth and elevation
            if distance > 0:
                normalized = diff / distance
                elevation = float(np.degrees(np.arcsin(np.clip(normalized[2], -1, 1))))
                azimuth = float(np.degrees(np.arctan2(normalized[0], normalized[1])))
            else:
                azimuth = self.DEFAULTS['azimuth']
                elevation = self.DEFAULTS['elevation']
            
            # Calculate roll from up vector
            roll = 0.0  # Simplified - would need more complex calculation for true roll
            
            return {
                'azimuth': azimuth,
                'elevation': elevation,
                'roll': roll,
                'distance': distance,
                'fov': float(camera.view_angle) if hasattr(camera, 'view_angle') else self.DEFAULTS['fov'],
                'center': tuple(focal),
                'position': tuple(position),
                'up': tuple(up),
            }
        except Exception:
            return {}

    def restore_state(self, state):
        if self.plotter is None or not state:
            return
        
        try:
            camera = self.plotter.camera
            
            # If we have direct position/focal/up, use those
            if 'position' in state and 'center' in state:
                camera.position = state['position']
                camera.focal_point = state['center']
                if 'up' in state:
                    camera.up = state['up']
            else:
                # Calculate position from spherical coordinates
                azimuth = np.radians(state.get('azimuth', self.DEFAULTS['azimuth']))
                elevation = np.radians(state.get('elevation', self.DEFAULTS['elevation']))
                distance = state.get('distance', self.DEFAULTS['distance'])
                
                center = state.get('center', self._center) or self._center
                
                # Spherical to Cartesian
                x = distance * np.cos(elevation) * np.sin(azimuth)
                y = distance * np.cos(elevation) * np.cos(azimuth)
                z = distance * np.sin(elevation)
                
                position = (center[0] + x, center[1] + y, center[2] + z)
                
                camera.position = position
                camera.focal_point = center
                camera.up = (0, 0, 1)
            
            # Set FOV if available
            fov = state.get('fov', self.DEFAULTS['fov'])
            if hasattr(camera, 'view_angle'):
                camera.view_angle = fov
            
            self.plotter.render()
            
        except Exception:
            pass

    def update_from_params(self, params):
        if self.plotter is None:
            return
        self.ensure_center()  # Just update cached center, don't move camera
        
        try:
            azimuth = np.radians(params.get('azimuth', self.DEFAULTS['azimuth']))
            elevation = np.radians(params.get('elevation', self.DEFAULTS['elevation']))
            distance = params.get('distance', self.DEFAULTS['distance'])
            fov = params.get('fov', self.DEFAULTS['fov'])
            
            center = self._center
            
            # Spherical to Cartesian
            x = distance * np.cos(elevation) * np.sin(azimuth)
            y = distance * np.cos(elevation) * np.cos(azimuth)
            z = distance * np.sin(elevation)
            
            position = (center[0] + x, center[1] + y, center[2] + z)
            
            camera = self.plotter.camera
            camera.position = position
            camera.focal_point = center
            camera.up = (0, 0, 1)
            
            if hasattr(camera, 'view_angle'):
                camera.view_angle = fov
            
            self.plotter.render()
            
        except Exception:
            pass

    def reset(self):
        self.restore_state(self.DEFAULTS)

    def to_config(self):
        config = self.capture_state()
        config['volume_extent'] = self.extent
        return config

    def apply_config(self, config):
        self.restore_state(config)


__all__ = ["CameraController"]
