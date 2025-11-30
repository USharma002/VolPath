class CameraController:
    """Encapsulates camera state so it can be swapped or extended later"""

    DEFAULTS = {'azimuth': 45, 'elevation': 30, 'roll': 0, 'distance': 2.0, 'fov': 60}

    def __init__(self, camera, extent):
        self.camera = camera
        self.extent = extent
        self.ensure_center()

    def ensure_center(self):
        if self.camera is None:
            return
        center = self.extent / 2.0
        self.camera.center = (center, center, center)

    def capture_state(self):
        if self.camera is None:
            return {}
        return {
            'azimuth': float(self.camera.azimuth),
            'elevation': float(self.camera.elevation),
            'roll': float(self.camera.roll),
            'distance': float(self.camera.distance),
            'fov': float(self.camera.fov),
            'center': self.camera.center
        }

    def restore_state(self, state):
        if self.camera is None or not state:
            return
        self.camera.azimuth = state.get('azimuth', self.DEFAULTS['azimuth'])
        self.camera.elevation = state.get('elevation', self.DEFAULTS['elevation'])
        self.camera.roll = state.get('roll', self.DEFAULTS['roll'])
        self.camera.distance = state.get('distance', self.DEFAULTS['distance'])
        self.camera.fov = state.get('fov', self.DEFAULTS['fov'])
        center = state.get('center')
        if center is not None:
            self.camera.center = center
        else:
            self.ensure_center()

    def update_from_params(self, params):
        if self.camera is None:
            return
        self.ensure_center()
        for key, value in params.items():
            setattr(self.camera, key, value)

    def reset(self):
        self.restore_state(self.DEFAULTS)

    def to_config(self):
        config = self.capture_state()
        config['volume_extent'] = self.extent
        return config

    def apply_config(self, config):
        self.restore_state(config)


__all__ = ["CameraController"]
