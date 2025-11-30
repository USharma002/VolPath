import numpy as np
from vispy import scene

from renderers.base import RendererBase


class ReferenceCube(RendererBase):
    """Wireframe cube to show the [0,1]^3 bounds"""

    def __init__(self, view, extent):
        super().__init__(view, extent)

    def render(self, color=(0.9, 0.9, 0.9, 0.6), visible=True):
        self.clear()
        corners = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ], dtype=np.uint32)

        self.visual = scene.visuals.Line(
            pos=corners,
            connect=edges,
            color=color,
            parent=self.view.scene if visible else None,
            method='gl'
        )
        self.visual.set_gl_state('translucent', depth_test=True, blend=True)
        self.set_active(visible)
        return self.visual


__all__ = ["ReferenceCube"]
