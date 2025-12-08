import numpy as np
import pyvista as pv

from renderers.base import RendererBase


class ReferenceCube(RendererBase):
    """Wireframe cube to show the [0,1]^3 bounds using PyVista"""

    def __init__(self, plotter, extent):
        super().__init__(plotter, extent)

    def render(self, color=(0.9, 0.9, 0.9, 0.6), visible=True):
        self.clear()
        
        # Create a box mesh and extract edges for wireframe
        box = pv.Box(bounds=[0, self.extent, 0, self.extent, 0, self.extent])
        edges = box.extract_all_edges()
        
        if not visible:
            return None
        
        try:
            # Convert color tuple to RGB (PyVista doesn't support alpha in line color)
            rgb_color = color[:3] if len(color) >= 3 else (0.9, 0.9, 0.9)
            opacity = color[3] if len(color) >= 4 else 0.6
            
            self.actor = self.plotter.add_mesh(
                edges,
                color=rgb_color,
                opacity=opacity,
                line_width=1,
                render_lines_as_tubes=False,
                show_scalar_bar=False,
            )
            
            self.set_active(visible)
            
        except Exception as e:
            print(f"Reference cube error: {e}")
            return None
        
        return self.actor


__all__ = ["ReferenceCube"]
