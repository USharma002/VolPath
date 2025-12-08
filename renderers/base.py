from abc import ABC, abstractmethod
import weakref


class RendererBase(ABC):
    """Base class for renderers so implementations stay swappable.
    
    Now uses PyVista as the backend instead of vispy.
    The 'plotter' is a pyvista.Plotter instance, and 'actor' replaces 'visual'.
    
    Memory optimized with __slots__ and proper cleanup.
    """
    
    __slots__ = ('_plotter_ref', 'extent', 'actor', '_active')

    def __init__(self, plotter, extent):
        # Use weakref to avoid circular references with plotter
        self._plotter_ref = weakref.ref(plotter) if plotter is not None else None
        self.extent = extent
        self.actor = None
        self._active = False
    
    @property
    def plotter(self):
        """Get plotter from weakref, returns None if plotter was garbage collected."""
        return self._plotter_ref() if self._plotter_ref is not None else None
    
    @plotter.setter
    def plotter(self, value):
        self._plotter_ref = weakref.ref(value) if value is not None else None

    def set_active(self, active: bool, render: bool = False):
        """Set visibility of the actor.
        
        Args:
            active: Whether to show or hide the actor
            render: Whether to trigger a render (default False to allow batch updates)
        """
        self._active = active
        if self.actor is not None:
            try:
                self.actor.SetVisibility(active)
            except Exception:
                pass
            if render:
                plotter = self.plotter
                if plotter is not None:
                    plotter.render()

    def clear(self, render: bool = False):
        """Remove the actor from the plotter.
        
        Args:
            render: Whether to trigger a render after clearing (default False)
        """
        plotter = self.plotter
        if self.actor is not None and plotter is not None:
            try:
                plotter.remove_actor(self.actor, render=render)
            except Exception:
                pass
        self.actor = None
        self._active = False

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.clear(render=False)
        except Exception:
            pass

    # Legacy property for backward compatibility during transition
    @property
    def visual(self):
        return self.actor

    @visual.setter
    def visual(self, value):
        self.actor = value


__all__ = ["RendererBase"]
