from abc import ABC, abstractmethod


class RendererBase(ABC):
    """Base class for renderers so implementations stay swappable"""

    def __init__(self, view, extent):
        self.view = view
        self.extent = extent
        self.visual = None
        self._active = False

    def set_active(self, active: bool):
        self._active = active
        if self.visual is not None:
            self.visual.parent = self.view.scene if active else None

    def clear(self):
        if self.visual is not None:
            self.visual.parent = None
            self.visual = None


__all__ = ["RendererBase"]
