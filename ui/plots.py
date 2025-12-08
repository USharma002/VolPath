from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Configure matplotlib once at module load for better performance
plt.style.use('dark_background')


class LossPlotWidget(FigureCanvas):
    """Matplotlib canvas to track neural training loss."""

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 2.5), dpi=100)
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#2d2d2d')
        super().__init__(self.fig)
        
        # Style the plot
        self.ax.set_title("Training Loss", fontsize=10, color='#e0e0e0', fontweight='bold')
        self.ax.set_xlabel("Step", fontsize=9, color='#b0b0b0')
        self.ax.set_ylabel("MSE Loss", fontsize=9, color='#b0b0b0')
        self.ax.grid(True, alpha=0.2, color='#555555', linestyle='--')
        
        # Gradient-colored line with glow effect
        (self.line,) = self.ax.plot([], [], color='#00d4ff', linewidth=2, 
                                     marker='', markersize=3, alpha=0.9)
        
        # Style axes
        self.ax.tick_params(labelsize=8, colors='#909090')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.spines['left'].set_color('#555555')
        
        self.steps = []
        self.losses = []
        self.fig.tight_layout(pad=1.5)
        
        # Throttle redraws - track last update time
        self._last_draw_time = 0
        self._draw_interval = 0.1  # Minimum 100ms between redraws

    def append(self, step, loss):
        import time
        self.steps.append(step)
        self.losses.append(loss)
        self.line.set_data(self.steps, self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Throttle redraws to reduce UI lag
        current_time = time.time()
        if current_time - self._last_draw_time >= self._draw_interval:
            self.draw_idle()
            self._last_draw_time = current_time

    def clear_plot(self):
        self.steps.clear()
        self.losses.clear()
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()


__all__ = ["LossPlotWidget"]
