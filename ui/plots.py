from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class LossPlotWidget(FigureCanvas):
    """Matplotlib canvas to track neural training loss."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(4, 2), dpi=100)
        super().__init__(self.fig)
        self.ax.set_title("Neural Loss", fontsize=9)
        self.ax.set_xlabel("Step", fontsize=8)
        self.ax.set_ylabel("MSE", fontsize=8)
        self.ax.grid(alpha=0.3)
        (self.line,) = self.ax.plot([], [], color="#ff7f0e", linewidth=1.5)
        self.ax.tick_params(labelsize=8)
        self.steps = []
        self.losses = []
        self.fig.tight_layout()

    def append(self, step, loss):
        self.steps.append(step)
        self.losses.append(loss)
        self.line.set_data(self.steps, self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def clear_plot(self):
        self.steps.clear()
        self.losses.clear()
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()


__all__ = ["LossPlotWidget"]
