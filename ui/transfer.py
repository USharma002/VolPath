import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ColorBarWidget(QWidget):
    """Compact color bar preview that honors opacity curve."""

    def __init__(self, transfer_function, parent=None):
        super().__init__(parent)
        self.tf = transfer_function
        self.setMinimumHeight(30)
        self.setMaximumHeight(40)

    def paintEvent(self, _event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(self.tf.colormap)
        opacity_array = self.tf.get_opacity_array()
        for i in range(width):
            norm_pos = i / width
            color_rgba = cmap(norm_pos)
            opacity = opacity_array[int(norm_pos * 255)]
            color = QColor.fromRgbF(color_rgba[0], color_rgba[1], color_rgba[2], opacity)
            painter.setPen(color)
            painter.drawLine(i, 0, i, height)
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawRect(0, 0, width - 1, width - 1)


class OpacityEditorWidget(QWidget):
    """Simple curve editor for opacity control points."""

    pointsChanged = pyqtSignal()

    def __init__(self, transfer_function, parent=None):
        super().__init__(parent)
        self.tf = transfer_function
        self.setMinimumHeight(150)
        self.selected_index = -1
        self.point_radius = 8
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        pos = event.position()
        norm_pos = self._pixel_to_norm(pos)
        for i, point in enumerate(self.tf.opacity_points):
            if self._is_near_point(pos, point):
                self.selected_index = i
                if event.button() == Qt.MouseButton.RightButton:
                    self.tf.remove_point(i)
                    self.selected_index = -1
                    self.pointsChanged.emit()
                    self.update()
                return
        if event.button() == Qt.MouseButton.LeftButton:
            self.tf.add_point(norm_pos.x(), norm_pos.y())
            for i, point in enumerate(self.tf.opacity_points):
                if abs(point[0] - norm_pos.x()) < 0.001:
                    self.selected_index = i
                    break
            self.pointsChanged.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if self.selected_index == -1:
            return
        norm_pos = self._pixel_to_norm(event.position())
        x_val = np.clip(norm_pos.x(), 0, 1)
        y_val = np.clip(norm_pos.y(), 0, 1)
        self.tf.update_point(self.selected_index, x_val, y_val)
        self.pointsChanged.emit()
        self.update()

    def mouseReleaseEvent(self, _event):
        self.selected_index = -1

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        grid_pen = QPen(QColor(60, 60, 60), 1)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            x = self.width() * i / 4
            y = self.height() * i / 4
            painter.drawLine(int(x), 0, int(x), self.height())
            painter.drawLine(0, int(y), self.width(), int(y))
        line_pen = QPen(QColor(100, 150, 255), 2)
        painter.setPen(line_pen)
        poly_points = [self._norm_to_pixel(p) for p in self.tf.opacity_points]
        painter.drawPolyline(QPolygonF(poly_points))
        for i, point in enumerate(self.tf.opacity_points):
            pixel_pos = self._norm_to_pixel(point)
            if i == self.selected_index:
                painter.setBrush(QBrush(QColor("yellow")))
            elif i == 0 or i == len(self.tf.opacity_points) - 1:
                painter.setBrush(QBrush(QColor("darkCyan")))
            else:
                painter.setBrush(QBrush(QColor("cyan")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(pixel_pos, self.point_radius, self.point_radius)

    def _pixel_to_norm(self, pos):
        x_val = pos.x() / self.width() if self.width() > 0 else 0
        y_val = 1.0 - (pos.y() / self.height()) if self.height() > 0 else 0
        return QPointF(x_val, y_val)

    def _norm_to_pixel(self, point):
        x_val = point[0] * self.width()
        y_val = (1.0 - point[1]) * self.height()
        return QPointF(x_val, y_val)

    def _is_near_point(self, pixel_pos, norm_point):
        point_pixel = self._norm_to_pixel(norm_point)
        dx = pixel_pos.x() - point_pixel.x()
        dy = pixel_pos.y() - point_pixel.y()
        return (dx * dx + dy * dy) < (self.point_radius * 1.5) ** 2


class TransferFunctionDialog(QDialog):
    """Modal dialog that aggregates the TF widgets and emits live updates."""

    transferFunctionChanged = pyqtSignal(object)

    def __init__(self, transfer_function, parent=None):
        super().__init__(parent)
        self.tf = transfer_function
        self.setWindowTitle("Transfer Function Editor")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)

        layout = QVBoxLayout(self)

        cmap_group = QGroupBox("Colormap")
        cmap_layout = QHBoxLayout()
        cmap_label = QLabel("Select Colormap:")
        self.cmap_combo = QComboBox()
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm', 'seismic', 'turbo']
        self.cmap_combo.addItems(cmaps)
        self.cmap_combo.setCurrentText(self.tf.colormap)
        self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        cmap_layout.addWidget(cmap_label)
        cmap_layout.addWidget(self.cmap_combo, stretch=1)
        cmap_group.setLayout(cmap_layout)
        layout.addWidget(cmap_group)

        colorbar_group = QGroupBox("Preview")
        colorbar_layout = QVBoxLayout()
        self.colorbar = ColorBarWidget(self.tf)
        colorbar_layout.addWidget(self.colorbar)
        colorbar_group.setLayout(colorbar_layout)
        layout.addWidget(colorbar_group)

        opacity_group = QGroupBox("Opacity Function")
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Left-click: Add | Right-click: Delete | Drag: Move"))
        self.opacity_editor = OpacityEditorWidget(self.tf)
        self.opacity_editor.pointsChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_editor)

        range_layout = QHBoxLayout()
        self.min_label = QLabel("Min: N/A")
        self.max_label = QLabel("Max: N/A")
        range_layout.addWidget(self.min_label, alignment=Qt.AlignmentFlag.AlignLeft)
        range_layout.addStretch()
        range_layout.addWidget(self.max_label, alignment=Qt.AlignmentFlag.AlignRight)
        opacity_layout.addLayout(range_layout)

        reset_btn = QPushButton("Reset to Linear")
        reset_btn.clicked.connect(self._reset_opacity)
        opacity_layout.addWidget(reset_btn)
        opacity_group.setLayout(opacity_layout)
        layout.addWidget(opacity_group)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        apply_btn = button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_btn.clicked.connect(lambda: self.transferFunctionChanged.emit(self.tf))
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def set_data_range(self, min_val, max_val):
        self.min_label.setText(f"Min: {min_val:.3e}")
        self.max_label.setText(f"Max: {max_val:.3e}")

    def _on_colormap_changed(self, colormap_name):
        self.tf.set_colormap(colormap_name)
        self.colorbar.update()
        self.transferFunctionChanged.emit(self.tf)

    def _on_opacity_changed(self):
        self.colorbar.update()
        self.transferFunctionChanged.emit(self.tf)

    def _reset_opacity(self):
        self.tf.reset()
        self.opacity_editor.update()
        self.colorbar.update()
        self.transferFunctionChanged.emit(self.tf)


__all__ = [
    "ColorBarWidget",
    "OpacityEditorWidget",
    "TransferFunctionDialog",
]
