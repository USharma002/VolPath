import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QPainter, QPen, QPolygonF, QImage
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

# Import cached colormap getter from transfer_function module
from transfer.transfer_function import get_cached_colormap


class ColorBarWidget(QWidget):
    """Compact color bar preview that honors opacity curve."""

    def __init__(self, transfer_function, parent=None):
        super().__init__(parent)
        self.tf = transfer_function
        self.setMinimumHeight(30)
        self.setMaximumHeight(40)
        self._cached_image = None
        self._cached_colormap = None
        self._cached_width = 0

    def paintEvent(self, _event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        if width <= 0 or height <= 0:
            return
        
        # Only rebuild if colormap changed or width changed significantly
        if (self._cached_image is None or 
            self._cached_colormap != self.tf.colormap or 
            abs(self._cached_width - width) > 10):
            self._rebuild_image(width, height)
        
        if self._cached_image is not None:
            painter.drawImage(0, 0, self._cached_image.scaled(width, height))
        
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawRect(0, 0, width - 1, height - 1)
    
    def _rebuild_image(self, width, height):
        """Rebuild the cached colorbar image."""
        cmap = get_cached_colormap(self.tf.colormap)
        opacity_array = self.tf.get_opacity_array()
        
        # Create image buffer
        self._cached_image = QImage(width, height, QImage.Format.Format_ARGB32)
        
        for i in range(width):
            norm_pos = i / width
            color_rgba = cmap(norm_pos)
            opacity = opacity_array[int(norm_pos * 255)]
            r, g, b = int(color_rgba[0] * 255), int(color_rgba[1] * 255), int(color_rgba[2] * 255)
            a = int(opacity * 255)
            color_val = (a << 24) | (r << 16) | (g << 8) | b
            for j in range(height):
                self._cached_image.setPixel(i, j, color_val)
        
        self._cached_colormap = self.tf.colormap
        self._cached_width = width


class OpacityEditorWidget(QWidget):
    """Simple curve editor for opacity control points with histogram background."""

    pointsChanged = pyqtSignal()

    def __init__(self, transfer_function, parent=None):
        super().__init__(parent)
        self.tf = transfer_function
        self.setMinimumHeight(150)
        self.selected_index = -1
        self.point_radius = 8
        self.setMouseTracking(True)
        self._histogram = None  # Normalized histogram values (0-1)
        self._histogram_bins = 256
        self._cached_histogram_image = None
        self._cached_colormap = None
        self._cached_size = (0, 0)

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

    def set_histogram(self, data):
        """Set histogram from data array. Computes normalized histogram for display."""
        if data is None or len(data) == 0:
            self._histogram = None
            self._cached_histogram_image = None
            return
        # Compute histogram - use pre-flattened if already 1D
        flat_data = data.ravel() if data.ndim > 1 else data
        hist, _ = np.histogram(flat_data, bins=self._histogram_bins, range=(flat_data.min(), flat_data.max()))
        # Normalize to 0-1 range using log scale for better visualization
        hist = hist.astype(np.float32)
        np.log1p(hist, out=hist)  # In-place log(1 + x) to handle zeros
        max_val = hist.max()
        if max_val > 0:
            hist /= max_val  # In-place division
        self._histogram = hist
        self._cached_histogram_image = None  # Invalidate cache
        self.update()

    def _rebuild_histogram_image(self, width, height):
        """Rebuild cached histogram image with colormap colors."""
        if self._histogram is None or len(self._histogram) == 0:
            self._cached_histogram_image = None
            return
        
        cmap = get_cached_colormap(self.tf.colormap)
        self._cached_histogram_image = QImage(width, height, QImage.Format.Format_ARGB32)
        self._cached_histogram_image.fill(QColor(25, 25, 30))
        
        bin_width = width / len(self._histogram)
        
        for i, h in enumerate(self._histogram):
            if h > 0.01:
                x = int(i * bin_width)
                bar_height = int(h * height)
                norm_x = i / len(self._histogram)
                color_rgba = cmap(norm_x)
                # Semi-transparent histogram bars
                r, g, b = int(color_rgba[0] * 255), int(color_rgba[1] * 255), int(color_rgba[2] * 255)
                a = 89  # ~0.35 * 255
                
                for px in range(max(0, x), min(width, int(x + bin_width) + 1)):
                    for py in range(height - bar_height, height):
                        if 0 <= py < height:
                            self._cached_histogram_image.setPixel(px, py, (a << 24) | (r << 16) | (g << 8) | b)
        
        self._cached_colormap = self.tf.colormap
        self._cached_size = (width, height)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw histogram background if available (cached)
        if self._histogram is not None and len(self._histogram) > 0:
            # Check if cache needs rebuilding
            if (self._cached_histogram_image is None or 
                self._cached_colormap != self.tf.colormap or
                abs(self._cached_size[0] - width) > 10 or
                abs(self._cached_size[1] - height) > 10):
                self._rebuild_histogram_image(width, height)
            
            if self._cached_histogram_image is not None:
                painter.drawImage(0, 0, self._cached_histogram_image.scaled(width, height))
        else:
            # Dark background with subtle gradient
            painter.fillRect(self.rect(), QColor(25, 25, 30))
        
        # Draw grid lines
        grid_pen = QPen(QColor(50, 50, 55), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        for i in range(1, 4):
            x = width * i // 4
            y = height * i // 4
            painter.drawLine(x, 0, x, height)
            painter.drawLine(0, y, width, y)
        
        # Draw opacity curve with glow effect
        poly_points = [self._norm_to_pixel(p) for p in self.tf.opacity_points]
        
        # Glow behind the curve
        glow_pen = QPen(QColor(80, 140, 255, 60), 6)
        painter.setPen(glow_pen)
        painter.drawPolyline(QPolygonF(poly_points))
        
        # Main curve
        line_pen = QPen(QColor(100, 180, 255), 2)
        painter.setPen(line_pen)
        painter.drawPolyline(QPolygonF(poly_points))
        
        # Draw control points
        for i, point in enumerate(self.tf.opacity_points):
            pixel_pos = self._norm_to_pixel(point)
            if i == self.selected_index:
                painter.setBrush(QBrush(QColor(255, 220, 50)))  # Yellow for selected
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            elif i == 0 or i == len(self.tf.opacity_points) - 1:
                painter.setBrush(QBrush(QColor(50, 180, 180)))  # Teal for endpoints
                painter.setPen(QPen(QColor(100, 220, 220), 1))
            else:
                painter.setBrush(QBrush(QColor(80, 200, 255)))  # Cyan for middle points
                painter.setPen(QPen(QColor(150, 230, 255), 1))
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
        # Full list of colormaps
        cmaps = [
            # Perceptually uniform
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            # Additional uniform
            'turbo', 'twilight', 'twilight_shifted',
            # Diverging
            'coolwarm', 'bwr', 'seismic', 'RdBu', 'RdYlBu', 'RdYlGn', 'PiYG', 'PRGn',
            # Rainbow-like
            'jet', 'rainbow', 'nipy_spectral', 'gist_rainbow', 'hsv',
            # Sequential (warm)
            'hot', 'afmhot', 'gist_heat', 'copper', 'YlOrRd', 'YlOrBr', 'Oranges', 'Reds',
            # Sequential (cool)
            'cool', 'Blues', 'YlGnBu', 'PuBuGn', 'GnBu', 'PuBu', 'BuPu',
            # Grayscale variants
            'Greys', 'bone', 'pink', 'binary', 'gray',
            # Other
            'terrain', 'ocean', 'cubehelix', 'gnuplot', 'gnuplot2',
        ]
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
        self.opacity_editor.update()  # Update histogram colors to match colormap
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
