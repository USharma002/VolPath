from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class MetadataDialog(QDialog):
    """Shows snapshot header attributes and dataset stats."""

    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Snapshot Metadata")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        header_table = QTableWidget(self)
        header_table.setColumnCount(2)
        header_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        header_table.horizontalHeader().setStretchLastSection(True)
        headers = metadata.get("headers", {})
        header_table.setRowCount(len(headers))
        for row, (key, value) in enumerate(sorted(headers.items())):
            header_table.setItem(row, 0, QTableWidgetItem(str(key)))
            header_table.setItem(row, 1, QTableWidgetItem(str(value)))
        tabs.addTab(header_table, "Header")

        datasets = metadata.get("datasets", [])
        dataset_table = QTableWidget(self)
        dataset_table.setColumnCount(4)
        dataset_table.setHorizontalHeaderLabels(["Dataset", "Length", "Shape", "DType"])
        dataset_table.horizontalHeader().setStretchLastSection(True)
        dataset_table.setRowCount(len(datasets))
        for row, entry in enumerate(sorted(datasets, key=lambda d: d["name"])):
            dataset_table.setItem(row, 0, QTableWidgetItem(entry["name"]))
            dataset_table.setItem(row, 1, QTableWidgetItem(str(entry["length"])) )
            dataset_table.setItem(row, 2, QTableWidgetItem(str(entry["shape"])) )
            dataset_table.setItem(row, 3, QTableWidgetItem(entry["dtype"]))
        tabs.addTab(dataset_table, "Datasets")


class ResolutionDialog(QDialog):
    """Prompt user for export resolution."""

    def __init__(self, default_width, default_height, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Resolution")
        self.setModal(True)
        layout = QFormLayout(self)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(128, 8192)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(int(max(128, default_width)))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(128, 8192)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(int(max(128, default_height)))

        layout.addRow("Width (pixels)", self.width_spin)
        layout.addRow("Height (pixels)", self.height_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_resolution(self):
        return self.width_spin.value(), self.height_spin.value()


class BatchConfigDialog(QDialog):
    """Dialog for configuring sequential rendering jobs with progress tracking."""

    def __init__(
        self,
        parent=None,
        current_resolution: int = 256,
        current_method: str = 'linear',
        current_epsilon: float = 1.0,
        allow_natural_neighbor: bool = False,
        batch_runner: Optional[Callable[[List[dict], str, str, Tuple[int, int], "BatchConfigDialog"], None]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Rendering Configuration")
        self.setGeometry(200, 200, 1100, 650)
        self.current_resolution = current_resolution
        self.current_method = current_method
        self.current_epsilon = current_epsilon
        self.allow_natural_neighbor = allow_natural_neighbor
        self.batch_runner = batch_runner

        self.configs: List[dict] = []
        self.is_rendering = False
        self._build_ui()
        self.add_config()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Batch Rendering Configuration")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        instructions = QLabel(
            "Configure multiple rendering jobs to run sequentially. "
            "Each job renders with a fixed canvas size and saves outputs."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(instructions)

        canvas_group = QGroupBox("Canvas Size (for batch renders)")
        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(QLabel("Width:"))
        self.canvas_width = QSpinBox()
        self.canvas_width.setRange(400, 4096)
        self.canvas_width.setValue(1920)
        self.canvas_width.setSingleStep(100)
        canvas_layout.addWidget(self.canvas_width)
        canvas_layout.addWidget(QLabel("Height:"))
        self.canvas_height = QSpinBox()
        self.canvas_height.setRange(400, 4096)
        self.canvas_height.setValue(1080)
        self.canvas_height.setSingleStep(100)
        canvas_layout.addWidget(self.canvas_height)
        canvas_layout.addStretch()
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Status", "Method", "Resolution", "Epsilon",
            "Colormap", "PNG", "NPY", "Actions"
        ])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        column_widths = [120, 160, 90, 70, 90, 50, 50, 80]
        for col, width in enumerate(column_widths):
            self.table.setColumnWidth(col, width)
        self.table.setMinimumHeight(250)
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        self.add_btn = QPushButton("➕ Add Configuration")
        self.add_btn.clicked.connect(self.add_config)
        button_layout.addWidget(self.add_btn)

        self.add_preset_btn = QPushButton("Add Preset...")
        self.add_preset_btn.clicked.connect(self.add_preset_menu)
        button_layout.addWidget(self.add_preset_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.textChanged.connect(self.update_summary)
        output_layout.addWidget(self.output_dir_edit)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.browse_btn)
        layout.addLayout(output_layout)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Filename Pattern:"))
        self.filename_pattern = QLineEdit("render_{method}_{resolution}")
        self.filename_pattern.setPlaceholderText("{method}_{resolution}_{epsilon}...")
        pattern_layout.addWidget(self.filename_pattern)
        help_btn = QPushButton("?")
        help_btn.setMaximumWidth(30)
        help_btn.clicked.connect(self.show_pattern_help)
        pattern_layout.addWidget(help_btn)
        layout.addLayout(pattern_layout)

        progress_group_layout = QVBoxLayout()
        self.overall_progress_label = QLabel("Overall Progress: Not started")
        self.overall_progress_label.setStyleSheet("font-weight: bold;")
        progress_group_layout.addWidget(self.overall_progress_label)
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setVisible(False)
        progress_group_layout.addWidget(self.overall_progress_bar)
        layout.addLayout(progress_group_layout)

        self.summary_label = QLabel("Configurations: 0 | Estimated time: -")
        self.summary_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        layout.addWidget(self.summary_label)

        button_box_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start Batch Render")
        self.start_btn.setStyleSheet(
            """
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
            """
        )
        self.start_btn.clicked.connect(self.start_batch_render)
        button_box_layout.addWidget(self.start_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close_dialog)
        button_box_layout.addWidget(self.close_btn)
        layout.addLayout(button_box_layout)

    def get_canvas_size(self) -> Tuple[int, int]:
        return (self.canvas_width.value(), self.canvas_height.value())

    def start_batch_render(self):
        configs = self.get_configurations()
        output_dir = self.get_output_dir()
        pattern = self.get_filename_pattern()
        canvas_size = self.get_canvas_size()

        if not output_dir:
            QMessageBox.warning(self, "Warning", "No output directory selected")
            return
        if len(configs) == 0:
            QMessageBox.warning(self, "Warning", "No configurations to render")
            return
        if not self.batch_runner:
            QMessageBox.critical(self, "Error", "No batch runner callback provided")
            return

        self.set_controls_enabled(False)
        self.is_rendering = True
        QApplication.processEvents()

        try:
            self.batch_runner(configs, output_dir, pattern, canvas_size, self)
        except Exception as exc:  # pragma: no cover - UI feedback path
            QMessageBox.critical(self, "Error", f"Batch render failed:\n{exc}")
            self.set_controls_enabled(True)
            self.is_rendering = False

    def close_dialog(self):
        if self.is_rendering:
            reply = QMessageBox.question(
                self,
                "Rendering in Progress",
                "Batch rendering is still running. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        self.reject()

    def set_controls_enabled(self, enabled):
        self.add_btn.setEnabled(enabled)
        self.add_preset_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)
        self.filename_pattern.setEnabled(enabled)
        self.canvas_width.setEnabled(enabled)
        self.canvas_height.setEnabled(enabled)
        if enabled:
            self.table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
            self.close_btn.setText("Close")
            self.close_btn.setStyleSheet("")
        else:
            self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.close_btn.setText("Cancel/Close")
            self.close_btn.setStyleSheet("color: red;")

    def on_batch_complete(self):
        self.is_rendering = False
        self.set_controls_enabled(True)
        self.overall_progress_label.setText("✓ Batch rendering complete!")
        self.overall_progress_label.setStyleSheet("color: green; font-weight: bold;")

    def add_config(self, method=None, resolution=None, epsilon=None, colormap='magma',
                   save_png=True, save_npy=True):
        row = self.table.rowCount()
        self.table.insertRow(row)

        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(2, 2, 2, 2)
        status_layout.setSpacing(2)
        status_label = QLabel("⏸ Pending")
        status_label.setStyleSheet("color: gray; font-size: 10px; font-weight: bold;")
        status_layout.addWidget(status_label)
        status_progress = QProgressBar()
        status_progress.setMaximumHeight(12)
        status_progress.setVisible(False)
        status_layout.addWidget(status_progress)
        self.table.setCellWidget(row, 0, status_widget)

        method_combo = QComboBox()
        methods = [
            'linear', 'nearest',
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic', 'rbf_linear',
            'rbf_gaussian', 'rbf_multiquadric', 'rbf_inverse_multiquadric',
            'rbf_inverse_quadratic'
        ]
        if self.allow_natural_neighbor:
            methods.append('natural_neighbor')
        method_combo.addItems(methods)
        method_combo.setCurrentText(method or self.current_method)
        method_combo.currentTextChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 1, method_combo)

        res_spin = QSpinBox()
        res_spin.setRange(32, 4096)
        res_spin.setValue(resolution if resolution else self.current_resolution)
        res_spin.setSingleStep(32)
        res_spin.valueChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 2, res_spin)

        eps_spin = QDoubleSpinBox()
        eps_spin.setRange(0.01, 100.0)
        eps_spin.setValue(epsilon if epsilon else self.current_epsilon)
        eps_spin.setSingleStep(0.1)
        eps_spin.setDecimals(2)
        eps_spin.valueChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 3, eps_spin)

        cmap_combo = QComboBox()
        cmap_combo.addItems(['magma', 'viridis', 'plasma', 'inferno', 'grays', 'hot', 'cool'])
        cmap_combo.setCurrentText(colormap)
        self.table.setCellWidget(row, 4, cmap_combo)

        png_check = QCheckBox()
        png_check.setChecked(save_png)
        png_widget = QWidget()
        png_layout = QHBoxLayout(png_widget)
        png_layout.addWidget(png_check)
        png_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 5, png_widget)

        npy_check = QCheckBox()
        npy_check.setChecked(save_npy)
        npy_widget = QWidget()
        npy_layout = QHBoxLayout(npy_widget)
        npy_layout.addWidget(npy_check)
        npy_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        npy_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 6, npy_widget)

        remove_btn = QPushButton("🗑️")
        remove_btn.setMaximumWidth(60)
        remove_btn.setStyleSheet("color: red;")
        remove_btn.clicked.connect(self.remove_config)
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.addWidget(remove_btn)
        action_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 7, action_widget)

        self.update_summary()

    def update_row_status(self, row, status, progress=None, style=None):
        status_widget = self.table.cellWidget(row, 0)
        if not status_widget:
            return
        status_label = status_widget.findChild(QLabel)
        status_progress = status_widget.findChild(QProgressBar)
        if status_label:
            status_label.setText(status)
            if style:
                status_label.setStyleSheet(style)
        if status_progress:
            if progress is not None:
                status_progress.setVisible(True)
                if progress == -1:
                    status_progress.setRange(0, 0)
                else:
                    if status_progress.maximum() == 0:
                        status_progress.setRange(0, 100)
                    status_progress.setValue(int(progress))
            else:
                status_progress.setVisible(False)

    def remove_config(self):
        button = self.sender()
        if not button:
            return
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 7)
            if widget and button in widget.findChildren(QPushButton):
                self.table.removeRow(r)
                break
        self.update_summary()

    def clear_all(self):
        reply = QMessageBox.question(
            self, "Clear All",
            "Remove all configurations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.table.setRowCount(0)
            self.update_summary()

    def add_preset_menu(self):
        menu = QMenu(self)
        presets = {
            "All Methods @ Current Resolution": self.preset_all_methods,
            "Resolution Comparison (Linear)": self.preset_resolution_comparison,
            "RBF Kernels Comparison": self.preset_rbf_comparison,
            "Quick Test (3 configs)": self.preset_quick_test,
        }
        for name, func in presets.items():
            action = menu.addAction(name)
            action.triggered.connect(func)
        menu.exec(self.add_preset_btn.mapToGlobal(self.add_preset_btn.rect().bottomLeft()))

    def preset_all_methods(self):
        methods = [
            'linear', 'nearest',
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic',
            'rbf_gaussian', 'rbf_multiquadric'
        ]
        if self.allow_natural_neighbor:
            methods.append('natural_neighbor')
        for method in methods:
            self.add_config(method=method, resolution=self.current_resolution)

    def preset_resolution_comparison(self):
        for res in [128, 256, 512, 768]:
            self.add_config(method='linear', resolution=res)

    def preset_rbf_comparison(self):
        rbf_kernels = [
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic', 'rbf_linear',
            'rbf_gaussian', 'rbf_multiquadric'
        ]
        for kernel in rbf_kernels:
            self.add_config(method=kernel, resolution=self.current_resolution)

    def preset_quick_test(self):
        self.add_config(method='linear', resolution=128)
        self.add_config(method='nearest', resolution=128)
        self.add_config(method='rbf_thin_plate_spline', resolution=128)

    def on_config_changed(self):
        self.update_summary()

    def update_summary(self):
        count = self.table.rowCount()
        if count == 0:
            self.summary_label.setText("Configurations: 0 | Estimated time: -")
            self.start_btn.setEnabled(False)
            return
        total_time = 0
        for row in range(count):
            res_spin = self.table.cellWidget(row, 2)
            if res_spin:
                resolution = res_spin.value()
                total_time += (resolution ** 3) / 100000
        minutes = int(total_time / 60)
        seconds = int(total_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        self.summary_label.setText(
            f"Configurations: {count} | Canvas: {self.canvas_width.value()}x{self.canvas_height.value()} | Est. time: ~{time_str}"
        )
        self.start_btn.setEnabled(count > 0 and len(self.output_dir_edit.text()) > 0)

    def update_overall_progress(self, current, total, current_name=""):
        self.overall_progress_bar.setVisible(True)
        self.overall_progress_bar.setMaximum(total)
        self.overall_progress_bar.setValue(current)
        if current >= total:
            self.overall_progress_label.setText(f"✓ Complete: {total}/{total} rendered")
            self.overall_progress_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            text = f"Overall Progress: {current}/{total} completed"
            if current_name:
                text += f" - Current: {current_name}"
            self.overall_progress_label.setText(text)
            self.overall_progress_label.setStyleSheet("color: blue; font-weight: bold;")

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            self.update_summary()

    def show_pattern_help(self):
        QMessageBox.information(
            self, "Filename Pattern Help",
            "Available placeholders:\n\n"
            "{method} - Interpolation method\n"
            "{resolution} - Grid resolution\n"
            "{epsilon} - Epsilon value (formatted)\n"
            "{colormap} - Colormap name\n"
            "{index} - Configuration index (0, 1, 2, ...)\n"
            "{timestamp} - Current timestamp\n\n"
            "Example: render_{method}_{resolution} → render_linear_256"
        )

    def get_configurations(self) -> List[dict]:
        configs = []
        for row in range(self.table.rowCount()):
            method_combo = self.table.cellWidget(row, 1)
            res_spin = self.table.cellWidget(row, 2)
            eps_spin = self.table.cellWidget(row, 3)
            cmap_combo = self.table.cellWidget(row, 4)
            png_widget = self.table.cellWidget(row, 5)
            png_check = png_widget.findChild(QCheckBox)
            npy_widget = self.table.cellWidget(row, 6)
            npy_check = npy_widget.findChild(QCheckBox)
            configs.append({
                'method': method_combo.currentText(),
                'resolution': res_spin.value(),
                'epsilon': eps_spin.value(),
                'colormap': cmap_combo.currentText(),
                'save_png': png_check.isChecked(),
                'save_npy': npy_check.isChecked(),
            })
        return configs

    def get_output_dir(self):
        return self.output_dir_edit.text()

    def get_filename_pattern(self):
        return self.filename_pattern.text()


__all__ = [
    "MetadataDialog",
    "ResolutionDialog",
    "BatchConfigDialog",
]
