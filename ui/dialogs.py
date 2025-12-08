from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
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
    """Dialog for configuring sequential rendering jobs with progress tracking.
    
    Supports multiple render modes: volume, scatter, neural, plenoxel.
    Each mode can have its own settings (resolution, method, colormap, etc.)
    """

    def __init__(
        self,
        parent=None,
        current_resolution: int = 256,
        current_method: str = 'linear',
        current_epsilon: float = 1.0,
        allow_natural_neighbor: bool = False,
        batch_runner: Optional[Callable[[List[dict], str, str, Tuple[int, int], "BatchConfigDialog"], None]] = None,
        has_neural_model: bool = False,
        has_plenoxel: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Rendering Configuration")
        self.setGeometry(100, 100, 1350, 750)
        self.current_resolution = current_resolution
        self.current_method = current_method
        self.current_epsilon = current_epsilon
        self.allow_natural_neighbor = allow_natural_neighbor
        self.batch_runner = batch_runner
        self.has_neural_model = has_neural_model
        self.has_plenoxel = has_plenoxel

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
            "Supports Volume (interpolated), Scatter, Neural, and Plenoxel modes. "
            "All renders use the same camera position for PSNR/metrics comparison."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(instructions)

        canvas_group = QGroupBox("Output Settings")
        canvas_layout = QHBoxLayout()
        canvas_layout.addWidget(QLabel("Canvas Size:"))
        self.canvas_width = QSpinBox()
        self.canvas_width.setRange(400, 4096)
        self.canvas_width.setValue(1920)
        self.canvas_width.setSingleStep(100)
        canvas_layout.addWidget(self.canvas_width)
        canvas_layout.addWidget(QLabel("×"))
        self.canvas_height = QSpinBox()
        self.canvas_height.setRange(400, 4096)
        self.canvas_height.setValue(1080)
        self.canvas_height.setSingleStep(100)
        canvas_layout.addWidget(self.canvas_height)
        canvas_layout.addWidget(QLabel("px"))
        canvas_layout.addStretch()
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)

        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Status", "Render Mode", "Interp. Method", "Resolution", "Epsilon",
            "Max Depth", "Min Pts", "Colormap", "Label", "PNG", "NPY", "Actions"
        ])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        column_widths = [100, 110, 150, 80, 70, 70, 70, 90, 120, 40, 40, 70]
        for col, width in enumerate(column_widths):
            self.table.setColumnWidth(col, width)
        self.table.setMinimumHeight(280)
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
        self.filename_pattern = QLineEdit("render_{mode}_{method}_{resolution}")
        self.filename_pattern.setPlaceholderText("{mode}_{method}_{resolution}...")
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

    def _get_interpolation_methods(self):
        """Return list of interpolation methods."""
        methods = [
            'linear', 'nearest',
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic', 'rbf_linear',
            'rbf_gaussian', 'rbf_multiquadric', 'rbf_inverse_multiquadric',
            'rbf_inverse_quadratic'
        ]
        if self.allow_natural_neighbor:
            methods.append('natural_neighbor')
        return methods

    def add_config(self, mode='volume', method=None, resolution=None, epsilon=None, 
                   colormap='magma', label='', save_png=True, save_npy=True,
                   max_depth=5, min_points=500):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Status column (col 0)
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

        # Render Mode column (col 1)
        mode_combo = QComboBox()
        modes = ['volume', 'scatter', 'arepovtk']
        if self.has_neural_model:
            modes.append('neural')
        if self.has_plenoxel:
            modes.append('plenoxel')
        mode_combo.addItems(modes)
        mode_combo.setCurrentText(mode if mode in modes else 'volume')
        mode_combo.currentTextChanged.connect(lambda m, r=row: self._on_mode_changed(r, m))
        self.table.setCellWidget(row, 1, mode_combo)

        # Interpolation Method column (col 2)
        method_combo = QComboBox()
        method_combo.addItems(self._get_interpolation_methods())
        method_combo.setCurrentText(method or self.current_method)
        method_combo.currentTextChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 2, method_combo)

        # Resolution column (col 3)
        res_spin = QSpinBox()
        res_spin.setRange(32, 4096)
        res_spin.setValue(resolution if resolution else self.current_resolution)
        res_spin.setSingleStep(32)
        res_spin.valueChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 3, res_spin)

        # Epsilon column (col 4)
        eps_spin = QDoubleSpinBox()
        eps_spin.setRange(0.01, 100.0)
        eps_spin.setValue(epsilon if epsilon else self.current_epsilon)
        eps_spin.setSingleStep(0.1)
        eps_spin.setDecimals(2)
        eps_spin.valueChanged.connect(self.on_config_changed)
        self.table.setCellWidget(row, 4, eps_spin)

        # Plenoxel Max Depth column (col 5)
        depth_spin = QSpinBox()
        depth_spin.setRange(1, 10)
        depth_spin.setValue(max_depth)
        depth_spin.setToolTip("Max octree depth for plenoxel grid")
        self.table.setCellWidget(row, 5, depth_spin)

        # Plenoxel Min Points column (col 6)
        min_pts_spin = QSpinBox()
        min_pts_spin.setRange(1, 100000)
        min_pts_spin.setValue(min_points)
        min_pts_spin.setSingleStep(100)
        min_pts_spin.setToolTip("Min particles per cell for plenoxel grid")
        self.table.setCellWidget(row, 6, min_pts_spin)

        # Colormap column (col 7)
        cmap_combo = QComboBox()
        cmap_combo.addItems(['magma', 'viridis', 'plasma', 'inferno', 'grays', 'hot', 'cool'])
        cmap_combo.setCurrentText(colormap)
        self.table.setCellWidget(row, 7, cmap_combo)

        # Label column (col 8)
        label_edit = QLineEdit(label)
        label_edit.setPlaceholderText("optional")
        self.table.setCellWidget(row, 8, label_edit)

        # PNG checkbox (col 9)
        png_check = QCheckBox()
        png_check.setChecked(save_png)
        png_widget = QWidget()
        png_layout = QHBoxLayout(png_widget)
        png_layout.addWidget(png_check)
        png_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        png_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 9, png_widget)

        # NPY checkbox (col 10)
        npy_check = QCheckBox()
        npy_check.setChecked(save_npy)
        npy_widget = QWidget()
        npy_layout = QHBoxLayout(npy_widget)
        npy_layout.addWidget(npy_check)
        npy_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        npy_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 10, npy_widget)

        # Remove button (col 11)
        remove_btn = QPushButton("🗑️")
        remove_btn.setMaximumWidth(60)
        remove_btn.setStyleSheet("color: red;")
        remove_btn.clicked.connect(self.remove_config)
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.addWidget(remove_btn)
        action_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, 11, action_widget)

        # Set initial mode-dependent visibility
        self._on_mode_changed(row, mode if mode in modes else 'volume')
        self.update_summary()

    def _on_mode_changed(self, row, mode):
        """Enable/disable columns based on render mode."""
        method_combo = self.table.cellWidget(row, 2)
        res_spin = self.table.cellWidget(row, 3)
        eps_spin = self.table.cellWidget(row, 4)
        depth_spin = self.table.cellWidget(row, 5)
        min_pts_spin = self.table.cellWidget(row, 6)
        
        if mode == 'volume':
            method_combo.setEnabled(True)
            res_spin.setEnabled(True)
            eps_spin.setEnabled(True)
            eps_spin.setToolTip("Epsilon for RBF interpolation")
            depth_spin.setEnabled(False)
            min_pts_spin.setEnabled(False)
        elif mode == 'scatter':
            method_combo.setEnabled(False)
            res_spin.setEnabled(False)
            eps_spin.setEnabled(False)
            depth_spin.setEnabled(False)
            min_pts_spin.setEnabled(False)
        elif mode == 'neural':
            method_combo.setEnabled(False)
            res_spin.setEnabled(True)  # Neural uses inference resolution
            eps_spin.setEnabled(False)
            depth_spin.setEnabled(False)
            min_pts_spin.setEnabled(False)
        elif mode == 'plenoxel':
            method_combo.setEnabled(False)
            res_spin.setEnabled(False)
            eps_spin.setEnabled(False)
            depth_spin.setEnabled(True)
            depth_spin.setToolTip("Max octree depth for plenoxel grid")
            min_pts_spin.setEnabled(True)
            min_pts_spin.setToolTip("Min particles per cell for plenoxel grid")
        elif mode == 'arepovtk':
            # ArepoVTK CPU ray marcher mode
            # Repurpose method combo for interpolation method
            method_combo.setEnabled(True)
            method_combo.clear()
            method_combo.addItems(['idw', 'natural_neighbor'])
            method_combo.setCurrentText('idw')
            method_combo.setToolTip("Interpolation: IDW (fast) or Natural Neighbor (Sibson, matches Illustris TNG)")
            res_spin.setEnabled(True)  # Grid resolution for Natural Neighbor
            res_spin.setToolTip("NN Grid Resolution (only for natural_neighbor mode)")
            res_spin.setValue(128)
            res_spin.setRange(32, 256)
            eps_spin.setEnabled(True)
            eps_spin.setToolTip("Step size as fraction of volume diagonal (smaller = higher quality, slower)")
            eps_spin.setValue(0.01)  # Default step size
            eps_spin.setRange(0.001, 0.5)
            eps_spin.setSingleStep(0.005)
            eps_spin.setDecimals(3)
            depth_spin.setEnabled(True)
            depth_spin.setToolTip("IDW Power (20 = power of 2.0, ArepoVTK default)")
            depth_spin.setValue(20)  # IDW power * 10
            depth_spin.setRange(5, 100)  # 0.5 to 10.0
            min_pts_spin.setEnabled(True)
            min_pts_spin.setToolTip("K neighbors for KD-tree IDW (more = smoother, slower)")
            min_pts_spin.setValue(32)  # Default k neighbors
            min_pts_spin.setRange(8, 256)

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
        QApplication.processEvents()

    def remove_config(self):
        button = self.sender()
        if not button:
            return
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 11)  # Actions column
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
            "All Modes (Current Settings)": self.preset_all_modes,
            "Volume: All Interpolation Methods": self.preset_all_methods,
            "Volume: Resolution Comparison (Linear)": self.preset_resolution_comparison,
            "Volume: RBF Kernels Comparison": self.preset_rbf_comparison,
            "Plenoxel: Depth Comparison": self.preset_plenoxel_depth,
            "Metrics Comparison (Volume + Neural + Scatter)": self.preset_metrics_comparison,
            "Quick Test (3 configs)": self.preset_quick_test,
        }
        for name, func in presets.items():
            action = menu.addAction(name)
            action.triggered.connect(func)
        menu.exec(self.add_preset_btn.mapToGlobal(self.add_preset_btn.rect().bottomLeft()))

    def preset_all_modes(self):
        """Add one config for each available mode."""
        self.add_config(mode='volume', label='volume_baseline')
        self.add_config(mode='scatter', label='scatter_baseline')
        if self.has_neural_model:
            self.add_config(mode='neural', label='neural_baseline')
        if self.has_plenoxel:
            self.add_config(mode='plenoxel', label='plenoxel_baseline')

    def preset_all_methods(self):
        methods = [
            'linear', 'nearest',
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic',
            'rbf_gaussian', 'rbf_multiquadric'
        ]
        if self.allow_natural_neighbor:
            methods.append('natural_neighbor')
        for method in methods:
            self.add_config(mode='volume', method=method, resolution=self.current_resolution, label=method)

    def preset_resolution_comparison(self):
        for res in [128, 256, 512, 768]:
            self.add_config(mode='volume', method='linear', resolution=res, label=f'linear_{res}')

    def preset_rbf_comparison(self):
        rbf_kernels = [
            'rbf_thin_plate_spline', 'rbf_cubic', 'rbf_quintic', 'rbf_linear',
            'rbf_gaussian', 'rbf_multiquadric'
        ]
        for kernel in rbf_kernels:
            self.add_config(mode='volume', method=kernel, resolution=self.current_resolution, label=kernel)

    def preset_plenoxel_depth(self):
        """Preset for comparing plenoxel grids at different depths."""
        if not self.has_plenoxel:
            QMessageBox.information(self, "Plenoxel", "Plenoxel mode not available. Build a plenoxel grid first.")
            return
        for depth in [3, 4, 5, 6, 7]:
            self.add_config(mode='plenoxel', max_depth=depth, min_points=500, label=f'plenoxel_d{depth}')

    def preset_metrics_comparison(self):
        """Preset for PSNR/SSIM comparison: same resolution across modes."""
        res = self.current_resolution
        self.add_config(mode='volume', method='linear', resolution=res, label='vol_linear')
        self.add_config(mode='volume', method='rbf_cubic', resolution=res, label='vol_rbf_cubic')
        self.add_config(mode='scatter', label='scatter_raw')
        if self.has_neural_model:
            self.add_config(mode='neural', resolution=res, label='neural')

    def preset_quick_test(self):
        self.add_config(mode='volume', method='linear', resolution=128, label='quick_linear')
        self.add_config(mode='volume', method='nearest', resolution=128, label='quick_nearest')
        self.add_config(mode='scatter', label='quick_scatter')

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
            mode_combo = self.table.cellWidget(row, 1)
            res_spin = self.table.cellWidget(row, 3)
            mode = mode_combo.currentText() if mode_combo else 'volume'
            if mode == 'volume' and res_spin:
                resolution = res_spin.value()
                total_time += (resolution ** 3) / 100000
            elif mode == 'neural' and res_spin:
                resolution = res_spin.value()
                total_time += (resolution ** 3) / 500000  # Neural is faster
            elif mode == 'scatter':
                total_time += 5  # Fixed estimate for scatter
            elif mode == 'plenoxel':
                total_time += 10  # Fixed estimate for plenoxel
        minutes = int(total_time / 60)
        seconds = int(total_time % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        self.summary_label.setText(
            f"Configurations: {count} | Canvas: {self.canvas_width.value()}×{self.canvas_height.value()} | Est. time: ~{time_str}"
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
                text += f" — Current: {current_name}"
            self.overall_progress_label.setText(text)
            self.overall_progress_label.setStyleSheet("color: blue; font-weight: bold;")
        QApplication.processEvents()

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            self.update_summary()

    def show_pattern_help(self):
        QMessageBox.information(
            self, "Filename Pattern Help",
            "Available placeholders:\n\n"
            "{mode} - Render mode (volume, scatter, neural, plenoxel)\n"
            "{method} - Interpolation method\n"
            "{resolution} - Grid resolution\n"
            "{epsilon} - Epsilon value (formatted)\n"
            "{max_depth} - Plenoxel max depth\n"
            "{min_points} - Plenoxel min points per cell\n"
            "{colormap} - Colormap name\n"
            "{label} - Custom label\n"
            "{index} - Configuration index (0, 1, 2, ...)\n"
            "{timestamp} - Current timestamp\n\n"
            "Example: render_{mode}_{method}_{resolution} → render_volume_linear_256\n"
            "Example: plenoxel_d{max_depth}_p{min_points} → plenoxel_d5_p500"
        )

    def get_configurations(self) -> List[dict]:
        configs = []
        for row in range(self.table.rowCount()):
            mode_combo = self.table.cellWidget(row, 1)
            method_combo = self.table.cellWidget(row, 2)
            res_spin = self.table.cellWidget(row, 3)
            eps_spin = self.table.cellWidget(row, 4)
            depth_spin = self.table.cellWidget(row, 5)
            min_pts_spin = self.table.cellWidget(row, 6)
            cmap_combo = self.table.cellWidget(row, 7)
            label_edit = self.table.cellWidget(row, 8)
            png_widget = self.table.cellWidget(row, 9)
            png_check = png_widget.findChild(QCheckBox)
            npy_widget = self.table.cellWidget(row, 10)
            npy_check = npy_widget.findChild(QCheckBox)
            mode = mode_combo.currentText()
            config = {
                'mode': mode,
                'method': method_combo.currentText(),
                'resolution': res_spin.value(),
                'epsilon': eps_spin.value(),
                'max_depth': depth_spin.value(),
                'min_points': min_pts_spin.value(),
                'colormap': cmap_combo.currentText(),
                'label': label_edit.text() if label_edit else '',
                'save_png': png_check.isChecked(),
                'save_npy': npy_check.isChecked(),
            }
            # Add ArepoVTK-specific parameters
            if mode == 'arepovtk':
                config['step_size'] = eps_spin.value()  # Epsilon field used for step size
                config['idw_power'] = depth_spin.value() / 10.0  # Max depth field used for IDW power
                config['k_neighbors'] = min_pts_spin.value()  # Min points field used for k neighbors
                config['interpolation_method'] = method_combo.currentText()  # 'idw' or 'natural_neighbor'
                config['nn_grid_resolution'] = res_spin.value()  # Grid resolution for Natural Neighbor
            configs.append(config)
        return configs

    def get_output_dir(self):
        return self.output_dir_edit.text()

    def get_filename_pattern(self):
        return self.filename_pattern.text()


class VolumeRenderSettingsDialog(QDialog):
    """Dialog for visualization settings (volume, neural, and scatter renderers)."""

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Visualization Settings")
        self.setModal(False)  # Non-modal so user can see changes
        self.resize(420, 520)
        
        self.parent_gui = parent
        current_settings = current_settings or {}
        
        layout = QVBoxLayout(self)
        
        # === Volume Rendering Section ===
        volume_group = QGroupBox("Volume Rendering (Volume & Neural)")
        volume_layout = QFormLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(['mip', 'minip', 'composite', 'additive'])
        self.method_combo.setCurrentText(current_settings.get('method', 'mip'))
        self.method_combo.currentTextChanged.connect(self._on_volume_setting_changed)
        volume_layout.addRow("Render Method:", self.method_combo)
        
        # Interpolation
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(['trilinear', 'nearest'])
        self.interp_combo.setCurrentText(current_settings.get('interpolation', 'trilinear'))
        self.interp_combo.currentTextChanged.connect(self._on_volume_setting_changed)
        volume_layout.addRow("Interpolation:", self.interp_combo)
        
        # Step Size
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(2)
        self.step_spin.setRange(0.05, 4.0)
        self.step_spin.setSingleStep(0.05)
        self.step_spin.setValue(current_settings.get('step_size', 0.5))
        self.step_spin.valueChanged.connect(self._on_volume_setting_changed)
        volume_layout.addRow("Step Size:", self.step_spin)
        
        # Opacity Scale
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setRange(0.1, 3.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setValue(current_settings.get('opacity_scale', 1.0))
        self.opacity_spin.valueChanged.connect(self._on_volume_setting_changed)
        volume_layout.addRow("Opacity Scale:", self.opacity_spin)
        
        # Shading
        self.shade_check = QCheckBox("Enable Shading")
        self.shade_check.setChecked(current_settings.get('shade', False))
        self.shade_check.stateChanged.connect(self._on_volume_setting_changed)
        volume_layout.addRow("", self.shade_check)
        
        volume_group.setLayout(volume_layout)
        layout.addWidget(volume_group)
        
        # === Scatter Plot Section ===
        scatter_group = QGroupBox("Scatter Plot")
        scatter_layout = QFormLayout()
        
        # Get current scatter settings from parent
        scatter_settings = current_settings.get('scatter', {})
        
        # Point Size
        self.scatter_point_size_spin = QSpinBox()
        self.scatter_point_size_spin.setRange(1, 20)
        self.scatter_point_size_spin.setValue(scatter_settings.get('point_size', 4))
        self.scatter_point_size_spin.valueChanged.connect(self._on_scatter_setting_changed)
        scatter_layout.addRow("Point Size:", self.scatter_point_size_spin)
        
        # Render as Spheres
        self.scatter_spheres_check = QCheckBox("Render as Spheres")
        self.scatter_spheres_check.setChecked(scatter_settings.get('render_as_spheres', True))
        self.scatter_spheres_check.setToolTip("Higher quality but slower")
        self.scatter_spheres_check.stateChanged.connect(self._on_scatter_setting_changed)
        scatter_layout.addRow("", self.scatter_spheres_check)
        
        # Show Scalar Bar
        self.scatter_scalar_bar_check = QCheckBox("Show Scalar Bar")
        self.scatter_scalar_bar_check.setChecked(scatter_settings.get('show_scalar_bar', False))
        self.scatter_scalar_bar_check.stateChanged.connect(self._on_scatter_setting_changed)
        scatter_layout.addRow("", self.scatter_scalar_bar_check)
        
        scatter_group.setLayout(scatter_layout)
        layout.addWidget(scatter_group)
        
        # Method descriptions
        desc_group = QGroupBox("Volume Method Descriptions")
        desc_layout = QVBoxLayout()
        desc = QLabel(
            "<b>mip</b>: Maximum Intensity Projection - shows brightest values<br>"
            "<b>minip</b>: Minimum Intensity Projection - shows darkest values<br>"
            "<b>composite</b>: Blends all values with opacity (translucent)<br>"
            "<b>additive</b>: Sums all values (good for emission)"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 9px;")
        desc_layout.addWidget(desc)
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_volume_setting_changed(self):
        """Auto-apply volume settings when they change."""
        self._apply_volume_settings()
    
    def _on_scatter_setting_changed(self):
        """Auto-apply scatter settings when they change."""
        self._apply_scatter_settings()
    
    def _apply_volume_settings(self):
        """Apply current settings to both volume and neural renderers."""
        if not self.parent_gui:
            return
            
        settings = self.get_volume_settings()
        
        def apply_to_renderer(renderer, name):
            """Apply settings to a single renderer."""
            if renderer is None:
                return False
                
            # Update method and step size
            renderer.update_render_settings(
                method=settings['method'],
                step_size=settings['step_size']
            )
            
            # Update interpolation and other settings
            renderer.interpolation = settings['interpolation']
            renderer.opacity_scale = settings['opacity_scale']
            renderer.shade = settings['shade']
            
            # Re-render if volume exists
            if renderer.actor is not None:
                renderer._apply_advanced_settings()
                return True
            return False
        
        # Apply to volume renderer
        volume_updated = False
        if hasattr(self.parent_gui, 'volume_renderer'):
            volume_updated = apply_to_renderer(self.parent_gui.volume_renderer, 'volume')
        
        # Apply to neural renderer (same settings for consistency)
        neural_updated = False
        if hasattr(self.parent_gui, 'neural_renderer'):
            neural_updated = apply_to_renderer(self.parent_gui.neural_renderer, 'neural')
        
        # Render if any were updated
        if (volume_updated or neural_updated) and hasattr(self.parent_gui, 'plotter'):
            self.parent_gui.plotter.render()
    
    def _apply_scatter_settings(self):
        """Apply scatter settings to scatter renderer."""
        if not self.parent_gui:
            return
        
        scatter_renderer = getattr(self.parent_gui, 'scatter_renderer', None)
        if scatter_renderer is None:
            return
        
        # Update settings
        scatter_renderer.set_point_size(self.scatter_point_size_spin.value())
        scatter_renderer.set_render_as_spheres(self.scatter_spheres_check.isChecked())
        scatter_renderer.set_show_scalar_bar(self.scatter_scalar_bar_check.isChecked())
        
        # Also update the panel spinboxes if they exist
        if hasattr(self.parent_gui, 'scatter_point_size_spin'):
            self.parent_gui.scatter_point_size_spin.blockSignals(True)
            self.parent_gui.scatter_point_size_spin.setValue(self.scatter_point_size_spin.value())
            self.parent_gui.scatter_point_size_spin.blockSignals(False)
        if hasattr(self.parent_gui, 'scatter_spheres_check'):
            self.parent_gui.scatter_spheres_check.blockSignals(True)
            self.parent_gui.scatter_spheres_check.setChecked(self.scatter_spheres_check.isChecked())
            self.parent_gui.scatter_spheres_check.blockSignals(False)
        if hasattr(self.parent_gui, 'scatter_scalar_bar_check'):
            self.parent_gui.scatter_scalar_bar_check.blockSignals(True)
            self.parent_gui.scatter_scalar_bar_check.setChecked(self.scatter_scalar_bar_check.isChecked())
            self.parent_gui.scatter_scalar_bar_check.blockSignals(False)
        
        # Re-render scatter if it exists
        if scatter_renderer.actor is not None:
            viz = getattr(self.parent_gui, 'visualization', None)
            if viz and self.parent_gui.viz_mode == 'scatter':
                viz.create_scatter_plot(force_visible=True)
            if hasattr(self.parent_gui, 'plotter'):
                self.parent_gui.plotter.render()
    
    def _reset_defaults(self):
        """Reset to default values."""
        # Volume defaults
        self.method_combo.setCurrentText('mip')
        self.interp_combo.setCurrentText('trilinear')
        self.step_spin.setValue(0.5)
        self.opacity_spin.setValue(1.0)
        self.shade_check.setChecked(False)
        # Scatter defaults
        self.scatter_point_size_spin.setValue(4)
        self.scatter_spheres_check.setChecked(True)
        self.scatter_scalar_bar_check.setChecked(False)  # Off by default
        # Apply all
        self._apply_volume_settings()
        self._apply_scatter_settings()
    
    def get_volume_settings(self):
        """Return current volume settings as dict."""
        return {
            'method': self.method_combo.currentText(),
            'interpolation': self.interp_combo.currentText(),
            'step_size': self.step_spin.value(),
            'opacity_scale': self.opacity_spin.value(),
            'shade': self.shade_check.isChecked(),
        }
    
    def get_scatter_settings(self):
        """Return current scatter settings as dict."""
        return {
            'point_size': self.scatter_point_size_spin.value(),
            'render_as_spheres': self.scatter_spheres_check.isChecked(),
            'show_scalar_bar': self.scatter_scalar_bar_check.isChecked(),
        }
    
    def get_settings(self):
        """Return all settings."""
        return {
            **self.get_volume_settings(),
            'scatter': self.get_scatter_settings(),
        }


class NeuralNetworkSettingsDialog(QDialog):
    """Dialog for neural network architecture and training parameters."""
    
    settingsChanged = pyqtSignal(dict)

    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Settings")
        self.setModal(False)
        self.resize(480, 650)
        
        self.parent_gui = parent
        current_settings = current_settings or {}
        
        layout = QVBoxLayout(self)
        
        # === Network Architecture Section ===
        arch_group = QGroupBox("Network Architecture")
        arch_layout = QFormLayout()
        
        # Number of levels
        self.n_levels_spin = QSpinBox()
        self.n_levels_spin.setRange(8, 24)
        self.n_levels_spin.setValue(current_settings.get('n_levels', 19))
        self.n_levels_spin.setToolTip("Number of hash grid levels (more = finer detail)")
        arch_layout.addRow("Hash Grid Levels:", self.n_levels_spin)
        
        # Features per level
        self.features_spin = QSpinBox()
        self.features_spin.setRange(1, 8)
        self.features_spin.setValue(current_settings.get('n_features_per_level', 4))
        self.features_spin.setToolTip("Features per hash level (more = richer representation)")
        arch_layout.addRow("Features per Level:", self.features_spin)
        
        # Hash table size (log2)
        self.hashmap_spin = QSpinBox()
        self.hashmap_spin.setRange(14, 24)
        self.hashmap_spin.setValue(current_settings.get('log2_hashmap_size', 22))
        self.hashmap_spin.setToolTip("Log2 of hash table size (22 = 4M entries, uses ~1GB VRAM)")
        arch_layout.addRow("Hash Table Size (log₂):", self.hashmap_spin)
        
        # Base resolution
        self.base_res_spin = QSpinBox()
        self.base_res_spin.setRange(4, 64)
        self.base_res_spin.setValue(current_settings.get('base_resolution', 16))
        self.base_res_spin.setToolTip("Base resolution of the hash grid")
        arch_layout.addRow("Base Resolution:", self.base_res_spin)
        
        # Per level scale
        self.level_scale_spin = QDoubleSpinBox()
        self.level_scale_spin.setDecimals(3)
        self.level_scale_spin.setRange(1.1, 2.5)
        self.level_scale_spin.setSingleStep(0.05)
        self.level_scale_spin.setValue(current_settings.get('per_level_scale', 1.447))
        self.level_scale_spin.setToolTip("Scale factor between levels (√2 ≈ 1.414 is common)")
        arch_layout.addRow("Per-Level Scale:", self.level_scale_spin)
        
        # MLP hidden layers
        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setRange(1, 8)
        self.hidden_layers_spin.setValue(current_settings.get('n_hidden_layers', 4))
        self.hidden_layers_spin.setToolTip("Number of hidden layers in the MLP")
        arch_layout.addRow("MLP Hidden Layers:", self.hidden_layers_spin)
        
        # MLP neurons
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(16, 256)
        self.neurons_spin.setSingleStep(16)
        self.neurons_spin.setValue(current_settings.get('n_neurons', 128))
        self.neurons_spin.setToolTip("Neurons per hidden layer (must be multiple of 16)")
        arch_layout.addRow("Neurons per Layer:", self.neurons_spin)
        
        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group)
        
        # === Training Parameters Section ===
        train_group = QGroupBox("Training Parameters")
        train_layout = QFormLayout()
        
        # Training steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 1000000)
        self.steps_spin.setSingleStep(5000)
        self.steps_spin.setValue(current_settings.get('steps', 50000))
        train_layout.addRow("Training Steps:", self.steps_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(512, 262144)
        self.batch_spin.setSingleStep(4096)
        self.batch_spin.setValue(current_settings.get('batch_size', 65536))
        train_layout.addRow("Batch Size:", self.batch_spin)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(1e-5, 1e-1)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(current_settings.get('lr', 1e-3))
        self.lr_spin.valueChanged.connect(self._on_setting_changed)
        train_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Log interval (original: 100)
        self.log_interval_spin = QSpinBox()
        self.log_interval_spin.setRange(1, 10000)
        self.log_interval_spin.setValue(current_settings.get('log_interval', 100))
        self.log_interval_spin.valueChanged.connect(self._on_setting_changed)
        train_layout.addRow("Log Interval:", self.log_interval_spin)
        
        # Preview interval (original: 250)
        self.preview_interval_spin = QSpinBox()
        self.preview_interval_spin.setRange(0, 10000)
        self.preview_interval_spin.setValue(current_settings.get('preview_interval', 250))
        self.preview_interval_spin.valueChanged.connect(self._on_setting_changed)
        train_layout.addRow("Preview Interval:", self.preview_interval_spin)
        
        # Preview resolution
        self.preview_res_spin = QSpinBox()
        self.preview_res_spin.setRange(32, 512)
        self.preview_res_spin.setSingleStep(32)
        self.preview_res_spin.setValue(current_settings.get('preview_resolution', 128))
        self.preview_res_spin.valueChanged.connect(self._on_setting_changed)
        train_layout.addRow("Preview Resolution:", self.preview_res_spin)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # === Inference Section ===
        inference_group = QGroupBox("Inference & Rendering")
        inference_layout = QFormLayout()
        
        # Inference batch (original: 32768)
        self.inference_batch_spin = QSpinBox()
        self.inference_batch_spin.setRange(1024, 524288)
        self.inference_batch_spin.setSingleStep(8192)
        self.inference_batch_spin.setValue(current_settings.get('inference_batch', 32768))
        self.inference_batch_spin.valueChanged.connect(self._on_setting_changed)
        inference_layout.addRow("Inference Batch:", self.inference_batch_spin)
        
        # Render resolution (original: 256)
        self.render_res_spin = QSpinBox()
        self.render_res_spin.setRange(64, 2048)
        self.render_res_spin.setSingleStep(64)
        self.render_res_spin.setValue(current_settings.get('render_resolution', 256))
        self.render_res_spin.valueChanged.connect(self._on_setting_changed)
        inference_layout.addRow("Render Resolution:", self.render_res_spin)
        
        # Display resolution (original: 256)
        self.display_res_spin = QSpinBox()
        self.display_res_spin.setRange(64, 2048)
        self.display_res_spin.setSingleStep(64)
        self.display_res_spin.setValue(current_settings.get('display_resolution', 256))
        self.display_res_spin.valueChanged.connect(self._on_setting_changed)
        inference_layout.addRow("Display Resolution:", self.display_res_spin)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        # === Presets Section ===
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout()
        
        small_btn = QPushButton("Small Dataset\n(<1M particles)")
        small_btn.clicked.connect(self._preset_small)
        preset_layout.addWidget(small_btn)
        
        medium_btn = QPushButton("Medium Dataset\n(1-20M particles)")
        medium_btn.clicked.connect(self._preset_medium)
        preset_layout.addWidget(medium_btn)
        
        large_btn = QPushButton("Large Dataset\n(>20M particles)")
        large_btn.clicked.connect(self._preset_large)
        preset_layout.addWidget(large_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # === Info text ===
        info_label = QLabel(
            "💡 <b>Tip:</b> For 100M+ particles, use Large preset. "
            "Hash table size 22+ and 128 neurons recommended. "
            "Low-memory mode checkbox in left panel reduces all values for ≤6GB GPUs."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_setting_changed(self):
        """Auto-apply settings to panel when any value changes."""
        self._apply_to_panel()
    
    def _preset_small(self):
        """Settings for small datasets (<1M particles) - original defaults."""
        # Architecture (original tinycudann defaults)
        self.n_levels_spin.setValue(16)
        self.features_spin.setValue(2)
        self.hashmap_spin.setValue(18)
        self.base_res_spin.setValue(16)
        self.level_scale_spin.setValue(1.5)
        self.hidden_layers_spin.setValue(2)
        self.neurons_spin.setValue(64)
        # Training (original defaults)
        self.steps_spin.setValue(20000)
        self.batch_spin.setValue(16384)
        self.lr_spin.setValue(1e-3)
        self.log_interval_spin.setValue(100)
        self.preview_interval_spin.setValue(250)
        self.preview_res_spin.setValue(128)
        # Inference (original defaults)
        self.inference_batch_spin.setValue(32768)
        self.render_res_spin.setValue(256)
        self.display_res_spin.setValue(256)
    
    def _preset_medium(self):
        """Settings for medium datasets (1-20M particles)."""
        # Architecture
        self.n_levels_spin.setValue(16)
        self.features_spin.setValue(2)
        self.hashmap_spin.setValue(20)
        self.base_res_spin.setValue(16)
        self.level_scale_spin.setValue(1.5)
        self.hidden_layers_spin.setValue(3)
        self.neurons_spin.setValue(64)
        # Training
        self.steps_spin.setValue(30000)
        self.batch_spin.setValue(32768)
        self.lr_spin.setValue(7e-4)
        self.log_interval_spin.setValue(150)
        self.preview_interval_spin.setValue(400)
        self.preview_res_spin.setValue(128)
        # Inference
        self.inference_batch_spin.setValue(98304)
        self.render_res_spin.setValue(384)
        self.display_res_spin.setValue(384)
    
    def _preset_large(self):
        """Settings for large datasets (>20M particles, optimized for 100M+)."""
        # Architecture
        self.n_levels_spin.setValue(19)
        self.features_spin.setValue(4)
        self.hashmap_spin.setValue(22)
        self.base_res_spin.setValue(16)
        self.level_scale_spin.setValue(1.447)
        self.hidden_layers_spin.setValue(4)
        self.neurons_spin.setValue(128)
        # Training
        self.steps_spin.setValue(50000)
        self.batch_spin.setValue(65536)
        self.lr_spin.setValue(5e-4)
        self.log_interval_spin.setValue(200)
        self.preview_interval_spin.setValue(500)
        self.preview_res_spin.setValue(128)
        # Inference
        self.inference_batch_spin.setValue(131072)
        self.render_res_spin.setValue(512)
        self.display_res_spin.setValue(512)
    
    def _apply_to_panel(self):
        """Apply settings to the neural panel spinboxes."""
        if not self.parent_gui:
            return
        
        params = getattr(self.parent_gui, 'neural_params', {})
        
        # Update training params
        if 'steps' in params:
            params['steps'].setValue(self.steps_spin.value())
        if 'batch_size' in params:
            params['batch_size'].setValue(self.batch_spin.value())
        if 'lr' in params:
            params['lr'].setValue(self.lr_spin.value())
        if 'log_interval' in params:
            params['log_interval'].setValue(self.log_interval_spin.value())
        if 'preview_interval' in params:
            params['preview_interval'].setValue(self.preview_interval_spin.value())
        if 'preview_resolution' in params:
            params['preview_resolution'].setValue(self.preview_res_spin.value())
        if 'inference_batch' in params:
            params['inference_batch'].setValue(self.inference_batch_spin.value())
        
        # Update render/display resolution
        if hasattr(self.parent_gui, 'neural_render_res_spin'):
            self.parent_gui.neural_render_res_spin.setValue(self.render_res_spin.value())
        if hasattr(self.parent_gui, 'neural_display_res_spin'):
            self.parent_gui.neural_display_res_spin.setValue(self.display_res_spin.value())
        
        # Store architecture settings for next training run
        self.parent_gui._neural_arch_settings = self.get_architecture_settings()
        
        self.settingsChanged.emit(self.get_all_settings())
    
    def get_architecture_settings(self):
        """Return architecture settings as dict."""
        return {
            'n_levels': self.n_levels_spin.value(),
            'n_features_per_level': self.features_spin.value(),
            'log2_hashmap_size': self.hashmap_spin.value(),
            'base_resolution': self.base_res_spin.value(),
            'per_level_scale': self.level_scale_spin.value(),
            'n_hidden_layers': self.hidden_layers_spin.value(),
            'n_neurons': self.neurons_spin.value(),
        }
    
    def get_training_settings(self):
        """Return training settings as dict."""
        return {
            'steps': self.steps_spin.value(),
            'batch_size': self.batch_spin.value(),
            'lr': self.lr_spin.value(),
            'log_interval': self.log_interval_spin.value(),
            'preview_interval': self.preview_interval_spin.value(),
            'preview_resolution': self.preview_res_spin.value(),
        }
    
    def get_inference_settings(self):
        """Return inference settings as dict."""
        return {
            'inference_batch': self.inference_batch_spin.value(),
            'render_resolution': self.render_res_spin.value(),
            'display_resolution': self.display_res_spin.value(),
        }
    
    def get_all_settings(self):
        """Return all settings combined."""
        return {
            **self.get_architecture_settings(),
            **self.get_training_settings(),
            **self.get_inference_settings(),
        }


__all__ = [
    "MetadataDialog",
    "ResolutionDialog",
    "BatchConfigDialog",
    "VolumeRenderSettingsDialog",
    "NeuralNetworkSettingsDialog",
]
