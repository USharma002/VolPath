"""UI helpers for constructing the sidebar control panel."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QCheckBox,
    QFormLayout,
    QSlider,
    QTextEdit,
    QProgressBar,
    QLineEdit,
    QStyle,
)

from config.optional import HAS_NATURAL_NEIGHBOR
from ui.plots import LossPlotWidget

__all__ = ["build_control_panel"]


def build_control_panel(gui, volume_extent):
    """Return the fully assembled left control panel for the main window."""
    panel = QWidget()
    layout = QVBoxLayout(panel)

    layout.addWidget(_build_viz_mode_group(gui))
    layout.addWidget(_build_file_group(gui))
    layout.addWidget(_build_plenoxel_group(gui))
    
    # Store references for mode-dependent visibility
    gui.interpolation_group = _build_interpolation_group(gui, volume_extent)
    layout.addWidget(gui.interpolation_group)
    
    # Subvolume section - visible in all modes
    layout.addWidget(_build_subvolume_group(gui))
    
    gui.neural_settings_group = _build_neural_group(gui)
    layout.addWidget(gui.neural_settings_group)
    
    layout.addWidget(_build_camera_group(gui))
    layout.addWidget(_build_log_group(gui))
    layout.addStretch()
    return panel


def _build_viz_mode_group(gui):
    group = QGroupBox("0. Visualization Mode")
    layout = QVBoxLayout()

    # First row: Volume and Scatter
    mode_layout = QHBoxLayout()
    gui.volume_mode_btn = QPushButton("Volume Rendering")
    gui.volume_mode_btn.setCheckable(True)
    gui.volume_mode_btn.setChecked(True)
    gui.volume_mode_btn.clicked.connect(lambda: gui.visualization.set_mode('volume'))
    mode_layout.addWidget(gui.volume_mode_btn)

    gui.scatter_mode_btn = QPushButton("Particle Scatter Plot")
    gui.scatter_mode_btn.setCheckable(True)
    gui.scatter_mode_btn.clicked.connect(lambda: gui.visualization.set_mode('scatter'))
    mode_layout.addWidget(gui.scatter_mode_btn)

    gui.neural_mode_btn = QPushButton("Neural Renderer")
    gui.neural_mode_btn.setCheckable(True)
    gui.neural_mode_btn.clicked.connect(lambda: gui.visualization.set_mode('neural'))
    mode_layout.addWidget(gui.neural_mode_btn)

    layout.addLayout(mode_layout)

    gui.viz_info_label = QLabel("Mode: Volume Rendering (interpolated)")
    gui.viz_info_label.setStyleSheet("color: #0066cc; font-size: 9px;")
    layout.addWidget(gui.viz_info_label)

    # Display options
    display_row = QHBoxLayout()
    gui.show_bounding_box_check = QCheckBox("Bounding Box")
    gui.show_bounding_box_check.setChecked(True)
    gui.show_bounding_box_check.toggled.connect(gui.on_bounding_box_toggled)
    display_row.addWidget(gui.show_bounding_box_check)
    
    gui.show_subvolume_box_check = QCheckBox("Subvolume Box")
    gui.show_subvolume_box_check.setChecked(True)
    gui.show_subvolume_box_check.toggled.connect(gui.on_subvolume_box_toggled)
    display_row.addWidget(gui.show_subvolume_box_check)
    display_row.addStretch()
    layout.addLayout(display_row)

    group.setLayout(layout)
    return group


def _build_file_group(gui):
    group = QGroupBox("1. Data Loading")
    layout = QVBoxLayout()

    file_row = QHBoxLayout()
    gui.file_label = QLabel("No file loaded")
    gui.file_label.setWordWrap(True)
    file_row.addWidget(gui.file_label, 1)

    gui.metadata_btn = QToolButton()
    icon = gui.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation)
    gui.metadata_btn.setIcon(icon)
    gui.metadata_btn.setToolTip("Show snapshot header and dataset details")
    gui.metadata_btn.clicked.connect(gui.show_metadata_dialog)
    gui.metadata_btn.setEnabled(False)
    file_row.addWidget(gui.metadata_btn, 0, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addLayout(file_row)

    load_btn = QPushButton("Load Snapshot...")
    load_btn.clicked.connect(gui.load_snapshot_file)
    layout.addWidget(load_btn)

    multi_file_info = QLabel("Supports: HDF5 (single/multi-file) and VTK (*.vtk/*.vtp/*.vti)")
    multi_file_info.setStyleSheet("color: gray; font-size: 8px;")
    multi_file_info.setWordWrap(True)
    layout.addWidget(multi_file_info)

    type_layout = QHBoxLayout()
    type_layout.addWidget(QLabel("Particle Type:"))
    gui.particle_type_combo = QComboBox()
    gui.particle_type_combo.addItems(["PartType0", "PartType1", "PartType4", "PartType5"])
    gui.particle_type_combo.currentTextChanged.connect(gui.on_particle_type_changed)
    type_layout.addWidget(gui.particle_type_combo)
    layout.addLayout(type_layout)

    coord_layout = QHBoxLayout()
    coord_layout.addWidget(QLabel("Coordinates:"))
    gui.coord_field = QLineEdit("Coordinates")
    coord_layout.addWidget(gui.coord_field)
    layout.addLayout(coord_layout)

    field_layout = QHBoxLayout()
    field_layout.addWidget(QLabel("Field:"))
    gui.field_combo = QComboBox()
    gui.field_combo.setEditable(False)
    gui.field_combo.currentTextChanged.connect(gui.on_field_changed)
    field_layout.addWidget(gui.field_combo)
    layout.addLayout(field_layout)

    transform_layout = QHBoxLayout()
    transform_layout.addWidget(QLabel("Transform:"))
    gui.transform_combo = QComboBox()
    gui.transform_combo.currentTextChanged.connect(gui.on_transform_changed)
    transform_layout.addWidget(gui.transform_combo)
    layout.addLayout(transform_layout)

    tf_btn_layout = QHBoxLayout()
    gui.tf_edit_btn = QPushButton("Edit Transfer Function...")
    gui.tf_edit_btn.clicked.connect(gui.open_transfer_function_dialog)
    tf_btn_layout.addWidget(gui.tf_edit_btn)
    tf_btn_layout.addStretch()
    layout.addLayout(tf_btn_layout)

    gui.data_info_label = QLabel("Particles: -")
    layout.addWidget(gui.data_info_label)

    group.setLayout(layout)
    return group


def _build_plenoxel_group(gui):
    group = QGroupBox("1b. Plenoxel Grid (experimental)")
    layout = QVBoxLayout()

    intro = QLabel(
        "Build a sparse voxel grid (Plenoxel) from the scatter points."
        " Use it to inspect occupied regions before running full volume renders."
    )
    intro.setWordWrap(True)
    intro.setStyleSheet("color: #666; font-size: 9px;")
    layout.addWidget(intro)

    gui.plenoxel_show_check = QCheckBox("Show plenoxel overlay in viewport")
    gui.plenoxel_show_check.setEnabled(False)
    gui.plenoxel_show_check.toggled.connect(gui.on_plenoxel_toggled)
    layout.addWidget(gui.plenoxel_show_check)

    min_points_row = QHBoxLayout()
    min_points_row.addWidget(QLabel("Min particles / cell:"))
    gui.plenoxel_min_points_spin = QSpinBox()
    gui.plenoxel_min_points_spin.setRange(1, 200000)
    gui.plenoxel_min_points_spin.setValue(500)
    min_points_row.addWidget(gui.plenoxel_min_points_spin)
    layout.addLayout(min_points_row)

    depth_row = QHBoxLayout()
    gui.plenoxel_min_depth_spin = QSpinBox()
    gui.plenoxel_min_depth_spin.setRange(0, 8)
    gui.plenoxel_min_depth_spin.setValue(1)
    gui.plenoxel_min_depth_spin.valueChanged.connect(gui.on_plenoxel_depth_changed)
    depth_row.addWidget(QLabel("Min depth"))
    depth_row.addWidget(gui.plenoxel_min_depth_spin)

    gui.plenoxel_max_depth_spin = QSpinBox()
    gui.plenoxel_max_depth_spin.setRange(1, 20)
    gui.plenoxel_max_depth_spin.setValue(5)
    gui.plenoxel_max_depth_spin.valueChanged.connect(gui.on_plenoxel_depth_changed)
    depth_row.addWidget(QLabel("Max depth"))
    depth_row.addWidget(gui.plenoxel_max_depth_spin)
    layout.addLayout(depth_row)

    gui.plenoxel_build_btn = QPushButton("Build Plenoxel Grid")
    gui.plenoxel_build_btn.setEnabled(False)
    gui.plenoxel_build_btn.clicked.connect(gui.rebuild_plenoxel_grid)
    layout.addWidget(gui.plenoxel_build_btn)

    gui.plenoxel_stats_label = QLabel("Plenoxel grid: not built")
    gui.plenoxel_stats_label.setStyleSheet("color: #555; font-size: 9px;")
    layout.addWidget(gui.plenoxel_stats_label)

    group.setLayout(layout)
    return group


def _build_interpolation_group(gui, volume_extent):
    group = QGroupBox("2. Interpolation Settings")
    layout = QVBoxLayout()

    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Method:"))
    gui.method_combo = QComboBox()
    methods = [
        'linear',
        'nearest',
        'rbf_thin_plate_spline',
        'rbf_cubic',
        'rbf_quintic',
        'rbf_linear',
        'rbf_gaussian',
        'rbf_multiquadric',
        'rbf_inverse_multiquadric',
        'rbf_inverse_quadratic',
    ]
    if HAS_NATURAL_NEIGHBOR:
        methods.append('natural_neighbor')
    gui.method_combo.addItems(methods)
    gui.method_combo.currentTextChanged.connect(gui.on_method_changed)
    method_layout.addWidget(gui.method_combo)
    layout.addLayout(method_layout)

    epsilon_layout = QHBoxLayout()
    epsilon_layout.addWidget(QLabel("RBF Epsilon:"))
    gui.epsilon_spin = QDoubleSpinBox()
    gui.epsilon_spin.setRange(0.01, 100.0)
    gui.epsilon_spin.setValue(1.0)
    gui.epsilon_spin.setSingleStep(0.1)
    gui.epsilon_spin.setDecimals(2)
    epsilon_layout.addWidget(gui.epsilon_spin)
    layout.addLayout(epsilon_layout)

    gui.epsilon_label = QLabel("(for gaussian, multiquadric, etc.)")
    gui.epsilon_label.setStyleSheet("color: gray; font-size: 9px;")
    layout.addWidget(gui.epsilon_label)

    res_layout = QHBoxLayout()
    res_layout.addWidget(QLabel("Grid Resolution:"))
    gui.resolution_spin = QSpinBox()
    gui.resolution_spin.setRange(32, 4096)
    gui.resolution_spin.setValue(256)
    gui.resolution_spin.setSingleStep(32)
    gui.resolution_spin.valueChanged.connect(gui.on_resolution_changed)
    res_layout.addWidget(gui.resolution_spin)
    layout.addLayout(res_layout)

    gui.resolution_info = QLabel(f"Volume extent: {volume_extent:.0f}³ units (fixed)")
    gui.resolution_info.setStyleSheet("color: #888; font-size: 9px;")
    layout.addWidget(gui.resolution_info)

    gui.resolution_warning = QLabel("⚠ Resolution >512 may fail on some GPUs")
    gui.resolution_warning.setStyleSheet("color: orange; font-weight: bold; font-size: 9px;")
    gui.resolution_warning.setVisible(False)
    layout.addWidget(gui.resolution_warning)

    gui.progress_bar = QProgressBar()
    layout.addWidget(gui.progress_bar)

    gui.compute_btn = QPushButton("Compute Volume")
    gui.compute_btn.clicked.connect(gui.compute_volume)
    gui.compute_btn.setEnabled(False)
    layout.addWidget(gui.compute_btn)

    group.setLayout(layout)
    return group


def _build_subvolume_group(gui):
    """Build the subvolume selection group - visible in all visualization modes."""
    gui.subvolume_group = QGroupBox("Subvolume Zoom")
    gui.subvolume_group.setCheckable(True)
    gui.subvolume_group.setChecked(False)
    gui.subvolume_group.toggled.connect(gui.subvolume.on_toggled)
    sub_layout = QVBoxLayout()
    sub_layout.addWidget(QLabel("Use sliders to choose cube center (0-1) and edge length."))

    for axis, label_text in zip('xyz', ['X', 'Y', 'Z']):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label_text} Center"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, gui.subvolume_slider_steps)
        slider.setValue(gui.subvolume_slider_steps // 2)
        slider.valueChanged.connect(lambda _, ax=axis: gui.subvolume.on_slider_changed(ax))
        row.addWidget(slider, 1)
        value_label = QLabel("0.500")
        value_label.setFixedWidth(60)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value_label)
        gui.subvolume_center_sliders[axis] = slider
        gui.subvolume_center_labels[axis] = value_label
        sub_layout.addLayout(row)
    for axis in 'xyz':
        gui.subvolume.on_slider_changed(axis)

    size_row = QHBoxLayout()
    size_row.addWidget(QLabel("Box Size"))
    gui.subvolume_size_slider = QSlider(Qt.Orientation.Horizontal)
    gui.subvolume_size_slider.setRange(int(0.05 * gui.subvolume_slider_steps), gui.subvolume_slider_steps)
    gui.subvolume_size_slider.setValue(int(0.30 * gui.subvolume_slider_steps))
    gui.subvolume_size_slider.valueChanged.connect(gui.subvolume.on_size_changed)
    size_row.addWidget(gui.subvolume_size_slider, 1)
    gui.subvolume_size_label = QLabel("0.300")
    gui.subvolume_size_label.setFixedWidth(60)
    gui.subvolume_size_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    size_row.addWidget(gui.subvolume_size_label)
    sub_layout.addLayout(size_row)
    gui.subvolume.on_size_changed()

    gui.extract_subbox_btn = QPushButton("Extract Subbox…")
    gui.extract_subbox_btn.setEnabled(False)
    gui.extract_subbox_btn.clicked.connect(gui.subvolume.extract_subbox)
    sub_layout.addWidget(gui.extract_subbox_btn)

    gui.subvolume_group.setLayout(sub_layout)
    return gui.subvolume_group


def _build_neural_group(gui):
    group = QGroupBox("3. Neural Renderer")
    layout = QVBoxLayout()

    intro = QLabel("Train a tinycudann neural field and render it like other modes.")
    intro.setWordWrap(True)
    intro.setStyleSheet("color: #555; font-size: 9px;")
    layout.addWidget(intro)

    gui.neural_status_label = QLabel("Status: idle (load data)")
    gui.neural_status_label.setStyleSheet("color: #0066cc; font-weight: bold;")
    layout.addWidget(gui.neural_status_label)

    form = QFormLayout()
    gui.neural_params.clear()

    steps_spin = QSpinBox()
    steps_spin.setRange(100, 1000000)
    steps_spin.setSingleStep(5000)
    steps_spin.setValue(20000)
    gui.neural_params['steps'] = steps_spin
    form.addRow("Training Steps", steps_spin)

    batch_spin = QSpinBox()
    batch_spin.setRange(512, 262144)
    batch_spin.setSingleStep(4096)
    batch_spin.setValue(16384)
    gui.neural_params['batch_size'] = batch_spin
    form.addRow("Batch Size", batch_spin)

    lr_spin = QDoubleSpinBox()
    lr_spin.setDecimals(5)
    lr_spin.setRange(1e-5, 1e-1)
    lr_spin.setSingleStep(1e-4)
    lr_spin.setValue(1e-3)
    gui.neural_params['lr'] = lr_spin
    form.addRow("Learning Rate", lr_spin)

    log_interval_spin = QSpinBox()
    log_interval_spin.setRange(1, 10000)
    log_interval_spin.setValue(100)
    log_interval_spin.setSingleStep(50)
    gui.neural_params['log_interval'] = log_interval_spin
    form.addRow("Log Interval", log_interval_spin)

    preview_interval_spin = QSpinBox()
    preview_interval_spin.setRange(0, 10000)
    preview_interval_spin.setValue(250)
    preview_interval_spin.setSingleStep(100)
    gui.neural_params['preview_interval'] = preview_interval_spin
    form.addRow("Preview Interval", preview_interval_spin)

    preview_res_spin = QSpinBox()
    preview_res_spin.setRange(32, 512)
    preview_res_spin.setValue(128)
    preview_res_spin.setSingleStep(32)
    gui.neural_params['preview_resolution'] = preview_res_spin
    form.addRow("Preview Resolution", preview_res_spin)

    inference_batch_spin = QSpinBox()
    inference_batch_spin.setRange(1024, 524288)
    inference_batch_spin.setValue(32768)
    inference_batch_spin.setSingleStep(8192)
    gui.neural_params['inference_batch'] = inference_batch_spin
    form.addRow("Inference Batch", inference_batch_spin)

    gui.neural_render_res_spin = QSpinBox()
    gui.neural_render_res_spin.setRange(64, 2048)
    gui.neural_render_res_spin.setValue(256)
    gui.neural_render_res_spin.setSingleStep(64)
    form.addRow("Render Resolution", gui.neural_render_res_spin)

    gui.neural_display_res_spin = QSpinBox()
    gui.neural_display_res_spin.setRange(64, 2048)
    gui.neural_display_res_spin.setValue(256)
    gui.neural_display_res_spin.setSingleStep(64)
    form.addRow("Neural Mode Grid", gui.neural_display_res_spin)

    layout.addLayout(form)

    res_hint = QLabel(
        "Both Render Resolution and Neural Mode Grid are applied when you press the "
        "'Render Neural Volume' button. Adjust them, then click the button to regenerate "
        "the neural field at the new size."
    )
    res_hint.setWordWrap(True)
    res_hint.setStyleSheet("color: #666; font-size: 9px;")
    layout.addWidget(res_hint)

    gui.neural_auto_preview_check = QCheckBox("Auto preview + update viewport during training")
    gui.neural_auto_preview_check.setChecked(True)
    layout.addWidget(gui.neural_auto_preview_check)

    gui.neural_low_memory_check = QCheckBox("Low-memory mode (≤6 GB GPU)")
    gui.neural_low_memory_check.setToolTip(
        "Reduces network width, hash levels, and batch sizes to keep training/inference within about 6 GB of VRAM."
        " Recommended for 1024³ renders or extremely large particle counts."
    )
    layout.addWidget(gui.neural_low_memory_check)

    button_layout = QHBoxLayout()
    gui.neural_train_btn = QPushButton("Train / Resume")
    gui.neural_train_btn.clicked.connect(gui.handle_neural_train_button)
    button_layout.addWidget(gui.neural_train_btn)

    gui.neural_reset_btn = QPushButton("Reset Network")
    gui.neural_reset_btn.setEnabled(False)
    gui.neural_reset_btn.clicked.connect(gui.reset_neural_renderer)
    button_layout.addWidget(gui.neural_reset_btn)
    layout.addLayout(button_layout)

    render_layout = QHBoxLayout()
    gui.neural_render_btn = QPushButton("Render Neural Volume")
    gui.neural_render_btn.setEnabled(False)
    gui.neural_render_btn.clicked.connect(lambda: gui.render_neural_volume(source='manual button'))
    render_layout.addWidget(gui.neural_render_btn)
    layout.addLayout(render_layout)

    # Save/Load weights buttons
    weights_layout = QHBoxLayout()
    gui.neural_save_btn = QPushButton("💾 Save Weights")
    gui.neural_save_btn.setEnabled(False)
    gui.neural_save_btn.setToolTip("Save trained neural network weights to a file")
    gui.neural_save_btn.clicked.connect(gui.save_neural_weights)
    weights_layout.addWidget(gui.neural_save_btn)

    gui.neural_load_btn = QPushButton("📂 Load Weights")
    gui.neural_load_btn.setToolTip("Load pre-trained neural network weights from a file")
    gui.neural_load_btn.clicked.connect(gui.load_neural_weights)
    weights_layout.addWidget(gui.neural_load_btn)
    layout.addLayout(weights_layout)

    gui.neural_loss_plot = LossPlotWidget()
    layout.addWidget(gui.neural_loss_plot)

    gui.neural_preview_label = QLabel("Last preview: —")
    gui.neural_preview_label.setStyleSheet("color: #444; font-size: 9px;")
    layout.addWidget(gui.neural_preview_label)

    layout.addStretch()
    group.setLayout(layout)
    gui._update_neural_controls_state()
    return group


def _build_camera_group(gui):
    group = QGroupBox("4. Camera Controls")
    layout = QVBoxLayout()

    gui.cam_params = {}
    params = [
        ("Azimuth", 0, 360, 45),
        ("Elevation", -90, 90, 30),
        ("Roll", -180, 180, 0),
        ("Distance", 0.5, 100, 2.5),
        ("FOV", 0, 120, 60),
    ]

    for name, min_val, max_val, default in params:
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel(f"{name}:"))
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(5 if name in ["Azimuth", "Elevation", "Roll"] else 10)
        spinbox.valueChanged.connect(gui.update_camera)
        gui.cam_params[name.lower()] = spinbox
        param_layout.addWidget(spinbox)
        layout.addLayout(param_layout)

    preset_layout = QHBoxLayout()
    for label, az, el in [("Front", 0, 0), ("Top", 0, 90), ("Side", 90, 0), ("Iso", 45, 30)]:
        btn = QPushButton(label)
        btn.clicked.connect(lambda checked, a=az, e=el: gui.set_camera_preset(a, e))
        preset_layout.addWidget(btn)
    layout.addLayout(preset_layout)

    reset_btn = QPushButton("Reset Camera")
    reset_btn.clicked.connect(gui.reset_camera)
    layout.addWidget(reset_btn)

    save_cam_layout = QHBoxLayout()
    save_cam_btn = QPushButton("Save Camera")
    save_cam_btn.clicked.connect(gui.save_camera_config)
    load_cam_btn = QPushButton("Load Camera")
    load_cam_btn.clicked.connect(gui.load_camera_config)
    save_cam_layout.addWidget(save_cam_btn)
    save_cam_layout.addWidget(load_cam_btn)
    layout.addLayout(save_cam_layout)

    group.setLayout(layout)
    return group


def _build_log_group(gui):
    group = QGroupBox("Log")
    layout = QVBoxLayout()

    gui.log_text = QTextEdit()
    gui.log_text.setReadOnly(True)
    gui.log_text.setMaximumHeight(150)
    layout.addWidget(gui.log_text)

    clear_btn = QPushButton("Clear Log")
    clear_btn.clicked.connect(gui.log_text.clear)
    layout.addWidget(clear_btn)

    group.setLayout(layout)
    return group
