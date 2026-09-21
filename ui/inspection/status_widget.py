"""Floating status panel shown next to the inspection overlay."""

import json
import os
import pygetwindow as gw
import numpy as np

import cv2
from PySide6.QtCore import Qt, Signal, QTimer, QSettings
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QPixmap,
    QRegion,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QLineEdit,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QSizePolicy,
)

from services.recipe_service import RecipeService, normalize_anomaly_threshold
from utils.paths import RECIPES_DIR, relativize_recipe_path, resolve_recipe_path
from services.marking_service import save_regions

STYLE_SHEET = """
#MainWidget{
    background: transparent;
    border:none;
}

#Container{
    background-color:#FFFFFF;
    border-radius:16px;
    border:1px solid #DDE4EE;
}

QLabel{
    border: none;
    outline: none;
    background: transparent;
    padding: 0;
    color: #4B5563;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

QPushButton{
    border: none;
    border-radius: 7px;
    background: transparent;
    color: #9CA3AF;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 2px;
}

QPushButton:hover{
    background-color: #F3F4F6;
    color: #1F2937;
}

QProgressBar {
    border: none;
    border-radius: 3px;
    background-color: #F3F4F6;
    max-height: 6px;
    text-visible: false;
}

QProgressBar::chunk {
    background-color: #4B5563;
    border-radius: 3px;
}

QComboBox {
    background-color: #F8FAFC;
    border: 1px solid #D9E2EC;
    border-radius: 8px;
    padding: 7px 10px;
    color: #374151;
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    font-size: 11px;
    font-weight: 600;
    min-width: 140px;
}

QComboBox:hover {
    background-color: #F1F5F9;
    color: #1F2937;
}

QComboBox::drop-down {
    border: none;
    background: transparent;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 4px;
    color: #4B5563;
    selection-background-color: #F3F4F6;
    selection-color: #1F2937;
}

QLabel:focus {
    border: none;
    outline: none;
}

QComboBox:focus { border: 1px solid #93C5FD; background-color: #FFFFFF; }

QTabWidget::pane {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    background: #FFFFFF;
    top: -1px;
}
QTabBar::tab {
    color: #64748B;
    background: transparent;
    min-width: 118px;
    padding: 8px 12px;
    margin-right: 4px;
    border-bottom: 2px solid transparent;
    font-size: 11px;
    font-weight: 700;
}
QTabBar::tab:hover { color: #2563EB; }
QTabBar::tab:selected { color: #2563EB; border-bottom: 2px solid #2563EB; }
QScrollArea { border: none; background: white; }
QScrollBar:vertical { background: #F8FAFC; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

STATE_COLORS = {
    "STARTING": ("#3B82F6", "STARTING"),
    "WAITING_SOURCE": ("#F59E0B", "WAITING SOURCE"),
    "READY": ("#10B981", "READY"),
    "DETECTING": ("#06B6D4", "DETECTING"),
    "ERROR": ("#EF4444", "ERROR"),
}

VERDICT_COLORS = {
    "READY": ("#2563EB", "#EFF6FF"),
    "LOADING MODEL": ("#D97706", "#FFFBEB"),
    "NO SOURCE": ("#6B7280", "#F3F4F6"),
    "PASS": ("#10B981", "#ECFDF5"),
    "UNKNOWN": ("#D97706", "#FFFBEB"),
    "NOT FOUND": ("#F59E0B", "#FFF7ED"),
    "SELECT PACKAGE": ("#2563EB", "#EFF6FF"),
    "WAITING SOURCE": ("#6B7280", "#F3F4F6"),
}

ICONS = {
    "pin": "\uE718",
    "unpin": "\uE77A",
    "minimize": "\uE921",
    "restore": "\uE70E",
    "crop": "\uE7A8",
    "close": "\uE8BB",
}


class _PanelScrollArea(QScrollArea):
    """Keep wrapped content within the viewport, including at high DPI."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.widget() is not None:
            self.widget().setFixedWidth(self.viewport().width())


class StatusWidget(QWidget):
    """Shows the live inspection status and lets the operator pick a recipe."""

    recipe_changed = Signal(object)
    recipe_request_changed = Signal(object, int)
    capture_mode_changed = Signal(bool)

    def __init__(self, overlay):
        super().__init__()
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.overlay = overlay
        self.current_recipe = None
        self.is_pinned = True
        self.is_minimized = False
        self.dragging = False
        self.drag_position = None
        self._heatmap_pixmap = None
        self._heatmap_source = None
        self._mini_thumb_source = None
        self._style_cache = {}
        self.capture_mode = False
        self.capture_frame = None
        self._capture_screen = None
        self._capture_busy = False
        self._screen_region = None
        self._capture_target = None
        self._loading_capture_settings = True
        self._capture_watch = QTimer(self)
        self._capture_watch.setInterval(180)
        self._capture_watch.timeout.connect(self._sync_capture_target)
        from services.capture_service import CaptureService
        self.capture_service = CaptureService()
        self.capture_settings = QSettings("VisionAI", "VisionAI")
        self.recipe_request_id = 0
        self.model_loading = False

        families = QFontDatabase.families()
        self.icon_font_family = (
            "Segoe Fluent Icons"
            if "Segoe Fluent Icons" in families
            else "Segoe MDL2 Assets"
        )
        self._load_ui_fonts()

        self.overlay.roi_saved.connect(self.on_roi_saved)
        self.overlay.roi_cancelled.connect(self.on_roi_cancelled)
        self.overlay.capture_area_saved.connect(self.on_capture_area_saved)

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )

        self.setFixedSize(410, 690)
        self.setObjectName("MainWidget")
        self.setStyleSheet(STYLE_SHEET)

        self._setup_ui()
        QApplication.instance().aboutToQuit.connect(self._flush_capture_settings)
        self.load_recipes()

        self.combo_pkg.currentIndexChanged.connect(self.on_package_changed)

    @staticmethod
    def _load_ui_fonts():
        """Load the bundled UI fonts so the panel looks identical on every PC."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        font_dir = os.path.join(project_root, "assets", "fonts")
        for filename in (
            "Poppins-Regular.ttf",
            "Poppins-Medium.ttf",
            "Poppins-SemiBold.ttf",
            "Poppins-Bold.ttf",
        ):
            font_path = os.path.join(font_dir, filename)
            if os.path.exists(font_path):
                QFontDatabase.addApplicationFont(font_path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)

        self.container = QFrame()
        self.container.setObjectName("Container")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.container.setGraphicsEffect(shadow)

        outer_layout.addWidget(self.container)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(20, 18, 20, 18)
        main_layout.setSpacing(14)

        # ----- Header and system status -----
        self.system_status = QLabel("● WAITING SOURCE")
        self.system_status.setStyleSheet(
            "color: #10B981; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;"
        )

        self.btn_pin = self._make_icon_button(
            ICONS["pin"], "Pinned - click to unpin"
        )
        self.btn_min = self._make_icon_button(
            ICONS["minimize"], "Minimize to compact view"
        )
        self.btn_roi = self._make_icon_button(
            ICONS["crop"], "Capture or edit inspection ROI"
        )
        self.btn_capture = QPushButton("Capture")
        self.btn_capture.setParent(self)
        self.btn_capture.hide()
        self.btn_capture.setCheckable(True)
        self.btn_capture.setStyleSheet("QPushButton{background:#EFF6FF;color:#2563EB;border:1px solid #BFDBFE;border-radius:6px;padding:4px 8px;font-weight:600;} QPushButton:checked{background:#2563EB;color:white;}")
        self.btn_capture.clicked.connect(self.toggle_capture_mode)
        self.btn_close = self._make_icon_button(
            ICONS["close"],
            "Close",
            extra_style=(
                "QPushButton:hover { background-color: #FEF2F2; color: #EF4444; }"
            ),
        )

        self.btn_pin.clicked.connect(self.toggle_pin)
        self.btn_min.clicked.connect(self.toggle_minimize)
        self.btn_close.clicked.connect(QApplication.quit)
        self.btn_roi.clicked.connect(self.on_roi_clicked)

        header = QHBoxLayout()
        header.addWidget(self.system_status)
        header.addStretch()
        header.addWidget(self.btn_pin)
        header.addWidget(self.btn_min)
        header.addWidget(self.btn_roi)
        header.addWidget(self.btn_close)
        main_layout.addLayout(header)

        # ----- Body -----
        self.body_container = QWidget()
        body_layout = QVBoxLayout(self.body_container)
        body_layout.setContentsMargins(12, 14, 12, 14)
        body_layout.setSpacing(10)
        body_layout.setAlignment(Qt.AlignTop)

        self._setup_package_selector(body_layout)
        self._setup_result_card(body_layout)
        self._setup_metrics_grid(body_layout)
        # Kept as a non-visible compatibility value for integrations/tests.
        self.marking_detail = QLabel()
        self.marking_detail.hide()

        line_mid = QFrame()
        line_mid.setFrameShape(QFrame.HLine)
        line_mid.setStyleSheet(
            "background-color: #E5E7EB; max-height: 1px; border: none;"
        )
        body_layout.addWidget(line_mid)

        self._setup_heatmap_panel(body_layout)

        line_bottom = QFrame()
        line_bottom.setFrameShape(QFrame.HLine)
        line_bottom.setStyleSheet(
            "background-color: #E5E7EB; max-height: 1px; border: none;"
        )
        body_layout.addWidget(line_bottom)

        self._setup_metadata_bar(body_layout)
        self._setup_capture_panel(body_layout)
        # Keep capture controls in a real page instead of appending them to
        # the inspection content. This also prevents long inspection text
        # from changing the window geometry.
        body_layout.removeWidget(self.capture_panel)
        self.capture_panel.setParent(None)
        self.capture_page = QWidget()
        capture_layout = QVBoxLayout(self.capture_page)
        capture_layout.setContentsMargins(12, 14, 12, 14)
        capture_layout.addWidget(self.capture_panel)
        capture_layout.addStretch()
        self.capture_panel.show()
        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        self.pages.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for content, title in ((self.body_container, "Inspection"), (self.capture_page, "Capture")):
            content.setStyleSheet("background-color: white;")
            scroll = _PanelScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(content)
            self.pages.addTab(scroll, title)
        self.pages.currentChanged.connect(self._on_page_changed)
        main_layout.addWidget(self.pages, 1)

        # ----- Mini mode card (shown when minimized) -----
        self.mini = self._build_mini_card()
        self.mini.hide()
        main_layout.addWidget(self.mini)

    def _make_icon_button(self, glyph, tooltip, extra_style=""):
        button = QPushButton(glyph)
        button.setFont(QFont(self.icon_font_family, 11))
        button.setFixedSize(28, 28)
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(
            f"QPushButton {{ font-family: '{self.icon_font_family}'; }} {extra_style}"
        )
        return button

    def _setup_capture_panel(self, body_layout):
        self.capture_panel = QFrame()
        self.capture_panel.setStyleSheet("QFrame{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;} QLabel{border:none;} QLineEdit{background:white;border:1px solid #CBD5E1;border-radius:6px;padding:6px;} QPushButton{background:#2563EB;color:white;padding:7px 10px;border-radius:6px;}")
        layout = QVBoxLayout(self.capture_panel); layout.setContentsMargins(12,12,12,12); layout.setSpacing(8)
        title = QLabel("CAPTURE MODE"); title.setStyleSheet("font-weight:700;color:#334155;")
        hint = QLabel("Image size × zoom; X/Y offset from centre."); hint.setStyleSheet("color:#64748B;font-size:11px;")
        hint.setWordWrap(True)
        row = QHBoxLayout(); self.capture_folder = QLineEdit(); self.capture_folder.setReadOnly(True); self.capture_folder.setPlaceholderText("Choose output folder")
        browse = QPushButton("Browse"); browse.clicked.connect(self.choose_capture_folder); row.addWidget(self.capture_folder); row.addWidget(browse)
        actions = QHBoxLayout(); self.capture_area_btn = QPushButton("Set image area"); self.capture_area_btn.clicked.connect(self.set_capture_area)
        self.capture_take_btn = QPushButton("Take photo"); self.capture_take_btn.clicked.connect(self.take_capture)
        self.capture_take_btn.setEnabled(False)
        self.capture_area_status = QLabel("Waiting for LCmicro image area")
        self.capture_area_status.setWordWrap(True)
        self.capture_preview = QLabel("Your latest photo will appear here")
        self.capture_preview.setAlignment(Qt.AlignCenter)
        self.capture_preview.setFixedHeight(180)
        self.capture_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.capture_preview.setStyleSheet("background:#0F172A;color:#94A3B8;border:none;border-radius:8px;")
        self.capture_count = QLabel("0 photos saved"); self.capture_count.setStyleSheet("color:#64748B;font-size:11px;")
        self.capture_count.setWordWrap(True)
        self.capture_count.setMinimumWidth(0)
        self.capture_count.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.capture_folder.setMinimumWidth(0)
        self.capture_take_btn.setMinimumHeight(42)
        self.capture_area_btn.setMinimumHeight(36)
        self.capture_area_btn.setStyleSheet("QPushButton{background:white;color:#2563EB;border:1px solid #BFDBFE;padding:8px;border-radius:6px;} QPushButton:hover{background:#EFF6FF;}")
        layout.setSpacing(6)
        layout.addWidget(QLabel("SAVE PHOTOS TO"))
        layout.addLayout(row)
        self.capture_area_btn.hide()
        self.capture_size_toggle = QPushButton("IMAGE GEOMETRY  ▾")
        self.capture_size_toggle.setCheckable(True)
        self.capture_size_toggle.setChecked(True)
        self.capture_size_toggle.setStyleSheet("QPushButton{text-align:left;background:transparent;color:#334155;font-weight:700;padding:5px 0;} QPushButton:hover{color:#2563EB;}")
        self.capture_size_toggle.clicked.connect(self._toggle_capture_geometry)
        layout.addWidget(self.capture_size_toggle)
        self.capture_size_panel = QWidget()
        self.capture_size_panel.setObjectName("CaptureGeometry")
        self.capture_size_panel.setStyleSheet("QWidget#CaptureGeometry{background:transparent;} QLabel{background:transparent;border:none;} QSpinBox,QDoubleSpinBox{background:white;border:1px solid #CBD5E1;border-radius:5px;padding:3px;color:#334155;} QPushButton:disabled{background:#CBD5E1;color:#64748B;}")
        controls = QGridLayout(self.capture_size_panel)
        controls.setContentsMargins(0, 0, 0, 0)
        self.capture_width = QSpinBox()
        self.capture_height = QSpinBox()
        self.capture_zoom = QDoubleSpinBox()
        self.capture_x = QSpinBox()
        self.capture_y = QSpinBox()
        for control, value in ((self.capture_width, 2160), (self.capture_height, 1620)):
            control.setRange(4, 20000)
            control.setValue(value)
        self.capture_zoom.setRange(0.1, 400)
        self.capture_zoom.setDecimals(1)
        self.capture_zoom.setSuffix(" %")
        self.capture_zoom.setValue(51.4)
        for control in (self.capture_x, self.capture_y):
            control.setRange(-20000, 20000)
        for index, (name, control) in enumerate((("Width px", self.capture_width), ("Height px", self.capture_height), ("Zoom", self.capture_zoom), ("Offset X", self.capture_x), ("Offset Y", self.capture_y))):
            controls.addWidget(QLabel(name), (index // 3) * 2, index % 3)
            controls.addWidget(control, (index // 3) * 2 + 1, index % 3)
            control.valueChanged.connect(self._capture_setting_changed)
        layout.addWidget(self.capture_size_panel)
        layout.addWidget(self.capture_area_status)
        layout.addWidget(self.capture_take_btn)
        layout.addWidget(self.capture_count)
        layout.addWidget(self.capture_preview)
        self._load_capture_settings()
        body_layout.addWidget(self.capture_panel); self.capture_panel.hide()

    def _toggle_capture_geometry(self, expanded):
        self.capture_size_toggle.setChecked(expanded)
        self.capture_size_panel.setVisible(expanded)
        self.capture_preview.setFixedHeight(180 if expanded else 260)
        self.capture_size_toggle.setText("Frame settings  ▾" if expanded else "Frame settings  ▸")
        if not self._loading_capture_settings:
            self._save_capture_settings()

    def _capture_setting_changed(self, *_):
        self._save_capture_settings()
        self._refresh_capture_region()

    def _load_capture_settings(self):
        values = {
            self.capture_width: ("capture/width", 2160),
            self.capture_height: ("capture/height", 1620),
            self.capture_zoom: ("capture/zoom", 51.4),
            self.capture_x: ("capture/offset_x", -107),
            self.capture_y: ("capture/offset_y", 3),
        }
        for control, (key, default) in values.items():
            control.blockSignals(True)
            control.setValue(self.capture_settings.value(key, default, type=type(default)))
            control.blockSignals(False)
        folder = self.capture_settings.value("capture/folder", "", type=str)
        if folder and os.path.isdir(folder):
            self.capture_service.set_folder(folder)
            self.capture_folder.setText(folder)
        self._toggle_capture_geometry(
            self.capture_settings.value("capture/geometry_expanded", False, type=bool)
        )
        self._loading_capture_settings = False

    def _save_capture_settings(self):
        if self._loading_capture_settings:
            return
        for control, key in ((self.capture_width, "capture/width"), (self.capture_height, "capture/height"),
                             (self.capture_zoom, "capture/zoom"), (self.capture_x, "capture/offset_x"),
                             (self.capture_y, "capture/offset_y")):
            self.capture_settings.setValue(key, control.value())
        self.capture_settings.setValue("capture/geometry_expanded", self.capture_size_toggle.isChecked())
        if self.capture_service.folder:
            self.capture_settings.setValue("capture/folder", str(self.capture_service.folder))
        self.capture_settings.sync()

    def closeEvent(self, event):
        self._flush_capture_settings()
        try:
            QApplication.instance().aboutToQuit.disconnect(self._flush_capture_settings)
        except RuntimeError:
            pass
        self._capture_watch.stop()
        super().closeEvent(event)

    def _flush_capture_settings(self):
        for control in (self.capture_width, self.capture_height, self.capture_zoom, self.capture_x, self.capture_y):
            control.interpretText()
        self._save_capture_settings()

    def toggle_capture_mode(self):
        self.capture_mode = self.btn_capture.isChecked()
        self.pages.setCurrentIndex(1 if self.capture_mode else 0)
        self.capture_mode_changed.emit(self.capture_mode)

    def _on_page_changed(self, index):
        self.capture_mode = index == 1
        self.btn_capture.setChecked(self.capture_mode)
        self.set_capture_frame(None)
        self.overlay.capture_view_rect = None
        self.overlay.update_boxes([])
        self.capture_mode_changed.emit(self.capture_mode)
        if self.capture_mode:
            self._capture_screen = self.screen()
            self.system_status.setText("● CAPTURE MODE")
            self._inspection_overlay_flags = self.overlay.windowFlags()
            self.overlay.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            self.overlay.setGeometry(self._capture_screen.geometry())
            self._refresh_capture_region()
            self._capture_watch.start()
            self._sync_capture_target()
        else:
            self._capture_watch.stop()
            self.overlay.clearMask()
            self._capture_screen = None
            if hasattr(self, "_inspection_overlay_flags"):
                self.overlay.setWindowFlags(self._inspection_overlay_flags)
            self.overlay.hide()

    def choose_capture_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose capture folder")
        if folder:
            self.capture_service.set_folder(folder); self.capture_folder.setText(folder)
            self._save_capture_settings()
            self.capture_take_btn.setEnabled(self.capture_service.area is not None)

    def set_capture_area(self):
        if self.capture_frame is not None:
            self.overlay.set_capture_frame(self.capture_frame); self.overlay.start_capture_area_mode()

    def on_capture_area_saved(self, rect):
        if self.capture_frame is None:
            return
        height, width = self.capture_frame.shape[:2]
        try:
            self.capture_service.set_area(rect, (width, height))
            self.capture_count.setText("Image area ready  ·  0 photos saved")
        except Exception as error:
            self.capture_count.setText(str(error))

    def take_capture(self):
        if self.capture_mode and self._capture_screen is not None:
            self._sync_capture_target()
            if self._capture_target is None or not self.overlay.isVisible():
                return
            if self._capture_busy or self._screen_region is None or self.capture_service.folder is None:
                return
            self._capture_busy = True
            self._pending_screen = self._capture_screen
            self._pending_region = self._screen_region
            self._pending_target = self._capture_target
            self.capture_take_btn.setEnabled(False)
            self.overlay.hide()
            self.hide()
            QTimer.singleShot(120, self._finish_screen_capture)
            return
        if self.capture_frame is None: return
        try:
            path = self.capture_service.save(self.capture_frame)
            self.capture_preview.setPixmap(QPixmap(str(path)).scaled(
                self.capture_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.overlay.flash_capture()
            self.capture_count.setText(f"{self.capture_service.sequence} photos saved  ·  {path.name}")
        except Exception as error:
            self.capture_count.setText(str(error))

    def _finish_screen_capture(self):
        saved = False
        try:
            active = gw.getActiveWindow()
            if active is None or active._hWnd != self._pending_target._hWnd:
                raise RuntimeError("Capture cancelled: source window changed")
            pixmap = self._pending_screen.grabWindow(0)
            if pixmap.isNull():
                raise RuntimeError("Screen capture unavailable")
            image = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            rows = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
            frame = cv2.cvtColor(rows[:, :image.width() * 3].reshape(image.height(), image.width(), 3), cv2.COLOR_RGB2BGR)
            x, y, w, h = self._pending_region
            if x < 0 or y < 0 or x + w > image.width() or y + h > image.height():
                raise RuntimeError("Screen size changed — retry capture")
            self.capture_service.set_area((x, y, w, h), (image.width(), image.height()))
            path = self.capture_service.save(frame)
            self.capture_preview.setPixmap(QPixmap(str(path)).scaled(
                self.capture_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.capture_count.setText(f"{self.capture_service.sequence} photos saved · {path.name}")
            saved = True
        except Exception as error:
            self.capture_count.setText(str(error))
        finally:
            self._capture_busy = False
            self.show()
            if self.capture_mode:
                self._refresh_capture_region()
                self._sync_capture_target()
                if saved:
                    self.overlay.flash_capture()

    def set_capture_frame(self, frame, region=None):
        self.capture_frame = None if frame is None else frame.copy()
        self._refresh_capture_region()

    def _refresh_capture_region(self, *_):
        from services.capture_service import scaled_capture_rect
        self.capture_service.area = None
        self.overlay.capture_view_rect = None
        self._screen_region = None
        screen = self._capture_screen if self.capture_mode else None
        if screen is not None or self.capture_frame is not None:
            ratio = screen.devicePixelRatio() if screen is not None else 1.0
            if screen is not None:
                geometry = screen.geometry()
                self.overlay.setGeometry(geometry)
                width, height = round(geometry.width() * ratio), round(geometry.height() * ratio)
            else:
                height, width = self.capture_frame.shape[:2]
            try:
                region = scaled_capture_rect((width, height),
                    (self.capture_width.value(), self.capture_height.value()),
                    self.capture_zoom.value(), (self.capture_x.value(), self.capture_y.value()))
                self.capture_service.set_area(region, (width, height))
                self._screen_region = region
                self.overlay.capture_view_rect = tuple(round(value / ratio) for value in region)
                self.capture_area_status.setText(f"Ready · {region[2]} × {region[3]} px")
            except ValueError as error:
                self.capture_area_status.setText(str(error))
        else:
            self.capture_area_status.setText("Open Capture to position the screen frame")
        self.overlay.update()
        self.capture_take_btn.setEnabled(
            not self._capture_busy and self.capture_service.area is not None and self.capture_service.folder is not None)

    def _sync_capture_target(self):
        """Keep the capture frame on the active LCmicro/PowerPoint window only."""
        if not self.capture_mode or self._capture_busy:
            return
        try:
            active = gw.getActiveWindow()
            handle = getattr(active, "_hWnd", None)
            own = handle in (int(self.winId()), int(self.overlay.winId()))
            title = (getattr(active, "title", "") or "").lower()
            if own:
                target = self._capture_target
            else:
                target = active if ("lcmicro" in title or "powerpoint" in title) else None
            if target is None or target.isMinimized:
                self._capture_target = None
                self.overlay.hide()
                self.capture_take_btn.setEnabled(False)
                self.capture_area_status.setText("Open LCmicro or PowerPoint to show the frame")
                return
            self._capture_target = target
            self._refresh_capture_region()
            geometry = self._capture_screen.geometry()
            # pygetwindow reports native pixels; Qt overlay uses logical pixels.
            ratio = self._capture_screen.devicePixelRatio()
            x = round(target.left / ratio - geometry.x())
            y = round(target.top / ratio - geometry.y())
            width, height = round(target.width / ratio), round(target.height / ratio)
            self.overlay.setMask(QRegion(x, y, width, height))
            self.overlay.show()
            self.overlay.sync_above_window(target._hWnd)
            region = self.overlay.capture_view_rect
            if region is None or not (region[0] >= x and region[1] >= y and region[0]+region[2] <= x+width and region[1]+region[3] <= y+height):
                self.capture_take_btn.setEnabled(False)
                self.capture_area_status.setText("Keep the capture frame inside the source window")
        except Exception:
            self._capture_target = None
            self.overlay.hide()
            self.capture_take_btn.setEnabled(False)

    def _setup_package_selector(self, body_layout):
        self.pkg_container = QWidget()
        pkg_layout = QVBoxLayout(self.pkg_container)
        pkg_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_pkg_title = QLabel("PACKAGE / RECIPE")
        self.lbl_pkg_title.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px;"
        )

        self.combo_pkg = QComboBox()
        self.combo_pkg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_pkg.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_pkg.setMinimumContentsLength(12)

        pkg_layout.addWidget(self.lbl_pkg_title)
        pkg_layout.addWidget(self.combo_pkg)
        body_layout.addWidget(self.pkg_container)

    def _setup_result_card(self, body_layout):
        self.card = QFrame()
        self.card.setFixedHeight(56)
        self.card.setStyleSheet(
            """
            QFrame {
              border: 1px solid #D1D5DB;
              border-radius: 8px;
              background-color: #F3F4F6;
            }
            """
        )

        self.result = QLabel("NO SOURCE")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #6B7280; letter-spacing: 1.5px;"
        )

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(self.result)

        body_layout.addWidget(self.card)
        body_layout.addSpacing(2)

    def _setup_metrics_grid(self, body_layout):
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Unit counter paired with the orientation status, plus one long
        # message box for log and error output.
        self.box_det, self.detected = self._create_big_box("DETECTED", "UNT")
        self.box_ori, self.orientation = self._create_big_box("ORIENTATION", " ")
        self.box_marking, self.marking_status = self._create_big_box("MARKING", " ")
        self.box_score, self.score = self._create_big_box("SCORE", "")
        self.box_msg, self.message = self._create_long_box("MESSAGE")

        grid_layout.addWidget(self.box_det, 0, 0, 1, 2)
        grid_layout.addWidget(self.box_ori, 0, 2, 1, 2)
        grid_layout.addWidget(self.box_marking, 0, 4, 1, 2)
        grid_layout.addWidget(self.box_msg, 1, 0, 1, 6)
        body_layout.addLayout(grid_layout)
        body_layout.addSpacing(4)

    @staticmethod
    def _create_big_box(title, unit_text):
        box = QFrame()
        box.setStyleSheet(
            "QFrame { border: 1px solid #F3F4F6; border-radius: 8px;"
            " background-color: #F9FAFB; }"
        )
        box.setFixedHeight(68)

        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)

        lbl_title = QLabel(title.upper())
        lbl_title.setStyleSheet(
            "color: #9CA3AF; font-size: 10px; font-weight: 600;"
            " letter-spacing: 0.5px; background:transparent; border:none;"
        )

        val_layout = QHBoxLayout()
        lbl_val = QLabel("0")
        lbl_val.setStyleSheet(
            "color: #1F2937; font-size: 24px; font-weight: 700;"
            " background:transparent; border:none;"
        )
        lbl_unit = QLabel(unit_text)
        lbl_unit.setStyleSheet(
            "color: #9CA3AF; font-size: 9px; font-weight: 600;"
            " margin-bottom: 4px; background:transparent; border:none;"
        )

        val_layout.addWidget(lbl_val)
        val_layout.addWidget(lbl_unit, alignment=Qt.AlignBottom)
        val_layout.addStretch()

        layout.addWidget(lbl_title)
        layout.addLayout(val_layout)
        return box, lbl_val

    @staticmethod
    def _create_long_box(title):
        box = QFrame()
        box.setObjectName("MessageBox")
        box.setStyleSheet(
            "QFrame#MessageBox { border: 1px solid #F3F4F6; border-radius: 8px;"
            " background-color: #F9FAFB; }"
        )
        box.setMinimumHeight(64)

        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 4, 12, 4)

        lbl_title = QLabel(f"{title.upper()}:")
        lbl_title.setStyleSheet(
            "color: #6B7280; font-size: 11px; font-weight: 600;"
            " background:transparent; border:none;"
        )
        lbl_val = QLabel("-")
        lbl_val.setObjectName("MessageText")
        lbl_val.setWordWrap(True)
        lbl_val.setStyleSheet(
            "QLabel#MessageText { color: #1F2937; font-size: 11px;"
            " font-weight: 600; background: transparent; border: 0px;"
            " outline: 0px; padding: 0px; }"
        )

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        return box, lbl_val

    def _setup_heatmap_panel(self, body_layout):
        lbl_heatmap_title = QLabel("ANOMALY HEATMAP")
        lbl_heatmap_title.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px;"
        )
        body_layout.addWidget(lbl_heatmap_title)

        self.heatmap = QLabel("Waiting for the first inspection")
        self.heatmap.setAlignment(Qt.AlignCenter)
        self.heatmap.setWordWrap(True)
        self.heatmap.setMinimumWidth(0)
        self.heatmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.heatmap.setFixedHeight(150)
        self.heatmap.setStyleSheet(
            "QLabel { border: 1px solid #E5E7EB; border-radius: 8px;"
            " background-color: #111827; color: #6B7280;"
            " font-size: 11px; font-weight: 600; }"
        )
        body_layout.addWidget(self.heatmap)
        body_layout.addSpacing(4)

    def _setup_metadata_bar(self, body_layout):
        bottom_bar = QHBoxLayout()

        def create_meta_lbl(prefix, val, val_color="#6B7280"):
            lbl = QLabel()
            lbl.setText(
                f"<font color='#9CA3AF'>{prefix}:</font>"
                f" <font color='{val_color}'>{val}</font>"
            )
            lbl.setStyleSheet("font-size: 10px; font-weight: 600;")
            return lbl

        self.source = create_meta_lbl("SRC", "LCmicro")
        self.fps_meta = create_meta_lbl("FPS", "0.0 Hz", "#2563EB")
        # self.db_status = create_meta_lbl("DB", "CONNECTED", "#10B981")

        bottom_bar.addWidget(self.source)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.fps_meta)
        bottom_bar.addStretch()
        # bottom_bar.addWidget(self.db_status)
        body_layout.addLayout(bottom_bar)

    # ------------------------------------------------------------------
    # Recipe handling
    # ------------------------------------------------------------------

    def load_recipes(self):
        """Populate the package selector with every saved recipe."""
        self.combo_pkg.clear()
        self.combo_pkg.addItem("Select Package", None)

        if not RECIPES_DIR.exists():
            print(f"Recipe directory not found: {RECIPES_DIR}")
            return

        for root, _dirs, files in os.walk(RECIPES_DIR):
            for filename in files:
                if filename != "recipe.json":
                    continue

                file_path = os.path.join(root, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        recipe_data = json.load(f)
                except Exception as error:
                    print(f"Error loading {filename}: {error}")
                    continue

                recipe_data["_recipe_path"] = file_path
                self.combo_pkg.addItem(recipe_data.get("recipe_name", filename), recipe_data)

    def on_package_changed(self, index):
        recipe_data = self.combo_pkg.itemData(index)
        self.recipe_request_id += 1
        request_id = self.recipe_request_id
        if recipe_data is None:
            self.current_recipe = None
            self.model_loading = False
            self.recipe_changed.emit(None)
            self.recipe_request_changed.emit(None, request_id)
            self.message.setText("No recipe selected")
            return

        self.current_recipe = recipe_data
        self.model_loading = True
        self.recipe_changed.emit(recipe_data)
        self.recipe_request_changed.emit(recipe_data, request_id)

        print(
            f"[RECIPE] Requested {recipe_data.get('recipe_name', '-')} "
            f"(package: {recipe_data.get('package_type', '-')}, "
            f"family: {recipe_data.get('package_family', '-')}, "
            f"pins: {recipe_data.get('pin_count', 0)})"
        )

        self.message.setText(f"Loading: {recipe_data.get('recipe_name', '')}")
        self._update_result_card("LOADING MODEL", False, "", model_loading=True)

    # ------------------------------------------------------------------
    # Window behaviour
    # ------------------------------------------------------------------

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        position = self.pos()

        if self.is_pinned:
            self.setWindowFlags(
                Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
            )
            self.btn_pin.setText(ICONS["pin"])
            self.btn_pin.setToolTip("Pinned - click to unpin")
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
            self.btn_pin.setText(ICONS["unpin"])
            self.btn_pin.setToolTip("Unpinned - click to pin")

        self.show()
        self.move(position)

    def toggle_minimize(self):
        self.is_minimized = not self.is_minimized

        if self.is_minimized:
            self.pages.hide()
            self.body_container.hide()
            self.pkg_container.hide()
            self.mini.show()
            self._mini_thumb_source = None
            self._update_mini_thumb()
            self.btn_min.setText(ICONS["restore"])
            self.btn_min.setToolTip("Expand to full view")
            self.setFixedSize(300, 148)
        else:
            self.mini.hide()
            self.pages.show()
            self.body_container.show()
            self.pkg_container.show()
            self.btn_min.setText(ICONS["minimize"])
            self.btn_min.setToolTip("Minimize to compact view")
            self.setFixedSize(410, 690)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.is_pinned:
            self.dragging = True
            self.drag_position = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if self.dragging and not self.is_pinned:
            self.move(event.globalPosition().toPoint() - self.drag_position)

    def mouseReleaseEvent(self, event):
        self.dragging = False

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def update_status(
        self,
        system_state,
        inspection_result,
        count,
        fps,
        score=0.0,
        is_upside_down=False,
        fail_reason="",
        error_message="",
        heatmap=None,
        marking=None,
        model_loading=False,
        recipe_request_id=None,
    ):
        """Refresh every field of the status panel."""
        if (
            self.model_loading
            and not model_loading
            and recipe_request_id is None
            and inspection_result not in ("NO SOURCE", "SELECT PACKAGE")
        ):
            return
        if recipe_request_id is not None and recipe_request_id != self.recipe_request_id:
            return
        self.model_loading = bool(model_loading)
        if model_loading:
            inspection_result = "LOADING MODEL"
        self._update_system_state(system_state)
        self._update_result_card(
            inspection_result, is_upside_down, fail_reason, error_message, score,
            model_loading=model_loading
        )

        display_count = 0 if inspection_result == "SELECT PACKAGE" else count
        self.detected.setText(str(display_count))
        self.score.setText(f"{score:.3f}")
        marking = marking or {"status": "NOT CHECKED", "score": None}
        mark_status = marking["status"]
        color = "#10B981" if mark_status == "MATCH" else (
            "#EF4444" if mark_status in ("FAIL", "MISMATCH", "POSITION ERROR") else "#D97706")
        self.marking_status.setText(mark_status)
        self.marking_status.setStyleSheet(f"border:none;font-weight:700;color:{color};")
        similarity = marking.get("score")
        detail = "Similarity: —" if similarity is None else f"Similarity: {similarity:.3f}"
        if marking.get("object"):
            detail = f"IC #{marking['object']} · " + detail
        self.marking_detail.setText(detail + "\n" + marking.get("reason", ""))

        if heatmap is not None:
            self._show_heatmap(heatmap)
        else:
            self._heatmap_source = None
            self._heatmap_pixmap = None
            self._mini_thumb_source = None
            self.heatmap.clear()
            self.heatmap.setText("No anomaly heatmap for this inspection")
            self.mini_thumb.clear()

        self.fps_meta.setText(
            f"<font color='#9CA3AF'>FPS:</font>"
            f" <font color='#2563EB'>{fps:.1f} Hz</font>"
        )
        self._update_mini(inspection_result, display_count, fps, score)

    def _apply_style(self, key, widget, css):
        """Restyle a widget only when the stylesheet actually changes.

        ``setStyleSheet`` repolishes the widget even for an identical string,
        which is pure overhead on the 700 ms status-update path.
        """
        if widget is self.orientation:
            css += " border: 0px; outline: 0px; background: transparent; padding: 0px;"
        if self._style_cache.get(key) != css:
            self._style_cache[key] = css
            widget.setStyleSheet(css)

    def _update_system_state(self, state):
        color, text = STATE_COLORS.get(state.value, ("#6B7280", state.value))
        self.system_status.setText(f"● {text}")
        self._apply_style(
            "system_status",
            self.system_status,
            f"color:{color}; font-size: 13px; font-weight: 600;",
        )

    def _build_message(
        self, inspection_result, is_upside_down, fail_reason, score,
        error_message="", model_loading=False
    ):
        """Compose an informative operator message for the current verdict."""
        if model_loading:
            name = (self.current_recipe or {}).get("recipe_name", "")
            return f"Loading: {name}", "#D97706"
        if inspection_result == "READY":
            name = (self.current_recipe or {}).get("recipe_name", "")
            return f"Ready: {name}", "#2563EB"
        if inspection_result == "NO SOURCE":
            if error_message:
                return error_message, "#EF4444"
            return (
                fail_reason
                if fail_reason
                else "Open the LCmicro program window to begin inspection"
            ), "#6B7280"

        if inspection_result == "SELECT PACKAGE":
            if model_loading:
                return "Loading recipe model...", "#D97706"
            return "Select a product recipe", "#2563EB"

        if inspection_result == "NOT FOUND":
            return (
                "No package detected in the source view"
            ), "#F59E0B"

        if inspection_result == "UNKNOWN":
            return fail_reason or "Inspection could not be determined", "#D97706"

        if inspection_result == "PASS":
            if fail_reason.startswith("Last inspected sample"):
                return fail_reason, "#10B981"
            threshold = self._recipe_threshold()
            margin = self._percent(threshold - score, threshold)
            return (
                f"Package verified · score {score:.4f} ≤ limit"
                f" {threshold:.4f} · margin {margin:.0f}%"
            ), "#10B981"

        if is_upside_down:
            return f"{fail_reason} · top mark not matched", "#EF4444"

        if fail_reason.startswith("Score:"):
            threshold = self._recipe_threshold()
            excess = self._percent(score - threshold, threshold)
            return (
                f"Defect suspected · score {score:.4f} > limit"
                f" {threshold:.4f} (+{excess:.0f}%)"
            ), "#EF4444"

        if fail_reason.startswith(("Pixel region", "Pixel gate")):
            return f"Defect suspected · {fail_reason}", "#EF4444"

        return fail_reason, "#EF4444"

    def _recipe_threshold(self):
        recipe = self.current_recipe or {}
        return normalize_anomaly_threshold(recipe.get("anomaly_threshold"))

    @staticmethod
    def _percent(value, reference):
        if reference <= 0:
            return 0.0
        return max(value, 0.0) / reference * 100.0

    def _update_result_card(
        self,
        inspection_result,
        is_upside_down,
        fail_reason,
        error_message="",
        score=0.0,
        model_loading=False,
    ):
        self.result.setText(inspection_result)

        if inspection_result == "NO SOURCE":
            self.orientation.setText("NO SOURCE")
            self._apply_style(
                "orientation",
                self.orientation,
                "color: #EF4444; font-size: 16px; font-weight: 700;"
                " background:transparent;",
            )
        elif inspection_result == "NOT FOUND":
            self.orientation.setText("NOT FOUND")
            color="#F59E0B"

        elif inspection_result == "SELECT PACKAGE":
            self.orientation.setText("N/A")
            color = "#2563EB"

        elif inspection_result == "UNKNOWN":
            self.orientation.setText("UNKNOWN")
            self._apply_style(
                "orientation", self.orientation,
                "color: #D97706; font-size: 16px; font-weight: 700; background:transparent;",
            )

        elif is_upside_down:
            self.orientation.setText("UPSIDE DOWN")
            self._apply_style(
                "orientation",
                self.orientation,
                "color: #EF4444; font-size: 16px; font-weight: 700;"
                " background:transparent;",
            )
        else:
            self.orientation.setText("NORMAL")
            self._apply_style(
                "orientation",
                self.orientation,
                "color: #10B981; font-size: 16px; font-weight: 700;"
                " background:transparent;",
            )

        display_text, message_color = self._build_message(
            inspection_result, is_upside_down, fail_reason, score, error_message,
            model_loading=model_loading
        )

        self.message.setText(display_text)
        self._apply_style(
            "message",
            self.message,
            f"color: {message_color}; font-size: 11px; font-weight: 600;"
            " background:transparent;",
        )

        color, background = VERDICT_COLORS.get(
            inspection_result, ("#EF4444", "#FEF2F2")
        )

        self._apply_style(
            "result",
            self.result,
            f"font-size: 24px; font-weight: 700; color: {color}; letter-spacing: 1.5px;",
        )
        self._apply_style(
            "card",
            self.card,
            f"QFrame {{ border: 1px solid {color}; border-radius: 8px;"
            f" background-color: {background}; }}",
        )

    # ------------------------------------------------------------------
    # Mini card and heatmap rendering
    # ------------------------------------------------------------------

    def _build_mini_card(self):
        card = QFrame()
        card.setObjectName("MiniCard")
        card.setFixedHeight(64)
        card.setStyleSheet(
            "QFrame#MiniCard { border: 1px solid #10B981; border-radius: 8px;"
            " background-color: #ECFDF5; }"
        )

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.mini_dot = QLabel("●")
        self.mini_dot.setStyleSheet("color: #10B981; font-size: 15px;")

        text_column = QVBoxLayout()
        text_column.setSpacing(0)

        self.mini_verdict = QLabel("PASS")
        self.mini_verdict.setStyleSheet(
            "color: #10B981; font-size: 15px; font-weight: 700;"
            " letter-spacing: 0.5px; background:transparent; border:none;"
        )
        self.mini_detail = QLabel("-")
        self.mini_detail.setStyleSheet(
            "color: #6B7280; font-size: 10px; font-weight: 600;"
            " background:transparent; border:none;"
        )
        text_column.addWidget(self.mini_verdict)
        text_column.addWidget(self.mini_detail)

        self.mini_thumb = QLabel()
        self.mini_thumb.setFixedSize(48, 48)
        self.mini_thumb.setAlignment(Qt.AlignCenter)
        self.mini_thumb.setStyleSheet(
            "border: 1px solid #E5E7EB; border-radius: 6px;"
            " background-color: #111827;"
        )

        layout.addWidget(self.mini_dot)
        layout.addLayout(text_column)
        layout.addStretch()
        layout.addWidget(self.mini_thumb)
        return card

    def _update_mini(self, inspection_result, count, fps, score):
        color, background = VERDICT_COLORS.get(
            inspection_result, ("#EF4444", "#FEF2F2")
        )

        self.mini_verdict.setText(inspection_result)
        self._apply_style(
            "mini_dot",
            self.mini_dot,
            f"color: {color}; font-size: 15px;",
        )
        self._apply_style(
            "mini_verdict",
            self.mini_verdict,
            f"color: {color}; font-size: 15px; font-weight: 700;"
            " letter-spacing: 0.5px; background:transparent; border:none;",
        )
        self._apply_style(
            "mini_card",
            self.mini,
            f"QFrame#MiniCard {{ border: 1px solid {color}; border-radius: 8px;"
            f" background-color: {background}; }}",
        )

        if inspection_result in ("PASS", "FAIL"):
            detail = f"score {score:.3f} · {count} IC · {fps:.1f} Hz"
        else:
            detail = f"{count} IC in view · {fps:.1f} Hz"
        self.mini_detail.setText(detail)

        if self.mini.isVisible():
            self._update_mini_thumb()

    def _update_mini_thumb(self):
        if self._heatmap_pixmap is None or self._heatmap_pixmap.isNull():
            self.mini_thumb.setPixmap(QPixmap())
            return

        if self._mini_thumb_source is self._heatmap_pixmap:
            return

        thumb = self._heatmap_pixmap.scaled(
            48,
            48,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        left = max((thumb.width() - 48) // 2, 0)
        top = max((thumb.height() - 48) // 2, 0)
        thumb = thumb.copy(left, top, 48, 48)
        self.mini_thumb.setPixmap(self._rounded(thumb, 5))
        self._mini_thumb_source = self._heatmap_pixmap

    def _show_heatmap(self, image):
        if image is self._heatmap_source and not self.heatmap.pixmap().isNull():
            return

        pixmap = self._pixmap_from_bgr(image)
        if pixmap.isNull():
            return

        self._heatmap_pixmap = pixmap
        self._heatmap_source = image
        self._mini_thumb_source = None

        panel_width = max(self.heatmap.width(), 200)
        scaled = pixmap.scaled(
            panel_width,
            self.heatmap.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.heatmap.setPixmap(self._rounded(scaled, 6))

    @staticmethod
    def _pixmap_from_bgr(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    @staticmethod
    def _rounded(pixmap, radius):
        if pixmap.isNull():
            return pixmap

        output = QPixmap(pixmap.size())
        output.fill(Qt.transparent)

        painter = QPainter(output)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, pixmap.width(), pixmap.height(), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return output

    # ------------------------------------------------------------------
    # Top-mark template capture
    # ------------------------------------------------------------------

    def on_roi_clicked(self):
        if self.current_recipe is None:
            self.message.setText("Please select a recipe first.")
            return
        if self.overlay.roi_mode:
            self.overlay.finish_roi_mode()
            return
        self.overlay.roi_regions = {}
        self.overlay.roi_live_frame = (
            self.overlay.current_frame.copy() if self.overlay.current_frame is not None else None)
        template_path = resolve_recipe_path(self.current_recipe.get("top_mark_template"))
        if template_path and template_path.is_file():
            top = cv2.imdecode(np.frombuffer(template_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if top is None:
                self.message.setText("Cannot read Top-mark template")
                return
            # Display the stored template in the editor so nested coordinates
            # are independent of where the live IC has moved.
            available_w = max(100, self.overlay.width() - 48)
            available_h = max(100, self.overlay.height() - 190)
            scale = min(available_w / top.shape[1], available_h / top.shape[0], 2.0)
            w, h = max(4, int(top.shape[1]*scale)), max(4, int(top.shape[0]*scale))
            canvas = np.full((max(self.overlay.height(), h+190),
                               max(self.overlay.width(), w+48), 3), 241, np.uint8)
            canvas[165:165+h, 24:24+w] = cv2.resize(top, (w, h))
            self.overlay.set_frame(canvas)
            self.overlay.roi_regions["top_mark"] = (24, 165, w, h)
            self.overlay.roi_source.setText("Saved Top-mark reference")
            laser = self.current_recipe.get("laser_mark_roi") or {}
            if laser and self.current_recipe.get("laser_mark_template"):
                try:
                    x, y, lw, lh = [float(laser[k]) for k in ("x", "y", "w", "h")]
                    if min(x,y) >= 0 and min(lw,lh)>0 and x+lw<=1.001 and y+lh<=1.001:
                        self.overlay.roi_regions["laser_mark"] = (
                            24+round(x*w), 165+round(y*h), round(lw*w), round(lh*h))
                except (KeyError, ValueError, TypeError):
                    pass
        elif self.overlay.current_frame is None:
            self.message.setText("No image available")
            return
        self.overlay.set_laser_roi_available("top_mark" in self.overlay.roi_regions)
        self.overlay.start_roi_mode()
        self.combo_pkg.setEnabled(False)
        self.message.setText("Blue: Top-mark · Yellow: Laser-mark · Save when ready")

    def on_roi_cancelled(self):
        self.combo_pkg.setEnabled(True)
        self.message.setText("ROI editor closed")

    def on_roi_saved(self, payload):
        if self.current_recipe is None:
            return
        try:
            regions, dirty = payload
            data = save_regions(self.current_recipe["_recipe_path"],
                                self.overlay.current_frame, regions, dirty)
            updated = dict(data, _recipe_path=self.current_recipe["_recipe_path"])
            self.current_recipe = updated
            for index in range(self.combo_pkg.count()):
                item = self.combo_pkg.itemData(index)
                if item and item.get("_recipe_path") == updated["_recipe_path"]:
                    self.combo_pkg.setItemData(index, updated)
                    break
            self.recipe_changed.emit(dict(updated))
            self.recipe_request_changed.emit(dict(updated), self.recipe_request_id)
            self.overlay.finish_roi_mode()
            self.message.setText("Saved: " + ", ".join(sorted(dirty)))
        except Exception as error:
            self.overlay.roi_notice = "Save failed: " + str(error)
            self.overlay.update()
            self.message.setText(self.overlay.roi_notice)
