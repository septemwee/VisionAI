"""Floating status panel shown next to the inspection overlay."""

import json
import os
import numpy as np

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPath,
    QPixmap,
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
    border-radius:12px;
    border:1px solid #E5E7EB;
}

QLabel{
    border: none;
    color: #4B5563;
    font-family: 'Poppins', 'Segoe UI', system-ui, sans-serif;
}

QPushButton{
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #9CA3AF;
    font-family: 'Poppins', sans-serif;
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
    background-color: #F3F4F6;
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    color: #374151;
    font-family: 'Poppins', sans-serif;
    font-size: 11px;
    font-weight: 600;
    min-width: 140px;
}

QComboBox:hover {
    background-color: #E5E7EB;
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
"""

STATE_COLORS = {
    "STARTING": ("#3B82F6", "STARTING"),
    "WAITING_SOURCE": ("#F59E0B", "WAITING SOURCE"),
    "READY": ("#10B981", "READY"),
    "DETECTING": ("#06B6D4", "DETECTING"),
    "ERROR": ("#EF4444", "ERROR"),
}

VERDICT_COLORS = {
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


class StatusWidget(QWidget):
    """Shows the live inspection status and lets the operator pick a recipe."""

    recipe_changed = Signal(object)
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
        from services.capture_service import CaptureService
        self.capture_service = CaptureService()

        families = QFontDatabase.families()
        self.icon_font_family = (
            "Segoe Fluent Icons"
            if "Segoe Fluent Icons" in families
            else "Segoe MDL2 Assets"
        )

        self.overlay.roi_saved.connect(self.on_roi_saved)
        self.overlay.roi_cancelled.connect(self.on_roi_cancelled)
        self.overlay.capture_area_saved.connect(self.on_capture_area_saved)

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )

        self.setFixedSize(360, 610)
        self.setObjectName("MainWidget")
        self.setStyleSheet(STYLE_SHEET)

        self._setup_ui()
        self.load_recipes()

        self.combo_pkg.currentIndexChanged.connect(self.on_package_changed)

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
        main_layout.setSpacing(12)

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
        header.addWidget(self.btn_capture)
        header.addWidget(self.btn_close)
        main_layout.addLayout(header)

        # ----- Body -----
        self.body_container = QWidget()
        body_layout = QVBoxLayout(self.body_container)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(14)

        self._setup_package_selector(body_layout)
        self._setup_result_card(body_layout)
        self._setup_metrics_grid(body_layout)
        marking_card = QFrame()
        marking_card.setStyleSheet("QFrame{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;}")
        marking_layout = QVBoxLayout(marking_card)
        marking_layout.setContentsMargins(12, 8, 12, 8)
        title = QLabel("MARKING")
        title.setStyleSheet("border:none;font-weight:700;color:#334155;")
        self.marking_status = QLabel("Laser-mark: NOT CHECKED")
        self.marking_detail = QLabel("Similarity: —")
        for label in (self.marking_status, self.marking_detail):
            label.setStyleSheet("border:none;font-size:11px;")
            label.setWordWrap(True)
        marking_layout.addWidget(title)
        marking_layout.addWidget(self.marking_status)
        marking_layout.addWidget(self.marking_detail)
        body_layout.addWidget(marking_card)

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
        capture_layout.setContentsMargins(0, 0, 0, 0)
        capture_layout.addWidget(self.capture_panel)
        capture_layout.addStretch()
        self.capture_panel.show()
        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        self.pages.setFixedHeight(520)
        self.pages.addTab(self.body_container, "Inspection")
        self.pages.addTab(self.capture_page, "Capture")
        self.pages.currentChanged.connect(self._on_page_changed)
        main_layout.addWidget(self.pages)

        # ----- Mini mode card (shown when minimized) -----
        self.mini = self._build_mini_card()
        self.mini.hide()
        main_layout.addWidget(self.mini)

    def _make_icon_button(self, glyph, tooltip, extra_style=""):
        button = QPushButton(glyph)
        button.setFont(QFont(self.icon_font_family, 11))
        button.setFixedSize(26, 24)
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
        hint = QLabel("Save only the selected LCmicro image viewport."); hint.setStyleSheet("color:#64748B;font-size:11px;")
        row = QHBoxLayout(); self.capture_folder = QLineEdit(); self.capture_folder.setReadOnly(True); self.capture_folder.setPlaceholderText("Choose output folder")
        browse = QPushButton("Browse"); browse.clicked.connect(self.choose_capture_folder); row.addWidget(self.capture_folder); row.addWidget(browse)
        actions = QHBoxLayout(); self.capture_area_btn = QPushButton("Set image area"); self.capture_area_btn.clicked.connect(self.set_capture_area)
        self.capture_take_btn = QPushButton("Take photo"); self.capture_take_btn.clicked.connect(self.take_capture)
        self.capture_count = QLabel("0 photos saved"); self.capture_count.setStyleSheet("color:#64748B;font-size:11px;")
        actions.addWidget(self.capture_area_btn); actions.addWidget(self.capture_take_btn); actions.addWidget(self.capture_count)
        layout.addWidget(title); layout.addWidget(hint); layout.addLayout(row); layout.addLayout(actions)
        body_layout.addWidget(self.capture_panel); self.capture_panel.hide()

    def toggle_capture_mode(self):
        self.capture_mode = self.btn_capture.isChecked()
        self.pages.setCurrentIndex(1 if self.capture_mode else 0)
        self.capture_mode_changed.emit(self.capture_mode)

    def _on_page_changed(self, index):
        self.capture_mode = index == 1
        self.btn_capture.setChecked(self.capture_mode)
        self.capture_mode_changed.emit(self.capture_mode)

    def choose_capture_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose capture folder")
        if folder:
            self.capture_service.set_folder(folder); self.capture_folder.setText(folder)

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
        if self.capture_frame is None: return
        try:
            path = self.capture_service.save(self.capture_frame)
            self.capture_count.setText(f"{self.capture_service.sequence} photos saved  ·  {path.name}")
        except Exception as error:
            self.capture_count.setText(str(error))

    def set_capture_frame(self, frame):
        self.capture_frame = None if frame is None else frame.copy()

    def _setup_package_selector(self, body_layout):
        self.pkg_container = QWidget()
        pkg_layout = QHBoxLayout(self.pkg_container)
        pkg_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_pkg_title = QLabel("PRODUCT:")
        self.lbl_pkg_title.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px;"
        )

        self.combo_pkg = QComboBox()

        pkg_layout.addWidget(self.lbl_pkg_title)
        pkg_layout.addWidget(self.combo_pkg)
        pkg_layout.addStretch()
        body_layout.addWidget(self.pkg_container)

    def _setup_result_card(self, body_layout):
        self.card = QFrame()
        self.card.setFixedHeight(56)
        self.card.setStyleSheet(
            """
            QFrame {
              border: 1px solid #10B981;
              border-radius: 8px;
              background-color: #ECFDF5;
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
        self.box_score, self.score = self._create_big_box("SCORE", "")
        self.box_msg, self.message = self._create_long_box("MESSAGE")

        grid_layout.addWidget(self.box_det, 0, 0, 1, 2)
        grid_layout.addWidget(self.box_ori, 0, 2, 1, 2)
        grid_layout.addWidget(self.box_msg, 1, 0, 1, 4)
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
        box.setStyleSheet(
            "QFrame { border: 1px solid #F3F4F6; border-radius: 8px;"
            " background-color: #F9FAFB; }"
        )
        box.setFixedHeight(64)

        layout = QHBoxLayout(box)
        layout.setContentsMargins(12, 4, 12, 4)

        lbl_title = QLabel(f"{title.upper()}:")
        lbl_title.setStyleSheet(
            "color: #6B7280; font-size: 11px; font-weight: 600;"
            " background:transparent; border:none;"
        )
        lbl_val = QLabel("-")
        lbl_val.setWordWrap(True)
        lbl_val.setStyleSheet(
            "color: #1F2937; font-size: 11px; font-weight: 600;"
            " background:transparent; border:none;"
        )

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        layout.addStretch()
        return box, lbl_val

    def _setup_heatmap_panel(self, body_layout):
        lbl_heatmap_title = QLabel("ANOMALY HEATMAP")
        lbl_heatmap_title.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px;"
        )
        body_layout.addWidget(lbl_heatmap_title)

        self.heatmap = QLabel("Waiting for the first inspection")
        self.heatmap.setAlignment(Qt.AlignCenter)
        self.heatmap.setFixedHeight(118)
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
        if recipe_data is None:
            self.current_recipe = None
            self.recipe_changed.emit(None)
            self.message.setText("No recipe selected")
            return

        self.current_recipe = recipe_data
        self.recipe_changed.emit(recipe_data)

        print(
            f"[RECIPE] Loaded {recipe_data.get('recipe_name', '-')} "
            f"(package: {recipe_data.get('package_type', '-')}, "
            f"family: {recipe_data.get('package_family', '-')}, "
            f"pins: {recipe_data.get('pin_count', 0)})"
        )

        self.message.setText(f"Loaded: {recipe_data.get('recipe_name', '')}")

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
            self.body_container.show()
            self.pkg_container.show()
            self.btn_min.setText(ICONS["minimize"])
            self.btn_min.setToolTip("Minimize to compact view")
            self.setFixedSize(360, 610)

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
    ):
        """Refresh every field of the status panel."""
        self._update_system_state(system_state)
        self._update_result_card(
            inspection_result, is_upside_down, fail_reason, error_message, score
        )

        self.detected.setText(str(count))
        self.score.setText(f"{score:.3f}")
        marking = marking or {"status": "NOT CHECKED", "score": None}
        mark_status = marking["status"]
        color = "#10B981" if mark_status == "MATCH" else (
            "#EF4444" if mark_status in ("FAIL", "MISMATCH", "POSITION ERROR") else "#D97706")
        self.marking_status.setText("Laser-mark: " + mark_status)
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
        self._update_mini(inspection_result, count, fps, score)

    def _apply_style(self, key, widget, css):
        """Restyle a widget only when the stylesheet actually changes.

        ``setStyleSheet`` repolishes the widget even for an identical string,
        which is pure overhead on the 700 ms status-update path.
        """
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
        self, inspection_result, is_upside_down, fail_reason, score, error_message=""
    ):
        """Compose an informative operator message for the current verdict."""
        if inspection_result == "NO SOURCE":
            if error_message:
                return error_message, "#EF4444"
            return (
                fail_reason
                if fail_reason
                else "Open the LCmicro program window to begin inspection"
            ), "#6B7280"

        if inspection_result == "SELECT PACKAGE":
            return "Select a product recipe", "#2563EB"

        if inspection_result == "NOT FOUND":
            return (
                "No package detected in the source view"
            ), "#F59E0B"

        if inspection_result == "UNKNOWN":
            return fail_reason or "Inspection could not be determined", "#D97706"

        if inspection_result == "PASS":
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
            inspection_result, is_upside_down, fail_reason, score, error_message
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
            self.overlay.finish_roi_mode()
            self.message.setText("Saved: " + ", ".join(sorted(dirty)))
        except Exception as error:
            self.overlay.roi_notice = "Save failed: " + str(error)
            self.overlay.update()
            self.message.setText(self.overlay.roi_notice)
