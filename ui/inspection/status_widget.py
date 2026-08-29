"""Floating status panel shown next to the inspection overlay."""

import json
import os

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
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
)

from utils.paths import RECIPES_DIR

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
    background: transparent;
    color: #9CA3AF;
    font-family: 'Poppins', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 4px;
}

QPushButton:hover{
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


class StatusWidget(QWidget):
    """Shows the live inspection status and lets the operator pick a recipe."""

    recipe_changed = Signal(dict)

    def __init__(self, overlay):
        super().__init__()
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.overlay = overlay
        self.current_recipe = None
        self.is_pinned = True
        self.is_minimized = False
        self.dragging = False
        self.drag_position = None

        self.overlay.roi_saved.connect(self.on_roi_saved)
        self.overlay.roi_cancelled.connect(self.on_roi_cancelled)

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )

        self.resize(320, 510)
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
        self.system_status = QLabel("● READY")
        self.system_status.setStyleSheet(
            "color: #10B981; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;"
        )

        self.btn_pin = QPushButton(" ⚲ ")
        self.btn_min = QPushButton("—")
        self.btn_roi = QPushButton("✂")
        self.btn_close = QPushButton("✕")
        self.btn_close.setStyleSheet("QPushButton:hover { color: #EF4444; }")

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
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(14)

        self._setup_package_selector(body_layout)
        self._setup_result_card(body_layout)
        self._setup_metrics_grid(body_layout)

        line_mid = QFrame()
        line_mid.setFrameShape(QFrame.HLine)
        line_mid.setStyleSheet(
            "background-color: #E5E7EB; max-height: 1px; border: none;"
        )
        body_layout.addWidget(line_mid)

        self._setup_performance_zone(body_layout)

        line_bottom = QFrame()
        line_bottom.setFrameShape(QFrame.HLine)
        line_bottom.setStyleSheet(
            "background-color: #E5E7EB; max-height: 1px; border: none;"
        )
        body_layout.addWidget(line_bottom)

        self._setup_metadata_bar(body_layout)

        main_layout.addWidget(self.body_container)

        # ----- Mini mode summary (shown when minimized) -----
        self.summary = QLabel("PASS | IC : 0")
        self.summary.hide()
        self.summary.setStyleSheet(
            "color: #10B981; font-size: 16px; font-weight: 700; padding: 2px 0px;"
        )
        main_layout.addWidget(self.summary)

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

        self.result = QLabel("PASS")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #10B981; letter-spacing: 1.5px;"
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
        box.setFixedHeight(38)

        layout = QHBoxLayout(box)
        layout.setContentsMargins(12, 0, 12, 0)

        lbl_title = QLabel(f"{title.upper()}:")
        lbl_title.setStyleSheet(
            "color: #6B7280; font-size: 11px; font-weight: 600;"
            " background:transparent; border:none;"
        )
        lbl_val = QLabel("-")
        lbl_val.setStyleSheet(
            "color: #1F2937; font-size: 12px; font-weight: 700;"
            " background:transparent; border:none;"
        )

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        layout.addStretch()
        return box, lbl_val

    def _setup_performance_zone(self, body_layout):
        perf_layout = QGridLayout()
        perf_layout.setHorizontalSpacing(20)
        perf_layout.setVerticalSpacing(4)

        lbl_fps_title = QLabel("FPS LATENCY")
        lbl_fps_title.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #9CA3AF; letter-spacing: 0.5px;"
        )
        perf_layout.addWidget(lbl_fps_title, 0, 1)

        fps_val_layout = QHBoxLayout()
        self.fps = QLabel("18.3")
        self.fps.setStyleSheet("color: #2563EB; font-size: 14px; font-weight: 700;")
        lbl_hz = QLabel("Hz")
        lbl_hz.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: 600;")
        fps_val_layout.addWidget(self.fps)
        fps_val_layout.addWidget(lbl_hz)
        fps_val_layout.addStretch()

        perf_layout.addLayout(fps_val_layout, 1, 1)
        body_layout.addLayout(perf_layout)
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
        self.model = create_meta_lbl("MODEL", "YOLOv8")
        self.db_status = create_meta_lbl("DB", "CONNECTED", "#10B981")

        bottom_bar.addWidget(self.source)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.model)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.db_status)
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

        model_type = recipe_data.get("model_type", "Unknown")
        self.model.setText(
            f"<font color='#9CA3AF'>MODEL:</font> "
            f"<font color='#6B7280'>{model_type}</font>"
        )

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
            self.btn_pin.setText(" ⚲ ")
        else:
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
            self.btn_pin.setText(" ⟟ ")

        self.show()
        self.move(position)

    def toggle_minimize(self):
        self.is_minimized = not self.is_minimized

        if self.is_minimized:
            self.body_container.hide()
            self.pkg_container.hide()
            self.summary.show()
            self.resize(250, 75)
        else:
            self.summary.hide()
            self.body_container.show()
            self.pkg_container.show()
            self.resize(320, 510)

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
    ):
        """Refresh every field of the status panel."""
        self._update_system_state(system_state)
        self._update_result_card(inspection_result, is_upside_down, fail_reason)

        self.detected.setText(str(count))
        self.score.setText(f"{score:.3f}")
        self.fps.setText(f"{fps:.1f}")
        self.summary.setText(f"{inspection_result} | IC : {count}")

    def _update_system_state(self, state):
        color, text = STATE_COLORS.get(state.value, ("#6B7280", state.value))
        self.system_status.setText(f"● {text}")
        self.system_status.setStyleSheet(
            f"color:{color}; font-size: 13px; font-weight: 600;"
        )

    def _update_result_card(self, inspection_result, is_upside_down, fail_reason):
        self.result.setText(inspection_result)

        if is_upside_down:
            self.orientation.setText("UPSIDE DOWN")
            self.orientation.setStyleSheet(
                "color: #EF4444; font-size: 16px; font-weight: 700;"
                " background:transparent;"
            )
        else:
            self.orientation.setText("NORMAL")
            self.orientation.setStyleSheet(
                "color: #10B981; font-size: 16px; font-weight: 700;"
                " background:transparent;"
            )

        self.message.setWordWrap(True)

        if inspection_result == "PASS":
            display_text = fail_reason if fail_reason else "Product verified successfully."
            message_color = "#10B981"
        elif inspection_result == "NOT FOUND":
            display_text = "NOT FOUND"
            message_color = "#F59E0B"
        else:
            display_text = fail_reason if fail_reason else "Unknown inspection defect detected."
            message_color = "#EF4444"

        self.message.setText(display_text)
        self.message.setStyleSheet(
            f"color: {message_color}; font-size: 11px; font-weight: 600;"
            " background:transparent;"
        )

        result_colors = {
            "PASS": ("#10B981", "#ECFDF5"),
            "NOT FOUND": ("#F59E0B", "#FFF7ED"),
        }
        color, background = result_colors.get(
            inspection_result, ("#EF4444", "#FEF2F2")
        )

        self.result.setStyleSheet(
            f"font-size: 24px; font-weight: 700; color: {color}; letter-spacing: 1.5px;"
        )
        self.card.setStyleSheet(
            f"QFrame {{ border: 1px solid {color}; border-radius: 8px;"
            f" background-color: {background}; }}"
        )
        self.summary.setStyleSheet(
            f"color:{color}; font-size: 16px; font-weight: 700;"
        )

    # ------------------------------------------------------------------
    # Top-mark template capture
    # ------------------------------------------------------------------

    def on_roi_clicked(self):
        """Start ROI selection so the user can capture a top-mark template."""
        if self.current_recipe is None:
            self.message.setText("Please select a recipe first.")
            return

        self.overlay.start_roi_mode()
        self.message.setText("Draw ROI and press ENTER")

    def on_roi_cancelled(self):
        self.message.setText("ROI selection cancelled")

    def on_roi_saved(self, roi_rect):
        """Store the selected ROI image as the recipe's top-mark template."""
        if self.current_recipe is None:
            self.message.setText("No recipe selected")
            return

        frame = self.overlay.current_frame
        if frame is None:
            self.message.setText("No image available")
            return

        x, y, width, height = roi_rect
        roi = frame[y:y + height, x:x + width]

        recipe_path = self.current_recipe.get("_recipe_path")
        recipe_dir = os.path.dirname(recipe_path)
        template_path = os.path.join(recipe_dir, "top_mark_template.jpg")

        saved = cv2.imwrite(template_path, roi)

        with open(recipe_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["top_mark_template"] = template_path
        data["top_mark_roi"] = {"x": x, "y": y, "w": width, "h": height}

        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Top-mark template saved: {template_path} (ok={saved})")
        self.message.setText("Top-mark template saved")
