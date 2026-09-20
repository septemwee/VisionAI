"""Transparent overlay window drawn over the captured source window.

The overlay is deliberately NOT topmost. It is inserted directly above the
source window in the Z order (``sync_above_window``), so it stays visible
over the source while any window the user opens over the source covers it
as well; when the source disappears, the worker emits a hide.
"""

import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, QPointF, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QImage
from PySide6.QtWidgets import QPushButton, QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout

# SetWindowPos flags: keep position/size, never steal focus.
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010

# Z-order walk constants.
_GW_HWNDPREV = 3          # window directly above in the Z order
_GWL_EXSTYLE = -20
_WS_EX_TOPMOST = 0x00000008


if getattr(ctypes, "windll", None) is not None:
    ctypes.windll.user32.GetWindow.restype = ctypes.c_void_p
    ctypes.windll.user32.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    ctypes.windll.user32.GetWindowLongW.restype = ctypes.c_long
    ctypes.windll.user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    ctypes.windll.user32.SetWindowPos.restype = wintypes.BOOL
    ctypes.windll.user32.SetWindowPos.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]


class OverlayWindow(QWidget):
    """Draws detection results and hosts the interactive ROI editor."""

    roi_saved = Signal(tuple)
    roi_cancelled = Signal()
    capture_area_saved = Signal(tuple)
    capture_area_cancelled = Signal()

    def __init__(self):
        super().__init__()

        self.boxes = []
        self.current_frame = None
        self.capture_area_mode = False
        self.capture_view_rect = None
        self.capture_flash = False
        self.capture_flash_timer = QTimer(self)
        self.capture_flash_timer.setSingleShot(True)
        self.capture_flash_timer.timeout.connect(self._end_capture_flash)
        self.capture_rect = None
        self.capture_start = None
        self.capture_end = None

        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.roi_type = "top_mark"
        self.laser_roi_available = False
        self.roi_regions = {}
        self.roi_dirty = set()
        self.roi_live_frame = None
        self.roi_notice = ""
        self.roi_panel = QFrame(self)
        self.roi_panel.setObjectName("RoiPanel")
        self.roi_panel.setCursor(Qt.ArrowCursor)
        self.roi_panel.setStyleSheet("""
            QFrame#RoiPanel { background:#FFFFFF; border:1px solid #E2E8F0;
                             border-radius:12px; }
            QLabel { background:transparent; border:none; color:#64748B;
                     font-family:'Segoe UI'; font-size:11px; }
            QPushButton { background:#F1F5F9; color:#475569; border:1px solid #E2E8F0;
                          border-radius:7px; padding:6px 12px; font-family:'Segoe UI';
                          font-size:12px; font-weight:600; }
            QPushButton:hover { background:#E2E8F0; }
            QPushButton:checked { background:#EFF6FF; color:#2563EB; border-color:#93C5FD; }
            QPushButton:disabled { background:#F8FAFC; color:#94A3B8; border-color:#F1F5F9; }
            QPushButton#Laser:checked { background:#FFFBEB; color:#B45309; border-color:#FCD34D; }
            QPushButton#Save { background:#2563EB; color:white; border-color:#2563EB; }
            QPushButton#Save:hover { background:#1D4ED8; }
            QPushButton#Save:disabled { background:#CBD5E1; border-color:#CBD5E1; }
        """)
        layout = QVBoxLayout(self.roi_panel)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        title = QLabel("INSPECTION ROI")
        title.setStyleSheet("font-size:13px;font-weight:700;color:#334155;")
        heading.addWidget(title)
        self.roi_source = QLabel("Captured image")
        heading.addWidget(self.roi_source)
        heading.addStretch()
        self.roi_save = QPushButton("Save ROIs", self.roi_panel)
        self.roi_save.setObjectName("Save")
        self.roi_save.setToolTip("Save both regions and their templates (Enter)")
        self.roi_save.clicked.connect(self.submit_rois)
        self.roi_exit = QPushButton("Close", self.roi_panel)
        self.roi_exit.setToolTip("Close without saving (Esc)")
        self.roi_exit.clicked.connect(self.finish_roi_mode)
        heading.addWidget(self.roi_exit)
        heading.addWidget(self.roi_save)
        layout.addLayout(heading)
        selector = QHBoxLayout()
        self.roi_buttons = {}
        for name, text in (("top_mark", "Top-mark"), ("laser_mark", "Laser-mark")):
            button = QPushButton(text, self.roi_panel)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, value=name: self.set_roi_type(value))
            button.setToolTip("Blue: orientation reference" if name == "top_mark"
                              else "Yellow: expected pattern and position inside Top-mark")
            if name == "laser_mark":
                button.setObjectName("Laser")
            self.roi_buttons[name] = button
            selector.addWidget(button)
        self.roi_live = QPushButton("Use live image", self.roi_panel)
        self.roi_live.setToolTip("Replace the reference using the captured live image")
        self.roi_live.clicked.connect(self.use_live_image)
        selector.addWidget(self.roi_live)
        self.roi_clear_laser = QPushButton("Cancel Laser", self.roi_panel)
        self.roi_clear_laser.setToolTip("Remove the optional Laser-mark selection")
        self.roi_clear_laser.clicked.connect(self.clear_laser_roi)
        selector.addWidget(self.roi_clear_laser)
        selector.addStretch()
        self.roi_selection_status = QLabel("Laser-mark is optional")
        selector.addWidget(self.roi_selection_status)
        layout.addLayout(selector)
        self.roi_hint = QLabel("Select a region type, then drag on the image.")
        self.roi_hint.setWordWrap(True)
        layout.addWidget(self.roi_hint)
        for button in (*self.roi_buttons.values(), self.roi_save, self.roi_exit, self.roi_live, self.roi_clear_laser):
            button.setFocusPolicy(Qt.NoFocus)
        self.roi_panel.hide()
        self.is_drawing_roi = False

        # No WindowStaysOnTopHint: a topmost overlay would float above
        # every other window. The Z order is managed against the source
        # window instead (see ``sync_above_window``).
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def sync_above_window(self, hwnd):
        """Insert this overlay directly above ``hwnd`` in the Z order.

        Re-asserted on every overlay update: if the source window (or any
        other window) was raised in the meantime, the overlay is placed
        right above the source again, below anything newer. Windows-only;
        other platforms (and a missing/invalid handle) are a no-op that
        returns ``False``.
        """
        target = int(hwnd) if hwnd else 0
        if not target:
            return False

        user32 = getattr(getattr(ctypes, "windll", None), "user32", None)
        if user32 is None:
            return False

        try:
            overlay_hwnd = int(self.winId())
        except RuntimeError:
            return False

        # SetWindowPos places the positioned window directly BELOW its
        # hWndInsertAfter handle, so the anchor is the window directly
        # above the source. Walk up past topmost-band windows (taskbar,
        # always-on-top tools): anchoring on one would drag the overlay
        # into the topmost band. If the overlay itself is already directly
        # above the source, nothing to do. When the walk reaches the top
        # of the normal band, HWND_TOP (0) places the overlay directly
        # above the source.
        anchor = user32.GetWindow(target, _GW_HWNDPREV)
        for _ in range(256):
            if not anchor:
                anchor = 0  # HWND_TOP
                break
            if anchor == overlay_hwnd:
                return True
            if user32.GetWindowLongW(anchor, _GWL_EXSTYLE) & _WS_EX_TOPMOST:
                anchor = user32.GetWindow(anchor, _GW_HWNDPREV)
                continue
            break

        try:
            return bool(
                user32.SetWindowPos(
                    overlay_hwnd,
                    int(anchor) if anchor else 0,
                    0,
                    0,
                    0,
                    0,
                    _SWP_NOSIZE | _SWP_NOMOVE | _SWP_NOACTIVATE,
                )
            )
        except Exception as error:
            print(f"[OVERLAY] z-order sync failed: {error}")
            return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def set_frame(self, frame):
        if self.roi_mode or self.capture_area_mode:
            return
        self.current_frame = frame.copy()

    def set_capture_frame(self, frame):
        self.current_frame = None if frame is None else frame.copy()
        self.update()

    def start_capture_area_mode(self):
        self.capture_area_mode = True
        self.roi_panel.hide()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.raise_(); self.activateWindow(); self.setFocus(Qt.ActiveWindowFocusReason)
        self.setCursor(Qt.CrossCursor); self.update()

    def finish_capture_area_mode(self, save=False):
        if save and self.capture_rect:
            self.capture_area_saved.emit(tuple(self.capture_rect))
        else:
            self.capture_area_cancelled.emit()
        self.capture_area_mode = False
        self.capture_start = self.capture_end = self.capture_rect = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.unsetCursor(); self.update()

    def set_laser_roi_available(self, available):
        self.laser_roi_available = bool(available)
        self.roi_buttons["laser_mark"].setEnabled(bool(available))

    def clear_laser_roi(self):
        self.roi_regions.pop("laser_mark", None)
        self.roi_dirty.add("laser_mark")
        self.set_roi_type("top_mark")
        self.roi_notice = "Laser-mark cancelled. You can save Top-mark only."
        self._refresh_roi_controls()
        self.update()

    def submit_rois(self):
        if not self.roi_dirty:
            self.finish_roi_mode()
            return
        if "top_mark" not in self.roi_regions:
            self.roi_notice = "Draw Top-mark first"
            self.update()
            return
        self.roi_saved.emit((dict(self.roi_regions), set(self.roi_dirty)))

    def use_live_image(self):
        if self.roi_live_frame is None:
            return
        self.current_frame = self.roi_live_frame.copy()
        self.roi_regions = {}
        self.roi_dirty = set()
        self.set_laser_roi_available(False)
        self.set_roi_type("top_mark")
        self.roi_notice = "Draw a new Top-mark on the captured live image"
        self.roi_source.setText("Captured live image")
        self._refresh_roi_controls()
        self.update()

    def start_roi_mode(self):
        """Enable mouse interaction so the user can draw a top-mark ROI."""
        self.roi_mode = True
        self.roi_dirty = set()
        self.roi_notice = "Select a type, then drag. Redraw replaces only that type."
        self.roi_panel.setGeometry(12,10,max(320,self.width()-24),130)
        self.roi_panel.show()
        self.roi_panel.raise_()
        self.roi_live.setEnabled(self.roi_live_frame is not None)
        self._refresh_roi_controls()
        self.roi_save.show()
        self.roi_exit.show()
        self.roi_live.show()
        self.roi_clear_laser.show()
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None

        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)
        self.setCursor(Qt.CrossCursor)
        self.set_roi_type("top_mark")
        self.roi_buttons["top_mark"].show()
        self.roi_buttons["laser_mark"].show()
        for button in self.roi_buttons.values():
            button.raise_()
        self.update()

    def set_roi_type(self, roi_type):
        if roi_type not in ("top_mark", "laser_mark"):
            return
        if roi_type == "laser_mark" and not self.laser_roi_available:
            return
        self.roi_type = roi_type
        for name, button in self.roi_buttons.items():
            button.setChecked(name == roi_type)
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.update()

    def finish_roi_mode(self):
        """Disable ROI editing and make the overlay click-through again."""
        self.roi_mode = False
        self.roi_panel.hide()
        self.roi_cancelled.emit()
        self.roi_save.hide()
        self.roi_exit.hide()
        self.roi_live.hide()
        self.roi_clear_laser.hide()
        for button in self.roi_buttons.values():
            button.hide()
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_drawing_roi = False

        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.unsetCursor()
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.capture_view_rect is not None:
            painter.setPen(QPen(QColor("#38BDF8"), 2))
            painter.drawRect(*self.capture_view_rect)
            if self.capture_flash:
                painter.fillRect(*self.capture_view_rect, QColor(255, 255, 255, 85))
        elif self.capture_area_mode:
            self._paint_capture_area(painter)
        elif self.roi_mode:
            self._paint_roi_mode(painter)
        else:
            self._paint_detections(painter)

    def flash_capture(self):
        self.capture_flash = True
        self.capture_flash_timer.start(140)
        self.update()

    def _end_capture_flash(self):
        self.capture_flash = False
        self.update()

    def _paint_capture_area(self, painter):
        if self.current_frame is not None:
            frame = self.current_frame
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
            painter.drawImage(0, 0, image)
        painter.fillRect(0, 0, self.width(), 86, QColor("#F1F5F9"))
        painter.setPen(QPen(QColor("#334155"), 2))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(18, 30, "CAPTURE AREA  ·  Drag over the image viewport")
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(18, 55, "Enter: save area    Esc: cancel")
        if self.capture_rect:
            painter.setPen(QPen(QColor("#2563EB"), 3))
            painter.drawRect(*self.capture_rect)
        if self.capture_start and self.capture_end:
            x, y = min(self.capture_start.x(), self.capture_end.x()), min(self.capture_start.y(), self.capture_end.y())
            painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.DashLine))
            painter.drawRect(x, y, abs(self.capture_end.x()-self.capture_start.x()), abs(self.capture_end.y()-self.capture_start.y()))

    def _paint_roi_mode(self, painter):
        if self.current_frame is not None:
            frame = self.current_frame
            image = QImage(frame.data, frame.shape[1], frame.shape[0],
                           frame.strides[0], QImage.Format_BGR888)
            painter.drawImage(0, 0, image)
        painter.fillRect(0, 0, self.width(), 150, QColor("#F1F5F9"))
        for name, rect in self.roi_regions.items():
            painter.setPen(QPen(QColor("#60A5FA" if name == "top_mark" else "#FBBF24"), 2))
            painter.drawRect(*rect)
            painter.drawText(rect[0], max(130, rect[1] - 6), name.replace("_", "-"))
        if self.is_drawing_roi and self.roi_start and self.roi_end:
            painter.setPen(QPen(QColor("white"), 2, Qt.DashLine))
            x, y = min(self.roi_start.x(), self.roi_end.x()), min(self.roi_start.y(), self.roi_end.y())
            painter.drawRect(x, y, abs(self.roi_end.x() - self.roi_start.x()),
                             abs(self.roi_end.y() - self.roi_start.y()))

    def _paint_detections(self, painter):
        for box in self.boxes:
            points = self._expand_points(
                box["points"],
                scale=1.08,
            )
            label = box["label"]
            result = box.get("result", "DETECTED")
            stale = box.get("stale", False)

            color = self._result_color(result)

            polygon = QPolygonF()
            for x, y in points:
                polygon.append(QPointF(float(x), float(y)))

            painter.setPen(QPen(color, 2))
            painter.drawPolygon(polygon)

            if box.get("segments"):
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                for segment in box["segments"]:
                    contour = QPolygonF()
                    for x, y in segment:
                        contour.append(QPointF(float(x), float(y)))
                    painter.drawPolygon(contour)

            label_x = int(min(point[0] for point in points))
            label_y = int(min(point[1] for point in points)) - 8

            font = QFont("Poppins")
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(color)

            prefix = f"{label} | STALE" if stale else label

            if result == "SELECT PACKAGE":
                text = "PLEASE SELECT PACKAGE"
            elif result == "DETECTED":
                text = f"{prefix} | DETECTED"
            else:
                text = f"{prefix} | {result} | {box.get('score', 0.0):.3f}"

            painter.drawText(label_x, label_y, text)

    @staticmethod
    def _result_color(result):
        if result == "PASS":
            return QColor(0, 255, 0)
        if result == "FAIL":
            return QColor(255, 0, 0)
        if result == "SELECT PACKAGE":
            return QColor(0, 0, 255)
        return QColor(255, 180, 0)

    @staticmethod
    def _expand_points(points, scale=1.08):
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)

        expanded = []

        for x, y in points:
            new_x = center_x + (x - center_x) * scale
            new_y = center_y + (y - center_y) * scale
            expanded.append((new_x, new_y))

        return expanded

    # ------------------------------------------------------------------
    # Mouse and keyboard interaction (ROI mode only)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if self.capture_area_mode:
            if event.button() == Qt.LeftButton and event.position().y() >= 90:
                self.capture_start = self.capture_end = event.position().toPoint(); self.update()
            return
        if not self.roi_mode:
            return

        if event.button() == Qt.LeftButton and event.position().y() >= 150:
            self.roi_start = event.position().toPoint()
            self.roi_end = self.roi_start
            self.is_drawing_roi = True

    def mouseMoveEvent(self, event):
        if self.capture_area_mode:
            if self.capture_start:
                self.capture_end = event.position().toPoint(); self.update()
            return
        if not self.roi_mode or not self.is_drawing_roi:
            return

        self.roi_end = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.capture_area_mode:
            if event.button() == Qt.LeftButton and self.capture_start:
                self.capture_end = event.position().toPoint()
                x, y = min(self.capture_start.x(), self.capture_end.x()), min(self.capture_start.y(), self.capture_end.y())
                self.capture_rect = (x, y, abs(self.capture_end.x()-self.capture_start.x()), abs(self.capture_end.y()-self.capture_start.y()))
                self.capture_start = self.capture_end = None; self.update()
            return
        if not self.roi_mode:
            return

        if event.button() == Qt.LeftButton and self.roi_start is not None:
            self.is_drawing_roi = False
            self.roi_end = event.position().toPoint()

            x = min(self.roi_start.x(), self.roi_end.x())
            y = min(self.roi_start.y(), self.roi_end.y())
            width = abs(self.roi_end.x() - self.roi_start.x())
            height = abs(self.roi_end.y() - self.roi_start.y())

            self.roi_rect = (x, y, width, height)
            if width < 4 or height < 4:
                self.roi_notice = "ROI is too small"
                self.update()
                return
            if self.roi_type == "laser_mark":
                tx, ty, tw, th = self.roi_regions["top_mark"]
                if x < tx or y < ty or x + width > tx + tw or y + height > ty + th:
                    self.roi_notice = "Laser-mark must stay inside Top-mark"
                    self.update()
                    return
            elif "laser_mark" in self.roi_regions:
                lx, ly, lw, lh = self.roi_regions["laser_mark"]
                if lx < x or ly < y or lx + lw > x + width or ly + lh > y + height:
                    self.roi_notice = "Top-mark must contain the existing Laser-mark"
                    self.update()
                    return
            self.roi_regions[self.roi_type] = self.roi_rect
            self.roi_dirty.add(self.roi_type)
            self._refresh_roi_controls()
            self.set_laser_roi_available("top_mark" in self.roi_regions)
            self.roi_notice = "Selection ready. Switch type to draw another ROI, or Save."
            self.update()

    def keyPressEvent(self, event):
        if self.capture_area_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter): self.finish_capture_area_mode(True)
            elif event.key() == Qt.Key_Escape: self.finish_capture_area_mode(False)
            return
        if not self.roi_mode:
            return

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.submit_rois()
        elif event.key() == Qt.Key_Escape:
            self.finish_roi_mode()
    @property
    def roi_notice(self):
        return self._roi_notice

    @roi_notice.setter
    def roi_notice(self, text):
        self._roi_notice = text
        if hasattr(self, "roi_hint"):
            self.roi_hint.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "roi_panel"):
            self.roi_panel.setGeometry(12, 10, max(320,self.width()-24), 130)

    def _refresh_roi_controls(self):
        self.roi_save.setEnabled(bool(self.roi_dirty))
        top = "Top-mark ready" if "top_mark" in self.roi_regions else "Draw Top-mark first"
        laser = "Laser ready" if "laser_mark" in self.roi_regions else "Laser optional"
        self.roi_selection_status.setText(top + " / " + laser)
