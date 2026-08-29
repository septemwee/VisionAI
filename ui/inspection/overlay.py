"""Transparent overlay window drawn on top of the captured source window."""

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class OverlayWindow(QWidget):
    """Draws detection results and hosts the interactive ROI editor."""

    roi_saved = Signal(tuple)
    roi_cancelled = Signal()

    def __init__(self):
        super().__init__()

        self.boxes = []
        self.current_frame = None

        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_drawing_roi = False

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def set_frame(self, frame):
        self.current_frame = frame.copy()

    def start_roi_mode(self):
        """Enable mouse interaction so the user can draw a top-mark ROI."""
        self.roi_mode = True
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None

        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def finish_roi_mode(self):
        """Disable ROI editing and make the overlay click-through again."""
        self.roi_mode = False
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

        if self.roi_mode:
            self._paint_roi_mode(painter)
        else:
            self._paint_detections(painter)

    def _paint_roi_mode(self, painter):
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            20,
            30,
            "ROI MODE | Drag Mouse | ENTER=Save | ESC=Cancel",
        )

        if not (self.roi_start and self.roi_end):
            return

        x = min(self.roi_start.x(), self.roi_end.x())
        y = min(self.roi_start.y(), self.roi_end.y())
        width = abs(self.roi_end.x() - self.roi_start.x())
        height = abs(self.roi_end.y() - self.roi_start.y())

        # Darken everything outside the selected rectangle.
        painter.fillRect(0, 0, self.width(), y, QColor(0, 0, 0, 180))
        painter.fillRect(0, y + height, self.width(), self.height() - (y + height), QColor(0, 0, 0, 180))
        painter.fillRect(0, y, x, height, QColor(0, 0, 0, 180))
        painter.fillRect(x + width, y, self.width() - (x + width), height, QColor(0, 0, 0, 180))

        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawRect(x, y, width, height)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(x, y - 5, f"{width} x {height}")

    def _paint_detections(self, painter):
        for box in self.boxes:
            points = box["points"]
            label = box["label"]
            score = box.get("score", 0)
            result = box.get("result", "PASS")

            color = self._result_color(result)

            polygon = QPolygonF()
            for x, y in points:
                polygon.append(QPointF(float(x), float(y)))

            painter.setPen(QPen(color, 2))
            painter.drawPolygon(polygon)

            label_x = int(min(point[0] for point in points))
            label_y = int(min(point[1] for point in points)) - 8

            font = QFont("Poppins")
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(color)

            if result == "SELECT PACKAGE":
                text = "PLEASE SELECT PACKAGE"
            else:
                text = f"{label} | {result} | {score:.3f}"

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

    # ------------------------------------------------------------------
    # Mouse and keyboard interaction (ROI mode only)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if not self.roi_mode:
            return

        if event.button() == Qt.LeftButton:
            self.roi_start = event.position().toPoint()
            self.roi_end = self.roi_start
            self.is_drawing_roi = True

    def mouseMoveEvent(self, event):
        if not self.roi_mode or not self.is_drawing_roi:
            return

        self.roi_end = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.roi_mode:
            return

        if event.button() == Qt.LeftButton:
            self.is_drawing_roi = False
            self.roi_end = event.position().toPoint()

            x = min(self.roi_start.x(), self.roi_end.x())
            y = min(self.roi_start.y(), self.roi_end.y())
            width = abs(self.roi_end.x() - self.roi_start.x())
            height = abs(self.roi_end.y() - self.roi_start.y())

            self.roi_rect = (x, y, width, height)
            self.update()

    def keyPressEvent(self, event):
        if not self.roi_mode:
            return

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.roi_rect:
                self.roi_saved.emit(self.roi_rect)
            self.finish_roi_mode()
        elif event.key() == Qt.Key_Escape:
            self.roi_cancelled.emit()
            self.finish_roi_mode()
