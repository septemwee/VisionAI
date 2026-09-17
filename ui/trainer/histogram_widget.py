"""Minimal QPainter histogram used by the calibration panel (no new deps)."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

BINS = 24


class HistogramWidget(QWidget):
    """Draws the distribution of calibration scores plus the threshold line."""

    def __init__(self):
        super().__init__()
        self._scores = []
        self._threshold = None
        self.setMinimumHeight(140)

    def set_data(self, scores, threshold=None):
        """Store scores (0-1 fractions) and an optional threshold; repaint."""
        self._scores = [float(score) for score in scores]
        self._threshold = None if threshold is None else float(threshold)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#F9FAFB"))

            if not self._scores:
                painter.setPen(QColor("#6B7280"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "Run calibration to see the score distribution",
                )
                return

            counts = [0] * BINS
            for score in self._scores:
                index = int(min(max(score, 0.0), 0.999) * BINS)
                counts[index] += 1

            peak = max(counts) or 1
            body = self.rect().adjusted(10, 10, -10, -24)
            bar_width = body.width() / BINS

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#2563EB"))
            for index, count in enumerate(counts):
                if count <= 0:
                    continue
                height = body.height() * count / peak
                x = body.left() + index * bar_width
                painter.drawRect(
                    QRectF(x + 1, body.bottom() - height, max(bar_width - 2, 1), height)
                )

            if self._threshold is not None:
                clamped = min(max(self._threshold, 0.0), 1.0)
                x = body.left() + body.width() * clamped
                painter.setPen(QPen(QColor("#EF4444"), 2))
                painter.drawLine(
                    int(x), int(body.top()), int(x), int(body.bottom())
                )
                painter.setPen(QColor("#EF4444"))
                label_x = min(x + 4, body.right() - 70)
                painter.drawText(
                    int(label_x), int(body.bottom() - 4), f"thr {self._threshold:.3f}"
                )

            painter.setPen(QColor("#6B7280"))
            painter.drawText(int(body.left()), int(self.rect().bottom() - 6), "0")
            painter.drawText(
                int(body.right() - 8), int(self.rect().bottom() - 6), "1"
            )
        finally:
            painter.end()
