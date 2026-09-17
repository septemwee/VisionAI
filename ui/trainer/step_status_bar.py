"""Header widget showing S1-S7 step badges and a next-step shortcut."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from services.training_assistant import (
    STATUS_ACTION_NEEDED,
    STATUS_BLOCKED,
    STATUS_OK,
    STATUS_WARNING,
    STEP_IDS,
)
from ui.theme import make_button

STATUS_STYLES = {
    STATUS_OK: ("#10B981", "OK"),
    STATUS_WARNING: ("#F59E0B", "WARN"),
    STATUS_ACTION_NEEDED: ("#F59E0B", "TODO"),
    STATUS_BLOCKED: ("#EF4444", "BLOCK"),
    "skipped": ("#9CA3AF", "SKIP"),
}

STEP_SHORT_LABELS = {
    "S1": "Recipe",
    "S2": "Dataset",
    "S3": "ROI",
    "S4": "Recommend",
    "S5": "Train",
    "S6": "Calibrate",
    "S7": "Finalize",
}

BADGE_STYLESHEET = (
    "QLabel{{background:{color};color:white;padding:4px 10px;"
    "border-radius:10px;font-size:11px;font-weight:600;}}"
)


class _Badge(QLabel):
    """Clickable step badge."""

    clicked = Signal(str)

    def __init__(self, step_id):
        super().__init__(step_id)
        self._step_id = step_id
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit(self._step_id)


class StepStatusBar(QWidget):
    """S1-S7 badges plus a button jumping to the first non-OK step."""

    step_clicked = Signal(str)
    next_step_requested = Signal(str)

    def __init__(self):
        super().__init__()

        self._statuses = {}
        self._next_step_id = None

        row = QHBoxLayout()
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)

        self.badges = {}
        for step_id in STEP_IDS:
            badge = _Badge(step_id)
            badge.clicked.connect(self.step_clicked)
            badge.setToolTip(step_id)
            self.badges[step_id] = badge
            row.addWidget(badge)

        row.addStretch()

        self.next_button = make_button("Next: -", "secondary")
        self.next_button.clicked.connect(self._emit_next)
        row.addWidget(self.next_button)

        self.setLayout(row)

    def _emit_next(self):
        if self._next_step_id:
            self.next_step_requested.emit(self._next_step_id)

    def update_statuses(self, statuses):
        """Apply ``{step_id: {"status", "detail"}}`` from compute_step_status."""
        self._statuses = dict(statuses) if statuses else {}

        for step_id in STEP_IDS:
            entry = self._statuses.get(step_id) or {}
            status = entry.get("status", STATUS_ACTION_NEEDED)
            detail = str(entry.get("detail", ""))

            color, _ = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_ACTION_NEEDED])
            badge = self.badges[step_id]
            badge.setText(f"{step_id} {STEP_SHORT_LABELS[step_id]}")
            badge.setToolTip(f"{step_id} {STEP_SHORT_LABELS[step_id]}: {detail}")
            badge.setStyleSheet(BADGE_STYLESHEET.format(color=color))

        self._next_step_id = None
        for step_id in STEP_IDS:
            entry = self._statuses.get(step_id) or {}
            if entry.get("status") != STATUS_OK:
                self._next_step_id = step_id
                break

        if self._next_step_id:
            self.next_button.setText(
                f"Next: {self._next_step_id} "
                f"{STEP_SHORT_LABELS[self._next_step_id]}"
            )
            self.next_button.setEnabled(True)
        else:
            self.next_button.setText("All Steps OK")
            self.next_button.setEnabled(False)
