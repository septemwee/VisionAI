"""Professional workflow stepper for the trainer wizard."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from services.training_assistant import STATUS_ACTION_NEEDED, STATUS_BLOCKED, STATUS_OK, STEP_IDS
from ui.theme import make_button

STEP_SHORT_LABELS = {
    "S1": "Recipe", "S2": "Dataset", "S3": "ROI", "S4": "Strategy",
    "S5": "Train", "S6": "Validate", "S7": "Finalize",
}

STEPPER_STYLESHEET = """
QWidget#stepRail { background: transparent; }
QLabel#workflowLabel { color: #64748B; font-size: 11px; font-weight: 600; }
QLabel#stepBadge {
    min-width: 66px; padding: 7px 7px; color: #64748B;
    background: transparent; border: 1px solid transparent;
    border-radius: 9px; font-size: 11px; font-weight: 600;
}
QLabel#stepBadge:hover { background: #F8FAFC; color: #334155; }
QLabel#stepBadge[current="true"] {
    color: #1D4ED8; background: #EFF6FF; border-color: #BFDBFE;
}
QLabel#stepBadge[completed="true"] { color: #047857; }
QLabel#stepBadge[blocked="true"] { color: #B91C1C; background: #FEF2F2; }
"""


class _Badge(QLabel):
    clicked = Signal(str)

    def __init__(self, step_id):
        super().__init__(step_id)
        self._step_id = step_id
        self.setObjectName("stepBadge")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit(self._step_id)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self._step_id)
            return
        super().keyPressEvent(event)


class StepStatusBar(QWidget):
    """S1-S7 progress with a clear continuation action."""

    step_clicked = Signal(str)
    next_step_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self._statuses = {}
        self._next_step_id = None
        self.setObjectName("stepRail")
        self.setStyleSheet(STEPPER_STYLESHEET)

        root = QVBoxLayout(self)
        root.setSpacing(7)
        root.setContentsMargins(0, 0, 0, 0)
        workflow_label = QLabel("MODEL SETUP WORKFLOW")
        workflow_label.setObjectName("workflowLabel")
        root.addWidget(workflow_label)

        row = QHBoxLayout()
        row.setSpacing(2)
        row.setContentsMargins(0, 0, 0, 0)
        self.badges = {}
        for step_id in STEP_IDS:
            badge = _Badge(step_id)
            badge.clicked.connect(self.step_clicked)
            self.badges[step_id] = badge
            row.addWidget(badge)
        row.addStretch()
        self.next_button = make_button("Continue setup", "secondary")
        self.next_button.clicked.connect(self._emit_next)
        row.addWidget(self.next_button)
        root.addLayout(row)

    @staticmethod
    def _refresh(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _emit_next(self):
        if self._next_step_id:
            self.next_step_requested.emit(self._next_step_id)

    def update_statuses(self, statuses):
        self._statuses = dict(statuses) if statuses else {}
        for step_id in STEP_IDS:
            entry = self._statuses.get(step_id) or {}
            status = entry.get("status", STATUS_ACTION_NEEDED)
            badge = self.badges[step_id]
            badge.setText(f"{step_id[1:]}  {STEP_SHORT_LABELS[step_id]}")
            badge.setToolTip(str(entry.get("detail", "")))
            badge.setProperty("completed", status == STATUS_OK)
            badge.setProperty("blocked", status == STATUS_BLOCKED)
            self._refresh(badge)

        self._next_step_id = next((
            step_id for step_id in STEP_IDS
            if (self._statuses.get(step_id) or {}).get("status") != STATUS_OK
        ), None)
        if self._next_step_id:
            self.next_button.setText(f"Required next: {STEP_SHORT_LABELS[self._next_step_id]}")
            self.next_button.setEnabled(True)
        else:
            self.next_button.setText("Setup complete")
            self.next_button.setEnabled(False)

    def set_current_step(self, step_id):
        for candidate, badge in self.badges.items():
            current = candidate == step_id
            badge.setProperty("current", current)
            state = "current" if current else "completed" if badge.property("completed") else "pending"
            badge.setAccessibleName(f"Step {candidate[1:]}, {STEP_SHORT_LABELS[candidate]}, {state}")
            self._refresh(badge)
