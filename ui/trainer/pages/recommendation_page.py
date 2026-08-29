"""Model recommendation page of the trainer.

The recommendations shown here are placeholders; they will be replaced with
real similarity matching between the new package and existing recipes.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PAGE_STYLE = """
QWidget{
    background:#FFFFFF;
    font-family:'Poppins';
}

QLabel{
    color:#374151;
    border:none;
}

QFrame{
    background:white;
    border:1px solid #E5E7EB;
    border-radius:12px;
}

QPushButton{
    background:#2563EB;
    color:white;
    border:none;
    border-radius:8px;
    padding:10px;
    font-weight:600;
}

QPushButton:hover{
    background:#1D4ED8;
}

QPushButton:pressed{
    background:#1E40AF;
}
"""

# Placeholder recommendation entries: (name, similarity, status, status color).
RECOMMENDED_MODELS = [
    ("SO14_V1", "96%", "✅ Recommended", "#10B981"),
    ("SO14_V2", "91%", "🔵 Alternative", "#2563EB"),
    ("SO16_V1", "74%", "⚠ Review Required", "#F59E0B"),
]


class RecommendationPage(QWidget):
    """Suggests reusing an existing model or training a new one."""

    def __init__(self):
        super().__init__()

        self.selected_model = None
        self.train_new_model_flag = False

        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(PAGE_STYLE)

        layout = QVBoxLayout()

        title = QLabel("Model Recommendation")
        title.setStyleSheet(
            "border:none;font-size:24px;font-weight:700;color:#111827;"
        )
        layout.addWidget(title)
        layout.addSpacing(10)

        for model_name, similarity, status, status_color in RECOMMENDED_MODELS:
            layout.addWidget(
                self.create_model_card(model_name, similarity, status, status_color)
            )

        layout.addSpacing(10)

        selected_title = QLabel("Selected Model")
        selected_title.setStyleSheet(
            "border:none;font-size:16px;font-weight:700;color:#111827;"
        )
        layout.addWidget(selected_title)

        self.selection_label = QLabel("None")
        self.selection_label.setStyleSheet(
            "border:none;padding:10px;border-radius:8px;"
        )
        layout.addWidget(self.selection_label)

        self.btn_train_new = QPushButton("Train New Model")
        self.btn_train_new.clicked.connect(self.train_new_model)
        layout.addWidget(self.btn_train_new)

        layout.addStretch()
        self.setLayout(layout)

    def create_model_card(self, model_name, similarity, status, status_color):
        """Build one recommendation card."""
        card = QFrame()
        card_layout = QHBoxLayout()

        preview = QLabel("Package Preview")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumSize(220, 140)
        preview.setStyleSheet(
            """
            background:#F9FAFB;
            border:1px dashed #CBD5E1;
            border-radius:10px;
            color:#6B7280;
            font-weight:600;
            align
            """
        )

        info_layout = QVBoxLayout()

        name_label = QLabel(model_name)
        name_label.setStyleSheet(
            "border:none;font-size:20px;font-weight:700;color:#111827;"
        )

        similarity_label = QLabel(f"Similarity : {similarity}")
        similarity_label.setStyleSheet(
            f"font-size:16px;font-weight:600;border:none;color:{status_color};"
        )

        status_label = QLabel(status)
        status_label.setStyleSheet(
            f"color:{status_color};border:none;font-weight:600;"
        )

        description = QLabel("Existing Production Model")
        description.setStyleSheet("border:none;")

        btn_preview = QPushButton("Preview")
        btn_use = QPushButton("Use Model")
        btn_use.clicked.connect(lambda: self.use_model(model_name))

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_preview)
        button_layout.addWidget(btn_use)

        info_layout.addWidget(name_label)
        info_layout.addWidget(similarity_label)
        info_layout.addWidget(status_label)
        info_layout.addWidget(description)
        info_layout.addStretch()
        info_layout.addLayout(button_layout)

        card_layout.addWidget(preview)
        card_layout.addLayout(info_layout)
        card.setLayout(card_layout)

        return card

    def use_model(self, model_name):
        self.selected_model = model_name
        self.selection_label.setText(model_name)

    def train_new_model(self):
        self.selected_model = None
        self.selection_label.setText("Train New Model")
