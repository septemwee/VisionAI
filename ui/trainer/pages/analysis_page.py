"""Physical analysis page of the trainer.

This page is currently disabled in the wizard flow and only kept as a
placeholder for the upcoming real analysis step.
"""

from PySide6.QtWidgets import QLabel, QPushButton, QFrame, QVBoxLayout, QWidget


class AnalysisPage(QWidget):
    """Shows physical package attributes derived from the dataset."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(
            """
            QWidget{
                background:#FFFFFF;
                font-family:'Poppins';
            }

            QFrame{
                background:white;
                border:1px solid #E5E7EB;
                border-radius:12px;
            }
            """
        )

        layout = QVBoxLayout()

        title = QLabel("Physical Analysis")
        title.setStyleSheet("font-size:24px;font-weight:700;")
        layout.addWidget(title)

        card = QFrame()
        card_layout = QVBoxLayout()

        self.family_label = QLabel("Package Family : SO")
        self.pin_label = QLabel("Pin Count : 14")
        self.aspect_label = QLabel("Aspect Ratio : 2.21")
        self.status_label = QLabel("✅ Analysis Complete")

        card_layout.addWidget(self.family_label)
        card_layout.addWidget(self.pin_label)
        card_layout.addWidget(self.aspect_label)
        card_layout.addWidget(self.status_label)
        card.setLayout(card_layout)
        layout.addWidget(card)

        self.btn_analyze = QPushButton("Analyze Dataset")
        layout.addWidget(self.btn_analyze)

        self.setLayout(layout)
