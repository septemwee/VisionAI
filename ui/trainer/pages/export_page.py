"""Export page of the trainer (placeholder)."""


from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class ExportPage(QWidget):
    """Final wizard step. Exporting models is not implemented yet."""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Export")
        self.model_label = QLabel("PatchCore")
        self.export_button = QPushButton("Export")

        layout.addWidget(title)
        layout.addWidget(self.model_label)
        layout.addWidget(self.export_button)

        self.setLayout(layout)
