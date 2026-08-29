"""Training page of the trainer."""

import sys

from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.widgets.log_redirector import LogRedirector
from ui.workers.training_worker import TrainingWorker


class TrainingPage(QWidget):
    """Starts PatchCore training in a background thread and shows its log."""

    def __init__(self):
        super().__init__()

        self.parent_window = None
        self.worker = None

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()

        self.lbl_title = QLabel("PatchCore Training")
        root.addWidget(self.lbl_title)

        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        root.addWidget(self.btn_train)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)  # Busy indicator while training runs.
        self.progress.hide()
        root.addWidget(self.progress)

        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        root.addWidget(self.btn_clear_log)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        root.addWidget(self.txt_log)

        # Training output is written via print(), so stdout and stderr are
        # redirected into the log view while a run is active.
        self.redirector = LogRedirector()
        self.redirector.text_written.connect(self.append_log)

        self.setLayout(root)

    # ------------------------------------------------------------------
    # Log view
    # ------------------------------------------------------------------

    def append_log(self, text):
        self.txt_log.insertPlainText(text)

        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.txt_log.clear()

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(self):
        if not self.parent_window:
            QMessageBox.warning(self, "Warning", "Parent window not found.")
            return

        recipe = self.parent_window.current_recipe_data
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        self.txt_log.clear()
        self.append_log("\n===== TRAINING START =====\n\n")

        self.btn_train.setEnabled(False)
        self.progress.show()

        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.worker = TrainingWorker(recipe)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.error_signal.connect(self.on_training_error)
        self.worker.start()

    def on_training_finished(self, model_dir):
        self.append_log("\n===== TRAINING COMPLETE =====\n")
        self.finish_run()

        QMessageBox.information(
            self, "Success", f"Training Complete\n\n{model_dir}"
        )

    def on_training_error(self, error_message):
        self.append_log(f"\nERROR : {error_message}\n")
        self.finish_run()

        QMessageBox.critical(self, "Training Error", error_message)

    def finish_run(self):
        """Restore the console and re-enable the training button."""
        self.progress.hide()
        self.btn_train.setEnabled(True)

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
