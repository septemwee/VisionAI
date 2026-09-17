"""Training page of the trainer: PatchCore training + threshold calibration."""

import sys

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from services.training_assistant import (
    STATUS_OK,
    recommend_threshold,
    set_step_status,
    pre_flight,
    utc_now_iso,
)
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    caption_label,
    card_frame,
    make_button,
    section_label,
    title_label,
)
from ui.trainer.histogram_widget import HistogramWidget
from ui.widgets.log_redirector import LogRedirector
from ui.workers.evaluation_worker import EvaluationWorker
from ui.workers.training_worker import TrainingWorker
from utils.paths import resolve_recipe_path


class TrainingPage(QWidget):
    """Trains PatchCore in a background thread and calibrates the threshold."""

    def __init__(self):
        super().__init__()

        self.parent_window = None
        self.worker = None
        self.evaluation_worker = None

        self._scores = None
        self._proposal = None

        self.recipe_service = RecipeService()

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        root.setSpacing(SPACE_M)

        self.lbl_title = title_label("PatchCore Training")
        root.addWidget(self.lbl_title)

        training_row = QHBoxLayout()
        training_row.setSpacing(SPACE_S)

        self.btn_train = make_button("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        training_row.addWidget(self.btn_train)

        self.btn_clear_log = make_button("Clear Log", "ghost")
        self.btn_clear_log.clicked.connect(self.clear_log)
        training_row.addWidget(self.btn_clear_log)
        training_row.addStretch()

        root.addLayout(training_row)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)  # Busy indicator while training runs.
        self.progress.hide()
        root.addWidget(self.progress)

        self.txt_log = QPlainTextEdit()
        self.txt_log.setObjectName("logView")
        self.txt_log.setReadOnly(True)
        root.addWidget(self.txt_log, 1)

        root.addWidget(self._build_calibration_card(), 1)

        # Training output is written via print(), so stdout and stderr are
        # redirected into the log view while a run is active.
        self.redirector = LogRedirector()
        self.redirector.text_written.connect(self.append_log)

        self.setLayout(root)
        self.refresh_calibration_panel()

    def _build_calibration_card(self):
        """Build the post-training threshold calibration panel (S6)."""
        card = card_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        layout.setSpacing(SPACE_S)

        layout.addWidget(section_label("Threshold Calibration"))

        row = QHBoxLayout()
        row.setSpacing(SPACE_S)

        self.btn_calibrate = make_button("Calibrate Threshold", "secondary")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        row.addWidget(self.btn_calibrate)

        margin_caption = caption_label("Margin")
        row.addWidget(margin_caption)

        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(1.0, 3.0)
        self.margin_spin.setSingleStep(0.05)
        self.margin_spin.setValue(1.2)
        self.margin_spin.setFixedWidth(90)
        self.margin_spin.valueChanged.connect(self._recompute_proposal)
        row.addWidget(self.margin_spin)
        row.addStretch()
        layout.addLayout(row)

        self.histogram = HistogramWidget()
        layout.addWidget(self.histogram)

        self.lbl_stats = QLabel("Good-set scores — run calibration first.")
        layout.addWidget(self.lbl_stats)

        self.lbl_proposed = QLabel("Proposed threshold : -")
        self.lbl_proposed.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        layout.addWidget(self.lbl_proposed)

        self.btn_apply_calibration = make_button("Apply to Recipe")
        self.btn_apply_calibration.clicked.connect(self.apply_calibration)
        self.btn_apply_calibration.setEnabled(False)
        layout.addWidget(self.btn_apply_calibration)

        return card

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
    # Training control (S5)
    # ------------------------------------------------------------------

    def start_training(self):
        if not self.parent_window:
            QMessageBox.warning(self, "Warning", "Parent window not found.")
            return

        if self.is_training_running() or self.is_calibration_running():
            return

        recipe = self.parent_window.current_recipe_data
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        allowed, reasons = pre_flight(recipe)
        if not allowed:
            QMessageBox.warning(
                self,
                "Pre-flight Check",
                "Training is blocked:\n\n- " + "\n- ".join(reasons),
            )
            return

        self.txt_log.clear()
        self.append_log("\n===== TRAINING START =====\n\n")

        self.btn_train.setEnabled(False)
        self.progress.show()

        self._previous_stdout = sys.stdout
        self._previous_stderr = sys.stderr
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.worker = TrainingWorker(recipe)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.error_signal.connect(self.on_training_error)
        self.worker.start()

    def on_training_finished(self, model_dir):
        self.append_log("\n===== TRAINING COMPLETE =====\n")
        self.finish_run()

        self.refresh_calibration_panel()
        if self.parent_window and hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

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
        self.restore_output()

    def is_training_running(self):
        """Return True while the training worker thread is active."""
        return self.worker is not None and self.worker.isRunning()

    def restore_output(self):
        """Restore the console that was active before this run."""
        if sys.stdout is not self.redirector:
            return

        sys.stdout = self._previous_stdout or sys.__stdout__
        sys.stderr = self._previous_stderr or sys.__stderr__

    # ------------------------------------------------------------------
    # Threshold calibration (S6)
    # ------------------------------------------------------------------

    def refresh_calibration_panel(self):
        """Enable the calibration panel only for trained, idle recipes."""
        recipe = None
        if self.parent_window is not None:
            recipe = self.parent_window.current_recipe_data

        trained = bool(recipe and (recipe.get("model") or {}).get("trained"))
        busy = self.is_training_running() or self.is_calibration_running()

        self.btn_calibrate.setEnabled(trained and not busy)
        if not trained:
            self.btn_calibrate.setToolTip("Train the model first.")
        else:
            self.btn_calibrate.setToolTip("")

    def start_calibration(self):
        if not self.parent_window:
            QMessageBox.warning(self, "Warning", "Parent window not found.")
            return

        if self.is_training_running() or self.is_calibration_running():
            return

        recipe = self.parent_window.current_recipe_data
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        if not (recipe.get("model") or {}).get("trained"):
            QMessageBox.warning(
                self, "Warning", "Train the model before calibrating."
            )
            return

        prepared = resolve_recipe_path(recipe.get("prepared_dataset_path"))
        good_dir = prepared / "train" / "good" if prepared is not None else None
        if good_dir is None or not good_dir.exists():
            QMessageBox.warning(
                self,
                "Warning",
                "The prepared training crops were not found.\n"
                "Generate the training dataset first.",
            )
            return

        self.append_log("\n===== CALIBRATION START =====\n\n")
        self.append_log(f"Scoring good-image folder: {good_dir}\n")

        self.btn_calibrate.setEnabled(False)
        self._proposal = None
        self._scores = None
        self.btn_apply_calibration.setEnabled(False)

        self._previous_stdout = sys.stdout
        self._previous_stderr = sys.stderr
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.evaluation_worker = EvaluationWorker(recipe, good_dir)
        self.evaluation_worker.scores_ready.connect(self.on_scores_ready)
        self.evaluation_worker.error_signal.connect(self.on_calibration_error)
        self.evaluation_worker.start()

    def on_scores_ready(self, scores):
        self.append_log("\n===== CALIBRATION COMPLETE =====\n")
        self.restore_output()

        self._scores = scores
        self.refresh_calibration_panel()
        self._recompute_proposal()

    def on_calibration_error(self, error_message):
        self.append_log(f"\nERROR : {error_message}\n")
        self.restore_output()
        self.refresh_calibration_panel()

        QMessageBox.critical(self, "Calibration Error", error_message)

    def _recompute_proposal(self):
        """Recompute the threshold proposal from stored scores (pure math)."""
        if not self._scores:
            return

        try:
            proposed, trace = recommend_threshold(
                [score for _, score in self._scores],
                self.margin_spin.value(),
            )
        except ValueError:
            return

        self._proposal = (proposed, trace)

        stats = trace["good_summary"]
        stats_text = (
            f"Good-set scores — min {stats['min']:.3f} | "
            f"median {stats['median']:.3f} | P95 {stats['p95']:.3f} | "
            f"P99 {stats['p99']:.3f} | max {stats['max']:.3f} "
            f"({stats['count']} images)"
        )
        if trace["warning"]:
            stats_text += "   WARNING: good-image scores reach the 1.0 ceiling."
        self.lbl_stats.setText(stats_text)

        self.lbl_proposed.setText(
            f"Proposed threshold : {proposed:.3f}   |   "
            f"Expected false alarms on good set : "
            f"{trace['expected_false_alarms']}"
        )
        self.histogram.set_data([score for _, score in self._scores], proposed)

        # A threshold of exactly 1.0 can never be exceeded — live inspection
        # classifies it as an invalid calibrated value and fails every part.
        # Block the apply path here so a saturated proposal cannot be saved.
        saturated = proposed >= 1.0 or trace["warning"]
        self.btn_apply_calibration.setEnabled(not saturated)

    def apply_calibration(self):
        """Write the proposed threshold + calibration trace to the recipe."""
        if not self._proposal:
            return

        if not self.parent_window:
            QMessageBox.warning(self, "Warning", "Parent window not found.")
            return

        recipe = self.parent_window.current_recipe_data
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        proposed, trace = self._proposal
        if proposed >= 1.0 or trace.get("warning"):
            QMessageBox.warning(
                self,
                "Calibration Blocked",
                f"Proposed threshold {proposed:.3f} is unusable: good-image "
                "scores reach the 1.0 scale ceiling. Raise the margin, "
                "retrain with more/better good crops, or fix the capture "
                "conditions, then recalibrate.",
            )
            return

        trace = dict(trace)
        trace["evaluated_at"] = utc_now_iso()

        recipe["anomaly_threshold"] = proposed
        recipe["calibration"] = trace
        set_step_status(recipe, "S6", STATUS_OK)
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        if hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Calibration Applied",
            f"Threshold {proposed:.3f} saved to recipe "
            f"{recipe['recipe_name']}.",
        )

    def is_calibration_running(self):
        """Return True while the evaluation worker thread is active."""
        return (
            self.evaluation_worker is not None
            and self.evaluation_worker.isRunning()
        )

    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_calibration_panel()
