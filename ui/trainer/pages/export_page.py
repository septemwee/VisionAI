"""Finalize page of the trainer (S7): summary, validation and handoff."""

from PySide6.QtWidgets import (
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from services.training_assistant import apply_finalize, finalize_summary
from ui.theme import (
    SPACE_L,
    SPACE_M,
    caption_label,
    card_frame,
    make_button,
    title_label,
)


class ExportPage(QWidget):
    """Shows the final recipe summary and marks the recipe as validated."""

    def __init__(self):
        super().__init__()

        self.parent_window = None
        self.recipe_service = RecipeService()

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        layout.setSpacing(SPACE_M)

        layout.addWidget(title_label("Finalize & Export"))
        layout.addWidget(
            caption_label(
                "Review the trained recipe below, then mark it validated "
                "for production use. Remember to capture the top-mark "
                "template in the inspection app."
            )
        )

        summary_card = card_frame()
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)

        self.summary_view = QPlainTextEdit()
        self.summary_view.setObjectName("summaryView")
        self.summary_view.setReadOnly(True)
        self.summary_view.setMinimumHeight(200)
        self.summary_view.setPlaceholderText(
            "Select a recipe to see its finalize summary."
        )
        summary_layout.addWidget(self.summary_view)

        layout.addWidget(summary_card, 1)

        self.btn_finalize = make_button("Finalize Recipe")
        self.btn_finalize.clicked.connect(self.finalize)
        self.btn_finalize.setEnabled(False)
        layout.addWidget(self.btn_finalize)

        self.setLayout(layout)

    # ------------------------------------------------------------------

    def refresh_summary(self):
        """Rebuild the summary from the current recipe (S7 entry hook)."""
        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if not recipe:
            self.summary_view.setPlainText("No recipe selected.")
            self.btn_finalize.setEnabled(False)
            return

        summary = finalize_summary(recipe)
        self.summary_view.setPlainText(self._format_summary(summary))
        self.btn_finalize.setEnabled(True)

    @staticmethod
    def _format_summary(summary):
        lines = [
            f"Recipe : {summary['recipe_name'] or '-'}",
            f"Package : {summary['package_family'] or '-'} / "
            f"{summary['package_type'] or '-'} / "
            f"pins {summary['pin_count'] if summary['pin_count'] else '-'}",
            "",
            f"Model : {'trained' if summary['model_trained'] else 'NOT trained'}",
            f"Model path : {summary['model_path'] or '-'}",
            f"Crops : saved {summary['crops_saved'] if summary['crops_saved'] is not None else '-'}"
            f", flagged {summary['crops_rejected'] if summary['crops_rejected'] is not None else '-'}",
            "",
            f"Threshold : {summary['anomaly_threshold']}",
        ]

        calibration = summary["calibration"]
        if calibration:
            stats = calibration.get("good_summary") or {}
            lines.append(
                f"Calibration : P99 {calibration.get('p99', 0):.3f} x "
                f"margin {calibration.get('margin', 0):.2f} -> "
                f"{calibration.get('proposed', 0):.3f} "
                f"(expected false alarms {calibration.get('expected_false_alarms', 0)})"
            )
            lines.append(
                f"Good-set summary : min {stats.get('min', 0):.3f} | "
                f"median {stats.get('median', 0):.3f} | "
                f"P95 {stats.get('p95', 0):.3f} | "
                f"max {stats.get('max', 0):.3f} "
                f"({stats.get('count', 0)} images)"
            )
            lines.append(f"Calibrated at : {calibration.get('evaluated_at', '-')}")
        else:
            lines.append("Calibration : not run (default threshold in use)")

        analysis = summary["dataset_analysis"]
        if analysis:
            geometry = analysis.get("geometry") or {}
            size_stats = geometry.get("size_stats") or {}
            if size_stats:
                lines.append(
                    f"Dataset analysis : {geometry.get('images_with_detection', 0)}/"
                    f"{geometry.get('images_analyzed', 0)} images with package, "
                    f"median size {size_stats.get('median_width', 0):.0f}x"
                    f"{size_stats.get('median_height', 0):.0f}px"
                )
            else:
                lines.append("Dataset analysis : not analyzed")
        else:
            lines.append("Dataset analysis : not analyzed")

        lines += ["", f"Version : {summary['version'] or '-'}"]
        if summary["last_trained"]:
            lines.append(f"Last trained : {summary['last_trained']}")

        if summary["warnings"]:
            lines.append("")
            lines.append("Warnings:")
            for warning in summary["warnings"]:
                lines.append(f" - {warning}")

        return "\n".join(lines)

    def finalize(self):
        """Mark the recipe validated and bump its version."""
        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        version = apply_finalize(recipe)
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        if self.parent_window and hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Finalized",
            f"Recipe {recipe['recipe_name']} marked validated "
            f"(version {version}).\n\nReminder: capture the top-mark "
            "template in the inspection app before production use.",
        )

        self.refresh_summary()
