"""Model recommendation page of the trainer (S4, placed after ROI/crops)."""

from functools import partial

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from services.roi_service import ROIService
from services.training_assistant import (
    REUSE_SIMILARITY,
    REVIEW_SIMILARITY,
    recommend_entry,
    set_chosen_path,
)
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    SPACE_XS,
    caption_label,
    card_frame,
    make_button,
    section_label,
    title_label,
)
from ui.workers.augment_worker import AugmentWorker

MAX_CARDS = 5

STATUS_BADGE_STYLES = {
    "Recommended": ("#D1FAE5", "#047857"),
    "Review Required": ("#FEF3C7", "#B45309"),
    "Low Similarity": ("#F3F4F6", "#6B7280"),
}


class RecommendWorker(QThread):
    """Runs recommend_entry (YOLO inference + visual match) off the GUI thread."""

    finished_signal = Signal(object)  # recommendation result dict
    error_signal = Signal(str)

    def __init__(self, recipe, candidates, sample_image_path, detect_roi):
        super().__init__()
        self.recipe = recipe
        self.candidates = candidates
        self.sample_image_path = sample_image_path
        self.detect_roi = detect_roi

    def run(self):
        try:
            result = recommend_entry(
                self.recipe,
                self.candidates,
                sample_image_path=self.sample_image_path,
                detect_roi=self.detect_roi,
            )
            self.finished_signal.emit(result)
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))


class RecommendationPage(QWidget):
    """Recommends reusing / extending an existing model or training new.

    Reached only after the ROI step, so the visual match always runs on
    real crops: the raw sample is re-cropped with EACH candidate recipe's
    own ROI config before matching. Every path still requires the user's
    explicit confirmation.
    """

    def __init__(self):
        super().__init__()

        self.recipe_service = RecipeService()
        self.parent_window = None

        self._roi_service = None
        self._result = None
        self._cards = []
        self.augment_worker = None
        self._recommend_worker = None
        self._recommend_cache_key = None
        self._recommend_cache_result = None
        self._use_buttons = []

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        layout.setSpacing(SPACE_M)

        layout.addWidget(title_label("Model Recommendation"))
        layout.addWidget(
            caption_label(
                "Computed from your dataset crops and existing recipes "
                "(metadata + visual crop match). Confirm a path to continue."
            )
        )

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setSpacing(SPACE_S)
        layout.addLayout(self.cards_layout)

        layout.addSpacing(SPACE_S)

        layout.addWidget(section_label("Selected Path"))
        self.selection_label = QLabel("None")
        self.selection_label.setObjectName("contextChip")
        selection_row = QHBoxLayout()
        selection_row.addWidget(self.selection_label)
        selection_row.addStretch()
        layout.addLayout(selection_row)

        self.btn_train_new = make_button("Train New Model")
        self.btn_train_new.clicked.connect(self.train_new_model)
        layout.addWidget(self.btn_train_new)

        layout.addStretch()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Recommendation computation
    # ------------------------------------------------------------------

    def _load_candidates(self, current_recipe):
        candidates = []
        for name in self.recipe_service.get_recipe_list():
            if name == current_recipe.get("recipe_name"):
                continue
            loaded = self.recipe_service.load_recipe(name)
            if loaded:
                candidates.append(loaded)
        return candidates

    def _sample_image_path(self):
        dataset_page = getattr(self.parent_window, "dataset_page", None)
        if dataset_page is not None:
            return getattr(dataset_page, "first_image_path", None)
        return None

    def _ensure_roi_service(self):
        if self._roi_service is not None:
            return self._roi_service
        try:
            self._roi_service = ROIService()
        except Exception as error:
            print(f"[RECOMMEND] ROI service unavailable: {error}")
        return self._roi_service

    def refresh_recommendations(self):
        """Recompute the recommendation cards from the current context."""
        self._clear_cards()

        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if not recipe:
            self._result = None
            self._add_info_card("Select or create a recipe first.")
            return

        candidates = self._load_candidates(recipe)
        if not candidates:
            self._result = None
            self._add_info_card(
                "No existing trained recipes — use 'Train New Model'."
            )
            return

        sample_path = self._sample_image_path()
        roi_service = self._ensure_roi_service()
        detect_roi = roi_service.detect_roi if roi_service is not None else None

        cache_key = (
            sample_path,
            recipe.get("recipe_name"),
            tuple(candidate.get("recipe_name") for candidate in candidates),
        )
        if self._recommend_cache_key == cache_key and (
            self._recommend_cache_result is not None
        ):
            self._apply_recommendation(self._recommend_cache_result)
            return

        if self.is_recommend_running():
            return

        self.selection_label.setText("Computing recommendation...")
        self._pending_cache_key = cache_key
        self._set_use_buttons_enabled(False)
        self.btn_train_new.setEnabled(False)

        self._recommend_worker = RecommendWorker(
            recipe, candidates, sample_path, detect_roi
        )
        self._recommend_worker.finished_signal.connect(
            self.on_recommendation_finished
        )
        self._recommend_worker.error_signal.connect(self.on_recommendation_error)
        self._recommend_worker.finished.connect(self._cleanup_recommend_worker)
        self._recommend_worker.start()

    def is_recommend_running(self):
        """Return True while the recommendation worker thread is active."""
        return self._recommend_worker is not None and self._recommend_worker.isRunning()

    def on_recommendation_finished(self, result):
        self._recommend_cache_key = self._pending_cache_key
        self._recommend_cache_result = result
        self._apply_recommendation(result)

    def on_recommendation_error(self, error_message):
        print(f"[RECOMMEND] failed: {error_message}")
        self._result = None
        self._clear_busy_state()
        self._add_info_card("Recommendation failed — see the console log.")
        QMessageBox.critical(self, "Recommendation Error", error_message)

    def _cleanup_recommend_worker(self):
        if self._recommend_worker is not None:
            self._recommend_worker.deleteLater()
        self._recommend_worker = None
        self._pending_cache_key = None

    def _clear_busy_state(self):
        self._set_use_buttons_enabled(True)
        self.btn_train_new.setEnabled(True)

    def _set_use_buttons_enabled(self, enabled):
        for button in self._use_buttons:
            button.setEnabled(enabled)

    def _apply_recommendation(self, result):
        self._clear_cards()
        if result is None:
            self._clear_busy_state()
            self._add_info_card("Recommendation failed — see the console log.")
            return

        self._result = result

        shown = 0
        for candidate in result["candidates"]:
            self._add_candidate_card(candidate)
            shown += 1
            if shown >= MAX_CARDS:
                break

        decision_text = f"Recommendation : {result['decision']}"
        if result["chosen"]:
            decision_text += f" ({result['chosen']})"
        self.selection_label.setText(decision_text)
        self._clear_busy_state()

    def _clear_cards(self):
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards = []
        self._use_buttons = []

    def _add_info_card(self, message):
        card = card_frame()
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        label = caption_label(message)
        card_layout.addWidget(label)
        card_layout.addStretch()
        card.setLayout(card_layout)
        self.cards_layout.addWidget(card)
        self._cards.append(card)

    def _status_badge(self, text):
        background, foreground = STATUS_BADGE_STYLES.get(
            text, ("#F3F4F6", "#6B7280")
        )
        badge = QLabel(text)
        badge.setStyleSheet(
            f"background:{background};color:{foreground};border:none;"
            "border-radius:6px;padding:3px 10px;font-weight:600;font-size:11px;"
        )
        return badge

    def _add_candidate_card(self, candidate):
        """Build one recommendation card."""
        similarity = candidate["similarity"]
        if similarity >= REUSE_SIMILARITY:
            status = "Recommended"
            accent = "#047857"
        elif similarity >= REVIEW_SIMILARITY:
            status = "Review Required"
            accent = "#B45309"
        else:
            status = "Low Similarity"
            accent = "#6B7280"

        card = card_frame()
        card_layout = QHBoxLayout()
        card_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        card_layout.setSpacing(SPACE_L)

        preview = QLabel("Crop Match")
        preview.setAlignment(Qt.AlignCenter)
        preview.setObjectName("preview")
        preview.setMinimumSize(150, 84)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(SPACE_XS)

        name_label = QLabel(candidate["name"])
        name_label.setStyleSheet(
            "border:none;font-size:16px;font-weight:700;color:#111827;"
            "background:transparent;"
        )

        detail_bits = [f"Similarity : {similarity * 100:.0f}%"]
        visual = candidate.get("visual_similarity")
        if visual is None:
            detail_bits.append("metadata-only (no crops to match)")
        else:
            detail_bits.append(f"visual {visual * 100:.0f}%")
        detail_label = QLabel("  |  ".join(detail_bits))
        detail_label.setStyleSheet(
            f"border:none;font-size:12px;font-weight:600;color:{accent};"
            "background:transparent;"
        )

        badge_row = QHBoxLayout()
        badge_row.setSpacing(SPACE_S)
        badge_row.addWidget(self._status_badge(status))
        badge_row.addStretch()

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACE_S)
        btn_use = make_button("Reuse Model", "secondary")
        btn_use.clicked.connect(
            lambda checked=False, name=candidate["name"]: self.use_model(name)
        )
        btn_augment = make_button("Add to Memory", "secondary")
        btn_augment.clicked.connect(
            lambda checked=False, name=candidate["name"]: self.add_to_memory(name)
        )
        button_row.addWidget(btn_use)
        button_row.addWidget(btn_augment)
        button_row.addStretch()
        self._use_buttons.append(btn_use)

        info_layout.addWidget(name_label)
        info_layout.addWidget(detail_label)
        info_layout.addLayout(badge_row)
        info_layout.addLayout(button_row)

        card_layout.addWidget(preview)
        card_layout.addLayout(info_layout, 1)
        card.setLayout(card_layout)

        self.cards_layout.addWidget(card)
        self._cards.append(card)

    # ------------------------------------------------------------------
    # Path selection (user confirms every path)
    # ------------------------------------------------------------------

    def use_model(self, model_name):
        """Reuse path: continue with the chosen recipe for calibration."""
        target = self.recipe_service.load_recipe(model_name)
        if not target:
            QMessageBox.warning(self, "Warning", f"Recipe {model_name} not found.")
            return

        set_chosen_path(target, "reuse", model_name)
        self.recipe_service.save_recipe(model_name, target)

        self.selection_label.setText(f"Reuse : {model_name}")

        if self.parent_window:
            self.parent_window.update_context(target)
            self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Reuse Model",
            f"Recipe {model_name} selected. Continue to Training for "
            "threshold calibration (no retrain needed).",
        )

    def add_to_memory(self, model_name):
        """Augment path: append new crops to the chosen recipe, then retrain."""
        target = self.recipe_service.load_recipe(model_name)
        if not target:
            QMessageBox.warning(self, "Warning", f"Recipe {model_name} not found.")
            return

        if self.is_augment_running():
            return

        dataset_page = getattr(self.parent_window, "dataset_page", None)
        paths = []
        if dataset_page is not None and dataset_page.dataset_path:
            paths = dataset_page.image_paths(dataset_page.dataset_path)
        if not paths:
            QMessageBox.warning(
                self, "Warning", "Select a dataset first (Dataset Import)."
            )
            return

        if not target.get("prepared_dataset_path"):
            QMessageBox.warning(
                self,
                "Warning",
                f"Recipe {model_name} has no prepared dataset to append to.",
            )
            return

        self.selection_label.setText(f"Add to memory : {model_name} (working...)")
        self.augment_worker = AugmentWorker(target, paths)
        self.augment_worker.finished_signal.connect(
            partial(self._finish_augment, target)
        )
        self.augment_worker.error_signal.connect(self.on_augment_error)
        self.augment_worker.finished.connect(self._cleanup_augment_worker)
        self.augment_worker.start()

    def _cleanup_augment_worker(self):
        if self.augment_worker is not None:
            self.augment_worker.deleteLater()
            self.augment_worker = None

    def _finish_augment(self, target, count):
        set_chosen_path(target, "augment", target.get("recipe_name"))
        self.recipe_service.save_recipe(target["recipe_name"], target)

        self.selection_label.setText(
            f"Add to memory : {target['recipe_name']} (+{count} crops)"
        )

        if self.parent_window:
            self.parent_window.update_context(target)
            self.parent_window.refresh_step_bar()

        QMessageBox.information(
            self,
            "Add to Memory",
            f"Appended {count} crops to {target['recipe_name']}. "
            "Continue to Training → Start Training to retrain, "
            "then calibrate the threshold.",
        )

    def on_augment_error(self, error_message):
        self.selection_label.setText("Add to memory failed.")
        QMessageBox.critical(self, "Add-to-memory Error", error_message)

    def train_new_model(self):
        """Train-new path: continue with the current recipe."""
        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        set_chosen_path(recipe, "train_new", None)
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        self.selection_label.setText("Train New Model")

        if self.parent_window and hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

    def is_augment_running(self):
        """Return True while the augment worker thread is active."""
        return self.augment_worker is not None and self.augment_worker.isRunning()
