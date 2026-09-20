"""Main wizard window for the trainer application."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from services.training_assistant import compute_step_status
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_XL,
    TRAINER_STYLESHEET,
    load_ui_fonts,
    make_button,
    title_label,
)
from ui.trainer.pages.dataset_page import DatasetPage
from ui.trainer.pages.export_page import ExportPage
from ui.trainer.pages.recommendation_page import RecommendationPage
from ui.trainer.pages.recipe_page import RecipePage
from ui.trainer.pages.review_page import ReviewPage
from ui.trainer.pages.roi_page import ROIPage
from ui.trainer.pages.training_page import TrainingPage
from ui.trainer.scroll_page import scrollable_page
from ui.trainer.step_status_bar import StepStatusBar

# Page order matches the guided flow (S1-S7): the model recommendation is
# placed AFTER the ROI crop step so it always runs on real crops.
STEP_TITLES = [
    "Recipe Setup",
    "Dataset",
    "ROI Preparation",
    "Model Strategy",
    "Train & Validate",
    "Model Review",
    "Production Ready",
]
PAGE_STEP_IDS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def navigation_state(index):
    """Return concise wizard-navigation copy for a page index."""
    last = len(STEP_TITLES) - 1
    return {
        "previous_enabled": index > 0,
        "next_enabled": index < last,
        "next_text": (
            f"Continue to {STEP_TITLES[index + 1]}" if index < last
            else "Setup complete"
        ),
    }

# Index of the ROI Verification step, which needs the dataset loaded first.
ROI_STEP_INDEX = 2
RECOMMENDATION_STEP_INDEX = 3
EXPORT_STEP_INDEX = 6


class TrainerWindow(QWidget):
    """Guides the user through recipe creation, training and review."""

    def __init__(self):
        super().__init__()

        self.current_step = 0
        self.current_recipe_data = None

        self.setup_ui()

    def setup_ui(self):
        load_ui_fonts()
        self.setWindowTitle("VisionAI Trainer")
        # Preferred size, clamped to the available screen so the window
        # never opens larger than the desktop (e.g. 1366x768 laptops).
        self.setMinimumSize(960, 600)
        preferred_width, preferred_height = 1400, 850
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            preferred_width = min(preferred_width, int(available.width() * 0.9))
            preferred_height = min(preferred_height, int(available.height() * 0.9))
        self.resize(preferred_width, preferred_height)
        self.setStyleSheet(TRAINER_STYLESHEET)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(SPACE_XL, SPACE_L, SPACE_XL, SPACE_L)
        root_layout.setSpacing(SPACE_M)

        # ----- Product header: purpose + active recipe context -----
        header_row = QHBoxLayout()
        header_row.setSpacing(SPACE_M)

        header_copy = QVBoxLayout()
        header_copy.setSpacing(2)
        product_label = QLabel("VISIONAI  /  MODEL STUDIO")
        product_label.setObjectName("productEyebrow")
        header_copy.addWidget(product_label)
        header_copy.addWidget(title_label("Inspection Model Trainer"))
        subtitle = QLabel(
            "A guided workflow for preparing data, training, and validating a production recipe."
        )
        subtitle.setObjectName("productSubtitle")
        header_copy.addWidget(subtitle)
        header_row.addLayout(header_copy)
        header_row.addStretch()

        self.context_label = QLabel("Package : - | Recipe : -")
        self.context_label.setObjectName("contextChip")
        self.context_label.setFixedHeight(38)
        header_row.addWidget(self.context_label, 0, Qt.AlignRight | Qt.AlignVCenter)

        root_layout.addLayout(header_row)

        # ----- Step status bar (S1-S7 badges + next-step shortcut) -----
        workflow_card = QFrame()
        workflow_card.setObjectName("workflowCard")
        workflow_layout = QVBoxLayout(workflow_card)
        workflow_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        self.step_status_bar = StepStatusBar()
        self.step_status_bar.step_clicked.connect(self.go_to_step)
        self.step_status_bar.next_step_requested.connect(self.go_to_step)
        workflow_layout.addWidget(self.step_status_bar)
        root_layout.addWidget(workflow_card)

        # ----- Wizard pages -----
        self.pages = QStackedWidget()
        self.pages.setObjectName("trainerPages")

        self.recipe_page = RecipePage()
        self.recipe_page.parent_window = self

        self.dataset_page = DatasetPage()
        self.dataset_page.parent_window = self

        self.roi_page = ROIPage()
        self.roi_page.parent_window = self

        self.recommendation_page = RecommendationPage()
        self.recommendation_page.parent_window = self

        self.training_page = TrainingPage()
        self.training_page.parent_window = self

        self.review_page = ReviewPage()
        self.review_page.parent_window = self

        self.export_page = ExportPage()
        self.export_page.parent_window = self

        for page in (
            self.recipe_page,
            self.dataset_page,
            self.roi_page,
            self.recommendation_page,
            self.training_page,
            self.review_page,
            self.export_page,
        ):
            # Scrollable wrapper: pages scroll instead of overlapping
            # when the window is smaller than the page content.
            self.pages.addWidget(scrollable_page(page))

        self.pages.currentChanged.connect(self._on_page_changed)

        root_layout.addWidget(self.pages)

        # ----- Navigation: previous | step indicator | next -----
        footer = QFrame()
        footer.setObjectName("trainerFooter")
        nav_layout = QHBoxLayout(footer)
        nav_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        nav_layout.setSpacing(SPACE_M)

        self.btn_previous = make_button("Previous", "secondary")
        self.btn_previous.clicked.connect(self.previous_page)

        self.btn_next = make_button("Continue")
        self.btn_next.setMinimumWidth(120)
        self.btn_next.clicked.connect(self.next_page)

        self.step_label = QLabel()
        self.step_label.setStyleSheet(
            "font-size:12px;font-weight:700;color:#2563EB;background:transparent;"
        )

        nav_layout.addWidget(self.step_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_previous)
        nav_layout.addWidget(self.btn_next)
        root_layout.addWidget(footer)

        self.setLayout(root_layout)
        self.update_step_label()
        self.refresh_step_bar()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def previous_page(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.pages.setCurrentIndex(self.current_step)
            self.update_step_label()

    def next_page(self):
        if self.current_step >= self.pages.count() - 1:
            return

        next_step = self.current_step + 1

        if next_step == ROI_STEP_INDEX:
            self.roi_page.load_dataset(self.dataset_page.dataset_path)

            image_path = self.dataset_page.first_image_path
            if image_path:
                self.roi_page.load_image(image_path)

        self.current_step = next_step
        self.pages.setCurrentIndex(self.current_step)
        self.update_step_label()

    def go_to_step(self, step_id):
        """Jump to the page owning a step (from the step status bar)."""
        statuses = compute_step_status(self.current_recipe_data)
        entry = statuses.get(str(step_id))
        if not entry:
            return

        self.current_step = entry["page_index"]
        self.pages.setCurrentIndex(self.current_step)
        self.update_step_label()

    def _on_page_changed(self, index):
        """Refresh step-specific content whenever a page becomes current."""
        if index == RECOMMENDATION_STEP_INDEX:
            self.recommendation_page.refresh_recommendations()
        elif index == EXPORT_STEP_INDEX:
            self.export_page.refresh_summary()
        self.refresh_step_bar()

    def update_step_label(self):
        self.step_label.setText(
            f"Step {self.current_step + 1}/{len(STEP_TITLES)}"
            f"  •  {STEP_TITLES[self.current_step]}"
        )

        self.step_status_bar.set_current_step(PAGE_STEP_IDS[self.current_step])
        state = navigation_state(self.current_step)
        self.btn_previous.setEnabled(state["previous_enabled"])
        self.btn_next.setEnabled(state["next_enabled"])
        self.btn_next.setText(state["next_text"])

    def refresh_step_bar(self):
        """Recompute S1-S7 statuses and sync the calibration panel."""
        self.step_status_bar.update_statuses(
            compute_step_status(self.current_recipe_data)
        )
        self.training_page.refresh_calibration_panel()

    # ------------------------------------------------------------------
    # Recipe context
    # ------------------------------------------------------------------

    def update_context(self, recipe_data):
        """Update the header line after a recipe is selected or created."""
        self.current_recipe_data = recipe_data
        self.context_label.setText(
            f"Package : {recipe_data.get('package_family', '-')}"
            f" | Recipe : {recipe_data.get('recipe_name', '-')}"
        )
        self.refresh_step_bar()

    def clear_context(self):
        """Reset the header after the current recipe is deleted."""
        self.current_recipe_data = None
        self.context_label.setText("Package : - | Recipe : -")
        self.refresh_step_bar()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _busy_activities(self):
        """Names of background workers still running, if any."""
        busy = []
        if self.training_page.is_training_running():
            busy.append("Training")
        if self.training_page.is_calibration_running():
            busy.append("Threshold calibration")
        if self.dataset_page.is_analysis_running():
            busy.append("Dataset analysis")
        if self.roi_page.is_autotune_running():
            busy.append("ROI auto-tune")
        if self.recommendation_page.is_augment_running():
            busy.append("Add-to-memory")
        return busy

    def closeEvent(self, event):
        busy = self._busy_activities()
        if busy:
            QMessageBox.warning(
                self,
                "Background Work In Progress",
                "Still running: " + ", ".join(busy) + ".\n"
                "Please wait for it to finish before closing the trainer.",
            )
            event.ignore()
            return

        self.training_page.restore_output()
        event.accept()
