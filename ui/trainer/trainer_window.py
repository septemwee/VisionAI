"""Main wizard window for the trainer application."""

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.trainer.pages.dataset_page import DatasetPage
from ui.trainer.pages.export_page import ExportPage
from ui.trainer.pages.recommendation_page import RecommendationPage
from ui.trainer.pages.recipe_page import RecipePage
from ui.trainer.pages.review_page import ReviewPage
from ui.trainer.pages.roi_page import ROIPage
from ui.trainer.pages.training_page import TrainingPage

STEP_TITLES = [
    "Recipe Manager",
    "Dataset Import",
    "Model Recommendation",
    "ROI Verification",
    "Training",
    "Model Review",
    "Export",
]

# Index of the ROI Verification step, which needs the dataset loaded first.
ROI_STEP_INDEX = 3


class TrainerWindow(QWidget):
    """Guides the user through recipe creation, training and review."""

    def __init__(self):
        super().__init__()

        self.current_step = 0
        self.current_recipe_data = None

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("VisionAI Trainer")
        self.resize(1400, 850)

        self.setStyleSheet(
            """
            QWidget{
                background:#FFFFFF;
                font-family:'Poppins';
            }

            QLabel{
                color:#374151;
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
            """
        )

        root_layout = QVBoxLayout()

        # ----- Header -----
        title = QLabel("VisionAI Trainer")
        title.setStyleSheet("font-size:30px;font-weight:700;color:#111827;")
        root_layout.addWidget(title)

        self.context_label = QLabel("Package : - | Recipe : -")
        self.context_label.setStyleSheet(
            """
            background:#EFF6FF;
            color:#1D4ED8;
            padding:10px;
            border-radius:8px;
            font-size:14px;
            font-weight:600;
            """
        )
        root_layout.addWidget(self.context_label)

        # ----- Step indicator -----
        self.step_label = QLabel()
        self.step_label.setStyleSheet("color:#2563EB;font-size:14px;font-weight:700;")
        root_layout.addWidget(self.step_label)

        # ----- Wizard pages -----
        self.pages = QStackedWidget()

        self.recipe_page = RecipePage()
        self.recipe_page.parent_window = self

        self.dataset_page = DatasetPage()
        self.dataset_page.parent_window = self

        self.recommendation_page = RecommendationPage()

        self.roi_page = ROIPage()
        self.roi_page.parent_window = self

        self.training_page = TrainingPage()
        self.training_page.parent_window = self

        self.review_page = ReviewPage()
        self.review_page.parent_window = self

        self.export_page = ExportPage()

        for page in (
            self.recipe_page,
            self.dataset_page,
            self.recommendation_page,
            self.roi_page,
            self.training_page,
            self.review_page,
            self.export_page,
        ):
            self.pages.addWidget(page)

        root_layout.addWidget(self.pages)

        # ----- Navigation -----
        nav_layout = QHBoxLayout()

        self.btn_previous = QPushButton("◀ Previous")
        self.btn_next = QPushButton("Next ▶")
        self.btn_previous.clicked.connect(self.previous_page)
        self.btn_next.clicked.connect(self.next_page)

        nav_layout.addWidget(self.btn_previous)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        root_layout.addLayout(nav_layout)

        self.setLayout(root_layout)
        self.update_step_label()

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

    def update_step_label(self):
        self.step_label.setText(
            f"Step {self.current_step + 1}/{len(STEP_TITLES)}"
            f"  •  {STEP_TITLES[self.current_step]}"
        )

    def update_context(self, recipe_data):
        """Update the header line after a recipe is selected or created."""
        self.current_recipe_data = recipe_data
        self.context_label.setText(
            f"Package : {recipe_data.get('package_family', '-')}"
            f" | Recipe : {recipe_data.get('recipe_name', '-')}"
        )
