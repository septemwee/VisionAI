"""Dataset import page of the trainer."""

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from services.dataset_service import DatasetService
from services.recipe_service import RecipeService

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]


class DatasetPage(QWidget):
    """Lets the user pick a dataset folder and shows basic statistics."""

    def __init__(self):
        super().__init__()

        self.dataset_service = DatasetService()
        self.recipe_service = RecipeService()

        self.analysis_result = None
        self.dataset_path = ""
        self.first_image_path = None
        self.parent_window = None

        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(
            """
            QWidget{
                background:#FFFFFF;
                font-family:'Poppins';
            }

            QLabel{
                color:#374151;
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
        )

        layout = QVBoxLayout()

        # ----- Title -----
        title = QLabel("Dataset Import")
        title.setStyleSheet(
            "font-size:24px;font-weight:700;color:#111827;Border: None;"
        )
        layout.addWidget(title)

        # ----- Dataset folder card -----
        dataset_card = QFrame()
        dataset_layout = QVBoxLayout()

        dataset_title = QLabel("Dataset Folder")
        dataset_title.setStyleSheet("font-weight:600;Border: None;")

        self.dataset_path_label = QLabel("No Dataset Selected")
        self.dataset_path_label.setStyleSheet("Border: None;")

        self.btn_browse = QPushButton("Browse Dataset")
        self.btn_browse.clicked.connect(self.select_dataset)

        dataset_layout.addWidget(dataset_title)
        dataset_layout.addWidget(self.dataset_path_label)
        dataset_layout.addWidget(self.btn_browse)
        dataset_card.setLayout(dataset_layout)
        layout.addWidget(dataset_card)

        # ----- Summary card -----
        summary_card = QFrame()
        summary_layout = QVBoxLayout()

        summary_title = QLabel("Dataset Summary")
        summary_title.setStyleSheet("border: None;font-weight:600;")

        self.image_count_label = QLabel("Images : 0")
        self.resolution_label = QLabel("Resolution : -")
        self.corrupted_label = QLabel("Corrupted : 0")
        self.status_label = QLabel("Status : -")

        for label in (
            self.image_count_label,
            self.resolution_label,
            self.corrupted_label,
            self.status_label,
        ):
            label.setStyleSheet("Border: None;")

        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.image_count_label)
        summary_layout.addWidget(self.resolution_label)
        summary_layout.addWidget(self.corrupted_label)
        summary_layout.addWidget(self.status_label)
        summary_card.setLayout(summary_layout)
        layout.addWidget(summary_card)

        # ----- Status card -----
        status_card = QFrame()
        status_layout = QVBoxLayout()

        status_title = QLabel("Dataset Status")
        status_title.setStyleSheet("Border: None;font-weight:600;")

        self.ready_label = QLabel("Waiting for Dataset...")
        self.ready_label.setStyleSheet("Border: None;")

        status_layout.addWidget(status_title)
        status_layout.addWidget(self.ready_label)
        status_card.setLayout(status_layout)
        layout.addWidget(status_card)

        # ----- Physical analysis card -----
        analysis_card = QFrame()
        analysis_layout = QVBoxLayout()

        analysis_title = QLabel("Physical Analysis")
        analysis_title.setStyleSheet("border:none;font-weight:600;")

        self.family_label = QLabel("Package Family : -")
        self.shape_label = QLabel("Shape : -")
        self.pin_label = QLabel("Pin Count : -")
        self.aspect_label = QLabel("Aspect Ratio : -")

        for label in (
            self.family_label,
            self.shape_label,
            self.pin_label,
            self.aspect_label,
        ):
            label.setStyleSheet("border:none;")

        analysis_layout.addWidget(analysis_title)
        analysis_layout.addWidget(self.family_label)
        analysis_layout.addWidget(self.shape_label)
        analysis_layout.addWidget(self.pin_label)
        analysis_layout.addWidget(self.aspect_label)
        analysis_card.setLayout(analysis_layout)
        layout.addWidget(analysis_card)

        layout.addStretch()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------

    def select_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset")
        if not folder:
            return

        self.dataset_path = folder

        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if recipe:
            recipe["dataset_path"] = folder
            self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
            print(f"Dataset path saved to recipe: {recipe['recipe_name']}")

        self.dataset_path_label.setText(folder)
        self.update_statistics(folder)

    def update_statistics(self, folder):
        result = self.dataset_service.analyze_dataset(folder)
        self.analysis_result = result

        self.image_count_label.setText(f"Images : {result['image_count']}")
        self.resolution_label.setText(f"Resolution : {result['resolution']}")
        self.corrupted_label.setText(f"Corrupted : {result['corrupted']}")
        self.status_label.setText(f"Status : {result['status']}")

        if result["image_count"] >= 300:
            self.ready_label.setText("✅ Ready for Physical Analysis")
        elif result["image_count"] >= 100:
            self.ready_label.setText("⚠ Dataset Usable")
        else:
            self.ready_label.setText("❌ Dataset Too Small")

        # The physical analysis below is a placeholder until the real
        # analysis service is implemented.
        self.family_label.setText("Package Family : SO")
        self.shape_label.setText("Shape : Rectangle")
        self.pin_label.setText("Pin Count : 14")
        self.aspect_label.setText("Aspect Ratio : 2.21")

        image_files = []
        for extension in IMAGE_EXTENSIONS:
            image_files.extend(Path(folder).glob(extension))

        if image_files:
            self.first_image_path = str(image_files[0])
            print(f"First image: {self.first_image_path}")
