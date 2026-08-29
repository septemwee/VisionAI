"""ROI verification and training-dataset generation page of the trainer."""

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from services.recipe_service import RecipeService
from services.roi_service import ROIService
from utils.image_utils import letterbox
from utils.paths import RECIPES_DIR

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]

# Geometry filters applied to generated crops, relative to the average size.
WIDTH_TOLERANCE = 0.25
HEIGHT_TOLERANCE = 0.25
RATIO_TOLERANCE = 0.2

# Fallback values for ROI settings.
DEFAULT_WIDTH_SCALE = 1.0
DEFAULT_HEIGHT_SCALE = 1.0
DEFAULT_PADDING = 10
DEFAULT_ANGLE_OFFSET = 0.0


class ROIPage(QWidget):
    """Verifies detected ROIs and builds the normal-image training set."""

    def __init__(self):
        super().__init__()

        self.roi_service = ROIService()
        self.recipe_service = RecipeService()

        self.dataset_path = ""
        self.image_files = []
        self.current_index = 0
        self.current_image_path = None
        self.parent_window = None
        self.roi_data = None
        self.roi_overrides = {}

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()

        title = QLabel("ROI Dataset Preparation")
        title.setStyleSheet("font-size:24px;font-weight:700;")
        root.addWidget(title)

        # ----- Image navigation -----
        nav_layout = QHBoxLayout()

        self.btn_previous = QPushButton("◀ Previous")
        self.btn_next = QPushButton("Next ▶")
        self.image_index_label = QLabel("0 / 0")

        self.btn_previous.clicked.connect(self.previous_image)
        self.btn_next.clicked.connect(self.next_image)

        nav_layout.addWidget(self.btn_previous)
        nav_layout.addWidget(self.image_index_label)
        nav_layout.addWidget(self.btn_next)
        root.addLayout(nav_layout)

        # ----- Previews -----
        image_layout = QHBoxLayout()

        self.original_preview = QLabel("Original Image")
        self.original_preview.setAlignment(Qt.AlignCenter)
        self.original_preview.setMinimumSize(450, 320)

        self.roi_preview = QLabel("ROI Preview")
        self.roi_preview.setAlignment(Qt.AlignCenter)
        self.roi_preview.setMinimumSize(450, 320)

        image_layout.addWidget(self.original_preview)
        image_layout.addWidget(self.roi_preview)
        root.addLayout(image_layout)

        # ----- ROI controls -----
        control_card = QFrame()
        control_layout = QVBoxLayout(control_card)

        self.width_scale_spin = self._create_double_spin(0.5, 3.0, 0.1, DEFAULT_WIDTH_SCALE)
        self.height_scale_spin = self._create_double_spin(0.5, 3.0, 0.1, DEFAULT_HEIGHT_SCALE)
        self.padding_spin = self._create_int_spin(0, 100, DEFAULT_PADDING)
        self.angle_offset_spin = self._create_double_spin(-180, 180, 1, DEFAULT_ANGLE_OFFSET)

        for spin in (
            self.width_scale_spin,
            self.height_scale_spin,
            self.padding_spin,
            self.angle_offset_spin,
        ):
            spin.valueChanged.connect(self.refresh_preview)

        for label_text, spin in (
            ("Width Scale", self.width_scale_spin),
            ("Height Scale", self.height_scale_spin),
            ("Padding", self.padding_spin),
            ("Angle Offset", self.angle_offset_spin),
        ):
            control_layout.addWidget(QLabel(label_text))
            control_layout.addWidget(spin)

        root.addWidget(control_card)

        # ----- Detection info -----
        self.confidence_label = QLabel("Confidence : -")
        self.size_label = QLabel("Size : -")
        root.addWidget(self.confidence_label)
        root.addWidget(self.size_label)

        # ----- Actions -----
        self.btn_save_override = QPushButton("Save Override For This Image")
        self.btn_save_override.clicked.connect(self.save_override)
        root.addWidget(self.btn_save_override)

        self.btn_save_default = QPushButton("Save As Default")
        self.btn_save_default.clicked.connect(self.save_default)
        root.addWidget(self.btn_save_default)

        self.btn_generate = QPushButton("Generate Training Dataset")
        self.btn_generate.clicked.connect(self.generate_training_dataset)
        root.addWidget(self.btn_generate)

        root.addStretch()
        self.setLayout(root)

    @staticmethod
    def _create_double_spin(minimum, maximum, step, default):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(default)
        return spin

    @staticmethod
    def _create_int_spin(minimum, maximum, default):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(default)
        return spin

    # ------------------------------------------------------------------
    # Dataset browsing
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_path):
        if not dataset_path:
            self.image_files = []
            self.current_index = 0
            return

        self.dataset_path = dataset_path

        self.image_files = []
        for extension in IMAGE_EXTENSIONS:
            self.image_files.extend(Path(dataset_path).glob(extension))

        print(f"Loaded dataset: {dataset_path} ({len(self.image_files)} images)")

        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.image_files:
            return

        self.current_image_path = str(self.image_files[self.current_index])
        self.load_override(self.current_image_path)
        self.load_image(self.current_image_path)

        self.image_index_label.setText(
            f"{self.current_index + 1} / {len(self.image_files)}"
        )

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.original_preview.setPixmap(
            pixmap.scaled(self.original_preview.size(), Qt.KeepAspectRatio)
        )

        roi = self.roi_service.detect_roi(image_path)
        if roi is None:
            return

        self.roi_data = roi
        self.show_roi_preview(image_path, roi)

        self.confidence_label.setText(f"Confidence : {roi['confidence']:.3f}")
        self.size_label.setText(f"Size : {roi['width']:.0f} x {roi['height']:.0f}")

    # ------------------------------------------------------------------
    # ROI preview
    # ------------------------------------------------------------------

    def show_roi_preview(self, image_path, roi):
        image = cv2.imread(image_path)
        if image is None:
            return

        angle_deg = math.degrees(roi["angle"]) + self.angle_offset_spin.value()

        rect = (
            (float(roi["cx"]), float(roi["cy"])),
            (
                float(roi["width"]) * self.width_scale_spin.value()
                + self.padding_spin.value() * 2,
                float(roi["height"]) * self.height_scale_spin.value()
                + self.padding_spin.value() * 2,
            ),
            angle_deg,
        )

        box = cv2.boxPoints(rect).astype(int)

        preview = image.copy()
        cv2.drawContours(preview, [box], 0, (0, 255, 0), 3)

        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            preview.data,
            preview.shape[1],
            preview.shape[0],
            preview.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)

        self.roi_preview.setPixmap(
            pixmap.scaled(self.roi_preview.size(), Qt.KeepAspectRatio)
        )

    def refresh_preview(self):
        if self.current_image_path and self.roi_data:
            self.show_roi_preview(self.current_image_path, self.roi_data)

    # ------------------------------------------------------------------
    # ROI settings persistence
    # ------------------------------------------------------------------

    def current_settings(self):
        """Return the current spin-box values as a settings dict."""
        return {
            "width_scale": self.width_scale_spin.value(),
            "height_scale": self.height_scale_spin.value(),
            "padding": self.padding_spin.value(),
            "angle_offset": self.angle_offset_spin.value(),
        }

    def apply_settings(self, settings):
        self.width_scale_spin.setValue(settings.get("width_scale", DEFAULT_WIDTH_SCALE))
        self.height_scale_spin.setValue(
            settings.get("height_scale", DEFAULT_HEIGHT_SCALE)
        )
        self.padding_spin.setValue(settings.get("padding", DEFAULT_PADDING))
        self.angle_offset_spin.setValue(
            settings.get("angle_offset", DEFAULT_ANGLE_OFFSET)
        )

    def save_override(self):
        """Store per-image ROI settings in the recipe."""
        if not self.current_image_path:
            return

        recipe = self._current_recipe()
        if recipe is None:
            return

        image_name = Path(self.current_image_path).name
        self.roi_overrides[image_name] = self.current_settings()
        recipe["roi_overrides"] = self.roi_overrides

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
        print(f"Override saved for {image_name}")

    def load_override(self, image_path):
        """Apply the stored override for an image, or the defaults."""
        image_name = Path(image_path).name
        settings = self.roi_overrides.get(image_name)

        if settings is None:
            self.apply_settings(
                {
                    "width_scale": DEFAULT_WIDTH_SCALE,
                    "height_scale": DEFAULT_HEIGHT_SCALE,
                    "padding": DEFAULT_PADDING,
                    "angle_offset": DEFAULT_ANGLE_OFFSET,
                }
            )
            return

        self.apply_settings(settings)

    def save_default(self):
        """Store the current settings as the recipe-wide default."""
        recipe = self._current_recipe()
        if recipe is None:
            return

        recipe["roi_default"] = self.current_settings()
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        self.parent_window.current_recipe_data = recipe

    def _current_recipe(self):
        if self.parent_window is None:
            return None
        return self.parent_window.current_recipe_data

    # ------------------------------------------------------------------
    # Training dataset generation
    # ------------------------------------------------------------------

    def generate_training_dataset(self):
        """Crop, filter and standardise every image into a training set."""
        recipe = self._current_recipe()
        if recipe is None:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        default_config = recipe.get("roi_default", {})
        overrides = recipe.get("roi_overrides", {})

        output_dir = (
            RECIPES_DIR / recipe["recipe_name"] / "patchcore_dataset" / "train" / "good"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        all_crops = []
        roi_widths = []
        roi_heights = []

        for image_file in self.image_files:
            roi = self.roi_service.detect_roi(str(image_file))
            if roi is None:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                continue

            config = dict(default_config)
            config.update(overrides.get(image_file.name, {}))

            crop = self._crop_rotated_roi(image, roi, config)

            height, width = crop.shape[:2]
            roi_widths.append(width)
            roi_heights.append(height)

            all_crops.append(
                {
                    "image_name": image_file.name,
                    "crop": crop,
                    "width": width,
                    "height": height,
                    "ratio": width / height,
                }
            )

        if not all_crops:
            QMessageBox.warning(
                self, "Warning", "No valid ROIs were found in the dataset."
            )
            return

        target_width = max(int(np.median(roi_widths)), 1)
        target_height = max(int(np.median(roi_heights)), 1)

        avg_width = int(np.mean(roi_widths))
        avg_height = int(np.mean(roi_heights))
        avg_ratio = float(np.mean([item["ratio"] for item in all_crops]))

        min_width = avg_width * (1 - WIDTH_TOLERANCE)
        max_width = avg_width * (1 + WIDTH_TOLERANCE)
        min_height = avg_height * (1 - HEIGHT_TOLERANCE)
        max_height = avg_height * (1 + HEIGHT_TOLERANCE)
        min_ratio = avg_ratio * (1 - RATIO_TOLERANCE)
        max_ratio = avg_ratio * (1 + RATIO_TOLERANCE)

        print(f"Average ROI: {avg_width}x{avg_height}, target {target_width}x{target_height}")

        saved_count = 0
        rejected_count = 0

        for item in all_crops:
            within_bounds = (
                min_width <= item["width"] <= max_width
                and min_height <= item["height"] <= max_height
                and min_ratio <= item["ratio"] <= max_ratio
            )

            if not within_bounds:
                rejected_count += 1
                print(f"[REJECT] {item['image_name']} {item['width']}x{item['height']}")
                continue

            standardized_crop = letterbox(item["crop"], target_width, target_height)
            cv2.imwrite(str(output_dir / item["image_name"]), standardized_crop)
            saved_count += 1

        print(f"Saved: {saved_count}, rejected: {rejected_count}")

        recipe["roi_statistics"] = {
            "avg_width": avg_width,
            "avg_height": avg_height,
            "min_width": min_width,
            "max_width": max_width,
            "min_height": min_height,
            "max_height": max_height,
            "avg_area": avg_width * avg_height,
            "avg_ratio": avg_width / avg_height,
            "target_width": target_width,
            "target_height": target_height,
            "saved_count": saved_count,
            "rejected_count": rejected_count,
        }
        recipe["dataset_prepared"] = True
        recipe["prepared_dataset_path"] = str(output_dir.parent.parent)

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        QMessageBox.information(self, "Success", "Training dataset generated.")

    @staticmethod
    def _crop_rotated_roi(image, roi, config):
        """Rotate the image so the package is upright and crop around it."""
        width_scale = config.get("width_scale", DEFAULT_WIDTH_SCALE)
        height_scale = config.get("height_scale", DEFAULT_HEIGHT_SCALE)
        padding = config.get("padding", DEFAULT_PADDING)
        angle_offset = config.get("angle_offset", DEFAULT_ANGLE_OFFSET)

        angle_deg = math.degrees(roi["angle"]) + angle_offset

        matrix = cv2.getRotationMatrix2D(
            (roi["cx"], roi["cy"]), angle_deg, 1.0
        )
        rotated = cv2.warpAffine(
            image, matrix, (image.shape[1], image.shape[0])
        )

        half_width = int(roi["width"] * width_scale) // 2
        half_height = int(roi["height"] * height_scale) // 2
        cx = int(roi["cx"])
        cy = int(roi["cy"])

        x1 = max(0, cx - half_width - padding)
        y1 = max(0, cy - half_height - padding)
        x2 = min(rotated.shape[1], cx + half_width + padding)
        y2 = min(rotated.shape[0], cy + half_height + padding)

        return rotated[y1:y2, x1:x2]
