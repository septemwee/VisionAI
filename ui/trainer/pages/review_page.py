"""Model review page of the trainer."""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from services.inference_service import InferenceService
from services.patchcore_service import SCORE_SCALE
from services.recipe_service import RecipeService, normalize_anomaly_threshold
from ui.theme import (
    SPACE_L,
    SPACE_M,
    SPACE_S,
    card_frame,
    make_button,
    section_label,
    title_label,
)
from utils.image_utils import heatmap_overlay, unwrap_anomaly_map


class ReviewPage(QWidget):
    """Scores a test image with the trained model and shows heatmaps."""

    def __init__(self):
        super().__init__()

        self.parent_window = None
        self.current_image_path = None
        self.current_result = None

        self.inference_service = InferenceService()
        self.recipe_service = RecipeService()

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(SPACE_L, SPACE_L, SPACE_L, SPACE_L)
        root.setSpacing(SPACE_M)

        root.addWidget(title_label("Model Review"))

        controls_card = card_frame()
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(SPACE_L, SPACE_M, SPACE_L, SPACE_M)
        controls_layout.setSpacing(SPACE_S)

        controls_layout.addWidget(section_label("Verification"))

        load_row = QHBoxLayout()
        load_row.setSpacing(SPACE_S)

        self.btn_load = make_button("Load Image", "secondary")
        self.btn_load.clicked.connect(self.load_image)
        load_row.addWidget(self.btn_load)

        self.lbl_score = QLabel("Score : -")
        self.lbl_score.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        self.lbl_result = QLabel("Result : -")
        self.lbl_result.setStyleSheet(
            "border:none;background:transparent;font-weight:700;color:#111827;"
        )
        load_row.addWidget(self.lbl_score)
        load_row.addWidget(self.lbl_result)
        load_row.addStretch()

        controls_layout.addLayout(load_row)

        self.lbl_threshold = QLabel("0.60")
        self.lbl_threshold.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(60)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        self.lbl_opacity = QLabel("0.50")
        self.lbl_opacity.setStyleSheet(
            "border:none;background:transparent;font-weight:600;color:#111827;"
        )
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_opacity)

        sliders = QFormLayout()
        sliders.setHorizontalSpacing(SPACE_M)
        sliders.setVerticalSpacing(SPACE_S)
        sliders.addRow("Threshold", self.threshold_slider)
        sliders.addRow("", self.lbl_threshold)
        sliders.addRow("Heatmap Opacity", self.opacity_slider)
        sliders.addRow("", self.lbl_opacity)
        controls_layout.addLayout(sliders)

        self.btn_apply_threshold = make_button("Apply Threshold to Recipe")
        self.btn_apply_threshold.clicked.connect(self.apply_threshold_to_recipe)
        controls_layout.addWidget(self.btn_apply_threshold)

        root.addWidget(controls_card)

        # ----- Image previews -----
        image_layout = QHBoxLayout()
        image_layout.setSpacing(SPACE_M)

        self.lbl_original = QLabel("Original")
        self.lbl_heatmap = QLabel("Heatmap Overlay")
        self.lbl_segment = QLabel("Threshold Mask")

        for label in (self.lbl_original, self.lbl_heatmap, self.lbl_segment):
            label.setMinimumSize(280, 280)
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName("preview")
            image_layout.addWidget(label)

        root.addLayout(image_layout)
        self.setLayout(root)

    # ------------------------------------------------------------------
    # Image loading and prediction
    # ------------------------------------------------------------------

    def load_image(self):
        if not self.parent_window:
            return

        recipe = self.parent_window.current_recipe_data
        if not recipe:
            return

        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if not image_path:
            return

        self.current_image_path = image_path

        self.inference_service.load_model(recipe)

        threshold = normalize_anomaly_threshold(recipe.get("anomaly_threshold"))

        self.current_result = self.inference_service.predict(
            image_path, anomaly_threshold=threshold, recipe=recipe
        )

        self.refresh_visualization()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def refresh_visualization(self):
        if not self.current_image_path or not self.current_result:
            return

        image = cv2.imread(self.current_image_path)
        if image is None:
            return

        score = float(self.current_result["raw_score"])
        threshold = float(self.current_result["anomaly_threshold"])
        result_text = self.current_result["result"]

        self.lbl_score.setText(f"Score : {score:.4f}")
        self.lbl_result.setText(f"Result : {result_text}")

        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(round(threshold * 100)))
        self.threshold_slider.blockSignals(False)
        self.lbl_threshold.setText(f"{threshold:.2f}")

        anomaly_map = unwrap_anomaly_map(
            self.current_result["prediction"].anomaly_map.cpu().numpy()
        )

        opacity = self.opacity_slider.value() / 100.0
        overlay = heatmap_overlay(image, anomaly_map, opacity)
        if overlay is None:
            return

        # The map holds raw NN distances on the ~0-100 scale while the
        # recipe threshold is a 0-1 fraction; scale before masking so the
        # mask matches the rule the live inspection gate applies.
        binary_map = (anomaly_map >= threshold * SCORE_SCALE).astype(np.uint8)
        binary_map = cv2.resize(
            binary_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        segment = image.copy()
        segment[binary_map > 0] = [0, 0, 255]

        self.set_image(self.lbl_original, image)
        self.set_image(self.lbl_heatmap, overlay)
        self.set_image(self.lbl_segment, segment)

    def update_threshold(self):
        self.lbl_threshold.setText(f"{self.threshold_slider.value() / 100.0:.2f}")
        self.refresh_visualization()

    def apply_threshold_to_recipe(self):
        """Persist the tuned slider value as the recipe's threshold."""
        recipe = self.parent_window.current_recipe_data if self.parent_window else None
        if not recipe:
            QMessageBox.warning(self, "Warning", "Please select a recipe first.")
            return

        value = self.threshold_slider.value() / 100.0
        from services.validation_service import invalidate_validation
        invalidate_validation(recipe)
        recipe["anomaly_threshold"] = value
        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)

        QMessageBox.information(
            self,
            "Threshold Saved",
            f"Threshold {value:.2f} saved to recipe {recipe['recipe_name']}.",
        )

        if hasattr(self.parent_window, "refresh_step_bar"):
            self.parent_window.refresh_step_bar()

    def update_opacity(self):
        value = self.opacity_slider.value() / 100.0
        self.lbl_opacity.setText(f"{value:.2f}")
        self.refresh_visualization()

    @staticmethod
    def set_image(label, image):
        if image is None:
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        qimage = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        label.setPixmap(pixmap)
