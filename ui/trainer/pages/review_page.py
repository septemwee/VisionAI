"""Model review page of the trainer."""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from services.inference_service import InferenceService

DEFAULT_THRESHOLD = 0.60


class ReviewPage(QWidget):
    """Scores a test image with the trained model and shows heatmaps."""

    def __init__(self):
        super().__init__()

        self.parent_window = None
        self.current_image_path = None
        self.current_result = None

        self.inference_service = InferenceService()

        self.setup_ui()

    def setup_ui(self):
        root = QVBoxLayout()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        root.addWidget(self.btn_load)

        self.lbl_score = QLabel("Score : -")
        self.lbl_result = QLabel("Result : -")
        root.addWidget(self.lbl_score)
        root.addWidget(self.lbl_result)

        # ----- Threshold slider -----
        root.addWidget(QLabel("Threshold"))

        self.lbl_threshold = QLabel("20")
        root.addWidget(self.lbl_threshold)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(20)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        root.addWidget(self.threshold_slider)

        # ----- Heatmap opacity slider -----
        root.addWidget(QLabel("Heatmap Opacity"))

        self.lbl_opacity = QLabel("0.50")
        root.addWidget(self.lbl_opacity)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        root.addWidget(self.opacity_slider)

        # ----- Image previews -----
        image_layout = QHBoxLayout()

        self.lbl_original = QLabel("Original")
        self.lbl_heatmap = QLabel("Heatmap Overlay")
        self.lbl_segment = QLabel("Threshold Mask")

        for label in (self.lbl_original, self.lbl_heatmap, self.lbl_segment):
            label.setMinimumSize(400, 400)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "border:1px solid #D1D5DB;background:white;"
            )
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

        threshold = float(
            recipe.get("anomaly_threshold", DEFAULT_THRESHOLD) or DEFAULT_THRESHOLD
        )

        self.current_result = self.inference_service.predict(
            image_path, anomaly_threshold=threshold
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

        anomaly_map = self.current_result["prediction"].anomaly_map.cpu().numpy()
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[0]

        anomaly_map_norm = cv2.normalize(
            anomaly_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        opacity = self.opacity_slider.value() / 100.0
        overlay = cv2.addWeighted(image, 1.0 - opacity, heatmap, opacity, 0)

        binary_map = (anomaly_map >= threshold).astype(np.uint8)
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
        self.lbl_threshold.setText(str(self.threshold_slider.value()))
        self.refresh_visualization()

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
