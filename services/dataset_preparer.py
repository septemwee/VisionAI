"""Prepares raw datasets for PatchCore training by cropping detected ROIs."""

from pathlib import Path

import cv2

from services.roi_service import ROIService

SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


class DatasetPreparer:
    """Crops the detected IC region out of every image in a dataset."""

    def __init__(self):
        self.roi_service = ROIService()

    def prepare_patchcore_dataset(self, source_dataset, output_dataset):
        source_dataset = Path(source_dataset)
        output_dataset = Path(output_dataset)

        train_good = output_dataset / "train" / "good"
        train_good.mkdir(parents=True, exist_ok=True)

        image_count = 0

        for extension in SUPPORTED_EXTENSIONS:
            for image_path in source_dataset.glob(extension):
                roi = self.roi_service.detect_roi(str(image_path))
                if roi is None:
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                x1, y1, x2, y2 = self._roi_bounds(image, roi)
                crop = image[y1:y2, x1:x2]

                cv2.imwrite(str(train_good / image_path.name), crop)
                image_count += 1

        return {
            "image_count": image_count,
            "dataset_path": str(output_dataset),
        }

    @staticmethod
    def _roi_bounds(image, roi):
        cx = int(roi["cx"])
        cy = int(roi["cy"])
        half_width = int(roi["width"]) // 2
        half_height = int(roi["height"]) // 2

        x1 = max(0, cx - half_width)
        y1 = max(0, cy - half_height)
        x2 = min(image.shape[1], cx + half_width)
        y2 = min(image.shape[0], cy + half_height)

        return x1, y1, x2, y2
