"""Background worker that crops and QA-classifies images into a training set.

The per-image loop (ROI detection, rotated crop, QA classification and
letterboxed write) runs off the GUI thread so the window stays responsive
while a dataset is being prepared.
"""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from services.roi_service import ROIService
from services.training_assistant import classify_crop, qa_bounds
from utils.image_utils import crop_rotated_roi, letterbox


class CropWorker(QThread):
    """Crops, QA-classifies and standardises every image into a training set."""

    progress = Signal(int)  # 0-100
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self, image_paths, default_config, overrides, tolerances, output_dir,
        max_flagged_crops=60,
    ):
        super().__init__()
        self.image_paths = [Path(path) for path in image_paths]
        self.default_config = dict(default_config)
        self.overrides = dict(overrides)
        self.tolerances = dict(tolerances)
        self.output_dir = Path(output_dir)
        # Flagged crops beyond this cap are reported without their pixels,
        # so worker memory stays bounded on large datasets.
        self.max_flagged_crops = int(max_flagged_crops)

    def run(self):
        try:
            self.finished_signal.emit(self._generate())
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))

    def _generate(self):
        roi_service = ROIService()
        default_config = self.default_config
        overrides = self.overrides
        tolerances = self.tolerances
        output_dir = self.output_dir

        # Pass 1: collect only crop DIMENSIONS (plus what pass 2 needs to
        # re-crop from disk). Full-resolution crops are never retained in a
        # list, so worker memory stays flat regardless of dataset size.
        entries = []
        roi_widths = []
        roi_heights = []

        total = max(len(self.image_paths), 1)
        for index, image_file in enumerate(self.image_paths):
            roi = roi_service.detect_roi(str(image_file))
            if roi is None:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                continue

            config = dict(default_config)
            config.update(overrides.get(image_file.name, {}))

            crop = crop_rotated_roi(image, roi, config)

            height, width = crop.shape[:2]
            if height <= 0 or width <= 0:
                continue

            roi_widths.append(width)
            roi_heights.append(height)
            entries.append(
                {
                    "image_name": image_file.name,
                    "path": str(image_file),
                    "roi": roi,
                    "config": config,
                    "width": width,
                    "height": height,
                    "ratio": width / height,
                }
            )

            self.progress.emit(int((index + 1) * 60 / total))

        if not entries:
            return {"valid": False}

        target_width = max(int(np.median(roi_widths)), 1)
        target_height = max(int(np.median(roi_heights)), 1)

        avg_width = int(np.mean(roi_widths))
        avg_height = int(np.mean(roi_heights))
        avg_ratio = float(np.mean([item["ratio"] for item in entries]))

        bounds = qa_bounds(avg_width, avg_height, avg_ratio, tolerances)

        print(f"Average ROI: {avg_width}x{avg_height}, target {target_width}x{target_height}")

        saved_count = 0
        flagged = []

        # Pass 2: re-crop each image from disk so only the current crop is
        # held in memory.
        for index, item in enumerate(entries):
            reasons = classify_crop(item, bounds)

            crop = crop_rotated_roi(
                cv2.imread(item["path"]), item["roi"], item["config"]
            )

            if reasons:
                flagged.append(
                    {
                        "image_name": item["image_name"],
                        "crop": (
                            crop if len(flagged) < self.max_flagged_crops else None
                        ),
                        "reasons": reasons,
                    }
                )
                print(f"[FLAG] {item['image_name']} — {'; '.join(reasons)}")
            else:
                standardized_crop = letterbox(crop, target_width, target_height)
                cv2.imwrite(str(output_dir / item["image_name"]), standardized_crop)
                saved_count += 1

            self.progress.emit(60 + int((index + 1) * 40 / len(entries)))

        return {
            "valid": True,
            "output_dir": str(output_dir),
            "target_width": target_width,
            "target_height": target_height,
            "avg_width": avg_width,
            "avg_height": avg_height,
            "bounds": bounds,
            "saved_count": saved_count,
            "flagged": flagged,
        }
