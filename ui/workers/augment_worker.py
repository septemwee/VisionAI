"""Background worker that appends new crops to an existing recipe's memory.

This is the add-to-memory v1 path (see the training assistant plan): the new
good images are routed through the TARGET recipe's own ROI / crop / letterbox
pipeline so the crops match its training distribution, then appended to its
``train/good`` folder. The follow-up retrain + recalibration happen through
the normal training and calibration flows.
"""

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from services.roi_service import ROIService
from utils.image_utils import crop_rotated_roi, letterbox
from utils.paths import RECIPES_DIR, resolve_recipe_path


class AugmentWorker(QThread):
    """Crops source images with the target recipe's config and appends them."""

    finished_signal = Signal(int)  # number of appended crops
    error_signal = Signal(str)

    def __init__(self, target_recipe, source_image_paths):
        super().__init__()
        self.target_recipe = target_recipe
        self.source_image_paths = [str(path) for path in source_image_paths]

    def run(self):
        try:
            appended = self._append_crops()
            self.finished_signal.emit(appended)
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))

    def _append_crops(self):
        target = self.target_recipe

        prepared = resolve_recipe_path(target.get("prepared_dataset_path"))
        if prepared is None:
            raise ValueError("Target recipe has no prepared dataset to append to.")

        resolved = prepared.resolve()
        recipes_root = RECIPES_DIR.resolve()
        if resolved != recipes_root and recipes_root not in resolved.parents:
            raise ValueError(
                "Prepared dataset path must stay inside the recipes directory."
            )

        good_dir = prepared / "train" / "good"
        good_dir.mkdir(parents=True, exist_ok=True)

        stats = target.get("roi_statistics") or {}
        target_width = int(stats.get("target_width") or 256)
        target_height = int(stats.get("target_height") or 256)

        default_config = dict(target.get("roi_default") or {})
        overrides = dict(target.get("roi_overrides") or {})

        try:
            confidence_floor = float(target.get("confidence_threshold") or 0.85)
        except (TypeError, ValueError):
            confidence_floor = 0.85

        roi_service = ROIService()
        appended = 0

        for path in self.source_image_paths:
            path = Path(path)

            try:
                roi = roi_service.detect_roi(str(path))
            except Exception:
                continue
            if roi is None:
                continue

            # Never append a crop the detector itself is unsure about: a
            # low-confidence or background detection becomes "good" memory
            # and inflates the recalibrated threshold over real defects.
            if float(roi.get("confidence", 0.0)) < confidence_floor:
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue

            config = dict(default_config)
            config.update(overrides.get(path.name, {}))

            crop = crop_rotated_roi(image, roi, config)
            if crop.size == 0:
                continue

            letterboxed = letterbox(crop, target_width, target_height)
            cv2.imwrite(str(good_dir / path.name), letterboxed)
            appended += 1

        target["dataset_prepared"] = True
        print(f"[AUGMENT] appended {appended} crops to {target.get('recipe_name')}")
        return appended
