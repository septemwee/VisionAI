"""Build a Trainer dataset with the exact crop geometry used by Inspection."""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from services.roi_service import ROIService
from services.training_assistant import classify_crop, qa_bounds
from services.dataset_split_service import assign_good_splits, write_split_manifest
from utils.image_utils import crop_yolo_obb, prepare_inspection_crop


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

            points = roi.get("points")
            if points is None:
                raise ValueError("YOLO OBB corners are required for inspection-compatible crops")
            crop = crop_yolo_obb(image, points)

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
        valid_names = [item["image_name"] for item in entries
                       if not classify_crop(item, bounds)]
        if len(valid_names) != len(set(valid_names)):
            raise ValueError('Duplicate source filenames: rename them before preparing crops.')
        assignments, split_counts = assign_good_splits(valid_names)
        dataset_root = output_dir.parent.parent
        split_dirs = {
            name: dataset_root / name / "good"
            for name in ("train", "calibration", "test")
        }
        for directory in split_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        # Pass 2: re-crop each image from disk so only the current crop is
        # held in memory.
        for index, item in enumerate(entries):
            reasons = classify_crop(item, bounds)

            crop = crop_yolo_obb(cv2.imread(item["path"]), item["roi"]["points"])

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
                standardized_crop = prepare_inspection_crop(
                    cv2.imread(item["path"]), item["roi"]["points"],
                    target_width, target_height,
                )
                split = assignments[item["image_name"]]
                if not cv2.imwrite(str(split_dirs[split] / item["image_name"]), standardized_crop):
                    raise OSError(f"Cannot save crop: {item['image_name']}")
                saved_count += 1

            self.progress.emit(60 + int((index + 1) * 40 / len(entries)))

        return {
            "valid": True,
            "output_dir": str(output_dir),
            "split_counts": split_counts,
            "split_manifest": str(write_split_manifest(dataset_root, assignments, split_counts)),
            "target_width": target_width,
            "target_height": target_height,
            "avg_width": avg_width,
            "avg_height": avg_height,
            "bounds": bounds,
            "saved_count": saved_count,
            "flagged": flagged,
        }
