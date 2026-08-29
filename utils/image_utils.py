"""Shared image-processing helpers used by both applications."""

import cv2
import numpy as np


def crop_from_obb(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop the axis-aligned bounding rectangle of an oriented bounding box."""
    points = points.astype(np.int32)

    x_min = max(0, int(np.min(points[:, 0])))
    y_min = max(0, int(np.min(points[:, 1])))
    x_max = min(frame.shape[1], int(np.max(points[:, 0])))
    y_max = min(frame.shape[0], int(np.max(points[:, 1])))

    return frame[y_min:y_max, x_min:x_max]


def letterbox(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize an image while keeping its aspect ratio, then centre it on a
    black canvas of the requested size."""
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        return cv2.resize(image, (target_width, target_height))

    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height))

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return canvas
