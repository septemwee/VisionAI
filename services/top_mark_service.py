"""Orientation detection for IC packages via top-mark template matching."""

import cv2
import numpy as np

ORIENTATION_ANGLES = (0, 90, 180, 270)
TEMPLATE_SCALES = np.arange(0.7, 1.31, 0.05)
TOP_LOCATIONS = 50
MAX_SCORE_WEIGHT = 0.7
TOP_MEAN_WEIGHT = 0.3


def rotate_image(image, angle):
    """Rotate an image by one of the supported right angles."""
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    if angle not in rotations:
        raise ValueError(f"Unsupported angle: {angle}")

    flag = rotations[angle]
    return image.copy() if flag is None else cv2.rotate(image, flag)


def match_multi_scale(candidate, template):
    """Match a template against a candidate image at multiple scales.

    The combined score blends the best match value with the mean of the top
    locations, which makes it more stable than a single peak.
    """
    best_score = -999.0
    best_location = None
    best_size = None
    best_scale = None

    candidate_height, candidate_width = candidate.shape[:2]

    for scale in TEMPLATE_SCALES:
        resized = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

        height, width = resized.shape[:2]
        if height >= candidate_height or width >= candidate_width:
            continue

        result = cv2.matchTemplate(candidate, resized, cv2.TM_CCOEFF_NORMED)

        flat = result.flatten()
        if len(flat) < TOP_LOCATIONS:
            continue

        top_scores = np.partition(flat, -TOP_LOCATIONS)[-TOP_LOCATIONS:]
        top_mean = np.mean(top_scores)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        score = MAX_SCORE_WEIGHT * max_val + TOP_MEAN_WEIGHT * top_mean

        if score > best_score:
            best_score = score
            best_location = max_loc
            best_size = (width, height)
            best_scale = scale

    return best_score, best_location, best_size, best_scale


class TopMarkService:
    """Determines package orientation by matching a top-mark template."""

    def detect_orientation(self, roi, template_path):
        """Return ``(best_angle, best_score, score_gap)`` for the given ROI.

        ``score_gap`` is the difference between the best and second-best
        orientation scores and indicates how confident the match is.
        """
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(template_path)

        crop = self._preprocess(roi)
        template = self._preprocess(template)

        scores = {}
        for angle in ORIENTATION_ANGLES:
            score, _, _, _ = match_multi_scale(rotate_image(crop, angle), template)
            scores[angle] = score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_angle, best_score = ranked[0]
        score_gap = ranked[0][1] - ranked[1][1]

        return best_angle, best_score, score_gap

    @staticmethod
    def _preprocess(image):
        """Normalise contrast and reduce noise before template matching."""
        if image.ndim == 3:
            channels = image.shape[2]

            if channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                image = image[:, :, 0]
            else:
                raise ValueError(f"Unsupported image channels: {channels}")

        image = cv2.equalizeHist(image)
        return cv2.GaussianBlur(image, (5, 5), 0)
