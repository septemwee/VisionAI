"""Orientation detection for IC packages via top-mark template matching."""

import cv2
import numpy as np

from utils.paths import resolve_recipe_path

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


class TopMarkService:
    """Determines package orientation by matching a top-mark template."""

    def __init__(self):
        self._template_cache_key = None
        self._template_cache = None

    def detect_orientation(self, roi, template_path):
        """Return ``(best_angle, best_score, score_gap)`` for the given ROI.

        ``score_gap`` is the difference between the best and second-best
        orientation scores and indicates how confident the match is.
        """
        template_path = resolve_recipe_path(template_path)
        if template_path is None:
            raise FileNotFoundError("No top-mark template path was provided.")

        template = self._load_template(template_path)
        crop = self._preprocess(roi)

        scaled_templates = []
        for scale in TEMPLATE_SCALES:
            scaled_templates.append(
                cv2.resize(
                    template,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            )

        scores = {}
        for angle in ORIENTATION_ANGLES:
            rotated = rotate_image(crop, angle)
            scores[angle] = self._score_angle(rotated, scaled_templates)
            if scores[angle] == -999.0:
                # The detected package may be displayed smaller than the
                # saved ROI. Only resize further when no normal scale fits.
                height, width = rotated.shape[:2]
                template_height, template_width = template.shape[:2]
                fit_scale = 0.95 * min(
                    height / template_height, width / template_width
                )
                if 0 < fit_scale < TEMPLATE_SCALES[0]:
                    fitted = cv2.resize(
                        template, None, fx=fit_scale, fy=fit_scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    scores[angle] = self._score_angle(rotated, [fitted])

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_angle, best_score = ranked[0]
        score_gap = ranked[0][1] - ranked[1][1]

        return best_angle, best_score, score_gap

    def _load_template(self, template_path):
        """Load and preprocess the template once, cached by path and mtime."""
        try:
            mtime = template_path.stat().st_mtime
        except OSError:
            mtime = None
        cache_key = (str(template_path), mtime)

        if cache_key == self._template_cache_key:
            return self._template_cache

        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(str(template_path))

        self._template_cache = self._preprocess(template)
        self._template_cache_key = cache_key
        return self._template_cache

    @staticmethod
    def _score_angle(candidate, scaled_templates):
        """Score one rotated candidate against all template scale variants."""
        candidate_height, candidate_width = candidate.shape[:2]
        best_score = -999.0

        for resized in scaled_templates:
            height, width = resized.shape[:2]
            if height >= candidate_height or width >= candidate_width:
                continue

            result = cv2.matchTemplate(candidate, resized, cv2.TM_CCOEFF_NORMED)

            flat = result.flatten()
            if len(flat) < TOP_LOCATIONS:
                continue

            top_scores = np.partition(flat, -TOP_LOCATIONS)[-TOP_LOCATIONS:]
            top_mean = np.mean(top_scores)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            score = MAX_SCORE_WEIGHT * max_val + TOP_MEAN_WEIGHT * top_mean
            if score > best_score:
                best_score = score

        return best_score

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
