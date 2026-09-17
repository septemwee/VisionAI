"""YOLO-based region-of-interest detection for IC packages."""

from ultralytics import YOLO

from utils.paths import DETECTION_MODEL_PATH


class ROIService:
    """Detects oriented bounding boxes of IC packages in images."""

    def __init__(self):
        self.model = YOLO(str(DETECTION_MODEL_PATH))

    def detect_rois(self, image_path):
        """Return every OBB detection as a dict (possibly an empty list).

        Each dict contains ``cx``, ``cy``, ``width``, ``height``, ``angle``
        (radians), ``confidence``, ``class_id`` and ``class_name``. Class
        names come from the model itself; unknown or single-class models
        fall back to ``"package"``.
        """
        results = self.model(image_path, verbose=False)
        result = results[0]

        if result.obb is None or len(result.obb) == 0:
            return []

        names = getattr(self.model, "names", None)
        if not isinstance(names, dict):
            names = {}

        detections = []
        for index in range(len(result.obb.conf)):
            xywhr = result.obb.xywhr[index].cpu().numpy()
            if result.obb.cls is not None:
                class_id = int(result.obb.cls[index])
            else:
                class_id = -1

            detections.append(
                {
                    "cx": float(xywhr[0]),
                    "cy": float(xywhr[1]),
                    "width": float(xywhr[2]),
                    "height": float(xywhr[3]),
                    "angle": float(xywhr[4]),
                    "confidence": float(result.obb.conf[index]),
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, "package")),
                }
            )

        return detections

    def detect_roi(self, image_path):
        """Return the highest-confidence OBB as a dict, or ``None``.

        The returned dict contains ``cx``, ``cy``, ``width``, ``height``,
        ``angle`` (radians) and ``confidence``.
        """
        detections = self.detect_rois(image_path)
        if not detections:
            return None

        best = max(detections, key=lambda detection: detection["confidence"])
        return {
            "cx": best["cx"],
            "cy": best["cy"],
            "width": best["width"],
            "height": best["height"],
            "angle": best["angle"],
            "confidence": best["confidence"],
        }
