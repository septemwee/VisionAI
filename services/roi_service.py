"""YOLO-based region-of-interest detection for IC packages."""

from ultralytics import YOLO

from utils.paths import DETECTION_MODEL_PATH


class ROIService:
    """Detects the oriented bounding box of an IC package in an image."""

    def __init__(self):
        self.model = YOLO(str(DETECTION_MODEL_PATH))

    def detect_roi(self, image_path):
        """Return the highest-confidence OBB as a dict, or ``None``.

        The returned dict contains ``cx``, ``cy``, ``width``, ``height``,
        ``angle`` (radians) and ``confidence``.
        """
        results = self.model(image_path, verbose=False)
        result = results[0]

        if result.obb is None or len(result.obb) == 0:
            return None

        best_index = int(result.obb.conf.argmax())
        xywhr = result.obb.xywhr[best_index].cpu().numpy()
        confidence = float(result.obb.conf[best_index])

        return {
            "cx": float(xywhr[0]),
            "cy": float(xywhr[1]),
            "width": float(xywhr[2]),
            "height": float(xywhr[3]),
            "angle": float(xywhr[4]),
            "confidence": confidence,
        }
