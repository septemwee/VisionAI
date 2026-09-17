"""Background worker that derives ROI defaults from detected package sizes."""

from PySide6.QtCore import QThread, Signal

from services.roi_service import ROIService
from services.training_assistant import auto_tune_plan


class AutoTuneWorker(QThread):
    """Detects packages on a sample of images and proposes ROI defaults.

    Emits a plan dict (see ``training_assistant.auto_tune_plan``) or ``None``
    when too few packages were detected to base defaults on.
    """

    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, image_paths, sample_limit=30):
        super().__init__()
        self.image_paths = [str(path) for path in image_paths]
        self.sample_limit = int(sample_limit)

    def run(self):
        try:
            roi_service = ROIService()

            paths = self.image_paths
            if len(paths) > self.sample_limit:
                step = len(paths) / self.sample_limit
                paths = [paths[int(index * step)] for index in range(self.sample_limit)]

            detections = []
            for path in paths:
                try:
                    detections.extend(roi_service.detect_rois(path))
                except Exception:
                    continue

            self.finished_signal.emit(auto_tune_plan(detections))
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
