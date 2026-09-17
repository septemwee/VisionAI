"""Background worker for dataset quality screening and geometry analysis."""

from PySide6.QtCore import QThread, Signal

from services.roi_service import ROIService
from services.training_assistant import analyze_dataset_geometry, assess_dataset


class ScreeningWorker(QThread):
    """Runs quality screening plus a YOLO geometry pass over a dataset.

    Both passes run in one background worker: blur / brightness / size
    outlier flags from ``assess_dataset`` and per-image OBB detections fed
    into ``analyze_dataset_geometry``. Dataset size is never judged — the
    worker only produces informational facts.
    """

    progress = Signal(int)  # 0-100
    finished_signal = Signal(dict)  # {"quality": ..., "geometry": ...}
    error_signal = Signal(str)

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = [str(path) for path in image_paths]

    def run(self):
        try:
            quality = assess_dataset(
                self.image_paths,
                progress_cb=lambda done, total: self.progress.emit(
                    int(done * 50 / max(total, 1))
                ),
            )

            detections = []
            roi_service = ROIService()
            total = max(len(self.image_paths), 1)
            for index, path in enumerate(self.image_paths):
                try:
                    detections.append(roi_service.detect_rois(path))
                except Exception:
                    detections.append([])
                self.progress.emit(50 + int((index + 1) * 50 / total))

            geometry = analyze_dataset_geometry(detections)
            self.finished_signal.emit(
                {"quality": quality, "geometry": geometry}
            )
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
