"""Background worker for the quick dataset statistics pass."""

from PySide6.QtCore import QThread, Signal

from services.dataset_service import DatasetService


class StatsWorker(QThread):
    """Runs ``DatasetService.analyze_dataset`` off the UI thread.

    Opening every image with PIL can take noticeable time on large
    datasets; running it here keeps the trainer window responsive and
    lets the page show real progress while the folder is scanned.
    """

    progress = Signal(int)  # 0-100
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = str(dataset_path)

    def run(self):
        try:
            result = DatasetService().analyze_dataset(
                self.dataset_path,
                progress_cb=lambda done, total: self.progress.emit(
                    int(done * 100 / max(total, 1))
                ),
            )
            self.finished_signal.emit(result)
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
