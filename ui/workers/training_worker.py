"""Background worker that runs PatchCore training off the UI thread."""

from PySide6.QtCore import QThread, Signal

from services.patchcore_service import PatchCoreService


class TrainingWorker(QThread):
    """Trains a PatchCore model for a recipe and reports the outcome."""

    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, recipe):
        super().__init__()
        self.recipe = recipe

    def run(self):
        try:
            model_dir = PatchCoreService().train(self.recipe)
            self.finished_signal.emit(str(model_dir))
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
