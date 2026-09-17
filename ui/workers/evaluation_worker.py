"""Background worker that scores a folder of images for calibration."""

from PySide6.QtCore import QThread, Signal

from services.patchcore_service import PatchCoreService


class EvaluationWorker(QThread):
    """Loads a recipe's PatchCore model and scores a folder of good images.

    Console output from the service follows the LogRedirector pattern and is
    handled by the owning page; this worker only emits the structured result.
    """

    scores_ready = Signal(list)  # list of (image_name, score) tuples
    error_signal = Signal(str)

    def __init__(self, recipe, good_folder):
        super().__init__()
        self.recipe = recipe
        self.good_folder = str(good_folder)

    def run(self):
        try:
            service = PatchCoreService()
            service.load_model(self.recipe["model"]["path"])
            scores = service.evaluate_folder(self.good_folder)
            self.scores_ready.emit(scores)
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
