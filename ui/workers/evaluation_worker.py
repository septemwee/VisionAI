"""Background worker that scores a folder of images for calibration."""

from PySide6.QtCore import QThread, Signal

from services.patchcore_service import PatchCoreService
from services.synthetic_defect_service import generate_synthetic_validation
from tempfile import TemporaryDirectory
from copy import deepcopy
from pathlib import Path
from services.validation_service import (
    artifact_identity, audit_splits, calibrate_pixel_gate, summarize_pixel_test,
    summarize_test,
)
from services.training_assistant import recommend_threshold_with_anomalies


class EvaluationWorker(QThread):
    """Loads a recipe's PatchCore model and scores a folder of good images.

    Console output from the service follows the LogRedirector pattern and is
    handled by the owning page; this worker only emits the structured result.
    """

    scores_ready = Signal(object)
    error_signal = Signal(str)

    def __init__(self, recipe, good_folder):
        super().__init__()
        self.recipe = deepcopy(recipe)
        self.good_folder = str(good_folder)

    def run(self):
        try:
            dataset_root = Path(self.good_folder).parent.parent
            if Path(self.good_folder).parent.name != 'calibration':
                raise ValueError('Regenerate the dataset with an independent calibration split first.')
            audit = audit_splits(dataset_root)
            if self.recipe.get('training_dataset_audit') != audit:
                raise ValueError('The dataset is not the verified training dataset. Train again before validation.')
            identity = artifact_identity(self.recipe)
            service = PatchCoreService()
            service.load_model(self.recipe["model"]["path"])
            good_details = service.evaluate_folder_details(self.good_folder)
            good_scores = [(name, score) for name, score, _ in good_details]
            with TemporaryDirectory(prefix="visionai_synthetic_") as temporary:
                generate_synthetic_validation(self.good_folder, temporary)
                synthetic_details = service.evaluate_folder_details(temporary)
                synthetic_scores = [(name, score) for name, score, _ in synthetic_details]
            threshold, trace = recommend_threshold_with_anomalies(
                [score for _, score in good_scores], [score for _, score in synthetic_scores])
            gate = calibrate_pixel_gate(good_details, synthetic_details, threshold)
            # Select the threshold BEFORE opening the independent test images.
            test_good_details = service.evaluate_folder_details(dataset_root / 'test' / 'good')
            test_good = [(name, score) for name, score, _ in test_good_details]
            with TemporaryDirectory(prefix='visionai_test_') as temporary:
                generate_synthetic_validation(dataset_root / 'test' / 'good', temporary)
                test_synthetic_details = service.evaluate_folder_details(temporary)
                test_synthetic = [(name, score) for name, score, _ in test_synthetic_details]
            if identity != artifact_identity(self.recipe) or audit != audit_splits(dataset_root):
                raise ValueError('Model or dataset changed during validation. Run validation again.')
            report = {
                'model_identity': identity, 'dataset_audit': audit,
                'threshold': threshold,
                'good': summarize_test(test_good, threshold),
                'synthetic': summarize_test(test_synthetic, threshold),
                'pixel_gate': {
                    'config': {key: gate[key] for key in ('enabled', 'threshold_ratio', 'min_area_px', 'max_area_px')},
                    'calibration': gate,
                    'good': summarize_pixel_test(test_good_details, threshold, gate),
                    'synthetic': summarize_pixel_test(test_synthetic_details, threshold, gate),
                },
                'evidence': 'synthetic_only',
            }
            self.scores_ready.emit({"good": good_scores, "synthetic": synthetic_scores,
                                   'report': report, 'recipe_name': self.recipe['recipe_name']})
        except Exception as error:  # Reported to the UI via the error signal.
            self.error_signal.emit(str(error))
