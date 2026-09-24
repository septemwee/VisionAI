"""Standalone PatchCore inference used by the trainer's review page."""

from anomalib.data import PredictDataset
from anomalib.engine import Engine

from services.patchcore_service import IMAGE_SIZE, PatchCoreService, SCORE_SCALE
from services.verdict_service import evaluate_verdict, InspectionError


class InferenceService:
    """Loads a trained PatchCore model and scores single images."""

    def __init__(self):
        self.engine = Engine()
        self.model = None

    def load_model(self, recipe):
        """Load the PatchCore model referenced by a recipe.

        Delegates to ``PatchCoreService.load_model`` so the review path
        applies the same metadata shape/hash validation as live inspection;
        keeping one loader prevents the two paths from diverging.
        """
        service = PatchCoreService()
        service.load_model(recipe["model"]["path"])
        self.model = service.model

    def predict(self, image_path, anomaly_threshold=0.60, recipe=None):
        """Score one image and return the verdict as a dict.

        The verdict comes from the shared ``evaluate_verdict`` helper, so
        the review page applies the same rules — image threshold AND the
        localized-anomaly pixel gate — as live inspection.
        """
        try:
            if self.model is None:
                raise InspectionError("MODEL_ERROR", "PatchCore model is unavailable")
            predict_dataset = PredictDataset(path=image_path, image_size=IMAGE_SIZE)
            results = self.engine.predict(
                model=self.model, dataset=predict_dataset, return_predictions=True,
            )
            prediction = results[0]
            raw_score = float(prediction.pred_score[0]) / SCORE_SCALE
            anomaly_map = getattr(prediction, "anomaly_map", None)
            if anomaly_map is not None:
                anomaly_map = anomaly_map.cpu().numpy()
            is_defect, reason, gate_region_px = evaluate_verdict(
                raw_score, anomaly_map, anomaly_threshold, recipe or {}
            )
        except Exception as error:
            code = error.code if isinstance(error, InspectionError) else "INFERENCE_ERROR"
            return {"raw_score": None, "anomaly_threshold": anomaly_threshold,
                    "is_defect": None, "reason": str(error), "gate_region_px": 0,
                    "result": "ERROR", "error_code": code, "prediction": None}

        return {
            "raw_score": raw_score,
            "anomaly_threshold": anomaly_threshold,
            "is_defect": is_defect,
            "reason": reason,
            "gate_region_px": gate_region_px,
            "result": "FAIL" if is_defect else "PASS",
            "error_code": None,
            "prediction": prediction,
        }
