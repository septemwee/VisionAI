"""Standalone PatchCore inference used by the trainer's review page."""

from pathlib import Path

import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from safetensors.torch import load_file

from utils.paths import BACKBONE_WEIGHTS_PATH

IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"


class InferenceService:
    """Loads a trained PatchCore model and scores single images."""

    def __init__(self):
        self.engine = Engine()
        self.model = None

    def load_model(self, recipe):
        """Load the PatchCore model referenced by a recipe."""
        model = Patchcore(
            backbone=BACKBONE,
            pre_trained=False,
            post_processor=False,
            pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
            visualizer=False,
        )

        state_dict = load_file(str(BACKBONE_WEIGHTS_PATH))
        backbone = model.model.feature_extractor.feature_extractor
        backbone.load_state_dict(state_dict, strict=False)

        memory_bank = torch.load(
            Path(recipe["model"]["path"]) / "memory_bank.pt", map_location="cpu"
        )
        model.model.memory_bank = memory_bank

        self.model = model

    def predict(self, image_path, anomaly_threshold=0.60):
        """Score one image and return the verdict as a dict."""
        predict_dataset = PredictDataset(path=image_path, image_size=IMAGE_SIZE)

        results = self.engine.predict(
            model=self.model,
            dataset=predict_dataset,
            return_predictions=True,
        )

        prediction = results[0]
        raw_score = float(prediction.pred_score[0])
        is_defect = raw_score > anomaly_threshold

        return {
            "raw_score": raw_score,
            "anomaly_threshold": anomaly_threshold,
            "is_defect": is_defect,
            "result": "FAIL" if is_defect else "PASS",
            "prediction": prediction,
        }
