"""Training and inference wrapper around the Anomalib PatchCore model."""

import json
import tempfile
from pathlib import Path

import cv2
import torch
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from safetensors.torch import load_file

from services.recipe_service import RecipeService
from utils.paths import BACKBONE_WEIGHTS_PATH, RECIPES_DIR

IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"


class PatchCoreService:
    """Trains, saves, loads and runs PatchCore anomaly-detection models.

    A trained model is stored in the recipe directory as ``patchcore.pt``
    (model weights) plus ``memory_bank.pt`` (the PatchCore memory bank) and a
    small ``metadata.json`` descriptor.
    """

    def __init__(self):
        self.recipe_service = RecipeService()
        self.model = None
        self.engine = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, recipe):
        """Train a PatchCore model for the recipe and store it on disk."""
        dataset_path = Path(recipe["prepared_dataset_path"])

        datamodule = Folder(
            name="IC",
            root=str(dataset_path),
            normal_dir="train/good",
            num_workers=0,
        )

        model = Patchcore(
            backbone=BACKBONE,
            pre_trained=False,
            pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
            visualizer=False,
        )

        self._load_local_backbone(model)

        engine = Engine()
        engine.fit(model=model, datamodule=datamodule)

        self.model = model
        self.engine = engine
        print("TRAINING COMPLETE")

        model_dir = RECIPES_DIR / recipe["recipe_name"] / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        self.save_model(model, model_dir)
        self.update_recipe(recipe, model_dir)

        return model_dir

    @staticmethod
    def _load_local_backbone(model):
        """Load backbone weights from the bundled safetensors file."""
        state_dict = load_file(str(BACKBONE_WEIGHTS_PATH))
        backbone = model.model.feature_extractor.feature_extractor
        backbone.load_state_dict(state_dict, strict=False)
        print("LOCAL BACKBONE LOADED")

    def save_model(self, model, model_dir):
        """Persist the model in the format expected by ``load_model``."""
        model_dir = Path(model_dir)

        torch.save(model.model.memory_bank, model_dir / "memory_bank.pt")
        torch.save(model.state_dict(), model_dir / "patchcore.pt")

        metadata = {
            "model_type": "patchcore",
            "backbone": BACKBONE,
            "memory_bank_shape": list(model.model.memory_bank.shape),
        }

        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print("MODEL SAVED")

    def update_recipe(self, recipe, model_dir):
        """Record the trained model location in the recipe file."""
        recipe["model"] = {
            "trained": True,
            "type": "patchcore",
            "backbone": BACKBONE,
            "path": str(model_dir),
        }

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
        print("RECIPE UPDATED")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_model(self, model_path):
        """Load a trained model from a recipe model directory."""
        model_path = Path(model_path)

        model = Patchcore(
            backbone=BACKBONE,
            pre_trained=False,
            post_processor=False,
            visualizer=False,
        )

        state_dict = torch.load(model_path / "patchcore.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        model.model.memory_bank = torch.load(
            model_path / "memory_bank.pt", map_location="cpu"
        )

        model.eval()

        self.model = model
        self.engine = Engine()
        print("PATCHCORE LOADED")

    def predict(self, image):
        """Return the anomaly score for a single BGR image."""
        if self.model is None:
            raise RuntimeError("No PatchCore model is loaded.")

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            cv2.imwrite(temp_path, image)

            predict_dataset = PredictDataset(path=temp_path, image_size=IMAGE_SIZE)
            predictions = self.engine.predict(model=self.model, dataset=predict_dataset)

            score = float(predictions[0].pred_score[0])
            print(f"[PATCHCORE] score={score:.4f}")
            return score
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def evaluate_folder(self, folder_path):
        """Score every image in a folder and print summary statistics."""
        predict_dataset = PredictDataset(path=str(folder_path))
        predictions = self.engine.predict(model=self.model, dataset=predict_dataset)

        scores = []
        for prediction in predictions:
            score = float(prediction.pred_score[0])
            image_name = Path(prediction.image_path[0]).name
            print(f"{image_name} {score:.4f}")
            scores.append(score)

        if not scores:
            print("No images were evaluated.")
            return

        print(f"\nMIN: {min(scores)}")
        print(f"MAX: {max(scores)}")
        print(f"AVG: {sum(scores) / len(scores)}")
