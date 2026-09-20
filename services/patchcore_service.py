"""Training and inference wrapper around the Anomalib PatchCore model."""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from safetensors.torch import load_file

from services.recipe_service import RecipeService
from utils.paths import BACKBONE_WEIGHTS_PATH, RECIPES_DIR, relativize_recipe_path, resolve_recipe_path

IMAGE_SIZE = (256, 256)
BACKBONE = "wide_resnet50_2"

Folder = None
PredictDataset = None
Engine = None
Patchcore = None

# Anomalib's raw pred_score (post_processor=False) is a nearest-neighbour
# distance that lands on a ~0-100 scale in this setup. All scores exposed by
# this service are divided by SCORE_SCALE so the app works with 0-1
# fractions, matching the anomaly_threshold convention.
SCORE_SCALE = 100.0

# Display-only EMA factor over consecutive anomaly maps: the fraction of
# the PREVIOUS map kept in the blended result. Verdict logic deliberately
# uses the raw map.
ANOMALY_MAP_SMOOTHING = 0.6

def _import_anomalib():
    global Folder
    global PredictDataset
    global Engine
    global Patchcore

    if Patchcore is not None:
        return

    from anomalib.models import Patchcore as _Patchcore

    Patchcore = _Patchcore


def _sha256_of(path):
    """SHA-256 hex digest of a file; None when the file is unreadable."""
    import hashlib

    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


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
        self._smoothed_anomaly_map = None
        self._model_cache = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, recipe):
        _import_anomalib()
        from anomalib.data import Folder
        from anomalib.engine import Engine

        """Train a PatchCore model for the recipe and store it on disk."""
        dataset_path = resolve_recipe_path(recipe["prepared_dataset_path"])
        from services.validation_service import audit_splits
        training_audit = audit_splits(dataset_path)

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

        name = RecipeService._validate_recipe_name(recipe["recipe_name"])
        from uuid import uuid4
        model_dir = RECIPES_DIR / name / 'model_versions' / uuid4().hex
        model_dir.mkdir(parents=True, exist_ok=True)

        self.save_model(model, model_dir)
        if training_audit != audit_splits(dataset_path):
            raise ValueError('Dataset changed during training. Prepare data and train again.')
        recipe['training_dataset_audit'] = training_audit
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
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "memory_bank_sha256": _sha256_of(model_dir / "memory_bank.pt"),
            "patchcore_sha256": _sha256_of(model_dir / "patchcore.pt"),
        }

        with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print("MODEL SAVED")

    def update_recipe(self, recipe, model_dir):
        """Record the trained model location in the recipe file."""
        from services.validation_service import invalidate_validation
        invalidate_validation(recipe)
        recipe["model"] = {
            "trained": True,
            "type": "patchcore",
            "backbone": BACKBONE,
            "path": relativize_recipe_path(model_dir),
        }

        self.recipe_service.save_recipe(recipe["recipe_name"], recipe)
        print("RECIPE UPDATED")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_model(self, model_path):
        _import_anomalib()
        """Load a trained model from a recipe model directory."""
        model_path = resolve_recipe_path(model_path)
        if model_path is None:
            raise FileNotFoundError("No PatchCore model path was provided.")

        cache_key = str(model_path.resolve())
        cached = self._model_cache.get(cache_key)
        if cached is not None:
            self.model = cached
            self.engine = None
            self._smoothed_anomaly_map = None
            print("PATCHCORE CACHE HIT")
            return

        model = Patchcore(
            backbone=BACKBONE,
            pre_trained=False,
            post_processor=False,
            pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
            visualizer=False,
        )

        state_dict = torch.load(
            model_path / "patchcore.pt", map_location="cpu", weights_only=True
        )
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"[PATCHCORE] State dict mismatch "
                f"(missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)})"
            )

        memory_bank = torch.load(
            model_path / "memory_bank.pt", map_location="cpu", weights_only=True
        )

        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, ValueError):
                metadata = {}

            expected_shape = metadata.get("memory_bank_shape")
            if expected_shape and list(memory_bank.shape) != list(expected_shape):
                raise ValueError(
                    f"Memory bank shape {list(memory_bank.shape)} does not match "
                    f"the recorded shape {list(expected_shape)} in metadata.json."
                )

        model.model.memory_bank = memory_bank

        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, ValueError):
                metadata = {}

            expected_shape = metadata.get("memory_bank_shape")
            if expected_shape and list(memory_bank.shape) != list(expected_shape):
                raise ValueError(
                    f"Memory bank shape {list(memory_bank.shape)} does not match "
                    f"the recorded shape {list(expected_shape)} in metadata.json."
                )

            # Integrity: verify the recorded hashes when present so a stale
            # or tampered artifact cannot load silently. Metadata written by
            # an older save_model has no hashes; that only warns, while a
            # MISMATCH always fails the load.
            for field, filename in (
                ("memory_bank_sha256", "memory_bank.pt"),
                    ("patchcore_sha256", "patchcore.pt"),
                ):
                expected_hash = metadata.get(field)
                if expected_hash is None:
                    print(
                        f"[PATCHCORE] metadata.json has no {field} — "
                        f"skipping integrity check for {filename}"
                    )
                    continue
                if os.environ.get("VISIONAI_VERIFY_MODEL", "0") == "1":
                    actual = _sha256_of(model_path / filename)
                    if actual != expected_hash:
                        raise ValueError(
                            f"{filename} integrity check failed: metadata records "
                            f"{expected_hash} but the file hashes to {actual}."
                        )

        model.model.memory_bank = memory_bank

        model.eval()

        self.model = model
        self._model_cache[cache_key] = model
        self.engine = None
        self._smoothed_anomaly_map = None
        print("PATCHCORE LOADED")

    def _predict_single(self, image):
        """Run PatchCore on one BGR image and return the raw prediction.

        The image is pre-processed in memory with the model's own transform
        (the same Compose the predict-time callback applies), so the live
        tick avoids the per-frame JPEG temp-file round-trip and the per-tick
        dataset/engine rebuild.
        """
        if self.model is None:
            raise RuntimeError("No PatchCore model is loaded.")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        )
        pre_processed = self.model.pre_processor.transform(tensor)
        return self.model.model(pre_processed.unsqueeze(0))

    def predict(self, image):
        """Return the anomaly score for a single BGR image as a 0-1 fraction."""
        prediction = self._predict_single(image)

        score = float(prediction.pred_score[0]) / SCORE_SCALE
        print(f"[PATCHCORE] score={score:.4f}")
        return score

    def predict_full(self, image):
        """Return ``(score, raw_anomaly_map)`` for one BGR image.

        ``anomaly_map`` is the raw numpy map (possibly ``(1, 1, H, W)``) for
        verdict logic (pixel gate) and for display mapping; any heatmap
        rendering happens in the callers.
        """
        # Inspection is inference-only. Explicitly disabling autograd avoids
        # retaining tensors and cuts CPU/memory overhead on every ROI.
        with torch.inference_mode():
            prediction = self._predict_single(image)

        score = float(prediction.pred_score[0]) / SCORE_SCALE

        anomaly_map = getattr(prediction, "anomaly_map", None)
        if anomaly_map is not None:
            anomaly_map = anomaly_map.cpu().numpy()

        print(f"[PATCHCORE] score={score:.4f} map={anomaly_map is not None}")
        return score, anomaly_map

    def predict_full_batch(self, images):
        """Return scores and anomaly maps for several BGR images in one pass."""
        if not images:
            return [], []
        if self.model is None:
            raise RuntimeError("No PatchCore model is loaded.")

        tensors = []
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
            tensors.append(self.model.pre_processor.transform(tensor))

        with torch.inference_mode():
            prediction = self.model.model(torch.stack(tensors, dim=0))
        scores = [float(value) / SCORE_SCALE for value in prediction.pred_score]
        maps = getattr(prediction, "anomaly_map", None)
        if maps is None:
            anomaly_maps = [None] * len(images)
        else:
            maps = maps.cpu().numpy()
            anomaly_maps = [maps[index:index + 1] for index in range(len(images))]

        return scores, anomaly_maps

    def display_anomaly_map(self, anomaly_map):
        """Return the EMA-smoothed copy of a map for DISPLAY purposes.

        Consumes one smoothing step per call; verdict logic deliberately
        keeps the raw map.
        """
        if anomaly_map is None:
            return None
        return self._smooth_anomaly_map(anomaly_map)

    def _smooth_anomaly_map(self, anomaly_map):
        """Exponentially smooth consecutive anomaly maps for display.

        PatchCore distances are noisy at the patch level, so an unsmoothed
        map makes the heatmap's hot region wander between frames even on a
        static scene. The EMA keeps the visualization stable; verdict logic
        deliberately keeps the raw map. State resets when the map shape
        changes or a new model is loaded.
        """
        previous = self._smoothed_anomaly_map
        if previous is None or previous.shape != anomaly_map.shape:
            self._smoothed_anomaly_map = anomaly_map.copy()
            return self._smoothed_anomaly_map

        smoothed = (
            ANOMALY_MAP_SMOOTHING * previous
            + (1.0 - ANOMALY_MAP_SMOOTHING) * anomaly_map
        )
        self._smoothed_anomaly_map = smoothed
        return smoothed

    def evaluate_folder(self, folder_path):
        """Score every image in a folder.

        Prints the per-image scores and summary statistics, and RETURNS the
        per-image ``(name, score)`` pairs (0-1 fractions) so calibration and
        other tooling can consume them.
        """
        return [(name, score) for name, score, _ in self.evaluate_folder_details(folder_path)]

    def evaluate_folder_details(self, folder_path):
        """Score every image and retain its raw anomaly map for calibration."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(str(folder))
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in
                       {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not paths:
            raise ValueError("No calibration images found")
        scored = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable calibration image: {path}")
            score, anomaly_map = self.predict_full(image)
            from utils.pixel_gate import unwrap_anomaly_map
            import numpy as np
            amap = unwrap_anomaly_map(anomaly_map)
            if not math.isfinite(score) or amap is None or amap.ndim != 2 or not amap.size or not np.isfinite(amap).all():
                raise ValueError(f"Invalid model output: {path}")
            image_name = str(path.relative_to(folder))
            print(f"{image_name} {score:.4f}")
            scored.append((image_name, score, amap.copy()))

        if not scored:
            print("No images were evaluated.")
            return []

        scores = [score for _, score, _ in scored]
        print(f"\nMIN: {min(scores)}")
        print(f"MAX: {max(scores)}")
        print(f"AVG: {sum(scores) / len(scores)}")
        return scored
