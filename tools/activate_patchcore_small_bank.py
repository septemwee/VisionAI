"""Create, validate, and activate a calibrated 512-vector PatchCore bank.

The active recipe changes only after independent synthetic acceptance passes.
The original model directory is never modified.
"""
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from services.patchcore_service import PatchCoreService
from services.recipe_service import RecipeService
from services.synthetic_defect_service import generate_synthetic_validation
from services.training_assistant import recommend_threshold_with_anomalies
from services.validation_service import (
    artifact_identity,
    audit_splits,
    calibrate_pixel_gate,
    check_acceptance,
    summarize_pixel_test,
    summarize_test,
)
from utils.image_utils import unwrap_anomaly_map
from utils.paths import relativize_recipe_path, resolve_recipe_path


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes/TJA1041_SO14/recipe.json"
REPORT_DIR = ROOT / "reports/patchcore_experiment_20260921"
BANK_PATH = REPORT_DIR / "bank_512_512.pt"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_folder(service, folder, label):
    paths = sorted(path for path in Path(folder).rglob("*")
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    details = []
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        score, anomaly_map = service.predict_full(image)
        amap = unwrap_anomaly_map(anomaly_map)
        if (amap is None or amap.ndim != 2 or not amap.size
                or not np.isfinite(amap).all() or not np.isfinite(score)):
            raise ValueError(f"Invalid PatchCore output: {path}")
        details.append((path.name, float(score), amap.copy()))
        if index % 20 == 0 or index == len(paths):
            print(f"{label}: {index}/{len(paths)}", flush=True)
    return details


def create_candidate(source_dir):
    target = source_dir.parent / uuid4().hex
    target.mkdir(parents=True, exist_ok=False)
    # The PatchCore state is immutable. A hard link prevents a needless 301 MB
    # duplicate while the candidate gets its own independently hashed bank.
    os.link(source_dir / "patchcore.pt", target / "patchcore.pt")
    bank = torch.load(BANK_PATH, map_location="cpu", weights_only=True)
    torch.save(bank, target / "memory_bank.pt")
    metadata = {
        "model_type": "patchcore",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "image_size": [512, 512],
        "num_neighbors": 1,
        "memory_bank_shape": list(bank.shape),
        "memory_bank_method": "two_stage_projected_farthest_first_v1",
        "candidate_source": str(source_dir),
        "candidate_bank_source": str(BANK_PATH.relative_to(ROOT)),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "memory_bank_sha256": sha256(target / "memory_bank.pt"),
        "patchcore_sha256": sha256(target / "patchcore.pt"),
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def main():
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    dataset_root = resolve_recipe_path(recipe["prepared_dataset_path"])
    before = audit_splits(dataset_root)
    if recipe.get("training_dataset_audit") != before:
        raise ValueError("Recipe dataset audit is stale; prepare/train before activating a candidate.")
    source_dir = resolve_recipe_path(recipe["model"]["path"])
    if not BANK_PATH.is_file():
        raise FileNotFoundError(BANK_PATH)
    torch.set_num_threads(4)
    candidate_dir = create_candidate(source_dir)
    print(f"candidate: {candidate_dir.name}", flush=True)
    service = PatchCoreService()
    try:
        service.load_model(candidate_dir)
        service.model.model.num_neighbors = 1
        calibration_good = score_folder(service, dataset_root / "calibration/good", "calibration good")
        with TemporaryDirectory(prefix="visionai_candidate_cal_") as temporary:
            generate_synthetic_validation(dataset_root / "calibration/good", temporary)
            calibration_synthetic = score_folder(service, temporary, "calibration synthetic")
        threshold, calibration = recommend_threshold_with_anomalies(
            [score for _, score, _ in calibration_good],
            [score for _, score, _ in calibration_synthetic],
        )
        gate = calibrate_pixel_gate(calibration_good, calibration_synthetic, threshold)
        test_good = score_folder(service, dataset_root / "test/good", "test good")
        with TemporaryDirectory(prefix="visionai_candidate_test_") as temporary:
            generate_synthetic_validation(dataset_root / "test/good", temporary)
            test_synthetic = score_folder(service, temporary, "test synthetic")
        candidate_recipe = copy.deepcopy(recipe)
        candidate_recipe["model"] = {
            "trained": True,
            "type": "patchcore",
            "backbone": "wide_resnet50_2",
            "path": relativize_recipe_path(candidate_dir),
        }
        candidate_recipe["anomaly_threshold"] = threshold
        candidate_recipe["pixel_gate"] = {key: gate[key] for key in (
            "enabled", "threshold_ratio", "min_area_px", "max_area_px")}
        candidate_recipe["verdict_policy"] = "image_and_pixel_v1"
        candidate_recipe["calibration"] = dict(
            calibration,
            good_set_role="held_out_calibration",
            evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pixel_gate=candidate_recipe["pixel_gate"],
            pixel_gate_metrics=gate,
        )
        candidate_recipe["acceptance_report"] = {
            "model_identity": artifact_identity(candidate_recipe),
            "dataset_audit": before,
            "threshold": threshold,
            "good": summarize_test([(name, score) for name, score, _ in test_good], threshold),
            "synthetic": summarize_test([(name, score) for name, score, _ in test_synthetic], threshold),
            "pixel_gate": {
                "config": candidate_recipe["pixel_gate"],
                "calibration": gate,
                "good": summarize_pixel_test(test_good, threshold, gate),
                "synthetic": summarize_pixel_test(test_synthetic, threshold, gate),
            },
            "evidence": "synthetic_only",
        }
        candidate_recipe["validation_scope"] = "held_out_good_and_synthetic"
        candidate_recipe["training_dataset_audit"] = before
        candidate_recipe["validated"] = True
        candidate_recipe["last_trained"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        candidate_recipe["version"] = "1.6"
        if audit_splits(dataset_root) != before:
            raise ValueError("Dataset changed during candidate validation.")
        check_acceptance(candidate_recipe)
        RecipeService().save_recipe(candidate_recipe["recipe_name"], candidate_recipe)
        report = {"activated": True, "candidate_dir": str(candidate_dir),
                  "threshold": threshold, "pixel_gate": gate,
                  "acceptance": candidate_recipe["acceptance_report"]}
        print("CANDIDATE ACTIVATED", flush=True)
    except Exception as error:
        report = {"activated": False, "candidate_dir": str(candidate_dir), "error": str(error)}
        print(f"CANDIDATE NOT ACTIVATED: {error}", flush=True)
        raise
    finally:
        (REPORT_DIR / "small_bank_activation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
