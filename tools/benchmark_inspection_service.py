"""Measure the real service and rendering on prepared recipe crops.

No recipe/model artifacts are overwritten. Camera/YOLO acquisition is excluded.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from services.patchcore_service import PatchCoreService
from utils.paths import resolve_recipe_path
from utils.image_utils import heatmap_overlay


def main():
    out = Path("reports/patchcore_experiment_20260921")
    recipe = json.loads(Path("recipes/TJA1041_SO14/recipe.json").read_text())
    sweep = json.loads((out / "results.json").read_text())
    candidate = next(c for c in sweep["results"]
                     if (c["size"], c["bank"], c["neighbors"]) == (512, 512, 1))
    images = [cv2.imread(r["file"]) for r in candidate["rows"]
              if r["role"] == "test" and r["kind"] == "good"][:4]
    torch.set_num_threads(4)
    service = PatchCoreService()
    started = time.perf_counter()
    service.load_model(recipe["model"]["path"])
    load_seconds = time.perf_counter() - started
    original_bank = service.model.model.memory_bank
    original_neighbors = service.model.model.num_neighbors
    small_bank = torch.load(out / "bank_512_512.pt", weights_only=True, map_location="cpu")
    results = []
    try:
        for threads in (4, 8):
            torch.set_num_threads(threads)
            for label, bank, neighbors, threshold in (
                ("baseline", original_bank, original_neighbors, recipe["anomaly_threshold"]),
                ("candidate", small_bank, 1, candidate["threshold"]),
            ):
                service.model.model.memory_bank = bank
                service.model.model.num_neighbors = neighbors
                service.predict_full(images[0])
                samples = []
                for img in images:
                    start = time.perf_counter()
                    score, amap = service.predict_full(img)
                    inference_ms = (time.perf_counter() - start) * 1000
                    heatmap = heatmap_overlay(img, amap)
                    assert heatmap.shape == img.shape
                    samples.append(dict(inference_ms=inference_ms,
                                        total_ms=(time.perf_counter()-start)*1000,
                                        score=score, image_fail=score > threshold))
                row = dict(model=label, threads=threads, bank=len(bank), samples=samples,
                           median_ms=float(np.median([s["total_ms"] for s in samples])))
                results.append(row)
                print(row, flush=True)
                (out / "inspection_service_runtime.json").write_text(json.dumps(
                    dict(load_seconds=load_seconds, input_size=512,
                         scope="prepared crops; predict_full + heatmap; excludes capture, YOLO, marking, segments and Qt paint",
                         candidate_pixel_gate_calibrated=False, results=results), indent=2))
    finally:
        service.model.model.memory_bank = original_bank
        service.model.model.num_neighbors = original_neighbors
    print("COMPLETE; original model restored in process; no recipe writes", flush=True)


if __name__ == "__main__":
    main()
