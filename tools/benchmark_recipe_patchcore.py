"""Read-only recipe experiment; writes artifacts only under the chosen output."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from services.validation_service import audit_splits
from services.synthetic_defect_service import generate_synthetic_validation
from services.training_assistant import recommend_threshold_with_anomalies
from utils.paths import BACKBONE_WEIGHTS_PATH, resolve_recipe_path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="recipes/TJA1041_SO14/recipe.json")
    parser.add_argument("--out", default="reports/patchcore_experiment_20260921")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    root = resolve_recipe_path(recipe["prepared_dataset_path"])
    before = audit_splits(root)
    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    np.random.seed(20260921)
    print("audit", before, flush=True)
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    from anomalib.models import Patchcore
    from sklearn.metrics import roc_auc_score
    print("imports complete", flush=True)
    model = PatchcoreModel(layers=["layer2", "layer3"], pre_trained=False).eval()
    weights = load_file(str(BACKBONE_WEIGHTS_PATH))
    mismatch = model.feature_extractor.feature_extractor.load_state_dict(weights, strict=False)
    del weights
    print("backbone loaded", mismatch, flush=True)
    sources = {role: sorted((root/role/"good").glob("*.jpg")) for role in before["counts"]}
    # Match the audit's supported extensions, not just JPEG.
    for role in sources:
        sources[role] = sorted(p for p in (root/role/"good").rglob("*")
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    examples = {}
    for role in ("calibration", "test"):
        synthetic = out / ("synthetic_" + role)
        manifest = generate_synthetic_validation(root/role/"good", synthetic)
        examples[role] = [(p, "good") for p in sources[role]] + [
            (synthetic/item["file"], item["type"]) for item in manifest]
    results = []
    banks = {}
    for size in (256, 384, 512):
        transform = Patchcore.configure_pre_processor(image_size=(size, size)).transform

        def tensor(image):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()/255
            return transform(x).unsqueeze(0)

        def embedding(image):
            features = model.feature_extractor(tensor(image))
            features = {k: model.feature_pooler(v) for k, v in features.items()}
            merged = model.generate_embedding(features)
            return model.reshape_embedding(merged)

        with torch.inference_mode():
            started = time.perf_counter()
            pool = []
            total_patches = 0
            generator = torch.Generator().manual_seed(20260921)
            for index, path in enumerate(sources["train"]):
                emb = embedding(cv2.imread(str(path)))
                total_patches += len(emb)
                selected = torch.randperm(len(emb), generator=generator)[:128]
                pool.append(emb[selected].clone())
                if index % 20 == 0:
                    print(f"size={size} train={index}/{len(sources['train'])}", flush=True)
            pool = torch.cat(pool)
            # Explicit resource-bounded two-stage coreset: stratified random
            # candidates per training image, then projected farthest-first.
            projection = torch.randn(pool.shape[1], 32, generator=generator)/np.sqrt(32)
            projected = pool @ projection
            distance = torch.full((len(pool),), float("inf"))
            chosen = []
            candidate = 0
            for _ in range(2048):
                chosen.append(candidate)
                distance = torch.minimum(distance, (projected-projected[candidate]).square().sum(1))
                distance[chosen] = -1
                candidate = int(distance.argmax())
            for count in (512, 2048):
                banks[count] = pool[chosen[:count]].clone()
                torch.save(banks[count], out/f"bank_{size}_{count}.pt")
            train_seconds = time.perf_counter()-started
            del pool, projected, projection
            print(f"size={size} bank ready train_seconds={train_seconds:.2f}", flush=True)
            configs = {(count, k): dict(size=size, bank=count, neighbors=k,
                train_seconds=train_seconds, candidate_patches=128*len(sources['train']),
                total_train_patches=total_patches, rows=[]) for count in banks for k in (1,9)}
            for role in ("calibration", "test"):
                # Test is scored only after each candidate's calibration threshold is frozen.
                for index, (path, kind) in enumerate(examples[role]):
                    image = cv2.imread(str(path))
                    started = time.perf_counter()
                    emb = embedding(image)
                    feature_ms = (time.perf_counter()-started)*1000
                    for count, bank in banks.items():
                        model.memory_bank = bank
                        started = time.perf_counter()
                        scores, locations = model.nearest_neighbors(emb, 1)
                        search_ms = (time.perf_counter()-started)*1000
                        for k in (1,9):
                            model.num_neighbors = k
                            started = time.perf_counter()
                            score = float(model.compute_anomaly_score(scores[None], locations[None], emb)[0])/100
                            configs[count,k]["rows"].append(dict(role=role, kind=kind, file=str(path),
                                score=score, feature_ms=feature_ms, search_ms=search_ms,
                                score_ms=(time.perf_counter()-started)*1000))
                    if index % 25 == 0:
                        print(f"size={size} {role}={index}/{len(examples[role])}", flush=True)
                if role == "calibration":
                    for config in configs.values():
                        rows = config["rows"]
                        threshold, trace = recommend_threshold_with_anomalies(
                            [r['score'] for r in rows if r['kind']=='good'],
                            [r['score'] for r in rows if r['kind']!='good'])
                        config.update(threshold=threshold, calibration=trace)
                    dump(out/f"frozen_calibration_{size}.json", list(configs.values()))
            for config in configs.values():
                rows = [r for r in config['rows'] if r['role']=='test']
                config['test_auc'] = roc_auc_score([r['kind']!='good' for r in rows], [r['score'] for r in rows])
                config['test'] = {kind: dict(count=sum(r['kind']==kind for r in rows),
                    above=sum(r['kind']==kind and r['score']>config['threshold'] for r in rows))
                    for kind in sorted({r['kind'] for r in rows})}
                # Independent end-to-end single-ROI timings, including map.
                model.memory_bank = banks[config['bank']]
                model.num_neighbors = config['neighbors']
                latency = []
                for path in sources['test'][:8]:
                    image = cv2.imread(str(path))
                    started = time.perf_counter()
                    prediction = model(tensor(image))
                    prediction.anomaly_map.cpu().numpy()
                    latency.append((time.perf_counter()-started)*1000)
                config['latency_median_ms'] = float(np.median(latency))
                config['latency_p95_ms'] = float(np.percentile(latency,95))
                config['latency_samples_ms'] = latency
                results.append(config)
            dump(out/"results.json", dict(audit=before, torch=torch.__version__,
                threads=torch.get_num_threads(), seed=20260921, results=results))
            print(f"size={size} COMPLETE", flush=True)
    if audit_splits(root) != before:
        raise RuntimeError("Dataset changed during experiment")
    print("EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
