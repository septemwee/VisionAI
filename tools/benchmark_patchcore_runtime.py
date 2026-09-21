"""Benchmark saved baseline and calibration-selected candidate, without writes to recipes."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from utils.paths import resolve_recipe_path

out = Path('reports/patchcore_experiment_20260921')
data = json.loads((out/'results.json').read_text())
configs = data['results']
feasible = [c for c in configs if c['calibration']['synthetic_detection_rate'] >= .8
            and c['calibration']['expected_false_alarms'] == 0]
selected = min(feasible, key=lambda c:c['latency_median_ms']) if feasible else max(
    configs, key=lambda c:(c['calibration']['synthetic_detection_rate'], -c['latency_median_ms']))
recipe = json.loads(Path('recipes/TJA1041_SO14/recipe.json').read_text())
folder = resolve_recipe_path(recipe['model']['path'])
torch.set_num_threads(4)
model = PatchcoreModel(layers=['layer2','layer3'], pre_trained=False).eval()
state = torch.load(folder/'patchcore.pt', map_location='cpu', weights_only=True, mmap=True)
bank = torch.load(folder/'memory_bank.pt', map_location='cpu', weights_only=True, mmap=True)
model.memory_bank = bank
keys = {k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')}
loaded = model.load_state_dict(keys, strict=False)
if loaded.missing_keys:
    raise RuntimeError(str(loaded))
from safetensors.torch import load_file
from utils.paths import BACKBONE_WEIGHTS_PATH
local = load_file(str(BACKBONE_WEIGHTS_PATH))
backbone = model.feature_extractor.feature_extractor.state_dict()
if not all(k in local and torch.equal(value, local[k]) for k,value in backbone.items()):
    raise RuntimeError('Saved baseline backbone differs from experimental backbone')
del local, state, keys
paths = [Path(r['file']) for r in selected['rows'] if r['role']=='test' and r['kind']=='good'][:8]
images = [cv2.imread(str(p)) for p in paths]
measurements = []
baseline_test = []
for label, size, memory, neighbors in (
    ('saved_baseline',512,bank,9),
    ('calibration_selected',selected['size'],torch.load(out/f"bank_{selected['size']}_{selected['bank']}.pt",
                                                    weights_only=True),selected['neighbors'])):
    model.memory_bank = memory
    model.num_neighbors = neighbors
    transform = Patchcore.configure_pre_processor(image_size=(size,size)).transform
    for threads in (2,4,8):
        torch.set_num_threads(threads)
        times, scores = [], []
        with torch.inference_mode():
            for image in [images[0]] + images:
                started = time.perf_counter()
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2,0,1).contiguous().float()/255
                prediction = model(transform(tensor)[None])
                score = float(prediction.pred_score[0])/100
                prediction.anomaly_map.cpu().numpy()
                times.append((time.perf_counter()-started)*1000)
                scores.append(score)
        measurements.append(dict(model=label,size=size,bank=len(memory),neighbors=neighbors,threads=threads,
             median_ms=float(np.median(times[1:])),p95_ms=float(np.percentile(times[1:],95)),
             latency_samples_ms=times[1:],scores=scores[1:]))
        print(measurements[-1],flush=True)
        (out/'runtime.json').write_text(json.dumps(dict(selected={k:selected[k] for k in ('size','bank','neighbors')},
            measurements=measurements, baseline_threshold=recipe['anomaly_threshold'], baseline_test=baseline_test),indent=2))
    if label == 'saved_baseline':
        from services.verdict_service import evaluate_verdict
        torch.set_num_threads(4)
        with torch.inference_mode():
            rows = [r for r in selected['rows'] if r['role']=='test']
            for index,row in enumerate(rows):
                image = cv2.imread(row['file'])
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2,0,1).contiguous().float()/255
                prediction = model(transform(tensor)[None])
                score = float(prediction.pred_score[0])/100
                fail, reason, _ = evaluate_verdict(score, prediction.anomaly_map.cpu().numpy(),
                                                  recipe['anomaly_threshold'], recipe)
                baseline_test.append(dict(file=row['file'],kind=row['kind'],score=score,
                    image_fail=score>recipe['anomaly_threshold'],full_verdict_fail=fail,reason=reason))
                if index%20==0:
                    print('baseline test',index,len(rows),flush=True)
        (out/'runtime.json').write_text(json.dumps(dict(selected={k:selected[k] for k in ('size','bank','neighbors')},
            measurements=measurements, baseline_threshold=recipe['anomaly_threshold'],baseline_test=baseline_test),indent=2))
