# VisionAI Code Audit — Problem List & Fix Plan

Analysis + planning only (no code changed yet). Scope reviewed: `main.py`,
`trainer_main.py`, `services/*`, `ui/**`, `utils/*`, `trainers/*`,
`config/model_registry.json`, `recipes/TJA1041_SO14/recipe.json`.

---

## Resolved design decisions (user-confirmed)

1. **Threshold semantics (C2):** `anomaly_threshold` is a **0–1 fraction**
   compared directly against the raw PatchCore score. Migrate the existing
   recipe value `60` → `0.60`. Unify all defaults to `0.60`.
2. **Lost target behavior (C5):** during the 1.5 s detection-memory window,
   show the last boxes/verdict but **do not re-inspect** stale crops. After
   timeout → `NOT FOUND`.
3. **Performance (H1):** full refactor — pipeline moves to a **QThread
   worker** with signal-based results, and `PatchCoreService.predict` is
   replaced by **direct tensor inference** (no temp JPEG, no per-call
   `PredictDataset`).

## Default decisions (documented, low risk — challenge if wrong)

- **Multi-box scenes:** inspect only the highest-confidence fresh box
  (recipe `target_count: 1`); other boxes show detection only, no copied
  score.
- **Orientation:** any non-zero best-match angle fails. 180° →
  "upside down"; 90°/270° → "rotated". Add a `score_gap` gate: below gate →
  "ORIENTATION UNKNOWN" → FAIL (fail-closed, consistent with existing
  design).
- **Dead code:** move unused files to `archive/` instead of deleting (git
  status of this checkout is uncertain).
- **New finding:** no `requirements.txt`/`pyproject.toml` exists; add one
  pinned to the installed versions (captured via `pip freeze` at
  implementation time).

---

## CRITICAL — correctness / data loss

### C1. Unhandled exception loop when recipe has no trained model
- Where: `main.py:265` (`self.inspect(...)`) + `services/patchcore_service.py:142`
  (`predict` raises `RuntimeError`).
- Trigger: recipe without a model (or broken model) + detected package →
  exception in the timer callback every tick.
- Fix: wrap the `self.inspect(...)` call in try/except → return
  `("FAIL", 0.0, f"Inspection error: {error}", False)`. With H1, move this
  handling into the worker and emit `error`.

### C2. Threshold unit bug — defective parts always PASS (decision 1)
- Where: `recipes/TJA1041_SO14/recipe.json` (`"anomaly_threshold": 60`),
  `main.py:307-311`, `ui/trainer/pages/review_page.py:19,51-57`,
  `services/recipe_service.py:89-93`, recipe template (`""`).
- Fix:
  1. Validate/clamp on load: values must be in (0, 1]; anything else → warn
     + fallback `0.60`. A legacy value >1 is treated as a percent and divided
     by 100 (handles `60` → `0.60` migration automatically).
  2. Migrate the checked-in recipe file to `0.60`.
  3. Single source of truth for the default (one constant, imported by
     `main.py`, `review_page.py`, `recipe_service.py`; delete
     `RecipeService.get_threshold`'s 0.5 default with the method).
  4. Review-page slider: map 0–100 slider → 0–1 threshold consistently and
     label it `0.60` style.
- **Correction (field-verified):** raw `pred_score` in anomalib 2.4.2 with
  `post_processor=False` is an unnormalized nearest-neighbour distance on a
  ~0–100 scale — known-good training images score 53.2–54.8. The recipe's
  original `60` was a percent threshold on that scale; the audit's earlier
  "0.1–5" assumption was wrong. To keep the confirmed 0–1 fraction
  convention, scores are now normalized to decimals at the service boundary
  (`SCORE_SCALE = 100.0` in `patchcore_service.predict` / `evaluate_folder`
  and `inference_service.predict`). Good parts score ≈0.53–0.55 → PASS at
  `0.60`; the original threshold meaning is preserved.

### C3. Re-saving a recipe in the trainer wipes trained-model data
- Where: `ui/trainer/pages/recipe_page.py:194-218`.
- Trigger: load trained recipe → Save → `model`, `roi_statistics`,
  `prepared_dataset_path`, `top_mark_template`, `top_mark_roi`,
  `dataset_prepared`, thresholds all reset to blank template values.
- Fix: on save, load the existing recipe (if present) and update only the
  edited metadata fields; preserve everything else. Validate `pin_count`
  (`int("abc")` currently raises → show a warning instead).

### C4. Newly saved top-mark template not used until recipe re-selected
- Where: `ui/inspection/status_widget.py:604-634` updates only the JSON on
  disk; in-memory dicts in the widget and `InspectionApp.current_recipe`
  still lack `top_mark_template` (`main.py:327` reads the in-memory dict).
- Fix: after a successful save, update the in-memory recipe dicts
  (combo item data + `InspectionApp.current_recipe`) with the new keys.

### C5. Stale detections inspected against the wrong frame (decision 2)
- Where: `main.py:177-187` (`apply_detection_memory`) + `main.py:280-283`
  (one score copied onto all boxes).
- Fix: tag boxes `fresh=True/False`. Inspect only fresh, highest-confidence
  boxes. Stale boxes keep the last verdict with a `"STALE"` marker on the
  overlay label; no new inspection runs on them. After the 1.5 s window →
  `NOT FOUND`, clear memory.

---

## HIGH — robustness / performance

### H1. Worker thread + fast predict (decision 3)
- Current: `main.py` QTimer runs mss capture + YOLO + PatchCore inline on the
  GUI thread; `patchcore_service.predict` writes a temp JPEG, builds a
  `PredictDataset` + `Engine.predict` per image (`main.py:73-75`,
  `patchcore_service.py:139-158`).
- Fix:
  1. New `ui/inspection/pipeline_worker.py` (QThread or QObject+moveToThread)
     owning mss, window lookup, YOLO, and the three services. Loop with the
     current 700 ms pacing; emits queued signals to the GUI thread:
     `boxes_ready(list)`, `result_ready(dict)`, `state_changed(state, msg)`,
     `error(str)`. `InspectionApp` becomes thin: forwards recipe selection to
     the worker, updates overlay/status on the GUI thread only. Qt objects
     (`OverlayWindow`, `StatusWidget`) are touched exclusively on the GUI
     thread.
  2. `PatchCoreService.predict`: run the loaded model directly on a
     preprocessed tensor (use the model's own pre-processor so
     train/inference preprocessing stays identical). Keep the temp-file
     `Engine.predict` path as a fallback behind a flag in case the installed
     anomalib version's direct-predict API differs — verify the API at
     implementation time before removing it.
  3. Stop/teardown: `closeEvent`-equivalent (app quit) must stop the worker
     loop before mss/model teardown.

### H2. Errors are invisible to the operator
- Where: `services/state_manager.py` (message stored, never shown);
  `status_widget.update_status` ignores it; ERROR overwritten next tick.
- Fix: pass `get_message()` into `update_status`, display in the MESSAGE box;
  keep ERROR sticky until the next successful cycle.

### H3. Mixed absolute/relative paths resolved against CWD
- Where: recipe file stores relative `recipes\...` paths (`model.path`,
  `prepared_dataset_path`, `top_mark_template`) while
  `PatchCoreService.train` writes absolute ones; broken when launched from a
  different CWD.
- Fix: one path helper (extend `utils/paths.py`) that resolves stored recipe
  paths against `PROJECT_ROOT` when not absolute; apply it at every load
  site (`main.py:102`, `top_mark_service`, `review_page`,
  `patchcore_service`). Normalize existing recipes on load (no destructive
  rewrite required).
- Also merge root resolvers: `main.py:350` uses
  `resource_path("models/detection/best.pt")` while
  `utils.paths.DETECTION_MODEL_PATH` is unused — keep one mechanism
  (`resource_path` for PyInstaller bundling, delegating to the same
  constants).

### H4. Trainer can be closed while the training thread runs
- Where: `ui/trainer/trainer_window.py` has no `closeEvent`;
  `ui/workers/training_worker.py` keeps running → "QThread: Destroyed while
  thread is still running"; `training_page.py:94-95` leaves
  `sys.stdout/stderr` redirected if the run never finishes.
- Fix: `closeEvent` on `TrainerWindow` → disable close or wait/abort worker;
  restore stdout/stderr in `finally` and on close.

### H5. ROI template save ignores failures / missing recipe path
- Where: `status_widget.on_roi_saved` — `cv2.imwrite` result ignored;
  `_recipe_path` may be absent (`os.path.dirname(None)` raises); failed
  write still stores a broken template path.
- Fix: check `saved`, show an error message, skip JSON update on failure,
  guard `_recipe_path`.

---

## MEDIUM — quality / consistency

- **M1.** `overlay.py:174-187` — `mouseReleaseEvent` crashes if release
  arrives without a prior in-overlay press (`roi_start is None`). Guard it.
- **M2.** `roi_page.py` — `self.roi_overrides` is memory-only;
  `load_override` never reads `recipe["roi_overrides"]`, so saved overrides
  don't drive the preview while dataset generation does use them. Load
  overrides from the recipe in `load_dataset`.
- **M3.** `roi_page.generate_training_dataset` — output dir is not cleared
  before writing → stale/rejected crops contaminate the training set. Wipe/
  recreate `train/good` first.
- **M4.** `status_widget.load_recipes` (`os.walk` + raw JSON) duplicates
  `RecipeService` and can list nested/duplicate recipes. Route through
  `RecipeService`.
- **M5.** `status_widget.on_package_changed` — selecting the "Select Package"
  placeholder (data None) returns early, so a recipe can never be deselected.
  Clear state on None.
- **M6. Confirmed dead code** (grep-verified: zero references):
  - `services/dataset_preparer.py` (superseded by `roi_page` generation)
  - `services/analysis_service.py` (stub, unused)
  - `trainers/dataset_analyzer.py` (third duplicate of dataset-stats logic)
  - `RecipeService.rename_recipe`, `RecipeService.get_threshold`,
    `RecipeService.update_model_path` (writes dead field `anomaly_model`)
  - `ui/trainer/pages/analysis_page.py`, `export_page` button — keep as
    acknowledged placeholders (do not wire).
  → Move the unused modules to `archive/`; delete the dead methods.
- **M7. Model loading hardening** (`patchcore_service.py:115-137`,
  `inference_service.py:24-43`):
  - `torch.load(...)` without `weights_only=True` (code-execution risk from
    recipe model files).
  - `load_state_dict(strict=False)` hides mismatched keys — log/verify
    missing/unexpected keys.
  - `load_model` doesn't configure the pre-processor (relies on anomalib
    default == `IMAGE_SIZE`) — make it explicit.
  - Verify `post_processor=False` at inference vs default at training keeps
    score semantics identical.
- **M8. Orientation semantics** (see default decisions): per-angle messaging
  (180° = upside down, 90°/270° = rotated) + `score_gap` gate with
  fail-closed "ORIENTATION UNKNOWN".
- **M9. Misc:**
  - `main.py:223` — zero-size window early-return leaves overlay/status
    stale; handle like the `window is None` branch.
  - `main.py:346-355` — missing `models/detection/best.pt` → raw traceback;
    show a clear error and exit.
  - Status bar fakes ("LCmicro", "YOLOv8", "DB CONNECTED") — wire to real
    values or remove.
  - `review_page.py:153` — mask compares raw per-pixel anomaly map values
    against the image-level score threshold (different scales); normalize
    the map or use a percentile for the mask.
  - FPS label shows inference latency only (1/elapsed), not loop rate —
    rename or compute real loop rate in the worker.
  - DPI scaling: overlay geometry (Qt logical px) vs `mss` capture (physical
    px via pygetwindow) can misalign on display scaling ≠100% — verify once
    on a scaled monitor; make the process DPI-aware if needed.
  - Add `requirements.txt` (pinned via `pip freeze` of the working env).

---

## Implementation order

1. **Phase 1 — critical fixes:** C1, C2, C3, C4, C5 (no threading changes).
2. **Phase 2 — threading + robustness:** H1 (worker + fast predict), H2, H3,
   H4, H5.
3. **Phase 3 — cleanup:** M1–M9 in one pass (including archive moves and
   `requirements.txt`).

## Validation plan

- Phase 1:
  - Recipe without model + detected package → no crash, FAIL + error message.
  - Recipe with `anomaly_threshold: 0.60` → known-defect sample FAILs, good
    sample PASSes; legacy `60` value auto-migrates/warns.
  - Load trained recipe → Save twice → `model`, `roi_statistics`,
    `top_mark_template` preserved in `recipe.json`.
  - Save top-mark ROI while recipe active → orientation check active
    immediately (no re-selection).
  - Briefly cover target window → verdict holds (stale marker), no
    wrong-region score; after 1.5 s → NOT FOUND.
- Phase 2: UI responsive during detection and training; closing apps
  mid-run is clean; launch from a different CWD → model/template load OK;
  errors visible in the MESSAGE box.
- Phase 3: no recipe-list duplicates; ROI preview honors saved overrides;
  regenerated dataset contains only current crops; app imports fine after
  archive moves.

## Risks / notes

- Anomalib direct-predict API must be verified against the installed version
  before deleting the temp-file path (H1 keeps a fallback).
- TrainingWorker trains in a fresh `PatchCoreService`; the trained model is
  not shared with a running inspection app (separate processes today — keep
  it that way; the inspection app picks models up via recipe re-selection).
- `runtime/audit/visionai.sqlite3` exists but nothing in code references a
  DB; the "DB CONNECTED" label is cosmetic (covered in M9).

## Out of scope

- Real physical-analysis service, model-recommendation similarity matching,
  export step (acknowledged placeholders in code/docs).
- Any DB integration.
