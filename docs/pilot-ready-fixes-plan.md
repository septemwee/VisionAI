# Pre-Pilot Fix Plan (VisionAI)

Source audit: `docs/visionai-code-audit-plan.md` (full findings + phases).
This plan scopes what must be fixed **before the first real-user test**.

## Context & constraint

- Real users test in a few days, running the **full workflow: both apps**
  (inspection `main.py` + trainer `trainer_main.py`).
- Priority: stability and correct verdicts over performance. No core-pipeline
  refactors land before the pilot.
- Full audit phases 2–3 (threading, cleanup) are **deferred until after the
  pilot** (see "Deferred" below).

## Resolved decisions

1. **Test scope:** both apps, full workflow → trainer data-loss (C3) and
   trainer close crash (H4) are in scope.
2. **H1 threading/fast-predict refactor: deferred** until after the pilot
   (regression risk on the core pipeline right before testing). The ~0.5 s
   per-cycle UI freeze stays for the pilot.
3. **Threshold semantics (from audit):** `anomaly_threshold` = 0–1 fraction,
   default `0.60`; legacy `60` auto-migrates.

## Default decisions (challenge if wrong)

- **Minimal path fix only** (H3-lite): resolve non-absolute recipe paths
  (`model.path`, `prepared_dataset_path`, `top_mark_template`) against
  `PROJECT_ROOT` at load sites. No root-resolver merge, no recipe rewrite.
- **Pilot logging:** redirect `print` output to a timestamped log file under
  `runtime/logs/` (keep console output). Needed so pilot issues are
  diagnosable afterwards.
- **No dead-code removal / archive moves before the pilot** (defer to
  post-pilot cleanup to avoid churn).
- **Orientation logic unchanged** (fail-closed as today); the `score_gap`
  gate (M8) is deferred — changing decision logic pre-pilot alters behavior
  users haven't seen yet.

---

## Task list (ordered)

1. **C1 — stop the crash loop** (`main.py:265`,
   `services/patchcore_service.py:142`): wrap `self.inspect(...)` in
   try/except → return `("FAIL", 0.0, f"Inspection error: {error}", False)`.
   Verify no-recipe/no-model + detected package no longer raises each tick.
2. **C2 — threshold units** (`main.py:307-311`,
   `ui/trainer/pages/review_page.py:19,51-57`,
   `services/recipe_service.py:89-93`, recipe template):
   - One shared constant `DEFAULT_ANOMALY_THRESHOLD = 0.60`; remove the 0.5
     default and the `""` template value (use the constant).
   - On load: values must be in (0, 1]; values >1 treated as percent ÷100
     (migrates `60` → `0.60`), else fallback + console warning.
   - Update `recipes/TJA1041_SO14/recipe.json` to `0.60`.
   - Review-page slider maps 0–100 → 0–1 and labels `0.60` style.
   - **Scores normalized to 0–1 decimals at the service boundary**
     (field-verified: raw `pred_score` is a ~0–100 nearest-neighbour
     distance; good parts score 53–55 raw). `SCORE_SCALE = 100.0` applied in
     `patchcore_service.predict` / `evaluate_folder` and
     `inference_service.predict`, so every display and the PASS/FAIL
     comparison use consistent decimals.
3. **C3 — recipe save must not wipe trained data**
   (`ui/trainer/pages/recipe_page.py:194-218`): on save, load existing
   recipe and update only edited metadata fields (name/family/type/size/
   version/type_name/pin_count/notes); preserve `model`,
   `roi_statistics`, `prepared_dataset_path`, `top_mark_*`, thresholds,
   `dataset_prepared`. Validate `pin_count` (warn on non-numeric instead of
   crash).
4. **C4 — saved top-mark template applies immediately**
   (`ui/inspection/status_widget.py:604-634`): after successful save, update
   the in-memory recipe dicts (combo item data +
   `InspectionApp.current_recipe`) with `top_mark_template` /
   `top_mark_roi` so orientation checking starts without re-selection.
5. **C5 — no inspection of stale detections** (`main.py:177-187,280-283`):
   tag boxes fresh/stale; inspect only fresh highest-confidence box; stale
   boxes keep last verdict + `"STALE"` overlay marker; after 1.5 s →
   `NOT FOUND`, clear memory; stop copying one score onto all boxes.
6. **H2 — operator-visible errors** (`services/state_manager.py`,
   `ui/inspection/status_widget.py:509-567`): pass
   `state_manager.get_message()` into `update_status`, show in MESSAGE box,
   keep ERROR sticky until the next successful cycle.
7. **H3-lite — CWD-proof recipe paths** (`utils/paths.py`, load sites
   `main.py:102`, `patchcore_service.load_model`, `top_mark_service`,
   `review_page`): helper that resolves non-absolute stored paths against
   `PROJECT_ROOT`.
8. **H4 — safe trainer close during training** (`ui/trainer/
   trainer_window.py`, `ui/trainer/pages/training_page.py`): add
   `closeEvent` → block or wait/abort `TrainingWorker`; restore
   `sys.stdout/stderr` in `finally` and on close.
9. **H5 — robust ROI template save** (`status_widget.py:604-634`): check
   `cv2.imwrite` result, show error + skip JSON update on failure, guard
   missing `_recipe_path`.
10. **Quick UX guards** (small, user-triggerable crashes):
    - M1: `overlay.py:174-187` guard `roi_start is None` on release.
    - M5: `status_widget.on_package_changed` — clear recipe state when the
      placeholder (None data) is selected.
11. **Graceful startup failure** (`main.py:346-355`): missing
    `models/detection/best.pt` → clear error dialog + exit instead of
    traceback.
12. **Pilot logging**: lightweight handler that tees `stdout/stderr` prints
    to `runtime/logs/pilot-<timestamp>.log`; log file per session, keep
    console behavior.
13. **`requirements.txt`**: pin current environment via `pip freeze` so the
    test machine setup is reproducible.

## Expectation note for the pilot

Verified against the trained TJA1041_SO14 model: raw PatchCore scores are
~0–100 nearest-neighbour distances (known-good parts score 53–55 raw), and
the original recipe threshold `60` was a percent threshold on that scale.
After the task-2 fixes the app is consistent end to end: scores are
normalized to 0–1 decimals at the service boundary and compared against the
`0.60` fraction threshold, which equals the original `60` percent threshold.
Verdict semantics are therefore unchanged from the original recipe intent —
what changed is that scores/thresholds now display and compare consistently
as 0–1 decimals everywhere (SCORE box, console, review page). If `0.60`
proves too strict/loose on pilot parts, it is tuned per recipe after the
pilot.

## Deferred (post-pilot)

- H1 worker-thread refactor + direct tensor predict (incl. anomalib API
  verification, fallback path).
- Remaining H3 consolidation (root resolvers), M2, M3, M4, M6 (dead code →
  `archive/`), M7 (model-loading hardening), M8 (orientation gate), M9 misc
  (status-bar fakes, review mask scaling, FPS label, DPI check).

## Validation checklist (run before handing to users)

1. Inspection app: recipe without model + detected package → FAIL + message,
   no crash (C1); error text visible in MESSAGE box (H2).
2. Verdicts: good part PASS, defective part FAIL at `0.60`; SCORE box,
   console and review page all show 0–1 decimals (e.g. `0.548`); legacy `60`
   value warns and behaves as `0.60` (C2).
3. Cover target briefly → verdict holds with STALE marker, no wrong-region
   score; after 1.5 s → NOT FOUND (C5).
4. Save top-mark ROI while recipe active → orientation check active
   immediately (C4); cancel/save with no prior press does not crash (M1).
5. Trainer: load trained recipe → Save twice → `model`, `roi_statistics`,
   `top_mark_template`, thresholds preserved (C3); non-numeric pin count →
   warning, no crash.
6. Trainer: start training → close window → clean exit, no QThread crash;
   console restored (H4).
7. Launch both apps from a different working directory (e.g. `cd C:\` then
   absolute path) → model + template load fine (H3-lite).
8. Confirm `runtime/logs/` captures session output (task 12).

## Risks

- C2 changes live verdicts (intended — see expectation note).
- H3-lite must not rewrite recipe files destructively; resolution happens at
  load time only.
- Keep diffs small and test after each task; no refactors ride along.
