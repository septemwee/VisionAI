# Training Assistant Plan — Semi-Automatic Trainer with Calibration

Date: 2026-09-01. Confirmed decisions: good-images-only calibration (P99 ×
margin, no defect-folder mode for now), P99 + adjustable margin as the
default threshold policy, **consolidated layout — no new pages (8 total)**,
**S0 recommendation entry (reuse / add-to-memory / train-new) on the
existing recommendation page**.

## Goal

Turn the existing trainer pages into a guided, semi-automatic flow in which
every step is checked by the system, shows an OK / ACTION NEEDED / WARNING
status, and proposes the next action — all inside the existing pages. The
centerpiece is an automatic calibration step that converts training results
into a measured `anomaly_threshold` (0–1 fraction) instead of a hand-picked
default, plus an intelligent entry that recommends whether to reuse an
existing model, add more images to its memory, or train a new one.

## Layout: features folded into existing pages (8 pages total)

| Assistant feature | Lives in |
|---|---|
| Entry recommendation: reuse / add-to-memory / train-new | `recommendation_page` (S0) — placeholder cards replaced by a computed recommender |
| Step guidance: S0–S6 status badges, "next suggested step" button, gating | New small widget `ui/trainer/step_status_bar.py` shown in the `TrainerWindow` header next to the context label (a widget, not a page) |
| Threshold calibration (histogram, P99 × margin, expected false alarms, Apply-to-recipe) | `training_page` — a post-training section that enables when training completes (page already owns the log + worker infra) |
| Finalize/summary (S6) | `export_page` — the placeholder becomes the real finalize step |
| Dataset quality screening | `dataset_page` — "Screen quality" button + flagged-move list |
| ROI auto-tune + crop QA | `roi_page` — "Auto-tune defaults" button + accept/review-flagged list |

The raw pages remain the single source of truth; the step bar only guides
and gates.

## S0 — Recommendation entry (recommendation_page)

Replaces the hardcoded placeholder cards with a computed recommender:

- **Signals (all from existing pieces):**
  - Package match: metadata similarity (`package_family`, `package_type`,
    `pin_count`) across existing trained recipes.
  - Visual match (per recipe, like-for-like): take ONE raw sample image
    from the new dataset, run `roi_service.detect_roi`, crop it with THAT
    recipe's `roi_default` config and letterbox to its target size, then
    multi-scale match it against the recipe's reference crop. Comparing
    crop-to-crop keeps the similarity meaningful without requiring the full
    S3 crop pass first.
  - Model health: trained flag + last calibration stats (P99 headroom);
    later: near-threshold score frequency from pilot logs.
- **Degraded mode:** if `detect_roi` fails on the sample image (no package
  found / no dataset yet), S0 reports "not analyzable yet — run dataset
  intake and ROI first" and the step bar routes to S2/S3; once crops exist,
  S0 re-runs on the real crops as a high-fidelity confirmation.
- **Augment gate:** for the augment path, the decision is only committed
  after S3 has cropped ALL new images with recipe X's config and the crop
  QA passes (full-fidelity check before append + retrain).
- **Decision rules (proposed → user confirms):**
  - Similarity >= 0.85 → "Reuse model X" (recalibrate, capture top-mark) or
    "Add images to X's memory" when the user has more good shots.
  - 0.60–0.85 → "Review required" (side-by-side crop drill-down).
  - < 0.60 or no trained recipes → "Train new model".
- **Add-to-memory path, v1 (now):** route the new good images through the
  existing recipe's ROI/crop/letterbox pipeline (so crops match the current
  training distribution), append to `train/good`, retrain, recalibrate
  (new P99 → updated threshold). Reuses S3 + S4 + S5 with an existing
  recipe; no new infrastructure.
- **Add-to-memory v2 (post-pilot):** true incremental memory-bank merge —
  extract patch embeddings for the new images only, union with the saved
  memory bank, re-run coreset subsampling, re-save. Anomalib exposes no
  incremental bank API; v1 is the safe stand-in.

## Architecture

- **New pure-logic service** `services/training_assistant.py` (no Qt):
  recommendation matching, dataset screening, ROI plan/QA classification,
  threshold recommendation, pre-flight checks, finalization, step-status
  computation.
- **New widget** `ui/trainer/step_status_bar.py`: S0–S6 badges + next-step
  button; driven by `training_assistant.compute_step_status(recipe)`.
- **New worker** `ui/workers/evaluation_worker.py` (QThread): scores a
  folder with `PatchCoreService` for calibration; console capture follows
  the LogRedirector pattern; covered by the trainer close-event guard.
- **Worker for screening**: dataset screening of 300+ images must not
  freeze the UI — small QThread worker emitting flags when done.
- **Shared crop helper**: move `roi_page._crop_rotated_roi` into
  `utils/image_utils.py` as `crop_rotated_roi(image, roi, config)` so
  `roi_page` and the assistant share one implementation.
- **Wizard state in the recipe**: `assistant_state` dict (per-step status +
  timestamps + calibration results + chosen S0 path) persisted via
  `RecipeService` helpers, so reopening the trainer resumes where the user
  left off.

## Step definitions and gating

| Step | OK condition | System assistance (where) |
|---|---|---|
| S0 Recommend | user chose a path (reuse / augment / new) | computed recommender (recommendation_page) |
| S1 Recipe | recipe exists + required metadata non-empty | completeness check + missing-field list (recipe_page; step bar badge) |
| S2 Dataset | folder selected, `image_count >= 300` (WARNING 100–299, BLOCK < 100) | "Screen quality": blur (Laplacian variance), brightness, size outliers → suggested-move list to `quarantine/` (user confirms; never deletes) (dataset_page) |
| S3 ROI/crops | `dataset_prepared` true and saved crops >= 10 | "Auto-tune defaults": scales/tolerances from score & size percentiles instead of ฆfixed 25%/20%; per-crop QA classification with reject reasons; "accept all / review flagged" (roi_page) |
| S4 Train | S2 not BLOCKED and S3 OK | pre-flight checklist gates the Train button; post-train auto-score of the training set (training_page) |
| S5 Calibrate | model trained | post-training section in training_page: scores `train/good` via worker → histogram + min/median/P95/P99/max; proposed threshold = clamp(P99 × margin, (0, 1]); expected false alarms = count(good > threshold); margin spin (1.0–3.0, default 1.2); one-click apply → writes `anomaly_threshold` + `calibration` trace block |
| S6 Finalize | calibration applied | export_page: summary (counts, threshold, calibration stats, warnings); bump `last_trained`/version; set `validated` flag; reminder to capture the top-mark template in the inspection app |

Next-step suggestion = first step whose status is not OK; the step-bar
button navigates to the owning page. For the augment path, S2–S4 operate on
the EXISTING recipe's dataset (new images flow through S3's crop pipeline).

## Threshold math (P99 + margin)

- `PatchCoreService.evaluate_folder` extended to RETURN per-image
  `(name, score)` pairs (0–1 fractions) while keeping its console output.
- `p99 = percentile(good_scores, 99)`; `proposed = min(1.0, p99 * margin)`;
  if `proposed < p99 + 0.02` use `p99 + 0.02`; clamp to > 0.
- Warning if `max(good_scores) >= 1.0` (training set itself at the scale
  ceiling — model too noisy).
- Trace: recipe `calibration` = {p99, margin, proposed,
  expected_false_alarms, evaluated_at, good_summary}.

## Implementation order (each task independently safe + tested)

0. **T0 S0 recommender logic + page wiring** — matching helpers in
   `training_assistant.py` (metadata + visual similarity), recommendation
   cards computed from real recipes; augment path routing; offscreen tests
   with synthetic recipes and synthetic image sets.
1. **T1 Shared crop helper** — extract `crop_rotated_roi` to
   `utils/image_utils.py`; `roi_page` uses it; offscreen test proves
   identical crop output on a synthetic ROI.
2. **T2 `evaluate_folder` return contract** — return `(name, score)` list;
   keep prints; harness test with a stubbed engine.
3. **T3 `training_assistant.py` core logic** — `assess_dataset` (synthetic
   blurry/bright/dark/outlier images), `recommend_threshold` (table-driven:
   normal, tight, ceiling, tiny-set), `pre_flight`, `finalize`,
   `compute_step_status` (gating + next-step).
4. **T4 Recipe assistant-state helpers** — get/set step statuses on a temp
   recipe; unit tests.
5. **T5 Step-status bar + TrainerWindow integration** — S0–S6 badges,
   next-step navigation, gating; offscreen test with stubbed services.
6. **T6 Calibration section in training_page + EvaluationWorker** — worker
   scoring, histogram rendering (QPainter, no new dependency), margin
   slider, apply-to-recipe; extend `TrainerWindow.closeEvent` guard to the
   evaluation worker; offscreen test with stubbed worker + real math.
7. **T7 Dataset screening in dataset_page + worker** — flagged-move list,
   quarantine move on confirmation; offscreen test with synthetic images.
8. **T8 ROI auto-tune in roi_page** — percentile-based defaults + flagged
   list; offscreen test.
9. **T9 Finalize in export_page + summary** — offscreen test on temp recipe.
10. **T10 Regression sweep** — re-run all existing offscreen suites
    (trainer close, recipe save, inspection suites) to prove no behavior
    change in existing paths.

## Risks & controls

- Change volume pre-pilot → mitigated by: everything is an extension of
  existing pages (no new pages, no embedding); each task keeps existing
  behavior; regression suite after each task.
- Calibration uses self-scores of the training set (optimistic) → the
  margin (default 1.2) compensates; pilot line data will validate;
  defect-folder mode and true memory-bank merging can be added later
  without UI changes.
- Screening moves files to `quarantine/` only after explicit confirmation;
  never deletes.
- Scoring/screening run in workers; both are covered by the trainer
  close-event guard.
- S0 visual matching must not fire on false "reuse" recommendations —
  thresholds conservative (>= 0.85 reuse) and user confirms every path.

## Out of scope (post-pilot backlog)

- Defect-folder separation calibration (best-F1 mode) — clean extension
  point in `recommend_threshold`.
- True incremental memory-bank merge (v2 augment path).
- Model recommendation similarity matching beyond v1 signals, physical
  analysis, export formats.
