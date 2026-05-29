# Model Evaluation Results — Single-POV Golf Swing Reconstruction

**Date:** 2026-05-23  **Author:** Banjo  **Status:** baseline scoping, no fine-tuning yet

## TL;DR — start with MediaPipe **Lite** (yes, Lite, not Heavy)

| Decision | Recommendation |
|---|---|
| **Primary 2D baseline** | `mediapipe_lite` |
| **Mobile / on-device fallback** | `movenet_lightning` |
| **If Team 2 needs maximum smoothness** | `mediapipe_heavy` (but worst at swing-event detection) |
| **If robustness > accuracy** | `vitpose_base` (99.9% detection rate, always finds a pose) |

The biggest finding: **the cheap MediaPipe variant beats the expensive one
on the metric that matters most for golf**. The coworker's `pose_landmarker_heavy.task`
choice is actually the *worst-performing MediaPipe variant* on swing-event
accuracy — by a wide margin.

---

## How we evaluated

- **Dataset:** all 1,400 labeled GolfDB clips (`Data/videos_160/`)
- **Models tested (7):** MediaPipe Heavy / Lite, MoveNet Thunder / Lightning,
  YOLOv8n / YOLOv8m, ViTPose-Base
- **Total inferences:** ~9,800 (model × clip) pairs
- **Schema:** every model projected onto canonical COCO-17 (17 keypoints)
- **Hardware:** RTX 4080 SUPER for GPU models; CPU for MediaPipe + MoveNet
  (their Windows builds are CPU-only)

### Metrics (9 total)

| Group | Metric | Direction | What it tells us |
|---|---|---|---|
| Reference-based | `pce_at_5` | ↑ | % of 8 swing events recovered from landmarks within ±5 frames |
| | `pce_at_3` | ↑ | Same, tighter tolerance |
| | `pce_at_1` | ↑ | Same, ±1 frame |
| Reference-free | `bone_cv_mean` | ↓ | Bone-length stability across the swing |
| | `jitter_mean_px` | ↓ | Frame-to-frame smoothness of landmarks |
| | `implausible_frac_mean` | ↓ | % of frames with degenerate joint angles |
| | `left_ankle_planting_std_px` | ↓ | Foot sliding during the swing |
| Operational | `detection_rate` | ↑ | % of (frame, keypoint) cells with confident output |
| | `fps_inference` | ↑ | Throughput on our hardware |

The PCE detector uses a deliberately simple wrist-trajectory heuristic
applied identically to every model — so a *better* PCE reflects better
underlying landmark quality, not a smarter event detector.

---

## Full leaderboard (median per model, n=1,400 clips)

Sorted by `pce_at_5` (descending):

| # | Model | FPS | Detect% | BoneCV | Jitter | Implaus% | FootStd | **PCE@5** | PCE@3 | PCE@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **mediapipe_lite** | 107 | 94.3 | 0.203 | 0.99 | 14.2 | 1.81 | **0.090** | 0.070 | 0.043 |
| 2 | movenet_lightning | 140 | 88.9 | 0.197 | 1.20 | 14.7 | 1.67 | 0.072 | 0.055 | 0.030 |
| 3 | yolov8m_pose | 120 | 86.2 | 0.208 | 2.01 | 14.0 | 1.57 | 0.057 | 0.040 | 0.019 |
| 4 | vitpose_base | 129 | **99.9** | 0.214 | 1.11 | 16.1 | **1.28** | 0.056 | 0.042 | 0.022 |
| 5 | yolov8n_pose | **214** | 89.7 | 0.233 | 3.22 | 14.9 | 3.31 | 0.054 | 0.037 | 0.016 |
| 6 | mediapipe_heavy | 24 | 94.3 | **0.182** | **0.70** | 19.4 | 0.98 | 0.052 | 0.037 | 0.018 |
| 7 | movenet_thunder | 95 | 88.2 | **0.182** | 1.06 | 15.8 | 1.17 | 0.049 | 0.036 | 0.018 |

Bolded = best in column.

---

## Key findings

### 1. Cheap models beat expensive ones on the golf-specific metric

MediaPipe **Lite** outperforms MediaPipe **Heavy** on PCE@5 by **73%**
(0.090 vs 0.052) — *and* it runs 4× faster on CPU. MoveNet Lightning
beats MoveNet Thunder by 47%. This is the opposite of what you'd expect
from generic pose-estimation benchmarks (COCO mAP, MPJPE) where the
heavier models dominate.

**Hypothesis why:** the swing-event detector uses wrist-Y argmin/argmax.
Heavy models produce *smoother* wrist trajectories — but smoothing
slightly displaces the actual peak frame. Lite models are jitterier
but their peak coincides more closely with the true event frame. In
other words: smoothing is helpful for visual rendering but *hurts*
sharp event localization.

### 2. There's no single winner — it's a Pareto trade

- **MediaPipe Heavy**: smoothest landmarks (jitter 0.70 px, bone CV 0.182)
  but worst on joint plausibility (19.4% degenerate) AND worst on PCE.
  *Smooth ≠ correct.*
- **YOLOv8n**: 2× faster than everything else (214 FPS) but worst on
  jitter (3.22 px) and foot stability.
- **ViTPose**: 99.9% detection rate — always finds a pose. Best foot
  stability. But middle of the pack on accuracy.
- **MoveNet Lightning**: top-2 on accuracy AND fast AND smallest model
  size. The dark horse for mobile.

### 3. Absolute PCE is low across all models (0.049 – 0.090)

Even the winner gets ~91% of swing events wrong (within a 5-frame
tolerance). This means **none of these models, used naively, can power
a golf coaching app on their own**. To make the system actually work
we will need at least one of:

1. **A smarter event detector** — current logic is just argmin/argmax
   on wrist Y. A small temporal model (1D CNN or LSTM over landmark
   trajectories) trained on GolfDB labels would likely 3–5× the PCE.
2. **Fine-tuning** the pose backbone on golf-domain frames.
3. **Both** — fine-tune the backbone and the event detector together.

The good news: any of these improvements amortizes across models, so
locking in *which* 2D backbone we use is a low-risk decision.

---

## Recommendation: `mediapipe_lite` as the primary 2D baseline

Reasons in priority order:

1. **Best PCE** of the 7 models (0.090). The event-detection signal
   downstream depends on this.
2. **Strong detection rate** (94.3%) — same as Heavy, much better than
   YOLO/MoveNet variants.
3. **Fast enough for on-device** at 107 FPS on CPU (Heavy is 24 FPS,
   which is borderline-too-slow for a 30 FPS phone camera).
4. **Small footprint** — the model file is ~3 MB (vs 30 MB for Heavy)
   which matters for a mobile app.
5. **Same MediaPipe API and integration** as the existing baseline —
   no engineering switching cost. We literally swap one file path.

### When to deviate

- **MediaPipe Lite's jitter (0.99 px) might still be too high for
  Team 2's UE5 Control Rig** — if their first end-to-end demo
  shows visible flicker, switch to **MediaPipe Heavy** and accept
  the PCE hit. We can compensate with a smarter event detector.
- **If we ship to phone and the inference cost is too high**, drop
  to **MoveNet Lightning** (140 FPS, only slightly lower PCE 0.072).
  Both are CPU-only on Windows but both have well-supported TF Lite
  + Core ML exports for iOS / Android.
- **If we run into clips where pose isn't detected at all** (e.g.
  unusual camera angles, occlusion), **ViTPose-Base** provides a
  robust fallback (99.9% detection rate). Could be used as a
  second-pass model when MediaPipe returns low confidence.

---

## Important caveats (be honest about these)

1. **Mixed-source data for YOLOv8n.** Clips 0–540 were generated by
   a pre-patch adapter that suffered GPU memory thrashing on long
   slow-mo clips. Outputs look reasonable but technically inconsistent
   with the chunked-batch version that produced clips 541–1399. If we
   want to be rigorous before sharing externally, delete those 540
   cached files and let the patched adapter re-run them (~12 min).

2. **PCE detector is intentionally simple.** A 1D-CNN or LSTM over
   landmark trajectories would change the absolute numbers. The
   *relative* ranking between models is what matters for picking a
   backbone.

3. **No 3D evaluation.** GolfDB doesn't ship 3D ground truth. We
   measure 2D landmark quality only. The downstream UE5 pipeline
   needs 3D — which means either:
   - Team 2 adds their own 2D→3D lift (current plan), or
   - We add a 3D mesh recovery model (HMR2.0, WHAM) to a future
     benchmark round. **Recommend doing this before locking the
     architecture.**

4. **No real-phone data.** GolfDB clips are pre-cropped to the
   golfer's bounding box and resized to 160×160. Real phone uploads
   won't be. The pose models' actual performance on real input
   could be different — especially YOLO, which usually assumes a
   wider field of view.

5. **No mobile latency measured.** All FPS numbers are on the 4080
   workstation / its CPU. Phone CPU + TF Lite export will be slower.

---

## Suggested next steps

1. **Decide & lock backbone** — propose: MediaPipe Lite. ETA: this
   week's meeting.
2. **Build a smarter swing-event detector** — small 1D-CNN over the
   landmark trajectories. GolfDB has 1,400 labeled examples, plenty
   for a small model. ETA: 1 week.
3. **Wire MediaPipe Lite into UE5 demo** — Team 2 imports the CSV
   landmark stream and verifies the swing looks plausible at 30 FPS.
   ETA: end of next week.
4. **Add a 3D contender** — HMR2.0 or MotionBERT. Required before we
   commit to "2D-only" architecture. ETA: 2 weeks.
5. **Collect 10–20 phone clips** in real shooting conditions and
   re-run this benchmark on them. We don't know yet whether GolfDB
   results generalize to real input. ETA: 1 week.

---

## Files & artifacts

| What | Where |
|---|---|
| Full evaluation notebook | `Scripts/model_evaluation_baseline.ipynb` |
| Reusable metrics + adapter library | `Scripts/eval_utils.py` |
| 7 model adapters | `Scripts/adapters/` |
| Aggregator (run after benchmarks) | `Scripts/compute_metrics.py` |
| Aggregated metrics (9,800 rows) | `Data/all_metrics.parquet` |
| Per-clip cached landmarks | `Data/eval_runs/<model>/<id>.parquet` |
| Per-model overlay MP4s (6 clips × 7 models) | `Data/overlays/<model>/<id>_<model>.mp4` |
| Side-by-side grid MP4s (the meeting demo) | `Data/overlays/sidebyside/<id>_compare.mp4` |
| Selection rationale for the 6 demo clips | `Data/overlays/manifest.json` |
| Frozen Python environment | `requirements_eval.txt` |
