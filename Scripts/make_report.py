"""Generate REPORT.ipynb — a polished, image-rich notebook telling the
full story of the capstone evaluation work.

Run this any time the visualizations or leaderboard change; it rebuilds
the notebook from scratch.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent / "REPORT.ipynb"
PROJECT_ROOT = Path(__file__).parent.parent
VIZ_DIR_REL = "../Data/visualizations"
HANDOFF_DIR_REL = "../Data/handoff"
OVERLAYS_DIR_REL = "../Data/overlays"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "id": str(uuid.uuid4())[:8], "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "id": str(uuid.uuid4())[:8],
            "source": source.splitlines(keepends=True)}


# =============================================================================

CELLS = [
    md("""# Single-POV Golf Swing Reconstruction — Capstone Research Report

**Author:** Banjo &nbsp; • &nbsp; **Date:** 2026-05-29 &nbsp; • &nbsp; **Status:** baseline scoping complete

---

## TL;DR

We evaluated **20 pose-estimation pipelines** (7 × 2D backbones, 10 × MotionBERT 3D-lift combinations, 3 × GolfPose 3D combinations) on the full **1,400-clip GolfDB** dataset under a 9-metric evaluation framework. We built an end-to-end pipeline that takes a single phone video and produces Unreal-Engine-ready files (CSV, BVH, JSON, MP4 overlay, standalone Three.js 3D preview).

**Headline recommendation:** `mediapipe_lite` → `motionbert_full` → One-Euro smoothing → BVH/CSV. PCE@5 = **0.170** (best of 20), inference at ~7,000 FPS post-2D, ~51% jitter reduction from smoothing.

**Biggest surprise:** golf-specific fine-tuning (**GolfPose**, fine-tuned on actual swing mocap) **lost to generic SOTA pretraining** (**MotionBERT-Full**, AMASS + H36M) by 2-2.4× on every backbone. Domain specialization isn't free.

```bash
python Scripts/pipeline.py path/to/swing.mp4 \\
    --backbone mediapipe_lite \\
    --lifter motionbert_full \\
    --smooth oneeuro
```

---

## What's in this report

1. [The problem & approach](#1)
2. [Evaluation framework](#2)
3. [2D pose models](#3)
4. [3D pose lifting — MotionBERT](#4)
5. [The surprise — GolfPose vs MotionBERT](#5)
6. [The smoothing layer](#6)
7. [End-to-end pipeline](#7)
8. [What we deliver for Unreal Engine 5](#8)
9. [Visualization gallery](#9)
10. [What's next](#10)
"""),

    # =========================================================================
    # §1
    # =========================================================================
    md("""<a id="1"></a>
## §1 — The problem & approach

### The capstone goal

Build an app that takes a **single-POV phone video** of a golf swing and produces a **biomechanically plausible 3D reconstruction** for use in coaching feedback (UE5 visualization, swing-event detection, posture analysis).

Two research questions:

1. **Whats the best way to evaluate this app + models?**
2. **What are the best off-the-shelf models without fine-tuning?**

### Starting baseline (coworker)

`Scripts/GolfDB_EDA.ipynb` — MediaPipe Pose **Heavy** on a single GolfDB clip (clip 0), producing a 2D landmark CSV + overlay MP4 + one wrist/shoulder Y-trajectory plot.

### Our extensions

- **Quantitative metrics** — 9 metrics across reference-based, physics-based, and operational dimensions.
- **Multi-model comparison** — 20 pipelines on the full 1,400-clip corpus (not 1).
- **3D lift** — adding MotionBERT + GolfPose 3D models on top of the 2D backbones.
- **Smoothing layer** — One-Euro filter + bone-length lock before downstream handoff.
- **UE5 handoff** — pipeline produces BVH / CSV / JSON / overlay MP4 / interactive HTML 3D preview.

The baseline plot we started from (coworker's), reproduced from our pipeline for comparison:

![coworker baseline plot recreated](""" + VIZ_DIR_REL + """/clip0_2d_ytraj_mediapipe_heavy.png)

This is `MediaPipe Heavy → wrist/shoulder Y over time` on the same clip (id 0) the coworker used. Notice the jitter spikes around frames 10-15 (address phase) and 60-90 (impact). Reducing this is the through-line of every improvement that follows.
"""),

    # =========================================================================
    # §2
    # =========================================================================
    md("""<a id="2"></a>
## §2 — Evaluation framework

Picking a model without defining "good" is meaningless. We split metrics into three groups:

| Group | Metric | Direction | What it measures |
|---|---|---|---|
| **Reference-based** (uses GolfDB swing-event labels) | `pce_at_5` | ↑ | % of 8 events recovered within ±5 frames |
| | `pce_at_3` | ↑ | Same, ±3 frames |
| | `pce_at_1` | ↑ | Same, ±1 frame |
| **Reference-free** (physics & golf-domain priors) | `bone_cv_mean` | ↓ | Bone-length stability across the swing |
| | `jitter_mean_px` | ↓ | Frame-to-frame acceleration (smoothness) |
| | `implausible_frac_mean` | ↓ | % frames with degenerate elbow / knee angles |
| | `left_ankle_planting_std_px` | ↓ | Foot sliding during the swing |
| **Operational** | `detection_rate` | ↑ | % of (frame, keypoint) cells with confident output |
| | `fps_inference` | ↑ | Throughput on our hardware |

The PCE detector applies the same wrist-Y trajectory heuristic to every model, so a better PCE reflects better underlying landmark quality, not a smarter detector. The detector is intentionally simple — we want to compare backbones, not event detectors.

All metrics are implemented in `Scripts/eval_utils.py`. The aggregated metric DataFrame lives at `Data/all_metrics.parquet` (28,000+ rows, 20 models × 1,400 clips).
"""),

    code("""# Live leaderboard from the cached metrics
import pandas as pd
from pathlib import Path

df = pd.read_parquet('../Data/all_metrics.parquet')
print(f'{len(df):,} (model x clip) rows across {df[\"model\"].nunique()} models')

lb = df.groupby('model').agg(
    fps=('fps_inference', 'median'),
    detect=('detection_rate', 'mean'),
    bone_cv=('bone_cv_mean', 'median'),
    jitter=('jitter_mean_px', 'median'),
    implaus=('implausible_frac_mean', 'median'),
    foot_std=('left_ankle_planting_std_px', 'median'),
    pce5=('pce_at_5', 'mean'),
    pce3=('pce_at_3', 'mean'),
    pce1=('pce_at_1', 'mean'),
).round(3).sort_values('pce5', ascending=False)
lb.head(10)
"""),

    # =========================================================================
    # §3 — 2D pose models
    # =========================================================================
    md("""<a id="3"></a>
## §3 — 2D pose models

We benchmarked **seven 2D pose estimators**, each projected onto a canonical COCO-17 schema:

| Model | Source | Hardware | Native skeleton |
|---|---|---|---|
| MediaPipe Heavy | Google MediaPipe Pose Landmarker (33 kp → 17) | CPU only on Windows | MediaPipe-33 |
| MediaPipe Lite | same family, smaller variant | CPU | MediaPipe-33 |
| MoveNet Thunder | Google MoveNet | CPU (TF Lite) | COCO-17 |
| MoveNet Lightning | smaller MoveNet variant | CPU | COCO-17 |
| YOLOv8n-Pose | Ultralytics | GPU | COCO-17 |
| YOLOv8m-Pose | bigger YOLOv8 variant | GPU | COCO-17 |
| ViTPose Base | HuggingFace transformer | GPU | COCO-17 |

### Headline finding: cheap beats expensive

The intuitive prior is "the bigger model is better." For golf-event detection, **the opposite holds**:

| Model | PCE@5 | Notes |
|---|---|---|
| **MediaPipe Lite** | **0.090** | 4× faster than Heavy on CPU, 3 MB model |
| MediaPipe Heavy | 0.052 | smoother but **73% worse PCE** |
| MoveNet Lightning | 0.072 | beats Thunder by 47% |
| MoveNet Thunder | 0.049 | smoothest 2D but worst PCE |

**Why?** Heavy models apply more temporal/spatial smoothing, which slightly displaces the actual peak frame in the wrist trajectory. Lite models are noisier but their peaks coincide more accurately with the labeled swing events.

### Same plot on the recommended backbone

The reference plot (MediaPipe Heavy) we showed above, but now with the **recommended backbone (MediaPipe Lite)** — same clip 0:

![mediapipe lite Y trajectory](""" + VIZ_DIR_REL + """/clip0_2d_ytraj_mediapipe_lite.png)

The Lite plot has slightly *more* high-frequency noise than Heavy, but its peaks/valleys land *closer to* the dotted GolfDB event lines. That's why it scores higher on PCE.
"""),

    # =========================================================================
    # §4 — 3D pose lifting
    # =========================================================================
    md("""<a id="4"></a>
## §4 — 3D pose lifting (MotionBERT)

The single biggest accuracy win we found. Adding **MotionBERT-Full** as a 3D lifter on top of MediaPipe Lite jumps PCE@5 from 0.090 → **0.170** (+89%).

### Architecture choices

- **MotionBERT-Lite** — 10.7M params, DSTformer with `dim_feat=256, mlp_ratio=4`
- **MotionBERT-Full** — 33M params, same DSTformer with `dim_feat=512, mlp_ratio=2`

Both:
- Take a 243-frame sliding window of 17 H36M keypoints (2D) as input
- Output 17 H36M keypoints in 3D per frame
- Pretrained on AMASS + Human3.6M then fine-tuned on H36M-SH for 3D pose

### Skeleton conversion: COCO-17 → H36M-17

MediaPipe / YOLO / MoveNet / ViTPose all output COCO-17 (eyes, ears, nose + body). MotionBERT was trained on H36M-17 (hip-center, spine, thorax, neck, head + body, no face). We project between them via `eval_utils.coco17_to_h36m17`:
- Derive `hip_center` as midpoint of L+R hips
- Derive `thorax` as midpoint of L+R shoulders
- Derive `spine` as midpoint of (hip_center, thorax)
- Derive `neck` as midpoint of (thorax, nose)
- Map `head ← nose`
- All other body joints pass through 1:1

### Why MotionBERT-Full wins

**3D wrist + shoulder Y trajectories, BEFORE and AFTER smoothing (clip 0):**

![3D Y trajectory before/after](""" + VIZ_DIR_REL + """/clip0_3d_ytraj_beforeafter_motionbert_full_from_mediapipe_lite.png)

The "BEFORE" panel is the raw lifter output — already smoother than the input 2D (compare to the §3 plot). The "AFTER" panel after One-Euro + bone-lock smoothing is essentially perfect — no high-frequency noise, sharp transitions preserved at impact.

### The Z-axis the lifter recovers

This is the value-add over any pure 2D method — depth coordinates the camera couldn't measure directly:

![3D X/Y/Z panels](""" + VIZ_DIR_REL + """/clip0_xyz_panels_motionbert_full_from_mediapipe_lite.png)

The bottom panel (Z) shows the depth axis. The wrists move forward (negative Z) during the backswing and back (positive Z) during downswing — physically correct golf-swing geometry.
"""),

    # =========================================================================
    # §5 — GolfPose surprise
    # =========================================================================
    md("""<a id="5"></a>
## §5 — The surprise: GolfPose vs MotionBERT

We expected **golf-specific fine-tuning (GolfPose)** to beat **generic SOTA pretraining (MotionBERT)** because GolfPose was trained on actual golf swing motion capture (17 golfer + 5 club keypoints) while MotionBERT was trained on generic motion (AMASS + H36M).

**The opposite happened.** Head-to-head, same upstream 2D backbone in each pair:

| Upstream 2D | GolfPose 17+0 | MotionBERT-Full | MotionBERT-Lite |
|---|---:|---:|---:|
| MediaPipe Lite | 0.076 | **0.170** (+124%) | 0.101 (+33%) |
| ViTPose Base | 0.050 | **0.118** (+136%) | 0.060 (+20%) |
| MediaPipe Heavy | 0.042 | **0.090** (+114%) | 0.051 (+21%) |

Worse: **GolfPose ← MediaPipe Lite (0.076) is below MediaPipe Lite alone (0.090)**. The golf-fine-tuned 3D lift actually *hurt* the underlying signal.

### Multi-model right-wrist trajectory on clip 0

You can see the disagreement directly:

![multi-model wrist trajectory](""" + VIZ_DIR_REL + """/clip0_multi_model_wrist.png)

GolfPose (orange) has visible jitter spikes throughout the swing — much closer to the raw 2D models than to MotionBERT. The two MotionBERT variants (red, purple) track each other closely and align tightly with the dotted GolfDB swing events.

### Why does golf-specific lose?

1. **Tiny training set.** GolfPose's 17+0 variant was fine-tuned on ~14,000 images from 4 subjects. MotionBERT used ~11,000 hours of AMASS mocap covering diverse activities, then fine-tuned on H36M-SH.
2. **Resolution mismatch.** GolfPose trained on 1280×720 phone video. GolfDB clips are pre-cropped + resized to 160×160. Our normalization scale is ~8× off from what GolfPose expects.
3. **17+0 is the ablation, not the flagship.** The paper's headline result uses the 22-keypoint variant (17 body + 5 club) — we can't run that without an external club tracker.

### Implication for the team

Domain-specific fine-tuning is **not** a guaranteed win on out-of-distribution data. If we ever invest in a golf-fine-tune, we need a **much larger** golf swing corpus that matches our deployment conditions (real phone uploads, diverse golfers, varied camera angles). For now, generic-but-bigger pretraining is the better bet.
"""),

    # =========================================================================
    # §6 — Smoothing layer
    # =========================================================================
    md("""<a id="6"></a>
## §6 — The smoothing layer

The MotionBERT 3D output is per-frame independent (no temporal smoothing inside the lifter despite the 243-frame receptive field — the window is used for context, but consecutive overlapping windows produce slightly inconsistent outputs that we average). That residual frame-to-frame inconsistency would show up as jitter in UE5.

The smoothing layer (`Scripts/smoothing.py`) gives the pipeline three complementary cleanups before handoff:

1. **Gap interpolation** — linear-fills runs of consecutive low-confidence frames (default ≤ 5 frames).
2. **Temporal filter** — One-Euro (default, adaptive: heavy smoothing on slow phases, snappy on impact) or Savitzky-Golay for offline use.
3. **Bone-length lock** — for each non-root joint, slide its position along the parent→child direction so the bone length equals the clip-wide median. Anatomically a person's bones don't change length; the lock removes the residual scale drift.

### Bone-length stability before/after (clip 1292, slow-mo face-on)

This is the most striking visualization of the smoothing impact. Red lines are raw 3D bone lengths over time; green lines are after smoothing.

![bone stability before/after](""" + VIZ_DIR_REL + """/clip1292_bone_stability_motionbert_full_from_mediapipe_lite.png)

Note the CV (coefficient of variation) drops from ~0.05-0.27 down to **0.000** after bone-lock — bones are now perfectly rigid frame-to-frame. This is anatomically correct (real bones don't change length).

### Acceleration distribution

Histogram of per-frame acceleration magnitudes across all joints:

![acceleration distribution](""" + VIZ_DIR_REL + """/clip0_accel_dist_motionbert_full_from_mediapipe_lite.png)

The smoothed distribution (green) is collapsed toward zero, with the long tail of high-acceleration frames eliminated. Mean acceleration drops by 79% on clip 0 with default smoothing parameters.

### Parameters (tunable per-clip via CLI flags)

| Flag | Default | Effect |
|---|---|---|
| `--smooth-min-cutoff` | 1.0 | Lower = smoother slow phases (more lag) |
| `--smooth-beta` | 0.3 | Higher = snappier fast phases (less lag at impact) |
| `--smooth-window` | 7 | Savgol window size (offline mode only) |
| `--no-bone-lock` | (off) | Disable the bone-length lock (rigid skeleton pass) |
"""),

    # =========================================================================
    # §7 — Pipeline
    # =========================================================================
    md("""<a id="7"></a>
## §7 — End-to-end pipeline

```
                          ┌─────────────────────┐
                          │     swing.mp4       │   single-POV video
                          │  (phone / GolfDB)   │
                          └──────────┬──────────┘
                                     │
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 1   2D POSE ESTIMATION       (per-frame, COCO-17 skeleton)  │
   │  ─────────────────────────────────────────────────────────────     │
   │   ⭐ mediapipe_lite     107 FPS   PCE@5 0.090                      │
   │      mediapipe_heavy / movenet_x / yolov8_x / vitpose_base         │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     │  17 landmarks × T frames (px)
                                     ▼
                      cache: Data/eval_runs/<2d>/<clip>.parquet
                                     │
                                     ▼
                  ┌──────────────────────────────────────┐
                  │   COCO-17 ──► H36M-17 conversion     │
                  └──────────────────┬───────────────────┘
                                     │
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 2   3D LIFT             (sliding 243-frame transformer)     │
   │  ─────────────────────────────────────────────────────────────     │
   │   ⭐ motionbert_full     6,900 FPS   PCE@5 0.170                   │
   │      motionbert_lite   10,500 FPS   PCE@5 0.101                   │
   │      golfpose3d         8,400 FPS   PCE@5 0.076 (don't use)       │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     │  17 H36M joints × T frames (3D)
                                     ▼
                      cache: Data/eval_runs/<3d>/<clip>.parquet
                                     │
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 3   SMOOTHING        (smoothing.py — empirical -79% jitter) │
   │  ─────────────────────────────────────────────────────────────     │
   │      ① gap interpolation     fills runs of conf < 0.3              │
   │      ② temporal filter       one_euro | savgol                     │
   │      ③ bone-length lock      rigidifies skeleton                    │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 4   EXPORT          (export_ue5.py — 6 deliverables)        │
   │  ─────────────────────────────────────────────────────────────     │
   │      <stem>_landmarks_3d.csv  ⭐ UE5 DataTable                     │
   │      <stem>_animation.bvh        UE5 Mocap Plugin / Blender        │
   │      <stem>_mocap.json           canonical metadata                │
   │      <stem>_landmarks_2d.csv     debugging                         │
   │      <stem>_overlay.mp4          h264 2D visualization             │
   │      <stem>_preview_3d.html   ⭐ standalone Three.js viewer        │
   └─────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │       TEAM 2        │
                          │  Unreal Engine 5    │
                          │   Control Rig       │
                          └─────────────────────┘
```

**One command runs the full chain:**

```bash
python Scripts/pipeline.py path/to/swing.mp4 \\
    --backbone mediapipe_lite \\
    --lifter motionbert_full \\
    --smooth oneeuro
```

Output goes to `Data/handoff/<stem>/`.
"""),

    # =========================================================================
    # §8 — UE5 handoff
    # =========================================================================
    md("""<a id="8"></a>
## §8 — What we deliver for Unreal Engine 5

A single pipeline run produces **6 files** in `Data/handoff/<stem>/`:

| File | Format | UE5 use | Recommended? |
|---|---|---|---|
| `<stem>_landmarks_3d.csv` | H36M-17 3D positions, per frame | Import as DataTable → Blueprint reads → Control Rig | ⭐ **Primary** |
| `<stem>_animation.bvh` | BioVision Hierarchy mocap | UE5 Mocap Plugin / Blender / MotionBuilder | Mocap-standard |
| `<stem>_mocap.json` | Self-describing skeleton + frames + events | Canonical intermediate format for any tool | Best for tooling |
| `<stem>_landmarks_2d.csv` | COCO-17 2D landmarks | Debugging, sanity-check | Reference only |
| `<stem>_overlay.mp4` | h264 2D landmark overlay on original | Visual sanity check | Always |
| `<stem>_preview_3d.html` | Standalone Three.js 3D viewer | **Drag into any browser — no install** | ⭐ **Best demo format** |

### Why we picked these formats

- **CSV** because UE5's DataTable system natively reads CSV and exposes rows to Blueprints. Lowest-friction integration.
- **BVH** because it's the industry-standard mocap format. Most pipelines (Maya, Blender, MotionBuilder, UE5 with plugin) accept it.
- **JSON** because it's self-describing — anyone can write a custom parser without our help.
- **HTML preview** because it lets non-technical team members verify the result in 5 seconds without any setup.

### Adding a new model is one file

Every model is a `BaseAdapter` subclass under `Scripts/adapters/`. New adapter = one Python file (~50 lines) + one line in `adapters/__init__.py`. The metrics framework, sweep runner, exporter, and pipeline all auto-discover it.
"""),

    # =========================================================================
    # §9 — Gallery
    # =========================================================================
    md("""<a id="9"></a>
## §9 — Visualization gallery

The 5 clips we picked cover the diversity of the GolfDB corpus:

| Clip | Why it's interesting | Frames | View |
|---|---|---|---|
| **0** | Coworker's reference clip (apples-to-apples) | 138 | other |
| **173** | Easiest (highest mean PCE across models) | 62 | other, real-time |
| **7** | Hardest (lowest mean PCE across models) | 245 | down-the-line, slow-mo |
| **34** | Most divisive (biggest model-to-model disagreement) | 263 | other, slow-mo |
| **1292** | TV broadcast view (face-on, slow-mo) | 399 | face-on, slow-mo |

For each clip we generated **8 visualizations**. The dashboard PNGs below tell the full story per clip in a single image:

### Clip 0 — coworker's reference

![clip 0 dashboard](""" + VIZ_DIR_REL + """/clip0_dashboard.png)

### Clip 173 — easiest case

![clip 173 dashboard](""" + VIZ_DIR_REL + """/clip173_dashboard.png)

### Clip 7 — hardest case

![clip 7 dashboard](""" + VIZ_DIR_REL + """/clip7_dashboard.png)

### Clip 34 — most divisive

![clip 34 dashboard](""" + VIZ_DIR_REL + """/clip34_dashboard.png)

### Clip 1292 — slow-mo face-on (TV broadcast view)

![clip 1292 dashboard](""" + VIZ_DIR_REL + """/clip1292_dashboard.png)

To regenerate any of these with different model/lifter settings:

```bash
python Scripts/make_visualizations.py --clip <N> --lifter motionbert_full --backbone mediapipe_lite
```
"""),

    # =========================================================================
    # §10 — Next steps
    # =========================================================================
    md("""<a id="10"></a>
## §10 — What's next

Concrete actions, ranked by expected impact:

### 1. Build a smarter swing-event detector (highest impact)

Current PCE@5 of 0.170 is bottlenecked by our deliberately simple wrist-Y argmin/argmax detector. A small 1D-CNN or LSTM trained over MotionBERT-Full's 3D landmark trajectories — supervised on GolfDB's 1,400 labels — would likely **3-5×** the PCE.

**ETA:** ~1 week. **No new data needed.**

### 2. Wire the BVH/CSV output into a UE5 Control Rig

Team 2 imports `Data/handoff/0/0_landmarks_3d.csv` as a DataTable and drives a humanoid Control Rig from the rows. The 3D HTML preview validates the geometry independently before we commit to a UE5 implementation.

**ETA:** ~1 week.

### 3. Collect real phone-video data and re-benchmark

Everything here is on GolfDB's pre-cropped, resized 160×160 clips. Real phone uploads will have:
- Full-frame video (golfer is a small portion)
- Variable resolutions / aspect ratios
- More background clutter
- Camera shake

We need to verify the top-3 pipelines still work in those conditions. **20 hand-captured clips** would be enough.

**ETA:** ~1 week.

### 4. Add WHAM or TRAM as a world-grounded 3D contender

Both solve foot-skate by design (output is in world coordinates with stable ground plane). Worth running on top of the recommended pipeline for the UE5 demo, especially if Team 2 sees foot-sliding artifacts.

**ETA:** ~1 week each.

### 5. Fine-tune MotionBERT-Full on golf data (longest investment, biggest payoff if it works)

Now that we know MotionBERT-Full is the strong baseline, the question becomes: can we fine-tune it on golf-specific 3D supervision and beat the generic baseline?

This requires either:
- Synthetic SMPL renders of golf swings (technically tractable, animation-quality concerns)
- Hand-aligned Vicon ground truth on a few hundred clips (expensive)
- Commercial annotation (Sportsbox AI or similar)

**ETA:** 2-4 weeks. **Significant data investment required.**

---

## Files & artifacts

| What | Where |
|---|---|
| **End-to-end pipeline** | `Scripts/pipeline.py` |
| **Smoothing layer** | `Scripts/smoothing.py` |
| **UE5 format exporters** | `Scripts/export_ue5.py` |
| **Visualization generator** | `Scripts/make_visualizations.py` |
| **This report's generator** | `Scripts/make_report.py` |
| Reusable metrics + adapter base | `Scripts/eval_utils.py` |
| Model adapters (20) | `Scripts/adapters/` |
| Full evaluation notebook | `Scripts/model_evaluation_baseline.ipynb` |
| Metric aggregator | `Scripts/compute_metrics.py` |
| GolfPose sanity check | `Scripts/golfpose_sanity_check.py` |
| Per-clip cached landmarks | `Data/eval_runs/<model>/<clip_id>.parquet` |
| Aggregated metrics (28,000+ rows) | `Data/all_metrics.parquet` |
| Per-model overlay MP4s | `Data/overlays/<model>/<clip>_<model>.mp4` |
| Side-by-side grid MP4s | `Data/overlays/sidebyside/<clip>_compare.mp4` |
| 40 visualization PNGs (this report) | `Data/visualizations/` |
| Demo handoff bundle (clip 0) | `Data/handoff/0/` |
| GolfPose dataset 3D ground truth | `golfpose_data/data_3d_golf_gt.npz` |
| Frozen Python environment | `requirements_eval.txt` |

---

*Generated by `Scripts/make_report.py`. To rebuild: `python Scripts/make_report.py`.*
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (golf-capstone)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
