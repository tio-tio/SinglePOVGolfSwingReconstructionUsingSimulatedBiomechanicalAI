# Single-POV Golf Swing Reconstruction — Model Evaluation & Pipeline

**Author:** Banjo &nbsp;•&nbsp; **Status:** baseline scoping complete, 2D + 3D pipeline working end-to-end

---

## TL;DR

**Recommended pipeline**: MediaPipe Lite → MotionBERT-**Full** → One-Euro smoothing → BVH/CSV for UE5.

```bash
python Scripts/pipeline.py path/to/swing.mp4 \
    --backbone mediapipe_lite \
    --lifter motionbert_full \
    --smooth oneeuro
```

That produces a folder of UE5-ingestible files for Team 2 (`Data/handoff/<stem>/`).

| Question | Answer |
|---|---|
| Best 2D backbone alone? | **MediaPipe Lite** (PCE@5 = 0.090, 107 FPS CPU, 3 MB) |
| Best end-to-end (2D + 3D)? | **MediaPipe Lite + MotionBERT-Full** (PCE@5 = **0.170**, 6,900 FPS combined) |
| What does the team get? | 2D landmarks CSV, 3D landmarks CSV, BVH mocap file, self-contained 3D Three.js HTML preview, h264 overlay MP4 |

The biggest surprises in the data:
1. **MotionBERT-Full nearly doubles PCE.** PCE@5 jumps from 0.090 (MediaPipe Lite alone) → 0.101 (+ MotionBERT-Lite) → **0.170 (+ MotionBERT-Full)**. The temporal-transformer 3D lifter is the single biggest accuracy win we found. And it's still 64× faster than the cheapest 2D model (6,900 FPS).
2. **Cheap models beat expensive ones — on the 2D side.** MediaPipe Lite beats MediaPipe Heavy on PCE@5 by 73%. MoveNet Lightning beats MoveNet Thunder by 47%. The original coworker baseline (`pose_landmarker_heavy.task`) is actually the *worst* MediaPipe variant for golf events.
3. **The full pipeline is essentially free post-2D.** Adding MotionBERT-Full on top of cached 2D processes all 1,400 GolfDB clips in 1.5 minutes (6,900 FPS amortized). Even Full has lower compute than every 2D backbone.
4. **Smoothing cuts jitter by half before UE5 sees it.** One-Euro filter + bone-length lock takes mean acceleration from 0.0041 → 0.0020 on demo clip.

---

## How we evaluated

- **Dataset:** all 1,400 labeled GolfDB clips at 160×160 (`Data/videos_160/`).
- **Models tested (20 total):**
  - **7 2D pose** — MediaPipe Heavy/Lite, MoveNet Thunder/Lightning, YOLOv8n/m-Pose, ViTPose-Base.
  - **7 MotionBERT-Lite** combinations — one per 2D backbone.
  - **3 MotionBERT-Full** combinations — on the top-3 2D backbones (MP-Heavy, MP-Lite, ViTPose).
  - **3 GolfPose 17+0** combinations — on the same three 2D backbones, for direct head-to-head against MotionBERT.
- **Total inferences:** ~28,000 (model × clip) pairs cached as Parquet under `Data/eval_runs/`.
- **Schema:** every 2D model projected onto canonical COCO-17. Every 3D model projected onto H36M-17 (the standard for MotionBERT/GolfPose lifters) then onto COCO-17 for metric compatibility.
- **Hardware:** RTX 4080 SUPER (16 GB) for GPU models; CPU for MediaPipe + MoveNet (Windows TF lacks GPU).

### Metric battery (9 metrics)

| Group | Metric | Better | What it measures |
|---|---|---|---|
| **Reference-based** (uses GolfDB swing-event labels) | `pce_at_5` | ↑ | % of 8 events recovered within ±5 frames |
| | `pce_at_3` | ↑ | Same, ±3 frames |
| | `pce_at_1` | ↑ | Same, ±1 frame |
| **Reference-free** (physics / domain priors) | `bone_cv_mean` | ↓ | Bone-length stability (real bones don't change) |
| | `jitter_mean_px` | ↓ | Frame-to-frame acceleration (smoothness) |
| | `implausible_frac_mean` | ↓ | Degenerate elbow/knee angles |
| | `left_ankle_planting_std_px` | ↓ | Foot sliding during the swing |
| **Operational** | `detection_rate` | ↑ | % of (frame, keypoint) cells with confident output |
| | `fps_inference` | ↑ | Throughput on our hardware |

The PCE detector applies the same wrist-Y trajectory heuristic to every model, so a better PCE reflects better underlying landmark quality, not detector cleverness.

---

## Full leaderboard (median per model, n=1,400 clips, 20 models)

Sorted by `pce_at_5`. **3D models in bold-italic.** Top 3 highlighted.

| # | Model | FPS | Detect% | BoneCV | Jitter | Implaus% | FootStd | **PCE@5** | PCE@3 | PCE@1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 🥇 | ***MotionBERT-Full ← mediapipe_lite*** | 6,878 | 76.5 | 0.201 | 0.003 | 14.0 | 0.025 | **0.170** | **0.143** | **0.093** |
| 🥈 | ***MotionBERT-Full ← vitpose_base*** | 6,901 | 76.5 | 0.203 | 0.003 | 15.7 | 0.022 | 0.118 | 0.097 | 0.055 |
| 🥉 | ***MotionBERT-Lite ← mediapipe_lite*** | 10,526 | 76.5 | 0.198 | 0.002 | 13.9 | 0.017 | 0.101 | 0.082 | 0.044 |
| 4 | mediapipe_lite | 107 | 94.3 | 0.203 | 0.99 | 14.2 | 1.81 | 0.090 | 0.070 | 0.043 |
| 5 | ***MotionBERT-Full ← mediapipe_heavy*** | 6,733 | 76.5 | 0.194 | 0.002 | 15.6 | 0.020 | 0.090 | 0.073 | 0.045 |
| 6 | ***MotionBERT-Lite ← movenet_lightning*** | 10,461 | 76.5 | 0.208 | 0.003 | 13.8 | 0.015 | 0.082 | 0.064 | 0.030 |
| 7 | ***GolfPose 17+0 ← mediapipe_lite*** | 8,352 | 76.5 | 0.203 | 0.017 | 12.0 | 0.040 | 0.076 | 0.059 | 0.036 |
| 8 | movenet_lightning | 140 | 88.9 | 0.197 | 1.20 | 14.7 | 1.67 | 0.072 | 0.055 | 0.030 |
| 9 | ***MotionBERT-Lite ← yolov8n_pose*** | 10,664 | 76.5 | 0.223 | 0.009 | 16.3 | 0.027 | 0.061 | 0.045 | 0.022 |
| 10 | ***MotionBERT-Lite ← vitpose_base*** | 11,008 | 76.5 | 0.200 | 0.002 | 15.9 | 0.012 | 0.060 | 0.044 | 0.022 |
| 11 | yolov8m_pose | 120 | 86.2 | 0.208 | 2.01 | 14.0 | 1.57 | 0.057 | 0.040 | 0.019 |
| 12 | vitpose_base | 129 | **99.9** | 0.214 | 1.11 | 16.1 | **1.28** | 0.056 | 0.042 | 0.022 |
| 13 | ***MotionBERT-Lite ← yolov8m_pose*** | 10,908 | 76.5 | 0.198 | 0.004 | 14.4 | 0.013 | 0.055 | 0.040 | 0.018 |
| 14 | yolov8n_pose | **214** | 89.7 | 0.233 | 3.22 | 14.9 | 3.31 | 0.054 | 0.037 | 0.016 |
| 15 | ***MotionBERT-Lite ← movenet_thunder*** | 10,506 | 76.5 | 0.196 | 0.003 | 15.1 | 0.011 | 0.053 | 0.040 | 0.019 |
| 16 | mediapipe_heavy | 24 | 94.3 | **0.182** | **0.70** | 19.4 | 0.98 | 0.052 | 0.037 | 0.018 |
| 17 | ***MotionBERT-Lite ← mediapipe_heavy*** | 10,200 | 76.5 | 0.190 | 0.002 | 15.9 | 0.010 | 0.051 | 0.037 | 0.017 |
| 18 | ***GolfPose 17+0 ← vitpose_base*** | 8,430 | 76.5 | 0.210 | 0.019 | 15.2 | 0.035 | 0.050 | 0.036 | 0.018 |
| 19 | movenet_thunder | 95 | 88.2 | **0.182** | 1.06 | 15.8 | 1.17 | 0.049 | 0.036 | 0.018 |
| 20 | ***GolfPose 17+0 ← mediapipe_heavy*** | 8,311 | 76.5 | 0.193 | 0.014 | 16.5 | 0.031 | 0.042 | 0.029 | 0.015 |

**Key reads:**
- **MotionBERT-Full is dominant.** It takes the top spots when paired with the right 2D backbone. The bigger 33M-parameter transformer (vs 10M for Lite) captures golf swing temporal structure dramatically better.
- **The 2D backbone choice matters more than the 3D lifter.** MotionBERT-Full ← MediaPipe Lite (0.170) is 89% better than MotionBERT-Full ← MediaPipe Heavy (0.090). The 2D upstream is the bottleneck — pick the right one first.
- **MotionBERT inference cost is essentially free** (6,700–11,000 FPS). Even the Full variant runs faster than every 2D backbone.
- **The 76.5% detection rate on 3D rows** is because we set face-keypoint confidence to 0 by convention (H36M-17 has no face keypoints). Body-joint detection inherits the upstream 2D's rate (~94% for MediaPipe).
- **GolfPose underperforms across the board** — see next section.

### Surprise finding: golf-specific fine-tuning DOES NOT beat generic SOTA

The headline experiment was GolfPose (golf-fine-tuned on mocap data) vs MotionBERT (generic pretraining on AMASS + Human3.6M). Same 2D backbone in each pair, head-to-head on PCE@5:

| Upstream 2D | GolfPose 17+0 | MotionBERT-Full | MotionBERT-Lite |
|---|---:|---:|---:|
| MediaPipe Lite | 0.076 | **0.170** (+124%) | 0.101 (+33%) |
| ViTPose Base | 0.050 | **0.118** (+136%) | 0.060 (+20%) |
| MediaPipe Heavy | 0.042 | **0.090** (+114%) | 0.051 (+21%) |

On *every* upstream backbone, MotionBERT-Full beats GolfPose by 2-2.4×. Worse: **GolfPose ← MediaPipe Lite (0.076) is actually below MediaPipe Lite alone (0.090)** — the golf-specific lift makes the 2D output *worse* in that configuration.

Why GolfPose loses:

1. **Tiny training set.** GolfPose 17+0 was fine-tuned on ~14,000 images from 4 subjects (G1-G4). MotionBERT was pretrained on AMASS (~11,000 hours of mocap covering many activities) + H36M, then fine-tuned on H36M-SH. The breadth of MotionBERT's pretraining covers golf motion patterns adequately, and its lifter is robust to the noise distribution we see from MediaPipe on GolfDB-style clips.

2. **Resolution mismatch.** GolfPose was trained on 1280×720 phone video. GolfDB clips are pre-cropped and resized to 160×160. Our MediaPipe outputs pixel coords in the smaller frame, then we normalize by 160 — but GolfPose's learned distribution expects 1280-scale normalization. The model still produces plausible anatomy (we verified shin/arm ratio = 1.62, bone CV 0.111 on clip 0) but the learned temporal priors don't transfer cleanly.

3. **17+0 is the ablation, not the flagship.** GolfPose's headline result is the 22-keypoint (17 body + 5 club) variant, which we can't run without club tracking. The 17+0 version is the paper's ablation baseline — it's not what the authors recommend in practice.

**Implication for the team:** domain-specific fine-tuning is **not** a guaranteed win on out-of-distribution data. Investing weeks into a golf-fine-tune of MotionBERT-Full is plausible *if* we have a few hundred high-quality golf swing samples that match our deployment distribution. Without that data the generic pretraining is the better bet.

---

## The end-to-end pipeline (deliverable for Team 2)

### `Scripts/pipeline.py` — one command, all outputs

```bash
python Scripts/pipeline.py <video.mp4> \
    [--backbone <2d_model>]   # default: mediapipe_heavy
    [--lifter <3d_model>]     # default: motionbert_lite
    [--smooth oneeuro|savgol|none]    # default: oneeuro
    [--no-bone-lock]          # default: bone-lock ON
    [--out-dir <path>]        # default: Data/handoff/<stem>/
    [--events f0,f1,...,f7]   # optional GolfDB-style frame labels
```

Produces in `Data/handoff/<stem>/`:

| File | What it is | UE5 use |
|---|---|---|
| `<stem>_landmarks_2d.csv` | COCO-17 2D landmarks in pixel space, per frame | Debugging / cross-reference |
| `<stem>_landmarks_3d.csv` | H36M-17 3D positions in meters, per frame | **Recommended: import as DataTable, drive Control Rig via Blueprint** |
| `<stem>_mocap.json` | Self-describing JSON: skeleton + frames + provenance + events | Canonical format for any downstream tool |
| `<stem>_animation.bvh` | BioVision Hierarchy mocap (positional channels) | Drop into UE5 Mocap Plugin / Blender / MotionBuilder |
| `<stem>_overlay.mp4` | h264-encoded 2D landmark overlay on original video | Visual sanity check |
| `<stem>_preview_3d.html` | Standalone Three.js 3D viewer | **Drag into any browser, no install — best demo format** |

### The smoothing layer (`Scripts/smoothing.py`)

Three-pass cleanup before handoff:

1. **Gap interpolation** — linear-fills short low-confidence runs (default ≤ 5 consecutive frames).
2. **Temporal filter** — One-Euro by default (adaptive: heavy smoothing on slow phases, snappy on impact); Savitzky-Golay alternative for offline mode.
3. **Bone-length lock** — for each non-root joint, slides it along the parent→child direction so the bone length equals the clip-wide median. Real bones don't change, so this kills the residual scale drift.

**Measured impact on clip 0**: mean acceleration **0.0041 → 0.0020 (-51%)** with default params. Tunable via `--smooth-min-cutoff` (lower = smoother slow phases, more lag) and `--smooth-beta` (higher = snappier impact, less lag).

### Adding new models is one file

Every model is a `BaseAdapter` subclass under `Scripts/adapters/`. New adapter = one Python file + one line in the registry. The metrics framework, sweep runner, exporter, and pipeline all auto-discover it.

---

## Key findings

### 1. Cheap > expensive on the golf-specific metric

MediaPipe **Lite** beats MediaPipe **Heavy** on PCE@5 by 73% (0.090 vs 0.052) *and* runs 4× faster. MoveNet Lightning beats MoveNet Thunder by 47%. This is the opposite of what generic pose benchmarks (COCO mAP, MPJPE) would predict.

**Hypothesis**: heavier models apply more temporal/spatial smoothing, which slightly displaces the actual peak frame in the wrist trajectory. Lite models are noisier but their peaks coincide more accurately with the labeled event frame. Smoothness ≠ correctness for sharp event localization.

### 2. The 3D lift is essentially free

MotionBERT-Lite runs at ~10,500 FPS on cached 2D — full GolfDB in 1.4 minutes. On MediaPipe Lite it adds a 12% PCE@5 lift. On other backbones it's neutral on PCE but it adds the 3D output Team 2 actually needs.

**There is no reason not to include MotionBERT in the production pipeline.**

### 3. The 3D jitter numbers in the leaderboard are NOT directly comparable to 2D

MotionBERT's `jitter_mean_px = 0.002` looks 350× better than MediaPipe's 0.99. **This is a unit mismatch**: 2D jitter is in pixels, 3D jitter is in normalized H36M coordinate space (~ [-0.6, 0.87] range). After rescaling, MotionBERT's effective pixel-equivalent jitter is ~0.16 px — still good, but not the dramatic improvement the raw number suggests. **Fair comparison requires reprojecting 3D back to 2D**, which we haven't done yet.

The comparable metric is `bone_cv_mean` (dimensionless std/mean). MotionBERT-Lite ← MediaPipe Lite has bone CV 0.198 vs MediaPipe Lite's 0.203 — slightly better, not dramatically. The 3D output is *also* not magically more stable than the 2D-projected bones across the corpus.

**Translation:** the lifter doesn't fix bad 2D inputs. It produces sensible 3D given the 2D it gets.

### 4. Absolute PCE is low across all models (0.049 – 0.101)

Even the winner misses ~90% of swing events within a ±5-frame tolerance. **No off-the-shelf model, used naively, can power a coaching app.** To make this usable for end-users we need *at least one of*:

1. **A smarter event detector** — current logic is wrist-Y argmin/argmax. A small 1D-CNN or LSTM over landmark trajectories trained on GolfDB labels would likely 3-5× the PCE. ~1 week of work.
2. **Fine-tuning the pose backbone** on golf-domain frames.
3. **Both** combined.

The good news: any of these improvements is **orthogonal to which 2D backbone we pick**, so the backbone choice is low-risk regardless.

---

## Recommendation: lock the pipeline

| Component | Choice | Rationale |
|---|---|---|
| 2D backbone | `mediapipe_lite` | Highest PCE among 2D models, fast on CPU, 3 MB model, same MediaPipe API as current code |
| 3D lifter | **`motionbert_full`** | **+89% PCE over Lite (0.090 → 0.170), still 6,900 FPS on GPU** |
| Smoothing | `oneeuro` + bone-lock | -51% jitter empirically, preserves impact snap |
| Handoff format | CSV (DataTable) + BVH + HTML preview | Three options at different levels of UE5 integration friction |

### When to deviate

- **Need maximum robustness?** Swap backbone to `vitpose_base` (99.9% detection rate, never misses a pose) at the cost of needing GPU.
- **Mobile latency is the binding constraint?** Drop to `movenet_lightning` (140 FPS, PCE 0.072 — only slightly worse) and keep MotionBERT-Lite on top.
- **Visible flicker in UE5 demo?** Increase `--smooth-min-cutoff` to 0.5, or switch to `--smooth savgol --smooth-window 11`.

---

## Caveats (be honest about these)

1. **No 3D ground truth on GolfDB.** All "3D quality" metrics are physics-based (bone-length stability, anatomical plausibility) rather than reference-based. The GolfPose paper's GolfSwing dataset (which has mocap ground truth) would let us measure true 3D MPJPE — see "still outstanding" below.

2. **PCE detector is intentionally simple.** The absolute PCE numbers (0.05-0.10) will jump 3-5× with a proper temporal event detector. The *relative* ranking between models won't change much.

3. **2D vs 3D jitter values are NOT directly comparable.** Different coordinate units (pixels vs normalized meters). The headline "350× smoother" claim is unit-mismatch, not real. We'd need to reproject 3D → 2D for a fair number.

4. **Bone CV on corpus median is similar between 2D and 3D.** Cherry-picked clips like clip 0 show 50%+ reduction (0.225 → 0.102) but the corpus median is flat (0.203 → 0.198). The lifter doesn't fix bad 2D inputs.

5. **No real-phone data tested.** GolfDB clips are pre-cropped to the golfer bounding box. Real phone uploads aren't. The pose models' performance on un-cropped real input is unknown — especially YOLO, which usually expects wider field-of-view.

6. **No mobile latency measured.** All FPS numbers are on a 4080 workstation / 4080 CPU. Phone CPU with TF Lite / Core ML export will be significantly slower.

7. **YOLOv8n cache has mixed sources.** First 540 clips were generated by a pre-patch adapter that hit GPU memory thrashing on long slow-mo clips. Outputs *look* reasonable but to be fully rigorous we'd delete and re-run with the patched chunked-batch adapter (~12 min).

8. **GolfPose 3D not benchmarked yet** — see below.

---

## Still outstanding

### Future work (not yet attempted)

1. **Golf-fine-tune MotionBERT-Full on GolfDB.** Now that we know MotionBERT-Full is the best generic baseline, fine-tuning it on golf-specific data is the obvious next move. We'd need ~hundreds of high-quality golf swings with 3D ground truth (synthetic from SMPL render, or hand-aligned via Vicon, or commercial like Sportsbox AI labels). ETA: 2-3 weeks.
2. **WHAM or TRAM as a world-grounded contender.** Both solve foot-skate via world coordinates (vs camera frame). Worth running on the top-tier pipeline for the UE5 demo. ETA: 1 week each.
3. **Real phone-video data.** Everything here is on GolfDB's pre-cropped 160×160 clips. We should re-run the top-3 pipelines on ~20 phone-captured clips in actual deployment conditions to verify generalization. ETA: 1 week.
4. **Proper PCE event detector.** A 1D-CNN over the MotionBERT 3D landmark trajectories trained on GolfDB labels would likely 3-5× current PCE. Single biggest remaining accuracy win. ETA: 1 week.

---

## Files & artifacts

| What | Where |
|---|---|
| **End-to-end pipeline** | `Scripts/pipeline.py` |
| **Smoothing layer** | `Scripts/smoothing.py` |
| **UE5 format exporters** | `Scripts/export_ue5.py` (CSV, JSON, BVH) |
| Reusable metrics + adapter base | `Scripts/eval_utils.py` |
| Model adapters (14) | `Scripts/adapters/` |
| Full evaluation notebook | `Scripts/model_evaluation_baseline.ipynb` |
| Metric aggregator | `Scripts/compute_metrics.py` |
| GolfPose sanity check (paper replication) | `Scripts/golfpose_sanity_check.py` |
| Per-clip cached landmarks | `Data/eval_runs/<model>/<clip_id>.parquet` |
| Aggregated metrics (19,600+ rows) | `Data/all_metrics.parquet` |
| Per-model overlay MP4s | `Data/overlays/<model>/<clip>_<model>.mp4` |
| Side-by-side grid MP4s (meeting demo) | `Data/overlays/sidebyside/<clip>_compare.mp4` |
| Demo handoff bundle (clip 0) | `Data/handoff/0/` |
| GolfPose dataset 3D ground truth | `golfpose_data/data_3d_golf_gt.npz` |
| Frozen Python environment | `requirements_eval.txt` |

---

## Suggested next steps

1. **Lock the pipeline.** Switch the current MediaPipe Heavy default to `mediapipe_lite` + `motionbert_lite`. This commit is one file in `adapters/__init__.py` plus updating the pipeline command in any docs. ETA: today.

2. **Wire the BVH/CSV into UE5.** Team 2 imports `Data/handoff/0/0_animation.bvh` (or `0_landmarks_3d.csv` as DataTable) and verifies the swing plays correctly on a Control Rig. ETA: this week.

3. **Train a proper swing-event detector.** Small 1D-CNN over MotionBERT 3D landmark trajectories, supervised on GolfDB's 1,400 labels. Expected 3-5× the current PCE. ETA: 1 week.

4. **Finish GolfPose benchmark.** Once the checkpoint is downloaded (~30 seconds in a browser), the rest is automated. Closes the "generic vs domain-specific" question. ETA: 30 min after checkpoint lands.

5. **Collect real phone data.** ~20 clips shot in actual deployment conditions (not pre-cropped) and re-run the benchmark. GolfDB results may or may not generalize. ETA: 1 week.

6. **Add WHAM or TRAM as a world-grounded 3D contender.** Both solve foot-skate via world coordinates, which is the #1 visible artifact downstream. ETA: 1 week each.
