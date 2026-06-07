# Checkpoint 1 — Presentation Assets

**For:** Motion Caddie GD.pptx (slides 1-18 are the real content; 19-47 are leftover template — delete them).

Everything in this folder is sized and exported for direct PowerPoint embed (videos are h264, images are 140 DPI PNG, sample data is plain CSV/JSON/BVH that opens in Excel/notepad).

---

## 🏆 The single most important asset

### `videos/01_pipeline_progression_slowmo.mp4`

This 1.2 MB MP4 tells the **entire pipeline story in one asset**. Slow-mo face-on swing, three panels side-by-side, frame-locked:

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  STAGE 1            │  STAGE 2            │  STAGE 3            │
│  2D LANDMARKS       │  3D LIFT (raw)      │  SMOOTHED 3D        │
│  (coworker style)   │  (jittery red)      │  (clean green)      │
│  mediapipe_heavy    │  motionbert_full    │  oneeuro+bone-lock  │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Drop on Slide 11 (Technical Pipeline v1) — set autoplay + loop.** This single video justifies the entire pipeline diagram on that slide.

If you only have time to add one thing, add this.

---

## 🥈 Second-most important: live 3D viewer

### `interactive/live_3d_preview.html`

**Open this in Chrome and use it as a live demo during the meeting.** Mouse-orbit the swing in 3D, pause/play, watch the labeled swing events tick by. It's standalone — no install, no internet, single file.

**Use during Slide 4 (Amateur golfer journey)** as a live demo for the "3D Replay" stage, or as a closing flourish before Q&A.

---

## 🥉 Third-most important: the dashboard PNG

### `images/01_full_story_dashboard.png`

Single image containing five sub-plots: 2D landmarks, 3D landmarks before/after, two bone-length traces, and the acceleration histogram. The yellow-boxed text on the right is the headline number: **"79% jitter reduction"**.

**Drop on Slide 12 or 17 (Evaluation Metrics).** It shows the metrics framework working end-to-end in one image.

---

## Full asset → slide mapping

### Videos (`videos/`)

| File | Slide | Why |
|---|---|---|
| **`01_pipeline_progression_slowmo.mp4`** | **11 (Technical Pipeline v1)** ⭐ | The whole story in one asset |
| `02_pipeline_progression_clip0.mp4` | 4 (Amateur golfer journey) — "Pose Overlay" + "3D Replay" steps | Same clip as coworker's reference |
| `03_smoothing_before_after.mp4` | 12 / 17 (Evaluation Metrics) — "Pose Stability" stage box | Visual proof of jitter reduction |
| `04_seven_model_comparison_slowmo.mp4` | 16 (Pipeline v2 / Baseline vs Stretch) | 7 pose models on the same swing, dramatizes the model-choice decision |
| `05_clip0_2d_overlay_only.mp4` | 7 (Example Swing) | Single-stage 2D overlay, matches coworker's simpler visualization |

### Images (`images/`)

| File | Slide | Why |
|---|---|---|
| **`01_full_story_dashboard.png`** | **12 / 17 (Evaluation Metrics)** ⭐ | Single-image full pipeline story |
| `02_bone_stability_proof_slowmo.png` | 9 (EDA: Features) | 8 bones over time — red bouncing → green flat. Proves the bone-lock smoother works |
| `03_smoothing_before_after_plot.png` | 12 (Pose Stability) | 3D Y-trajectory before/after smoothing |
| `04_multi_model_comparison.png` | 8 (Data reliability) | 5 models' wrist trajectory on same clip — model choice matters |
| `05_3d_xyz_depth_panels.png` | 10 (2D → 3D mapping) | Shows the Z-axis (depth) the lifter recovers — the value-add over 2D |
| `06_coworker_reference_2d_plot.png` | 7 or 15 | Recreates the exact plot from coworker's notebook |
| `07_recommended_backbone_2d_plot.png` | 7 or 12 | Same plot, but on the recommended backbone (MediaPipe Lite) |
| `08_acceleration_distribution.png` | 12 (Pose Stability metric box) | Histogram showing distribution-level smoothing impact |
| `09_progression_3panel_still.png` | 11 (Technical Pipeline) — as the visual anchor | Single still frame from the progression video — use if you can't embed video |

### Sample data (`sample_data/`) — for Slide 15

Slide 15 currently says "**ILLUSTRATIVE TARGET OUTPUT**". Replace the placeholder landmark values with real values from these files:

| File | Use |
|---|---|
| `landmarks_2d.csv` | Open in Excel, copy first 8 rows into the table on Slide 15 |
| `landmarks_3d.csv` | Same — pick whichever (2D or 3D) you want to feature |
| `animation.bvh` | Mention as the BVH mocap deliverable. Can drag into Blender to show it works |
| `mocap.json` | Reference as the "canonical metadata" format |

### Reports (`reports/`)

| File | Use |
|---|---|
| `RESULTS.md` | Speaker notes / Q&A reference. Don't embed — keep open on your laptop |
| `REPORT_standalone.html` | Send to teammates pre-meeting; open as the deep-dive companion |

---

## Slide-by-slide quick-reference checklist

Use this as your editing pass through the deck:

- [ ] **Slide 1 (Title)** — optional: drop `images/09_progression_3panel_still.png` as a hero
- [ ] **Slide 2 (Problem Statement)** — text-only, no media (see chat for tightened wording)
- [ ] **Slide 3 (Why It Matters)** — text-only, no media (see chat for tightened wording)
- [ ] **Slide 4 (Amateur golfer journey)**
  - Pose Overlay step → `videos/05_clip0_2d_overlay_only.mp4`
  - 3D-Style Replay step → `interactive/live_3d_preview.html` (open in browser, screenshot if you can't link)
  - Swing Phases step → `images/03_smoothing_before_after_plot.png` (showing labeled events)
- [ ] **Slide 5 (Competitive)** — text-only
- [ ] **Slide 6 (MVP Scope)** — text-only
- [ ] **Slide 7 (Example Swing)** → `videos/05_clip0_2d_overlay_only.mp4` OR `images/06_coworker_reference_2d_plot.png`
- [ ] **Slide 8 (Data reliability)** → `images/04_multi_model_comparison.png`
- [ ] **Slide 9 (EDA: Features)** → `images/02_bone_stability_proof_slowmo.png`
- [ ] **Slide 10 (2D → 3D mapping)** → `images/05_3d_xyz_depth_panels.png` + `images/03_smoothing_before_after_plot.png`
- [ ] **Slide 11 (Technical Pipeline)** → ⭐ `videos/01_pipeline_progression_slowmo.mp4` ⭐
- [ ] **Slide 12 (Evaluation Metrics)** → ⭐ `images/01_full_story_dashboard.png` ⭐ + `videos/03_smoothing_before_after.mp4`
- [ ] **Slide 13 (Plan / Risks)** — text-only (update with our findings — see chat for rewrite)
- [ ] **Slide 14 (Thank you)** — optional: `interactive/live_3d_preview.html` link for Q&A
- [ ] **Slide 15 (GolfDB → Pose Landmarks "illustrative")** → `sample_data/landmarks_3d.csv` rows + `videos/05_clip0_2d_overlay_only.mp4` frame still
- [ ] **Slide 16 (Pipeline v2)** → `videos/04_seven_model_comparison_slowmo.mp4`
- [ ] **Slide 17 (Eval Metrics v2)** → `images/01_full_story_dashboard.png` (or delete — it's a duplicate of 12)
- [ ] **Slide 18 (Plan / Risks v2)** — text-only (or delete — duplicate of 13)
- [ ] **Slides 19-47** → DELETE (leftover board-meeting template)

---

## How to embed videos in PowerPoint

1. Insert → Video → Video on My PC
2. Select the MP4 from this folder
3. With the video selected: **Playback tab** → check **Loop until Stopped** and set **Start: Automatically**
4. **Format tab** → resize to fill the placeholder

For the interactive HTML viewer:
- It's not a media type PowerPoint embeds. Two options:
  - **Live demo**: Alt-Tab to a Chrome window during the meeting
  - **Action button**: Insert → Action → Hyperlink to: `interactive/live_3d_preview.html` (relative if you keep the folder structure intact)

---

## Folder contents (16 files, ~10 MB total)

```
Checkpoint 1/
├── README.md                                    (this file)
├── videos/  (~4.2 MB)
│   ├── 01_pipeline_progression_slowmo.mp4       ⭐ THE one to embed
│   ├── 02_pipeline_progression_clip0.mp4
│   ├── 03_smoothing_before_after.mp4
│   ├── 04_seven_model_comparison_slowmo.mp4
│   └── 05_clip0_2d_overlay_only.mp4
├── images/  (2.4 MB)
│   ├── 01_full_story_dashboard.png              ⭐ THE png to embed
│   ├── 02_bone_stability_proof_slowmo.png
│   ├── 03_smoothing_before_after_plot.png
│   ├── 04_multi_model_comparison.png
│   ├── 05_3d_xyz_depth_panels.png
│   ├── 06_coworker_reference_2d_plot.png
│   ├── 07_recommended_backbone_2d_plot.png
│   ├── 08_acceleration_distribution.png
│   └── 09_progression_3panel_still.png
├── interactive/  (164 KB)
│   └── live_3d_preview.html                     ⭐ THE live demo
├── sample_data/  (680 KB) — for Slide 15
│   ├── landmarks_3d.csv
│   ├── landmarks_2d.csv
│   ├── animation.bvh
│   └── mocap.json
└── reports/  (4.6 MB)
    ├── RESULTS.md                               (open as Q&A reference)
    └── REPORT_standalone.html                   (share with teammates pre-meeting)
```
