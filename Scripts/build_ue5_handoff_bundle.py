"""Generate a ready-to-share UE5 handoff bundle.

Picks N stratified clips, runs the full pipeline (2D->3D->smoothing->export)
on each, writes a README explaining file formats + ingestion steps, then
zips the whole `Data/handoff/` folder for Drive/OneDrive upload.

Usage:
    python build_ue5_handoff_bundle.py                    # 12 clips, default models
    python build_ue5_handoff_bundle.py --n 20             # more clips
    python build_ue5_handoff_bundle.py --lifter motionbert_lite   # cheaper
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
HANDOFF_DIR  = PROJECT_ROOT / "Data" / "handoff"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"

# Always include the 5 demo clips we already use across the rest of the deck
DEMO_CLIPS = [0, 7, 34, 173, 1292]


def pick_clips(n: int, seed: int = 42) -> list[int]:
    """Demo clips + stratified sample of additional clips covering
    view × club × slow diversity."""
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    available = sorted([cid for cid in df.index
                        if (PROJECT_ROOT / "Data" / "videos_160" / f"{cid}.mp4").exists()
                        and cid not in DEMO_CLIPS])
    extras_needed = max(0, n - len(DEMO_CLIPS))
    if extras_needed == 0:
        return DEMO_CLIPS[:n]
    pool = df.loc[available].copy()
    pool["strat"] = (pool["view"].astype(str) + "_" +
                      pool["slow"].astype(str) + "_" + pool["club"].astype(str))
    from sklearn.model_selection import StratifiedShuffleSplit
    n_classes = pool["strat"].nunique()
    if extras_needed < n_classes:
        extras = pool.sample(n=extras_needed, random_state=seed).index.tolist()
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=extras_needed,
                                       random_state=seed)
        _, idx = next(sss.split(pool, pool["strat"]))
        extras = pool.iloc[idx].index.tolist()
    return DEMO_CLIPS + extras


README_TEMPLATE = """# Motion Caddie — UE5 Handoff Bundle

**Generated:** {generated_at}
**Pipeline:** {backbone_2d} → {lifter_3d} → One-Euro + bone-lock smoothing
**Clips included:** {n_clips}

Each subfolder (named by GolfDB clip ID) contains the same set of files
for one golf swing. All files are produced from a single pose-pipeline
run; pick whichever format is easiest for your UE5 ingestion code.

---

## File formats per clip

| File | Format | Use |
|---|---|---|
| `<clip>_landmarks_3d.csv` | CSV (per-frame H36M-17 joint positions in meters) | **Recommended primary.** Import as DataTable in UE5, Blueprint reads rows + drives Control Rig. |
| `<clip>_animation.bvh` | BioVision Hierarchy mocap | Drag into UE5 Mocap Plugin or Blender. Industry-standard mocap file. |
| `<clip>_mocap.json` | Self-describing JSON: skeleton + frames + events + provenance | Canonical metadata; build any custom Blueprint parser. |
| `<clip>_landmarks_2d.csv` | CSV of COCO-17 2D landmarks (pixel coords) | Debugging / sanity-check the 2D upstream. |
| `<clip>_overlay.mp4` | h264 video with 2D landmarks drawn on the source clip | Sanity-check that the pose tracking didn't drift. |
| `<clip>_preview_3d.html` | Standalone Three.js 3D viewer | Drop into any browser to preview before UE5 ingestion. |

## H36M-17 skeleton

The 3D landmarks use the canonical H36M-17 skeleton (Human3.6M dataset
convention). Joint indices and parent links:

```
 0 hip_center        (root)            parent: -1
 1 right_hip                            parent: 0
 2 right_knee                           parent: 1
 3 right_ankle                          parent: 2
 4 left_hip                             parent: 0
 5 left_knee                            parent: 4
 6 left_ankle                           parent: 5
 7 spine                                parent: 0
 8 thorax                               parent: 7
 9 neck                                 parent: 8
10 head                                 parent: 9
11 left_shoulder                        parent: 8
12 left_elbow                           parent: 11
13 left_wrist                           parent: 12
14 right_shoulder                       parent: 8
15 right_elbow                          parent: 14
16 right_wrist                          parent: 15
```

## CSV schema (`<clip>_landmarks_3d.csv`)

Columns: `frame_idx, time_s, joint_idx, joint_name, x, y, z, event_label`

- One row per (frame, joint) — 17 rows per frame.
- `(x, y, z)` are in H36M's normalized 3D space (~ meters, hip-centered).
- `event_label` is non-empty on frames where a GolfDB swing event occurs
  (address, toe_up, mid_backswing, top, mid_downswing, impact,
  mid_follow_through, finish).

## Recommended UE5 ingestion flow

1. Import `<clip>_landmarks_3d.csv` as a **Data Table** (struct with the
   columns above).
2. In Blueprint, iterate frames in `frame_idx` order; per frame, read all
   17 rows.
3. Map each `joint_name` to the corresponding bone in your Control Rig
   skeleton.
4. Set bone transforms directly OR feed `(x, y, z)` into an IK solver if
   you want anatomically constrained playback.
5. Use `event_label` to trigger phase callbacks (e.g. play swoosh on
   `impact`, show overlay text on `top`).

## Coordinate convention

The 3D output uses H36M-camera coordinates:
- `+x` = subject's right (camera-relative)
- `+y` = down (screen Y; flip for "up" in UE5)
- `+z` = depth into the scene (away from camera)

In UE5 (Z-up, left-handed), the equivalent transform is roughly:
- `ue_x = z`
- `ue_y = x`
- `ue_z = -y`

(Test against `<clip>_preview_3d.html` to confirm orientation; the HTML
viewer applies this same flip.)

## Notes

- **Smoothing is already applied** (One-Euro filter + per-clip bone-length
  lock). Bones won't change length frame-to-frame.
- **30 FPS assumed** unless `_mocap.json` says otherwise.
- All clips are GolfDB-sourced (face-on / down-the-line / other camera
  angles). Slow-mo clips are present and labeled in `_mocap.json`.
- Questions: open `_preview_3d.html` first to confirm the swing looks
  right before debugging UE5 ingestion.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--backbone", default="mediapipe_lite")
    p.add_argument("--lifter", default="motionbert_full")
    p.add_argument("--zip-name", default="motion_caddie_ue5_handoff.zip")
    args = p.parse_args()

    clips = pick_clips(args.n)
    print(f"[bundle] {len(clips)} clips to generate")
    print(f"  demo:   {DEMO_CLIPS}")
    print(f"  extras: {clips[len(DEMO_CLIPS):]}")

    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_py = Path(__file__).parent / "pipeline.py"
    py_exe = sys.executable

    n_ok = 0
    for cid in clips:
        video_path = PROJECT_ROOT / "Data" / "videos_160" / f"{cid}.mp4"
        if not video_path.exists():
            print(f"  -- clip {cid}: video missing")
            continue
        print(f"[bundle] running pipeline on clip {cid} ...")
        cp = subprocess.run(
            [py_exe, str(pipeline_py), str(video_path),
             "--backbone", args.backbone,
             "--lifter", args.lifter,
             "--smooth", "oneeuro"],
            capture_output=True, text=True, timeout=600,
        )
        if cp.returncode != 0:
            print(f"  -- clip {cid}: pipeline failed: {cp.stderr[:200]}")
            continue
        n_ok += 1

    print(f"\n[bundle] pipeline succeeded for {n_ok}/{len(clips)} clips")

    # Write README
    from datetime import datetime
    readme = README_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d"),
        backbone_2d=args.backbone,
        lifter_3d=args.lifter,
        n_clips=n_ok,
    )
    (HANDOFF_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"[bundle] wrote {HANDOFF_DIR / 'README.md'}")

    # Zip everything in Data/handoff/
    zip_path = PROJECT_ROOT / args.zip_name
    print(f"[bundle] zipping to {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in HANDOFF_DIR.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(HANDOFF_DIR.parent)
                zf.write(f, arcname=arcname)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[bundle] done: {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
