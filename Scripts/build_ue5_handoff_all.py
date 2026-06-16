"""Bulk-export the UE5 handoff bundle for ALL 1,400 GolfDB clips.

Skips per-clip Python startup + model loading (pipeline.py reloads
MediaPipe + MotionBERT every invocation). Loads everything once, then
iterates all clips, reading already-cached 3D landmarks and writing the
5/6 handoff files per clip.

Output: Data/handoff/<clip>/{<clip>_landmarks_2d.csv, _landmarks_3d.csv,
_mocap.json, _animation.bvh, _overlay.mp4, _preview_3d.html} per clip,
plus Data/handoff/README.md.

Then zips the whole tree to motion_caddie_ue5_handoff_full.zip at repo root.

Idempotent: skips clips whose 6-file bundle already exists.

Usage:
    python build_ue5_handoff_all.py
    python build_ue5_handoff_all.py --skip-overlay   # ~5x faster, no MP4s
    python build_ue5_handoff_all.py --max 100        # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import video_info, COCO17_IDX
from export_ue5 import (
    export_csv, export_json, export_bvh,
    load_3d_parquet_as_h36m,
)
from smoothing import smooth_sequence
from pipeline import write_2d_overlay, write_3d_preview_html

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
HANDOFF_DIR  = PROJECT_ROOT / "Data" / "handoff"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


def events_clip_local(events_row) -> list[int]:
    ev = np.asarray(events_row)
    return [int(x) for x in ev[1:9] - ev[0]]


SWING_EVENT_NAMES = ["address", "toe_up", "mid_backswing", "top",
                      "mid_downswing", "impact", "mid_follow_through", "finish"]


def process_one(clip_id: int, df_labels: pd.DataFrame,
                 backbone_2d: str, lifter_3d: str,
                 skip_overlay: bool = False, skip_preview: bool = False,
                 overwrite: bool = False) -> tuple[bool, str]:
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    if not video_path.exists():
        return False, "video missing"

    out_dir = HANDOFF_DIR / str(clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv2d_path  = out_dir / f"{clip_id}_landmarks_2d.csv"
    csv3d_path  = out_dir / f"{clip_id}_landmarks_3d.csv"
    json_path   = out_dir / f"{clip_id}_mocap.json"
    bvh_path    = out_dir / f"{clip_id}_animation.bvh"
    overlay_path = out_dir / f"{clip_id}_overlay.mp4"
    html_path   = out_dir / f"{clip_id}_preview_3d.html"

    needed = [csv3d_path, json_path, bvh_path]
    if not skip_overlay: needed.append(overlay_path)
    if not skip_preview: needed.append(html_path)
    if (not overwrite) and all(p.exists() for p in needed):
        return True, "skipped (already present)"

    info = video_info(video_path)
    fps = info["fps"] or 30.0

    # --- 2D landmarks cache -> CSV (always cheap) ---
    parquet_2d = CACHE_DIR / backbone_2d / f"{clip_id}.parquet"
    if parquet_2d.exists() and (overwrite or not csv2d_path.exists()):
        pd.read_parquet(parquet_2d).to_csv(csv2d_path, index=False)

    # --- 3D landmarks cache -> H36M-17 numpy ---
    parquet_3d = CACHE_DIR / f"{lifter_3d}_from_{backbone_2d}" / f"{clip_id}.parquet"
    if not parquet_3d.exists():
        return False, f"3D cache missing: {parquet_3d.name}"
    xyz_h36m = load_3d_parquet_as_h36m(parquet_3d)

    # --- Smooth ---
    xyz_smooth = smooth_sequence(xyz_h36m, method="oneeuro", fps=fps, bone_lock=True)

    # --- Events (if labeled in GolfDB) ---
    events_idx, events_names = None, None
    if clip_id in df_labels.index:
        events_idx = events_clip_local(df_labels.loc[clip_id, "events"])
        events_names = SWING_EVENT_NAMES

    # --- Exports ---
    export_csv(xyz_smooth, csv3d_path, fps=fps,
                events_frame_indices=events_idx, event_names=events_names)
    export_json(xyz_smooth, json_path, fps=fps,
                 source_video=str(video_path),
                 backbone_2d=backbone_2d,
                 lifter_3d=f"{lifter_3d}_from_{backbone_2d}",
                 events_frame_indices=events_idx, event_names=events_names)
    export_bvh(xyz_smooth, bvh_path, fps=fps)

    if not skip_overlay and (overwrite or not overlay_path.exists()):
        if parquet_2d.exists():
            try:
                write_2d_overlay(video_path, parquet_2d, overlay_path)
            except Exception as e:
                return True, f"overlay failed: {repr(e)[:80]}"

    if not skip_preview and (overwrite or not html_path.exists()):
        try:
            write_3d_preview_html(xyz_smooth, html_path, fps=fps,
                                    events_frame_indices=events_idx,
                                    event_names=events_names,
                                    title=f"clip {clip_id}")
        except Exception as e:
            return True, f"preview failed: {repr(e)[:80]}"

    return True, "ok"


README_TEMPLATE = """# Motion Caddie — UE5 Handoff Bundle (Full GolfDB)

**Generated:** {generated_at}
**Pipeline:** {backbone_2d} → {lifter_3d} → One-Euro + bone-lock smoothing
**Clips included:** {n_clips} (full GolfDB corpus)

Each subfolder (named by GolfDB clip ID) contains the same set of files
for one golf swing. Pick whichever format fits your UE5 ingestion code.

---

## File formats per clip

| File | Format | Use |
|---|---|---|
| `<clip>_landmarks_3d.csv` | CSV (per-frame H36M-17 joint positions, ~ meters) | **Recommended primary.** Import as DataTable in UE5, Blueprint reads rows + drives Control Rig. |
| `<clip>_animation.bvh` | BioVision Hierarchy mocap | Drag into UE5 Mocap Plugin or Blender. |
| `<clip>_mocap.json` | Self-describing JSON: skeleton + frames + events + provenance | Build any custom Blueprint parser. |
| `<clip>_landmarks_2d.csv` | CSV of COCO-17 2D landmarks (pixel coords) | Debugging the 2D upstream. |
| `<clip>_overlay.mp4` | h264 video with 2D landmarks drawn on the source clip | Sanity-check that pose tracking didn't drift. |
| `<clip>_preview_3d.html` | Standalone Three.js 3D viewer | Drop into any browser to preview before UE5 ingestion. |

## H36M-17 skeleton

```
 0 hip_center        (root)     parent: -1
 1 right_hip                    parent: 0
 2 right_knee                   parent: 1
 3 right_ankle                  parent: 2
 4 left_hip                     parent: 0
 5 left_knee                    parent: 4
 6 left_ankle                   parent: 5
 7 spine                        parent: 0
 8 thorax                       parent: 7
 9 neck                         parent: 8
10 head                         parent: 9
11 left_shoulder                parent: 8
12 left_elbow                   parent: 11
13 left_wrist                   parent: 12
14 right_shoulder               parent: 8
15 right_elbow                  parent: 14
16 right_wrist                  parent: 15
```

## CSV schema (`<clip>_landmarks_3d.csv`)

Columns: `frame_idx, time_s, joint_idx, joint_name, x, y, z, event_label`

- One row per (frame, joint) — 17 rows per frame.
- `(x, y, z)` are in H36M's normalized 3D space (~ meters, hip-centered).
- `event_label` is non-empty on frames where a GolfDB swing event occurs.

## UE5 ingestion (recommended)

1. Import `<clip>_landmarks_3d.csv` as a Data Table.
2. Per frame, read all 17 rows in `frame_idx` order.
3. Map `joint_name` to corresponding bones in your Control Rig.
4. Use `event_label` to trigger phase callbacks (impact swoosh, etc.).

## Coordinate convention

H36M-camera coords: `+x` right, `+y` down, `+z` depth.
For UE5 (Z-up, left-handed): `ue_x = z, ue_y = x, ue_z = -y`.
(Test against `<clip>_preview_3d.html`; the HTML viewer applies the same flip.)

## Notes

- **Smoothing already applied** (One-Euro filter + bone-length lock).
- **30 FPS assumed** unless `_mocap.json` specifies otherwise per clip.
- Sourced from GolfDB (1,400 professional swings, 580 source videos).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mediapipe_lite")
    p.add_argument("--lifter", default="motionbert_full")
    p.add_argument("--max", type=int, default=None, help="Process only first N clips (smoke test)")
    p.add_argument("--skip-overlay", action="store_true")
    p.add_argument("--skip-preview", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--zip-name", default="motion_caddie_ue5_handoff_full.zip")
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()

    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    clip_ids = sorted([cid for cid in df.index
                        if (VIDEO_DIR / f"{cid}.mp4").exists()])
    if args.max:
        clip_ids = clip_ids[: args.max]
    print(f"[bulk] {len(clip_ids)} clips to process")
    print(f"  backbone: {args.backbone}")
    print(f"  lifter:   {args.lifter}")
    print(f"  overlay:  {'skip' if args.skip_overlay else 'render'}")
    print(f"  preview:  {'skip' if args.skip_preview else 'render'}")

    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    t0 = time.perf_counter()
    failures: list[tuple[int, str]] = []

    for cid in tqdm(clip_ids, ncols=80):
        ok, msg = process_one(
            cid, df, args.backbone, args.lifter,
            skip_overlay=args.skip_overlay,
            skip_preview=args.skip_preview,
            overwrite=args.overwrite,
        )
        if ok and msg == "skipped (already present)":
            n_skipped += 1
        elif ok:
            n_ok += 1
        else:
            n_failed += 1
            failures.append((cid, msg))

    elapsed = time.perf_counter() - t0
    print(f"\n[bulk] processed {n_ok} new, skipped {n_skipped}, failed {n_failed}  "
          f"in {elapsed/60:.1f} min")
    if failures[:5]:
        print("  first failures:")
        for cid, msg in failures[:5]:
            print(f"    clip {cid}: {msg}")

    # README
    readme = README_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d"),
        backbone_2d=args.backbone,
        lifter_3d=args.lifter,
        n_clips=n_ok + n_skipped,
    )
    (HANDOFF_DIR / "README.md").write_text(readme, encoding="utf-8")

    # Zip
    if args.no_zip:
        return
    zip_path = PROJECT_ROOT / args.zip_name
    print(f"\n[bulk] zipping to {zip_path} ...")
    n_files_zipped = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in HANDOFF_DIR.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(HANDOFF_DIR.parent)
                zf.write(f, arcname=arcname)
                n_files_zipped += 1
    size_mb = zip_path.stat().st_size / 1e6
    print(f"[bulk] zipped {n_files_zipped} files: {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
