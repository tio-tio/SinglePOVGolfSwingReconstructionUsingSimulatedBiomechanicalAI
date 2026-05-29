"""Render overlay MP4s of cached landmark predictions for visual comparison.

Reads cached parquet landmarks from Data/eval_runs/<model>/<id>.parquet,
overlays them on the original video frames, and writes browser-friendly
h264 MP4s to Data/overlays/<model>/<id>_<model>.mp4.

Strategy: don't render all 9,800 (model x clip) combinations. Pick a small
representative set so the team can compare models side-by-side on the same
clip. Defaults pick 6 clips covering the diversity dimensions.

Usage:
    python render_overlays.py                        # default 6 clips x 7 models
    python render_overlays.py --clips 0 17 42 ...    # specific clips
    python render_overlays.py --models mediapipe_lite vitpose_base
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import COCO17_IDX, COCO17_NAMES

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
VIDEO_DIR    = DATA_DIR / "videos_160"
CACHE_DIR    = DATA_DIR / "eval_runs"
OUT_DIR      = DATA_DIR / "overlays"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"

# COCO-17 skeleton connections for drawing
COCO_LINKS = [
    (5, 7), (7, 9), (6, 8), (8, 10),     # arms
    (5, 6), (11, 12), (5, 11), (6, 12),  # torso
    (11, 13), (13, 15),                   # left leg
    (12, 14), (14, 16),                   # right leg
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
]

ALL_MODELS = [
    "mediapipe_heavy", "mediapipe_lite",
    "movenet_thunder", "movenet_lightning",
    "yolov8n_pose", "yolov8m_pose",
    "vitpose_base",
]


def pick_representative_clips(df: pd.DataFrame, all_metrics: pd.DataFrame, n: int = 6) -> list[int]:
    """Pick clips that cover the diversity dimensions we care about for
    qualitative comparison. We try to balance camera view, slow-mo, and
    difficulty (high vs low cross-model agreement)."""
    # difficulty: median PCE@5 across models per clip
    per_clip = all_metrics.groupby("clip_id")["pce_at_5"].agg(["mean", "std"]).reset_index()
    per_clip = per_clip.merge(df[["id", "view", "slow", "sex"]], left_on="clip_id", right_on="id")

    picks = []
    # 1. Easiest clip (highest mean PCE)
    picks.append(("easiest", int(per_clip.nlargest(1, "mean").iloc[0]["clip_id"])))
    # 2. Hardest clip (lowest mean PCE)
    picks.append(("hardest", int(per_clip.nsmallest(1, "mean").iloc[0]["clip_id"])))
    # 3. Most divisive (highest disagreement across models)
    picks.append(("divisive", int(per_clip.nlargest(1, "std").iloc[0]["clip_id"])))
    # 4. Slow-mo face-on (the classic golf TV view)
    cand = per_clip[(per_clip["view"] == "face-on") & (per_clip["slow"] == 1)]
    if len(cand):
        picks.append(("slowmo_faceon", int(cand.sample(1, random_state=42).iloc[0]["clip_id"])))
    # 5. Slow-mo down-the-line
    cand = per_clip[(per_clip["view"] == "down-the-line") & (per_clip["slow"] == 1)]
    if len(cand):
        picks.append(("slowmo_dtl", int(cand.sample(1, random_state=42).iloc[0]["clip_id"])))
    # 6. Real-time, non-standard view
    cand = per_clip[(per_clip["view"] == "other") & (per_clip["slow"] == 0)]
    if len(cand):
        picks.append(("realtime_other", int(cand.sample(1, random_state=42).iloc[0]["clip_id"])))

    # Dedupe while preserving the order/labels for caller info
    seen = set()
    out = []
    for label, cid in picks:
        if cid not in seen:
            out.append((label, cid))
            seen.add(cid)
    return out[:n]


def render_overlay(clip_id: int, model_name: str, label: str | None = None) -> Path | None:
    """Render one overlay MP4. Returns the output path, or None if the
    landmarks parquet isn't cached."""
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    cache_path = CACHE_DIR / model_name / f"{clip_id}.parquet"
    if not video_path.exists() or not cache_path.exists():
        return None

    lm_df = pd.read_parquet(cache_path)

    out_subdir = OUT_DIR / model_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_subdir / f"{clip_id}_{model_name}_tmp.mp4"
    final_path = out_subdir / f"{clip_id}_{model_name}.mp4"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))

    # pivot landmarks for fast per-frame lookup
    by_frame = {fi: g for fi, g in lm_df.groupby("frame")}

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if fi in by_frame:
            g = by_frame[fi]
            kp = {row.kp_idx: (int(row.x), int(row.y), float(row.conf)) for row in g.itertuples()}
            # skeleton lines
            for a, b in COCO_LINKS:
                if a in kp and b in kp and kp[a][2] >= 0.3 and kp[b][2] >= 0.3:
                    cv2.line(frame, (kp[a][0], kp[a][1]), (kp[b][0], kp[b][1]),
                              (50, 220, 50), 2, cv2.LINE_AA)
            # keypoint dots
            for ki, (x, y, c) in kp.items():
                if c >= 0.3:
                    cv2.circle(frame, (x, y), 3, (0, 0, 220), -1, cv2.LINE_AA)

        # text label
        banner = f"{model_name}" if label is None else f"{model_name}  [{label}]"
        cv2.rectangle(frame, (0, 0), (w, 14), (0, 0, 0), -1)
        cv2.putText(frame, banner, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()

    # h264 conversion for browser playback (matches coworker's pipeline).
    # Prefer system ffmpeg; fall back to the imageio-ffmpeg bundled binary.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(tmp_path), "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", "-loglevel", "error", str(final_path)],
                check=True, timeout=120,
            )
            tmp_path.unlink()
            return final_path
        except Exception as e:
            print(f"  ffmpeg failed for {clip_id}/{model_name}: {e}; keeping mp4v file")
            tmp_path.rename(final_path)
            return final_path
    # no ffmpeg
    tmp_path.rename(final_path)
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", type=int, default=None,
                   help="Specific clip IDs to render. Default: pick 6 representative.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Models to render. Default: all 7.")
    args = p.parse_args()

    df = pd.read_pickle(GOLFDB_PKL)
    metrics_path = DATA_DIR / "all_metrics.parquet"
    if not metrics_path.exists():
        print("[overlays] all_metrics.parquet not found - run compute_metrics.py first")
        return
    all_metrics = pd.read_parquet(metrics_path)

    if args.clips:
        clips = [(f"manual_{c}", c) for c in args.clips]
    else:
        clips = pick_representative_clips(df, all_metrics, n=6)

    models = args.models or ALL_MODELS

    # Manifest so the team knows what each clip is
    manifest = []
    for label, cid in clips:
        row = df[df.id == cid].iloc[0]
        manifest.append({
            "clip_id": cid,
            "selection_label": label,
            "view": row["view"],
            "slow_mo": bool(row["slow"]),
            "club": row["club"],
            "sex": row["sex"],
            "youtube_id": row["youtube_id"],
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("[overlays] manifest:")
    print(json.dumps(manifest, indent=2))

    print(f"\n[overlays] rendering {len(models)} models x {len(clips)} clips = {len(models) * len(clips)} videos")
    for label, cid in clips:
        for mname in models:
            out = render_overlay(cid, mname, label=label)
            if out:
                print(f"  ok  {out.relative_to(PROJECT_ROOT)}  ({out.stat().st_size // 1024} KB)")
            else:
                print(f"  --  {cid}/{mname}: missing input")

    print(f"\n[overlays] all videos in: {OUT_DIR}")


if __name__ == "__main__":
    main()
