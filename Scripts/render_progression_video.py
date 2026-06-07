"""Render 3-panel "progression" videos: 2D -> 3D-raw -> 3D-smoothed.

For each clip, produces a side-by-side MP4 showing the pipeline progression:

   ┌──────────────┬──────────────┬──────────────┐
   │              │              │              │
   │   2D OVERLAY │  RAW 3D LIFT │  SMOOTHED 3D │
   │   (input)    │  (jittery)   │  (UE5-ready) │
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘

Panel 1 mirrors the coworker's original visualization: MediaPipe Heavy 2D
landmarks drawn over the original GolfDB video frame.
Panels 2 + 3 are the raw vs smoothed 3D skeleton from MotionBERT-Full.

Output: Data/overlays/progression/<clip>_progression.mp4 (h264).

Usage:
    python render_progression_video.py                       # 5 curated clips
    python render_progression_video.py --clips 0 7 1292
    python render_progression_video.py --backbone-2d mediapipe_lite
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import COCO17_IDX, video_info
from export_ue5 import H36M17_NAMES, H36M17_PARENTS, load_3d_parquet_as_h36m
from smoothing import smooth_sequence

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
OUT_DIR      = PROJECT_ROOT / "Data" / "overlays" / "progression"

# COCO-17 skeleton connections (matches the existing overlay style)
COCO_LINKS = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
              (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]

# H36M-17 skeleton connections for 3D panels
H36M_LINKS = [
    (0, 1), (1, 2), (2, 3),     # right leg
    (0, 4), (4, 5), (5, 6),     # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine + head
    (8, 11), (11, 12), (12, 13),    # left arm
    (8, 14), (14, 15), (15, 16),    # right arm
]


def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def load_2d_landmarks(model: str, clip_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    p = CACHE_DIR / model / f"{clip_id}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    T = int(df["frame"].max()) + 1
    xy = np.zeros((T, 17, 2), dtype=np.float32)
    conf = np.zeros((T, 17), dtype=np.float32)
    for row in df.itertuples(index=False):
        xy[int(row.frame), int(row.kp_idx), 0] = row.x
        xy[int(row.frame), int(row.kp_idx), 1] = row.y
        conf[int(row.frame), int(row.kp_idx)] = row.conf
    return xy, conf


def project_xyz_to_canvas(xyz: np.ndarray, canvas_w: int, canvas_h: int,
                           padding: float = 0.15) -> np.ndarray:
    """Orthographic projection. Same logic as render_smoothing_video so
    panels 2 and 3 are visually directly comparable."""
    hip = xyz[:, 0, :].mean(axis=0)
    centered = xyz - hip[None, None, :]
    extent = max(np.abs(centered[..., 0]).max(),
                  np.abs(centered[..., 1]).max(), 1e-6)
    half_w = canvas_w * (1 - padding) / 2
    half_h = canvas_h * (1 - padding) / 2
    scale = min(half_w, half_h) / extent
    proj = np.zeros((xyz.shape[0], xyz.shape[1], 2), dtype=np.float32)
    proj[..., 0] = centered[..., 0] * scale + canvas_w / 2
    proj[..., 1] = centered[..., 1] * scale + canvas_h / 2
    return proj


def draw_3d_skeleton(canvas: np.ndarray, pts_2d: np.ndarray,
                      line_color: tuple, joint_color: tuple,
                      line_thickness: int = 2, joint_radius: int = 4):
    for a, b in H36M_LINKS:
        x1, y1 = int(pts_2d[a, 0]), int(pts_2d[a, 1])
        x2, y2 = int(pts_2d[b, 0]), int(pts_2d[b, 1])
        cv2.line(canvas, (x1, y1), (x2, y2), line_color, line_thickness, cv2.LINE_AA)
    for j in range(pts_2d.shape[0]):
        x, y = int(pts_2d[j, 0]), int(pts_2d[j, 1])
        cv2.circle(canvas, (x, y), joint_radius, joint_color, -1, cv2.LINE_AA)


def draw_2d_overlay(frame_bgr: np.ndarray, xy: np.ndarray, conf: np.ndarray,
                     line_color: tuple = (50, 220, 50),
                     joint_color: tuple = (0, 0, 220)):
    """Draw COCO-17 landmarks on a frame in-place. Matches existing overlay style."""
    for a, b in COCO_LINKS:
        if conf[a] >= 0.3 and conf[b] >= 0.3:
            cv2.line(frame_bgr,
                      (int(xy[a, 0]), int(xy[a, 1])),
                      (int(xy[b, 0]), int(xy[b, 1])),
                      line_color, 2, cv2.LINE_AA)
    for j in range(17):
        if conf[j] >= 0.3:
            cv2.circle(frame_bgr,
                        (int(xy[j, 0]), int(xy[j, 1])),
                        3, joint_color, -1, cv2.LINE_AA)


def label_panel(canvas: np.ndarray, title: str, sub: str = "",
                 title_color: tuple = (255, 255, 255)):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                 title_color, 2, cv2.LINE_AA)
    if sub:
        cv2.rectangle(canvas, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(canvas, sub, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                     (220, 220, 220), 1, cv2.LINE_AA)


def draw_arrow_between(canvas: np.ndarray, x_center: int, y_center: int,
                        size: int = 24, color: tuple = (200, 200, 200)):
    """Draw a small right-arrow at (x_center, y_center) inside a panel border."""
    half = size // 2
    cv2.arrowedLine(canvas,
                     (x_center - half, y_center),
                     (x_center + half, y_center),
                     color, 3, tipLength=0.5)


def render_one_clip(clip_id: int, backbone_2d: str, lifter: str,
                     panel_w: int = 480, panel_h: int = 480) -> Path | None:
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    parquet_2d = CACHE_DIR / backbone_2d / f"{clip_id}.parquet"
    parquet_3d = CACHE_DIR / f"{lifter}_from_{backbone_2d}" / f"{clip_id}.parquet"
    if not video_path.exists():
        print(f"  -- clip {clip_id}: video not found")
        return None
    if not parquet_2d.exists():
        print(f"  -- clip {clip_id}: 2D cache for {backbone_2d} not found")
        return None
    if not parquet_3d.exists():
        print(f"  -- clip {clip_id}: 3D cache ({lifter} from {backbone_2d}) not found")
        return None

    # Load 2D
    xy_2d, conf_2d = load_2d_landmarks(backbone_2d, clip_id)
    # Load + smooth 3D
    xyz_raw = load_3d_parquet_as_h36m(parquet_3d)
    info = video_info(video_path)
    fps = info["fps"] or 30.0
    xyz_smooth = smooth_sequence(xyz_raw, method="oneeuro", fps=fps, bone_lock=True)

    # Project both 3D arrays onto canvas
    proj_raw = project_xyz_to_canvas(xyz_raw,    panel_w, panel_h)
    proj_smt = project_xyz_to_canvas(xyz_smooth, panel_w, panel_h)

    # 2D landmarks are in input-video pixel coords (160x160).
    # We need to scale them to the panel size for drawing.
    # First, resize the video frame to panel size; then scale 2D coords by the same factor.
    in_w, in_h = info["width"], info["height"]
    scale_x = panel_w / in_w
    scale_y = panel_h / in_h
    xy_2d_panel = xy_2d.copy()
    xy_2d_panel[..., 0] *= scale_x
    xy_2d_panel[..., 1] *= scale_y

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    n_frames = info["n_frames"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path   = OUT_DIR / f"{clip_id}_progression_tmp.mp4"
    final_path = OUT_DIR / f"{clip_id}_progression.mp4"
    out_w = panel_w * 3
    out_h = panel_h
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (out_w, out_h))

    # Per-clip stats for footers
    raw_accel = np.linalg.norm(np.diff(xyz_raw,    n=2, axis=0), axis=-1).mean()
    smt_accel = np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).mean()
    pct = (1 - smt_accel / max(raw_accel, 1e-9)) * 100

    bone_indices = [(11, 12), (12, 13), (1, 2), (2, 3)]
    def bone_cv(xyz):
        cvs = []
        for a, b in bone_indices:
            d = np.linalg.norm(xyz[:, a] - xyz[:, b], axis=-1)
            cvs.append(d.std() / max(d.mean(), 1e-9))
        return float(np.mean(cvs))
    raw_bcv = bone_cv(xyz_raw)
    smt_bcv = bone_cv(xyz_smooth)

    # Confidence on 2D
    det_rate = (conf_2d >= 0.3).mean()

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fi >= xyz_raw.shape[0]: break

        # --- Panel 1: original video + 2D landmark overlay ---
        panel_2d = cv2.resize(frame, (panel_w, panel_h))
        draw_2d_overlay(panel_2d, xy_2d_panel[fi], conf_2d[fi])
        label_panel(panel_2d,
                     f"  STAGE 1: 2D LANDMARKS",
                     f"  {backbone_2d}   detect {det_rate*100:.0f}%   coworker-style")

        # --- Panel 2: raw 3D skeleton ---
        panel_raw = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
        for x in range(0, panel_w, 40):
            cv2.line(panel_raw, (x, 0), (x, panel_h), (50, 50, 50), 1)
        for y in range(0, panel_h, 40):
            cv2.line(panel_raw, (0, y), (panel_w, y), (50, 50, 50), 1)
        draw_3d_skeleton(panel_raw, proj_raw[fi],
                          line_color=(80, 80, 240),
                          joint_color=(80, 80, 240))
        label_panel(panel_raw,
                     f"  STAGE 2: 3D LIFT (raw)",
                     f"  {lifter}   accel {raw_accel:.4f}   bone CV {raw_bcv:.3f}")

        # --- Panel 3: smoothed 3D skeleton ---
        panel_smt = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
        for x in range(0, panel_w, 40):
            cv2.line(panel_smt, (x, 0), (x, panel_h), (50, 50, 50), 1)
        for y in range(0, panel_h, 40):
            cv2.line(panel_smt, (0, y), (panel_w, y), (50, 50, 50), 1)
        draw_3d_skeleton(panel_smt, proj_smt[fi],
                          line_color=(80, 240, 80),
                          joint_color=(80, 240, 80))
        label_panel(panel_smt,
                     f"  STAGE 3: SMOOTHED (UE5-ready)",
                     f"  oneeuro + bone-lock   accel {smt_accel:.4f}   -{pct:.0f}%")

        # Concatenate side-by-side
        out_frame = np.concatenate([panel_2d, panel_raw, panel_smt], axis=1)

        # Draw arrows between panels (small overlay near vertical mid-point)
        y_mid = panel_h // 2
        # Arrow from panel 1 -> 2 (right edge of panel 1)
        draw_arrow_between(out_frame, panel_w + 0, y_mid, size=30, color=(255, 255, 255))
        # Arrow from panel 2 -> 3
        draw_arrow_between(out_frame, panel_w * 2 + 0, y_mid, size=30, color=(255, 255, 255))

        writer.write(out_frame)
        fi += 1

    cap.release()
    writer.release()

    # H264 conversion
    ffp = ffmpeg_path()
    if ffp:
        try:
            subprocess.run([ffp, "-y", "-i", str(tmp_path), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", str(final_path)],
                            check=True, timeout=180)
            tmp_path.unlink()
            return final_path
        except Exception as e:
            print(f"  ffmpeg fallback (clip {clip_id}): {e}")
            tmp_path.rename(final_path)
            return final_path
    tmp_path.rename(final_path)
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", type=int,
                   default=[0, 173, 7, 34, 1292],
                   help="Clip IDs (default: 5 curated demo clips)")
    p.add_argument("--backbone-2d", default="mediapipe_heavy",
                   help="2D backbone whose landmarks appear in panel 1 "
                        "(default: mediapipe_heavy to match coworker's plot)")
    p.add_argument("--lifter", default="motionbert_full")
    args = p.parse_args()

    print(f"[progression-video] {len(args.clips)} clips")
    print(f"   panel 1 (2D)         : {args.backbone_2d}")
    print(f"   panel 2 (3D raw)     : {args.lifter} from {args.backbone_2d}")
    print(f"   panel 3 (3D smoothed): same + oneeuro + bone-lock")
    for cid in args.clips:
        print(f"\n  clip {cid}:")
        out = render_one_clip(cid, args.backbone_2d, args.lifter)
        if out:
            print(f"    OK  {out.relative_to(PROJECT_ROOT)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
