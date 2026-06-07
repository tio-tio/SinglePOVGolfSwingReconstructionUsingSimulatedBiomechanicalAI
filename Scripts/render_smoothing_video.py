"""Render before/after-smoothing comparison videos.

For each input clip, produces a 3-panel side-by-side MP4:

   ┌──────────────┬──────────────┬──────────────┐
   │              │              │              │
   │   ORIGINAL   │  RAW 3D      │  SMOOTHED 3D │
   │   VIDEO      │  (jittery)   │  (clean)     │
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘

The 3D skeletons are rendered as 2D projections via an orthographic view
of the (x, y) plane. Both panels use the same projection so any visible
difference is purely from the smoothing.

Output: Data/overlays/smoothing/<clip>_smoothing_compare.mp4 (h264).

Usage:
    python render_smoothing_video.py                       # default: 5 curated clips
    python render_smoothing_video.py --clips 0 7 1292
    python render_smoothing_video.py --backbone mediapipe_heavy --lifter motionbert_full
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
from eval_utils import video_info
from export_ue5 import H36M17_NAMES, H36M17_PARENTS, load_3d_parquet_as_h36m
from smoothing import smooth_sequence

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
OUT_DIR      = PROJECT_ROOT / "Data" / "overlays" / "smoothing"

# H36M-17 skeleton connections for drawing
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


def project_xyz_to_canvas(xyz: np.ndarray, canvas_w: int, canvas_h: int,
                           padding: float = 0.15) -> np.ndarray:
    """Orthographic projection of a (T, J, 3) clip into 2D pixel space on
    a canvas of size (canvas_w, canvas_h). Uses (x, y) of the 3D output,
    keeping y down (screen convention).

    Scale is fixed across the whole clip so motion is faithful (no
    rescaling per frame). Centered on the mean hip position over the clip.
    """
    # Center on mean hip
    hip = xyz[:, 0, :].mean(axis=0)  # (3,)
    centered = xyz - hip[None, None, :]

    # Pick scale from max extent across both x and y over the whole clip
    extent = max(
        np.abs(centered[..., 0]).max(),
        np.abs(centered[..., 1]).max(),
        1e-6,
    )
    half_w = canvas_w * (1 - padding) / 2
    half_h = canvas_h * (1 - padding) / 2
    scale = min(half_w, half_h) / extent

    # Project to pixel coords with center of canvas as origin
    proj = np.zeros((xyz.shape[0], xyz.shape[1], 2), dtype=np.float32)
    proj[..., 0] = centered[..., 0] * scale + canvas_w / 2
    proj[..., 1] = centered[..., 1] * scale + canvas_h / 2
    return proj


def draw_skeleton(canvas: np.ndarray, pts_2d: np.ndarray,
                   line_color: tuple = (50, 220, 50),
                   joint_color: tuple = (0, 0, 220),
                   line_thickness: int = 2, joint_radius: int = 4):
    """Draw a stick-figure skeleton on the given canvas (modifies in-place).
    pts_2d is (17, 2) pixel coords."""
    for a, b in H36M_LINKS:
        x1, y1 = int(pts_2d[a, 0]), int(pts_2d[a, 1])
        x2, y2 = int(pts_2d[b, 0]), int(pts_2d[b, 1])
        cv2.line(canvas, (x1, y1), (x2, y2), line_color, line_thickness, cv2.LINE_AA)
    for j in range(pts_2d.shape[0]):
        x, y = int(pts_2d[j, 0]), int(pts_2d[j, 1])
        cv2.circle(canvas, (x, y), joint_radius, joint_color, -1, cv2.LINE_AA)


def label_panel(canvas: np.ndarray, text: str, sub: str = ""):
    """Add a top banner with text + optional small subtitle to a panel."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                 (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.rectangle(canvas, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(canvas, sub, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                     (220, 220, 220), 1, cv2.LINE_AA)


def render_one_clip(clip_id: int, backbone: str, lifter: str,
                     panel_w: int = 480, panel_h: int = 480) -> Path | None:
    """Produce the 3-panel before/after MP4 for one clip."""
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    parquet_path = CACHE_DIR / f"{lifter}_from_{backbone}" / f"{clip_id}.parquet"
    if not video_path.exists():
        print(f"  -- clip {clip_id}: video not found")
        return None
    if not parquet_path.exists():
        print(f"  -- clip {clip_id}: 3D cache not found ({lifter} <- {backbone})")
        return None

    # Load + smooth 3D
    xyz_raw = load_3d_parquet_as_h36m(parquet_path)
    info = video_info(video_path)
    fps = info["fps"] or 30.0
    xyz_smooth = smooth_sequence(xyz_raw, method="oneeuro", fps=fps, bone_lock=True)

    # Project both onto same canvas
    proj_raw = project_xyz_to_canvas(xyz_raw, panel_w, panel_h)
    proj_smt = project_xyz_to_canvas(xyz_smooth, panel_w, panel_h)

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    n_frames = info["n_frames"]

    # Output writer (3 panels wide)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path   = OUT_DIR / f"{clip_id}_smoothing_compare_tmp.mp4"
    final_path = OUT_DIR / f"{clip_id}_smoothing_compare.mp4"
    out_w = panel_w * 3
    out_h = panel_h
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (out_w, out_h))

    # Compute per-clip stats to display
    raw_accel = np.linalg.norm(np.diff(xyz_raw,    n=2, axis=0), axis=-1).mean()
    smt_accel = np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).mean()
    pct = (1 - smt_accel / max(raw_accel, 1e-9)) * 100

    bone_indices = [(11, 12), (12, 13), (1, 2), (2, 3)]  # L upper arm, L forearm, R thigh, R shin
    def bone_cv(xyz):
        cvs = []
        for a, b in bone_indices:
            d = np.linalg.norm(xyz[:, a] - xyz[:, b], axis=-1)
            cvs.append(d.std() / max(d.mean(), 1e-9))
        return np.mean(cvs)
    raw_bcv = bone_cv(xyz_raw)
    smt_bcv = bone_cv(xyz_smooth)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fi >= xyz_raw.shape[0]: break

        # Panel 1: original video, resized to panel size
        panel_orig = cv2.resize(frame, (panel_w, panel_h))
        label_panel(panel_orig, "  ORIGINAL VIDEO", f"  clip {clip_id} (160x160 GolfDB)")

        # Panel 2: raw 3D skeleton
        panel_raw = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
        # subtle grid for ground
        for x in range(0, panel_w, 40):
            cv2.line(panel_raw, (x, 0), (x, panel_h), (50, 50, 50), 1)
        for y in range(0, panel_h, 40):
            cv2.line(panel_raw, (0, y), (panel_w, y), (50, 50, 50), 1)
        draw_skeleton(panel_raw, proj_raw[fi], line_color=(80, 80, 240),
                       joint_color=(80, 80, 240))
        label_panel(panel_raw, "  RAW 3D (no smoothing)",
                     f"  accel {raw_accel:.4f}   bone CV {raw_bcv:.3f}")

        # Panel 3: smoothed 3D skeleton
        panel_smt = np.full((panel_h, panel_w, 3), 30, dtype=np.uint8)
        for x in range(0, panel_w, 40):
            cv2.line(panel_smt, (x, 0), (x, panel_h), (50, 50, 50), 1)
        for y in range(0, panel_h, 40):
            cv2.line(panel_smt, (0, y), (panel_w, y), (50, 50, 50), 1)
        draw_skeleton(panel_smt, proj_smt[fi], line_color=(80, 240, 80),
                       joint_color=(80, 240, 80))
        label_panel(panel_smt, "  SMOOTHED (One-Euro + bone-lock)",
                     f"  accel {smt_accel:.4f}   bone CV {smt_bcv:.3f}   -{pct:.0f}%")

        # Concatenate side-by-side
        out_frame = np.concatenate([panel_orig, panel_raw, panel_smt], axis=1)

        # Footer: model info
        cv2.rectangle(out_frame, (0, out_h - 2), (out_w, out_h), (255, 255, 255), -1)
        writer.write(out_frame)
        fi += 1

    cap.release()
    writer.release()

    # H264 conversion for browser/PPT playback
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
                   help="Clip IDs (default: the 5 curated demo clips)")
    p.add_argument("--backbone", default="mediapipe_lite")
    p.add_argument("--lifter", default="motionbert_full")
    args = p.parse_args()

    print(f"[smoothing-video] {len(args.clips)} clips, backbone={args.backbone}, lifter={args.lifter}")
    for cid in args.clips:
        print(f"\n  clip {cid}:")
        out = render_one_clip(cid, args.backbone, args.lifter)
        if out:
            print(f"    OK  {out.relative_to(PROJECT_ROOT)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
