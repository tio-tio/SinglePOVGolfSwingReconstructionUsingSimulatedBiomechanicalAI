"""Render the 3D pose viewer as an MP4.

Replicates what `Data/handoff/<clip>/<clip>_preview_3d.html` shows but as
a slide-embeddable video. No browser, no Three.js, no internet — pure
matplotlib + OpenCV.

The video:
  - plays the swing twice (4-6 seconds total)
  - slowly rotates the camera 180 across the duration
  - overlays clip ID, frame counter, and current GolfDB event label

Output: Data/handoff/<clip>/<clip>_preview_3d.mp4 (h264).

Usage:
    python render_3d_turntable_mp4.py                       # default: clip 0
    python render_3d_turntable_mp4.py --clip 1292
    python render_3d_turntable_mp4.py --clip 0 --loops 3 --rotate 360
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import SWING_EVENTS, video_info
from export_ue5 import H36M17_NAMES, H36M17_PARENTS, load_3d_parquet_as_h36m
from smoothing import smooth_sequence

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
HANDOFF_DIR  = PROJECT_ROOT / "Data" / "handoff"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


# H36M-17 connectivity for drawing the skeleton
H36M_LINKS = [
    (0, 1), (1, 2), (2, 3),                # right leg
    (0, 4), (4, 5), (5, 6),                # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),       # spine + head
    (8, 11), (11, 12), (12, 13),           # left arm
    (8, 14), (14, 15), (15, 16),           # right arm
]


def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def events_for_clip(clip_id: int) -> dict[int, str] | None:
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    if clip_id not in df.index: return None
    ev = np.asarray(df.loc[clip_id, "events"])
    events_local = ev[1:9] - ev[0]
    return {int(f): name for f, name in zip(events_local, SWING_EVENTS)}


def render_turntable_mp4(clip_id: int, backbone: str, lifter: str,
                          loops: int = 2, rotate_total_deg: float = 180.0,
                          width: int = 720, height: int = 720,
                          fps_out: float = 30.0) -> Path | None:
    parquet_path = CACHE_DIR / f"{lifter}_from_{backbone}" / f"{clip_id}.parquet"
    if not parquet_path.exists():
        print(f"  -- no cache for {lifter} from {backbone}, clip {clip_id}")
        return None

    xyz_raw = load_3d_parquet_as_h36m(parquet_path)
    # smooth, just like the HTML preview shows the smoothed version
    video_path = PROJECT_ROOT / "Data" / "videos_160" / f"{clip_id}.mp4"
    fps_input = video_info(video_path)["fps"] if video_path.exists() else 30.0
    xyz = smooth_sequence(xyz_raw, method="oneeuro", fps=fps_input, bone_lock=True)

    T = xyz.shape[0]
    n_render = T * loops
    event_lookup = events_for_clip(clip_id) or {}

    # Set up output writer
    out_dir = HANDOFF_DIR / str(clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path   = out_dir / f"{clip_id}_preview_3d_tmp.mp4"
    final_path = out_dir / f"{clip_id}_preview_3d.mp4"
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps_out, (width, height))

    # Pre-compute axis limits using the full clip so the view doesn't jitter
    # Use H36M skeleton convention: X=right, Y=down, Z=forward
    # For plotting we want Y inverted so up is up
    pts = xyz.reshape(-1, 3)
    rng = max(pts[:, 0].ptp(), pts[:, 1].ptp(), pts[:, 2].ptp()) * 0.55
    cx, cy, cz = pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#1a1a1a")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for fi in range(n_render):
        clip_frame = fi % T
        loop_idx = fi // T
        progress = fi / max(1, n_render - 1)
        # Azimuth: start at -90 (face-on), sweep across rotate_total_deg
        azim = -90 + rotate_total_deg * progress
        elev = 10

        ax.clear()
        ax.set_facecolor("#1a1a1a")

        # XYZ for current frame
        p = xyz[clip_frame]  # (17, 3)
        # Plot with H36M's "Y is down" inverted
        xs, ys, zs = p[:, 0], -p[:, 1], p[:, 2]

        # Draw bones
        for a, b in H36M_LINKS:
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]],
                    color="#50DC50", linewidth=2.5, alpha=0.95)
        # Draw joints
        ax.scatter(xs, zs, ys, c="#FF5050", s=40, edgecolors="white",
                   linewidths=0.5, alpha=0.95)

        # Ground grid plane for visual reference
        floor_y = ys.min() - 0.05
        gx = np.linspace(cx - rng, cx + rng, 6)
        gz = np.linspace(cz - rng, cz + rng, 6)
        for gxi in gx:
            ax.plot([gxi, gxi], [gz[0], gz[-1]], [floor_y, floor_y],
                    color="#444444", linewidth=0.5)
        for gzi in gz:
            ax.plot([gx[0], gx[-1]], [gzi, gzi], [floor_y, floor_y],
                    color="#444444", linewidth=0.5)

        # Camera + axes
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(cx - rng, cx + rng)
        ax.set_ylim(cz - rng, cz + rng)
        ax.set_zlim(floor_y, floor_y + 2 * rng)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")
        ax.grid(False)

        # Render figure to a numpy RGB array
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame_rgba = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

        # Title bar + status overlay
        cv2.rectangle(frame_bgr, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(frame_bgr, f"  clip {clip_id}   {lifter}_from_{backbone}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom overlay: frame counter + current event
        cv2.rectangle(frame_bgr, (0, height - 28), (width, height), (0, 0, 0), -1)
        ev_label = event_lookup.get(clip_frame, "")
        bottom = f"  frame {clip_frame + 1}/{T}    loop {loop_idx + 1}/{loops}"
        if ev_label:
            bottom += f"      EVENT: {ev_label.upper()}"
        cv2.putText(frame_bgr, bottom, (8, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (80, 255, 255) if ev_label else (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

    writer.release()
    plt.close(fig)

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
            print(f"  ffmpeg fallback: {e}")
            tmp_path.rename(final_path)
            return final_path
    tmp_path.rename(final_path)
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=int, default=0)
    p.add_argument("--backbone", default="mediapipe_lite")
    p.add_argument("--lifter", default="motionbert_full")
    p.add_argument("--loops", type=int, default=2)
    p.add_argument("--rotate", type=float, default=180.0,
                   help="Total camera rotation in degrees across the video")
    args = p.parse_args()

    out = render_turntable_mp4(
        clip_id=args.clip, backbone=args.backbone, lifter=args.lifter,
        loops=args.loops, rotate_total_deg=args.rotate,
    )
    if out:
        print(f"OK  {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
