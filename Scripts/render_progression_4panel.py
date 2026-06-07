"""4-panel progression video: 2D -> 3D-raw -> 3D-smoothed -> 3D-turntable.

Layout (2x2):
   ┌──────────────────────┬──────────────────────┐
   │  STAGE 1             │  STAGE 2             │
   │  2D LANDMARKS        │  3D LIFT (raw)       │
   │  coworker-style      │  jittery red         │
   ├──────────────────────┼──────────────────────┤
   │  STAGE 3             │  STAGE 4             │
   │  SMOOTHED            │  3D TURNTABLE        │
   │  oneeuro+bone-lock   │  what UE5 receives   │
   └──────────────────────┴──────────────────────┘

Output: Data/overlays/progression_4panel/<clip>_progression_4panel.mp4 (h264).
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
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
OUT_DIR      = PROJECT_ROOT / "Data" / "overlays" / "progression_4panel"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"

COCO_LINKS = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
              (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]

H36M_LINKS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def load_2d_landmarks(model: str, clip_id: int):
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


def events_for_clip(clip_id: int) -> dict[int, str]:
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    if clip_id not in df.index: return {}
    ev = np.asarray(df.loc[clip_id, "events"])
    events_local = ev[1:9] - ev[0]
    return {int(f): n for f, n in zip(events_local, SWING_EVENTS)}


def project_xyz_ortho(xyz: np.ndarray, w: int, h: int, padding: float = 0.15):
    hip = xyz[:, 0, :].mean(axis=0)
    centered = xyz - hip[None, None, :]
    extent = max(np.abs(centered[..., 0]).max(),
                 np.abs(centered[..., 1]).max(), 1e-6)
    half_w = w * (1 - padding) / 2
    half_h = h * (1 - padding) / 2
    scale = min(half_w, half_h) / extent
    proj = np.zeros((xyz.shape[0], xyz.shape[1], 2), dtype=np.float32)
    proj[..., 0] = centered[..., 0] * scale + w / 2
    proj[..., 1] = centered[..., 1] * scale + h / 2
    return proj


def draw_2d_overlay(frame_bgr, xy, conf,
                     line_color=(50, 220, 50), joint_color=(0, 0, 220)):
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


def draw_3d_skeleton_ortho(canvas, pts_2d, line_color, joint_color):
    for a, b in H36M_LINKS:
        cv2.line(canvas,
                  (int(pts_2d[a, 0]), int(pts_2d[a, 1])),
                  (int(pts_2d[b, 0]), int(pts_2d[b, 1])),
                  line_color, 2, cv2.LINE_AA)
    for j in range(pts_2d.shape[0]):
        cv2.circle(canvas,
                    (int(pts_2d[j, 0]), int(pts_2d[j, 1])),
                    4, joint_color, -1, cv2.LINE_AA)


def label_panel(canvas, title, sub="", title_color=(255, 255, 255)):
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                 title_color, 2, cv2.LINE_AA)
    if sub:
        cv2.rectangle(canvas, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(canvas, sub, (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                     (220, 220, 220), 1, cv2.LINE_AA)


# Persistent matplotlib figure to render 3D turntable panel (faster than recreating)
class TurntableRenderer:
    def __init__(self, panel_w: int, panel_h: int, xyz_smooth: np.ndarray):
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.xyz = xyz_smooth
        pts = xyz_smooth.reshape(-1, 3)
        self.rng = max(pts[:, 0].ptp(), pts[:, 1].ptp(), pts[:, 2].ptp()) * 0.55
        self.cx, self.cy, self.cz = pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()
        self.fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100,
                              facecolor="#1a1a1a")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def render(self, frame_idx: int, azim: float) -> np.ndarray:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        p = self.xyz[frame_idx]
        xs, ys, zs = p[:, 0], -p[:, 1], p[:, 2]
        for a, b in H36M_LINKS:
            ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]],
                    color="#50DC50", linewidth=2.5, alpha=0.95)
        ax.scatter(xs, zs, ys, c="#FF5050", s=35, edgecolors="white",
                   linewidths=0.4, alpha=0.95)
        # Ground grid
        floor_y = ys.min() - 0.05
        gx = np.linspace(self.cx - self.rng, self.cx + self.rng, 5)
        gz = np.linspace(self.cz - self.rng, self.cz + self.rng, 5)
        for gxi in gx:
            ax.plot([gxi, gxi], [gz[0], gz[-1]], [floor_y, floor_y],
                    color="#444444", linewidth=0.5)
        for gzi in gz:
            ax.plot([gx[0], gx[-1]], [gzi, gzi], [floor_y, floor_y],
                    color="#444444", linewidth=0.5)
        ax.view_init(elev=12, azim=azim)
        ax.set_xlim(self.cx - self.rng, self.cx + self.rng)
        ax.set_ylim(self.cz - self.rng, self.cz + self.rng)
        ax.set_zlim(floor_y, floor_y + 2 * self.rng)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")
        ax.grid(False)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        frame_rgba = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

    def close(self):
        plt.close(self.fig)


def render_one_clip(clip_id: int, backbone_2d: str, lifter: str,
                     panel: int = 480) -> Path | None:
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    parquet_2d = CACHE_DIR / backbone_2d / f"{clip_id}.parquet"
    parquet_3d = CACHE_DIR / f"{lifter}_from_{backbone_2d}" / f"{clip_id}.parquet"
    if not (video_path.exists() and parquet_2d.exists() and parquet_3d.exists()):
        print(f"  -- clip {clip_id}: missing input cache")
        return None

    xy_2d, conf_2d = load_2d_landmarks(backbone_2d, clip_id)
    xyz_raw = load_3d_parquet_as_h36m(parquet_3d)
    info = video_info(video_path)
    fps = info["fps"] or 30.0
    xyz_smooth = smooth_sequence(xyz_raw, method="oneeuro", fps=fps, bone_lock=True)

    proj_raw = project_xyz_ortho(xyz_raw,    panel, panel)
    proj_smt = project_xyz_ortho(xyz_smooth, panel, panel)

    # 2D coords -> panel space
    in_w, in_h = info["width"], info["height"]
    xy_2d_panel = xy_2d.copy()
    xy_2d_panel[..., 0] *= panel / in_w
    xy_2d_panel[..., 1] *= panel / in_h

    cap = cv2.VideoCapture(str(video_path))
    n_frames = info["n_frames"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"{clip_id}_progression_4panel_tmp.mp4"
    final = OUT_DIR / f"{clip_id}_progression_4panel.mp4"
    out_w = panel * 2
    out_h = panel * 2
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (out_w, out_h))

    # Stats
    raw_accel = np.linalg.norm(np.diff(xyz_raw,    n=2, axis=0), axis=-1).mean()
    smt_accel = np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).mean()
    pct = (1 - smt_accel / max(raw_accel, 1e-9)) * 100
    det_rate = (conf_2d >= 0.3).mean()
    event_lookup = events_for_clip(clip_id)

    turntable = TurntableRenderer(panel, panel, xyz_smooth)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fi >= xyz_raw.shape[0]: break

        # --- Panel 1 (top-left): 2D landmarks on original ---
        panel_2d = cv2.resize(frame, (panel, panel))
        draw_2d_overlay(panel_2d, xy_2d_panel[fi], conf_2d[fi])
        label_panel(panel_2d,
                     "  STAGE 1: 2D LANDMARKS",
                     f"  {backbone_2d}  detect {det_rate*100:.0f}%")

        # --- Panel 2 (top-right): raw 3D orthographic ---
        panel_raw = np.full((panel, panel, 3), 30, dtype=np.uint8)
        for x in range(0, panel, 40):
            cv2.line(panel_raw, (x, 0), (x, panel), (50, 50, 50), 1)
        for y in range(0, panel, 40):
            cv2.line(panel_raw, (0, y), (panel, y), (50, 50, 50), 1)
        draw_3d_skeleton_ortho(panel_raw, proj_raw[fi],
                                (80, 80, 240), (80, 80, 240))
        label_panel(panel_raw,
                     "  STAGE 2: 3D LIFT (raw)",
                     f"  {lifter}  accel {raw_accel:.4f}")

        # --- Panel 3 (bottom-left): smoothed 3D orthographic ---
        panel_smt = np.full((panel, panel, 3), 30, dtype=np.uint8)
        for x in range(0, panel, 40):
            cv2.line(panel_smt, (x, 0), (x, panel), (50, 50, 50), 1)
        for y in range(0, panel, 40):
            cv2.line(panel_smt, (0, y), (panel, y), (50, 50, 50), 1)
        draw_3d_skeleton_ortho(panel_smt, proj_smt[fi],
                                (80, 240, 80), (80, 240, 80))
        label_panel(panel_smt,
                     "  STAGE 3: SMOOTHED",
                     f"  oneeuro+bone-lock  -{pct:.0f}% jitter")

        # --- Panel 4 (bottom-right): 3D turntable (rotating camera) ---
        progress = fi / max(1, xyz_raw.shape[0] - 1)
        azim = -90 + 180 * progress  # sweep 180 across the swing
        panel_3d = turntable.render(fi, azim)
        if panel_3d.shape[:2] != (panel, panel):
            panel_3d = cv2.resize(panel_3d, (panel, panel))
        # Overlay label + footer (matplotlib already renders the skeleton on dark bg)
        label_panel(panel_3d,
                     "  STAGE 4: 3D VIEWER",
                     f"  what UE5 receives  rotating view")
        # Event label overlay if active
        ev_name = event_lookup.get(fi, "")
        if ev_name:
            cv2.putText(panel_3d, ev_name.upper(),
                        (panel // 2 - 60, panel // 2 + panel // 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 255), 2, cv2.LINE_AA)

        # Stitch 2x2
        top = np.concatenate([panel_2d, panel_raw], axis=1)
        bot = np.concatenate([panel_smt, panel_3d], axis=1)
        out_frame = np.concatenate([top, bot], axis=0)

        writer.write(out_frame)
        fi += 1

    cap.release()
    writer.release()
    turntable.close()

    ffp = ffmpeg_path()
    if ffp:
        try:
            subprocess.run([ffp, "-y", "-i", str(tmp), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", str(final)],
                            check=True, timeout=300)
            tmp.unlink()
            return final
        except Exception as e:
            print(f"  ffmpeg fallback: {e}")
            tmp.rename(final)
            return final
    tmp.rename(final)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="+", type=int, default=[0, 1292])
    p.add_argument("--backbone-2d", default="mediapipe_heavy")
    p.add_argument("--lifter", default="motionbert_full")
    args = p.parse_args()

    print(f"[progression-4panel] {len(args.clips)} clips")
    for cid in args.clips:
        print(f"\n  clip {cid}:")
        out = render_one_clip(cid, args.backbone_2d, args.lifter)
        if out:
            print(f"    OK  {out.relative_to(PROJECT_ROOT)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
