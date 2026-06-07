"""Generate presentation-grade visualizations for the meeting.

Produces a folder of PNG files showing:
  1. The classic coworker-style 2D Y-trajectory plot (for reference)
  2. 3D Y-trajectory BEFORE and AFTER smoothing (the headline before/after)
  3. Bone-length stability over time, before/after smoothing
  4. Per-joint acceleration histogram before/after smoothing
  5. 3D X / Y / Z position panels (shows the value-add of the lift)
  6. Multi-model overlay (top 4 winners on the same plot)
  7. Combined dashboard PNG with the killer slide

Usage:
    python make_visualizations.py                       # default: clip 0
    python make_visualizations.py --clip 173
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import COCO17_IDX, SWING_EVENTS, video_info
from export_ue5 import H36M17_IDX, H36M17_PARENTS, load_3d_parquet_as_h36m
from smoothing import smooth_sequence

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "eval_runs"
OUT_DIR      = DATA_DIR / "visualizations"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140


# ===========================================================================
# Data loaders
# ===========================================================================

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


def load_3d_landmarks(model: str, clip_id: int) -> np.ndarray | None:
    p = CACHE_DIR / model / f"{clip_id}.parquet"
    if not p.exists(): return None
    return load_3d_parquet_as_h36m(p)


def get_events(clip_id: int, df_labels: pd.DataFrame) -> np.ndarray | None:
    if clip_id not in df_labels.index: return None
    ev = np.asarray(df_labels.loc[clip_id, "events"])
    return ev[1:9] - ev[0]


# ===========================================================================
# Plot 1: Coworker-style 2D Y-trajectory (reference)
# ===========================================================================

def plot_2d_yt(xy: np.ndarray, events: np.ndarray, clip_id: int, model: str, video_w: int = 160) -> Path:
    """Recreate the coworker's exact plot style: normalized Y position for
    LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER."""
    fig, ax = plt.subplots(figsize=(13, 6))
    joints = [
        ("LEFT_WRIST", COCO17_IDX["left_wrist"], "tab:blue"),
        ("RIGHT_WRIST", COCO17_IDX["right_wrist"], "tab:orange"),
        ("LEFT_SHOULDER", COCO17_IDX["left_shoulder"], "tab:green"),
        ("RIGHT_SHOULDER", COCO17_IDX["right_shoulder"], "tab:red"),
    ]
    for name, idx, color in joints:
        y_normalized = xy[:, idx, 1] / video_w
        ax.plot(np.arange(len(y_normalized)), y_normalized,
                 label=name, color=color, linewidth=1.4)
    if events is not None:
        for ev_name, ev_frame in zip(SWING_EVENTS, events):
            ax.axvline(int(ev_frame), color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Y Position")
    ax.set_title(f"{model} Pose Landmarks Over Time   (clip {clip_id})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_2d_ytraj_{model}.png"
    fig.savefig(out); plt.close(fig)
    return out


# ===========================================================================
# Plot 2: 3D Y-trajectory BEFORE and AFTER smoothing
# ===========================================================================

def plot_3d_yt_before_after(xyz_raw: np.ndarray, xyz_smooth: np.ndarray,
                              events: np.ndarray, clip_id: int, model: str) -> Path:
    """The headline visualization — same 4 landmarks (wrists, shoulders)
    Y-position over time, raw vs smoothed."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    joints = [
        ("LEFT_WRIST",    H36M17_IDX["left_wrist"],    "tab:blue"),
        ("RIGHT_WRIST",   H36M17_IDX["right_wrist"],   "tab:orange"),
        ("LEFT_SHOULDER", H36M17_IDX["left_shoulder"], "tab:green"),
        ("RIGHT_SHOULDER",H36M17_IDX["right_shoulder"],"tab:red"),
    ]
    for ax, xyz, title_suffix in [(ax1, xyz_raw, "BEFORE smoothing (raw 3D lift)"),
                                    (ax2, xyz_smooth, "AFTER smoothing (One-Euro + bone-lock)")]:
        for name, idx, color in joints:
            ax.plot(np.arange(xyz.shape[0]), xyz[:, idx, 1],
                     label=name, color=color, linewidth=1.4)
        if events is not None:
            for ev_name, ev_frame in zip(SWING_EVENTS, events):
                ax.axvline(int(ev_frame), color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
                ax.text(int(ev_frame), ax.get_ylim()[1] * 0.97, ev_name[:4],
                         rotation=90, fontsize=7, color="gray", va="top")
        ax.invert_yaxis()
        ax.set_ylabel("Y position (3D, normalized)")
        ax.set_title(f"{title_suffix}")
        ax.legend(loc="upper right", fontsize=9)
    ax2.set_xlabel("Frame")
    plt.suptitle(f"3D Wrist + Shoulder Y-trajectories   ·   clip {clip_id}   ·   {model}",
                  y=1.01, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_3d_ytraj_beforeafter_{model}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Plot 3: Bone-length stability over time, before/after smoothing
# ===========================================================================

def plot_bone_stability(xyz_raw: np.ndarray, xyz_smooth: np.ndarray,
                         clip_id: int, model: str) -> Path:
    """Show how bone-lock makes each bone length perfectly flat."""
    bones = [
        ("left upper arm",  H36M17_IDX["left_shoulder"],  H36M17_IDX["left_elbow"]),
        ("right upper arm", H36M17_IDX["right_shoulder"], H36M17_IDX["right_elbow"]),
        ("left forearm",    H36M17_IDX["left_elbow"],     H36M17_IDX["left_wrist"]),
        ("right forearm",   H36M17_IDX["right_elbow"],    H36M17_IDX["right_wrist"]),
        ("left thigh",      H36M17_IDX["left_hip"],       H36M17_IDX["left_knee"]),
        ("right thigh",     H36M17_IDX["right_hip"],      H36M17_IDX["right_knee"]),
        ("left shin",       H36M17_IDX["left_knee"],      H36M17_IDX["left_ankle"]),
        ("right shin",      H36M17_IDX["right_knee"],     H36M17_IDX["right_ankle"]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharey=False)
    for ax, (bname, a, b) in zip(axes.flat, bones):
        raw_len = np.linalg.norm(xyz_raw[:, a] - xyz_raw[:, b], axis=-1)
        smt_len = np.linalg.norm(xyz_smooth[:, a] - xyz_smooth[:, b], axis=-1)
        raw_cv = raw_len.std() / max(raw_len.mean(), 1e-9)
        smt_cv = smt_len.std() / max(smt_len.mean(), 1e-9)
        ax.plot(raw_len, label=f"raw (CV={raw_cv:.3f})", color="tab:red", alpha=0.7, linewidth=1)
        ax.plot(smt_len, label=f"smoothed (CV={smt_cv:.3f})", color="tab:green", linewidth=1.5)
        ax.set_title(bname, fontsize=10)
        ax.set_ylabel("length")
        ax.legend(fontsize=8, loc="upper right")
    plt.suptitle(f"Bone-length stability over time   ·   clip {clip_id}   ·   {model}\n"
                  f"(lower CV = more physically stable; smoothing target = ~constant)",
                  y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_bone_stability_{model}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Plot 4: Per-joint acceleration distribution
# ===========================================================================

def plot_acceleration_dist(xyz_raw: np.ndarray, xyz_smooth: np.ndarray,
                            clip_id: int, model: str) -> Path:
    """Histogram of per-frame acceleration magnitudes across all joints,
    before vs after smoothing. This is the dimensionless 'how much smoother
    did it get' visualization."""
    fig, ax = plt.subplots(figsize=(11, 5))
    a_raw = np.linalg.norm(np.diff(xyz_raw, n=2, axis=0), axis=-1).flatten()
    a_smt = np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).flatten()
    bins = np.linspace(0, max(a_raw.max(), a_smt.max()) * 0.5, 60)
    ax.hist(a_raw, bins=bins, alpha=0.55, label=f"raw  (mean {a_raw.mean():.4f})",
            color="tab:red", density=True)
    ax.hist(a_smt, bins=bins, alpha=0.55, label=f"smoothed  (mean {a_smt.mean():.4f})",
            color="tab:green", density=True)
    ax.axvline(a_raw.mean(),  color="tab:red", linestyle="--", linewidth=1)
    ax.axvline(a_smt.mean(),  color="tab:green", linestyle="--", linewidth=1)
    ax.set_xlabel("Per-frame acceleration magnitude  (3D units)")
    ax.set_ylabel("Density")
    pct = (1.0 - a_smt.mean() / a_raw.mean()) * 100
    ax.set_title(f"Per-joint acceleration distribution   ·   clip {clip_id}   ·   {model}\n"
                  f"Mean acceleration reduced by {pct:.0f}%   (raw -> smoothed)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_accel_dist_{model}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Plot 5: 3D X / Y / Z panels (shows the value-add of the lift)
# ===========================================================================

def plot_xyz_panels(xyz_smooth: np.ndarray, events: np.ndarray,
                     clip_id: int, model: str) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    joints = [
        ("LEFT_WRIST",    H36M17_IDX["left_wrist"],    "tab:blue"),
        ("RIGHT_WRIST",   H36M17_IDX["right_wrist"],   "tab:orange"),
        ("LEFT_SHOULDER", H36M17_IDX["left_shoulder"], "tab:green"),
        ("RIGHT_SHOULDER",H36M17_IDX["right_shoulder"],"tab:red"),
        ("HIP_CENTER",    H36M17_IDX["hip_center"],    "tab:purple"),
    ]
    axis_labels = ["X position (left/right)", "Y position (down/up — inverted)",
                    "Z position (toward camera / away)"]
    for axis_i, (ax, label) in enumerate(zip(axes, axis_labels)):
        for name, idx, color in joints:
            ax.plot(np.arange(xyz_smooth.shape[0]), xyz_smooth[:, idx, axis_i],
                     label=name, color=color, linewidth=1.4)
        if events is not None:
            for ev_name, ev_frame in zip(SWING_EVENTS, events):
                ax.axvline(int(ev_frame), color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
        if axis_i == 1: ax.invert_yaxis()
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Frame")
    plt.suptitle(f"Full 3D trajectory (X / Y / Z)   ·   clip {clip_id}   ·   {model}\n"
                  f"Z is the depth axis recovered by the lifter — the value-add over 2D",
                  y=1.005, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_xyz_panels_{model}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Plot 6: Multi-model wrist Y comparison
# ===========================================================================

def plot_multi_model_wrist(clip_id: int, events: np.ndarray, models: list[tuple[str, str, bool]]) -> Path:
    """models = list of (display_name, cache_name, is_3d)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = sns.color_palette("tab10", n_colors=len(models))
    for i, (display, cache_name, is_3d) in enumerate(models):
        if is_3d:
            xyz = load_3d_landmarks(cache_name, clip_id)
            if xyz is None: continue
            y = xyz[:, H36M17_IDX["right_wrist"], 1]
            y = (y - y.min()) / max(y.max() - y.min(), 1e-9)
        else:
            res = load_2d_landmarks(cache_name, clip_id)
            if res is None: continue
            xy, conf = res
            y = xy[:, COCO17_IDX["right_wrist"], 1] / 160.0
        ax.plot(np.arange(len(y)), y, label=display, color=palette[i], linewidth=1.3, alpha=0.85)
    if events is not None:
        for ev_name, ev_frame in zip(SWING_EVENTS, events):
            ax.axvline(int(ev_frame), color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
            ax.text(int(ev_frame), ax.get_ylim()[1] * 0.98, ev_name[:4],
                     rotation=90, fontsize=8, color="gray", va="top")
    ax.invert_yaxis()
    ax.set_xlabel("Frame"); ax.set_ylabel("Right wrist Y (normalized [0,1])")
    ax.set_title(f"Right-wrist Y trajectory across models   ·   clip {clip_id}\n"
                  f"How well does each model's wrist trace align with the labeled swing events (dotted)?")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    out = OUT_DIR / f"clip{clip_id}_multi_model_wrist.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Plot 7: Combined dashboard PNG
# ===========================================================================

def plot_dashboard(xy_2d: np.ndarray, xyz_raw: np.ndarray, xyz_smooth: np.ndarray,
                    events: np.ndarray, clip_id: int, lifter: str) -> Path:
    """One PNG telling the whole story for slides."""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)

    # (0,0:2) — 2D Y-traj
    ax = fig.add_subplot(gs[0, 0:2])
    joints2d = [("L wrist", COCO17_IDX["left_wrist"], "tab:blue"),
                ("R wrist", COCO17_IDX["right_wrist"], "tab:orange"),
                ("L shoulder", COCO17_IDX["left_shoulder"], "tab:green"),
                ("R shoulder", COCO17_IDX["right_shoulder"], "tab:red")]
    for n, idx, c in joints2d:
        ax.plot(xy_2d[:, idx, 1] / 160, label=n, color=c, linewidth=1.2)
    if events is not None:
        for ef in events: ax.axvline(int(ef), color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.invert_yaxis(); ax.set_title("①  2D landmark Y (MediaPipe Lite, normalized)", fontsize=11)
    ax.legend(fontsize=8, ncol=4, loc="lower right")

    # (0,2) — text summary
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    a_raw = np.linalg.norm(np.diff(xyz_raw, n=2, axis=0), axis=-1).mean()
    a_smt = np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).mean()
    pct = (1 - a_smt / a_raw) * 100
    summary = (f"clip {clip_id}\n\n"
                f"frames: {xy_2d.shape[0]}\n\n"
                f"pipeline:\n  2D: mediapipe_lite\n  3D: {lifter}\n  smoothing: one-euro + bone-lock\n\n"
                f"acceleration:\n  raw       {a_raw:.4f}\n  smoothed  {a_smt:.4f}\n  -{pct:.0f}% jitter")
    ax.text(0.05, 0.95, summary, fontsize=10, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="gray"))

    # (1,0:3) — 3D Y traj raw vs smooth
    ax = fig.add_subplot(gs[1, :])
    joints3d = [("L wrist", H36M17_IDX["left_wrist"], "tab:blue"),
                ("R wrist", H36M17_IDX["right_wrist"], "tab:orange"),
                ("L shoulder", H36M17_IDX["left_shoulder"], "tab:green"),
                ("R shoulder", H36M17_IDX["right_shoulder"], "tab:red")]
    for n, idx, c in joints3d:
        ax.plot(xyz_raw[:, idx, 1], color=c, linewidth=1.2, alpha=0.5, linestyle="--")
        ax.plot(xyz_smooth[:, idx, 1], color=c, linewidth=1.5, label=n)
    if events is not None:
        for ef in events: ax.axvline(int(ef), color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.invert_yaxis(); ax.set_title("②  3D landmark Y   ·   dashed = raw, solid = smoothed", fontsize=11)
    ax.set_xlabel("Frame")
    ax.legend(fontsize=9, ncol=4, loc="lower right")

    # (2,0) — bone length: left upper arm
    ax = fig.add_subplot(gs[2, 0])
    a, b = H36M17_IDX["left_shoulder"], H36M17_IDX["left_elbow"]
    raw_len = np.linalg.norm(xyz_raw[:, a] - xyz_raw[:, b], axis=-1)
    smt_len = np.linalg.norm(xyz_smooth[:, a] - xyz_smooth[:, b], axis=-1)
    ax.plot(raw_len, color="tab:red", alpha=0.7, label=f"raw CV={raw_len.std()/raw_len.mean():.3f}")
    ax.plot(smt_len, color="tab:green", linewidth=1.5, label=f"smooth CV={smt_len.std()/smt_len.mean():.3f}")
    ax.set_title("③  Left upper arm length", fontsize=10)
    ax.legend(fontsize=8)

    # (2,1) — bone length: left shin
    ax = fig.add_subplot(gs[2, 1])
    a, b = H36M17_IDX["left_knee"], H36M17_IDX["left_ankle"]
    raw_len = np.linalg.norm(xyz_raw[:, a] - xyz_raw[:, b], axis=-1)
    smt_len = np.linalg.norm(xyz_smooth[:, a] - xyz_smooth[:, b], axis=-1)
    ax.plot(raw_len, color="tab:red", alpha=0.7, label=f"raw CV={raw_len.std()/raw_len.mean():.3f}")
    ax.plot(smt_len, color="tab:green", linewidth=1.5, label=f"smooth CV={smt_len.std()/smt_len.mean():.3f}")
    ax.set_title("④  Left shin length", fontsize=10)
    ax.legend(fontsize=8)

    # (2,2) — accel histogram
    ax = fig.add_subplot(gs[2, 2])
    bins = np.linspace(0, max(a_raw, a_smt) * 8, 50)
    ax.hist(np.linalg.norm(np.diff(xyz_raw, n=2, axis=0), axis=-1).flatten(),
            bins=bins, alpha=0.55, color="tab:red", label="raw", density=True)
    ax.hist(np.linalg.norm(np.diff(xyz_smooth, n=2, axis=0), axis=-1).flatten(),
            bins=bins, alpha=0.55, color="tab:green", label="smoothed", density=True)
    ax.set_xlabel("|accel|"); ax.set_title("⑤  Acceleration distribution", fontsize=10)
    ax.legend(fontsize=8)

    plt.suptitle(f"GOLF SWING 3D RECONSTRUCTION DASHBOARD   ·   clip {clip_id}   ·   {lifter}",
                  y=1.005, fontsize=13, fontweight="bold")
    out = OUT_DIR / f"clip{clip_id}_dashboard.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=int, default=0, help="Clip ID")
    p.add_argument("--lifter", default="motionbert_full",
                   help="3D lifter: motionbert_full, motionbert_lite, golfpose3d")
    p.add_argument("--backbone", default="mediapipe_lite", help="2D backbone for the 3D lifter")
    p.add_argument("--ref-2d-model", default="mediapipe_heavy",
                   help="2D model used for the reference (top-of-dashboard) plot")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_labels = pd.read_pickle(GOLFDB_PKL).set_index("id")
    events = get_events(args.clip, df_labels)
    print(f"[viz] clip {args.clip}   events: {events.tolist() if events is not None else 'none'}")

    # Reference 2D (coworker style)
    ref_2d = load_2d_landmarks(args.ref_2d_model, args.clip)
    if ref_2d is None:
        print(f"[viz] no 2D cache for {args.ref_2d_model}/{args.clip}")
        return
    xy_ref, _ = ref_2d
    out1 = plot_2d_yt(xy_ref, events, args.clip, args.ref_2d_model)
    print(f"  wrote {out1.name}")

    # Also do the MediaPipe Lite 2D for the dashboard (the recommended backbone)
    mp_lite = load_2d_landmarks(args.backbone, args.clip)
    if mp_lite is None:
        print(f"[viz] no 2D cache for {args.backbone}/{args.clip}")
        return
    xy_lite, _ = mp_lite
    out1b = plot_2d_yt(xy_lite, events, args.clip, args.backbone)
    print(f"  wrote {out1b.name}")

    # 3D raw + smoothed
    cache_name = f"{args.lifter}_from_{args.backbone}"
    xyz_raw = load_3d_landmarks(cache_name, args.clip)
    if xyz_raw is None:
        print(f"[viz] no 3D cache for {cache_name}/{args.clip}")
        return
    info_path = PROJECT_ROOT / "Data" / "videos_160" / f"{args.clip}.mp4"
    fps = video_info(info_path).get("fps") or 30.0 if info_path.exists() else 30.0
    xyz_smooth = smooth_sequence(xyz_raw, method="oneeuro", fps=fps, bone_lock=True)

    out2 = plot_3d_yt_before_after(xyz_raw, xyz_smooth, events, args.clip, cache_name)
    print(f"  wrote {out2.name}")
    out3 = plot_bone_stability(xyz_raw, xyz_smooth, args.clip, cache_name)
    print(f"  wrote {out3.name}")
    out4 = plot_acceleration_dist(xyz_raw, xyz_smooth, args.clip, cache_name)
    print(f"  wrote {out4.name}")
    out5 = plot_xyz_panels(xyz_smooth, events, args.clip, cache_name)
    print(f"  wrote {out5.name}")

    # Multi-model wrist comparison (the head-to-head visual)
    out6 = plot_multi_model_wrist(args.clip, events, models=[
        ("MediaPipe Lite (2D)",                  "mediapipe_lite",                       False),
        ("MediaPipe Heavy (2D)",                 "mediapipe_heavy",                      False),
        ("MotionBERT-Full ← MP-Lite",            "motionbert_full_from_mediapipe_lite",  True),
        ("MotionBERT-Lite ← MP-Lite",            "motionbert_lite_from_mediapipe_lite",  True),
        ("GolfPose ← MP-Lite",                   "golfpose3d_from_mediapipe_lite",       True),
    ])
    print(f"  wrote {out6.name}")

    # Combined dashboard
    out7 = plot_dashboard(xy_lite, xyz_raw, xyz_smooth, events, args.clip, cache_name)
    print(f"  wrote {out7.name}")

    print(f"\n[viz] all written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
