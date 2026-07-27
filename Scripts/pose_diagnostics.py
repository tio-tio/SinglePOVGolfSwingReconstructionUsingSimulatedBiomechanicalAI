"""Pose-quality diagnostics for the MediaPipe -> 3D lift pipeline.

Answers two production questions per clip, from the cached landmark parquets:

  1. Is MediaPipe (lite) swapping left/right identities — especially the
     wrists, which sit on top of each other on the club grip for the entire
     swing? Swaps inject step discontinuities that survive the 3D lift and
     read as jitter in the replay, worst at address where nothing should move.
  2. Is one leg systematically "elevated" in the 3D output? Measured as the
     left-vs-right ankle height gap (camera y, y-down) during the low-motion
     address segment, plus the ankle depth gap to test the camera-tilt
     hypothesis (feet at different depths + pitched camera = fake height gap).

Per-clip artifacts (also written on every upload via pipeline.py):
  <stem>_pose_diag.json    all metrics below
  <stem>_pose_debug.mp4    L/R-colored MediaPipe skeleton overlay
                           (left=orange, right=cyan) with per-frame swap
                           banners and wrist confidence readouts

Batch mode aggregates the metrics over the cached eval clips:

  python pose_diagnostics.py --batch 200
  python pose_diagnostics.py path/to/clip.mp4            # single clip + video
  python pose_diagnostics.py --stem 1292                 # cached clip by stem
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import COCO17_NAMES, COCO17_IDX
from video_orientation import open_capture

L_SH, R_SH = COCO17_IDX["left_shoulder"], COCO17_IDX["right_shoulder"]
L_WR, R_WR = COCO17_IDX["left_wrist"], COCO17_IDX["right_wrist"]
L_AN, R_AN = COCO17_IDX["left_ankle"], COCO17_IDX["right_ankle"]
L_HP, R_HP = COCO17_IDX["left_hip"], COCO17_IDX["right_hip"]

# L/R pairs checked for identity swaps (index_left, index_right, name)
SWAP_PAIRS = (
    (L_WR, R_WR, "wrist"),
    (COCO17_IDX["left_elbow"], COCO17_IDX["right_elbow"], "elbow"),
    (L_AN, R_AN, "ankle"),
    (COCO17_IDX["left_knee"], COCO17_IDX["right_knee"], "knee"),
)


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------

def load_2d(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    """(T,17,2) xy pixels + (T,17) conf from a cached 2D parquet."""
    df = pd.read_parquet(parquet)
    T = int(df["frame"].max()) + 1
    xy = np.zeros((T, 17, 2), dtype=np.float32)
    conf = np.zeros((T, 17), dtype=np.float32)
    fr = df["frame"].to_numpy(int)
    kp = df["kp_idx"].to_numpy(int)
    xy[fr, kp, 0] = df["x"].to_numpy(np.float32)
    xy[fr, kp, 1] = df["y"].to_numpy(np.float32)
    conf[fr, kp] = df["conf"].to_numpy(np.float32)
    return xy, conf


def load_3d(parquet: Path) -> np.ndarray:
    """(T,17,3) xyz (COCO order, camera coords: x right, y DOWN, z depth)."""
    df = pd.read_parquet(parquet)
    T = int(df["frame"].max()) + 1
    xyz = np.zeros((T, 17, 3), dtype=np.float32)
    fr = df["frame"].to_numpy(int)
    kp = df["kp_idx"].to_numpy(int)
    for i, c in enumerate("xyz"):
        xyz[fr, kp, i] = df[c].to_numpy(np.float32)
    return xyz


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _torso_scale(xy: np.ndarray) -> float:
    """Median shoulder-midpoint -> hip-midpoint distance; pixel size normalizer."""
    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hp = (xy[:, L_HP] + xy[:, R_HP]) / 2
    d = np.linalg.norm(sh - hp, axis=1)
    d = d[d > 1e-3]
    return float(np.median(d)) if d.size else 1.0


def detect_swaps(xy: np.ndarray, conf: np.ndarray,
                 min_conf: float = 0.3, margin: float = 0.8) -> dict:
    """Continuity-based L/R identity-swap detection.

    A frame t is a swap candidate for a pair when matching (L_t->R_{t-1},
    R_t->L_{t-1}) is markedly cheaper than the straight assignment — i.e. the
    two detections jumped across each other faster than the limbs plausibly
    moved. Only scored when all four points clear min_conf and the pair is
    separated enough for the assignment to be meaningful (else "merged").
    """
    T = xy.shape[0]
    scale = _torso_scale(xy)
    out = {}
    for li, ri, name in SWAP_PAIRS:
        swap_frames = []
        merged = 0
        scored = 0
        sep = np.linalg.norm(xy[:, li] - xy[:, ri], axis=1) / scale
        for t in range(1, T):
            if min(conf[t, li], conf[t, ri], conf[t - 1, li], conf[t - 1, ri]) < min_conf:
                continue
            if sep[t] < 0.15:          # points on top of each other: unattributable
                merged += 1
                continue
            scored += 1
            straight = (np.linalg.norm(xy[t, li] - xy[t - 1, li])
                        + np.linalg.norm(xy[t, ri] - xy[t - 1, ri]))
            swapped = (np.linalg.norm(xy[t, li] - xy[t - 1, ri])
                       + np.linalg.norm(xy[t, ri] - xy[t - 1, li]))
            if swapped < straight * margin and straight / scale > 0.05:
                swap_frames.append(t)
        out[name] = {
            "swap_frames": swap_frames,
            "n_swaps": len(swap_frames),
            "swaps_per_100f": 100.0 * len(swap_frames) / max(T, 1),
            "frames_scored": scored,
            "frames_merged": merged,
            "merged_frac": merged / max(T - 1, 1),
        }
    return out


# Repair groups, decoded in order. Each group is a set of L/R joint pairs that
# swap TOGETHER, judged by the bones listed (anchors like shoulders/hips are
# far apart on the torso — MediaPipe essentially never confuses those). Whole-
# limb groups run first (MediaPipe's dominant failure mode swaps the entire
# arm/leg), then leaf-only groups catch wrist/ankle flickers against the
# already-repaired parents. Judging elbows/knees only in the whole-limb group
# keeps them from being ripped away from their own children (measured: an
# elbow-only pass regressed forearm consistency badly on some clips).
_LS, _RS = COCO17_IDX["left_shoulder"], COCO17_IDX["right_shoulder"]
_LE, _RE = COCO17_IDX["left_elbow"], COCO17_IDX["right_elbow"]
_LK, _RK = COCO17_IDX["left_knee"], COCO17_IDX["right_knee"]
REPAIR_GROUPS = (
    # name, pairs to swap jointly, bones scored
    ("arm", ((_LE, _RE), (L_WR, R_WR)),
     ((_LS, _LE), (_RS, _RE), (_LE, L_WR), (_RE, R_WR))),
    ("wrist", ((L_WR, R_WR),),
     ((_LE, L_WR), (_RE, R_WR))),
    ("leg", ((_LK, _RK), (L_AN, R_AN)),
     ((L_HP, _LK), (R_HP, _RK), (_LK, L_AN), (_RK, R_AN))),
    ("ankle", ((L_AN, R_AN),),
     ((_LK, L_AN), (_RK, R_AN))),
)


def repair_lr_swaps(xy: np.ndarray, conf: np.ndarray,
                    min_conf: float = 0.3,
                    bone_w: float = 0.5,
                    switch_pen: float = 0.06) -> tuple[np.ndarray, np.ndarray, dict]:
    """Undo L/R identity swaps via a 2-state Viterbi pass per limb pair.

    Neither single cue survives a golf swing alone (both were measured on the
    400-clip eval batch):
      - trajectory continuity is uninformative during the downswing (hands
        move too far per frame) — a greedy continuity tracker that guesses
        wrong once drags the wrong identity through the rest of the clip;
      - per-frame bone-length assignment fixes identity but flips freely on
        noisy frames, injecting position steps (2D wrist jitter UP in 92%
        of clips).
    So both are combined in a proper 2-state decode over {straight, swapped}:
      emission   = bone-length deviation of the assignment from the clip-median
                   lengths (a wrist labeled onto the wrong arm stretches its
                   forearm across the body — the invariant that survives fast
                   motion), weighted by `bone_w`;
      transition = continuity cost of the labeled positions between frames,
                   plus `switch_pen` (fraction of torso scale) per identity
                   flip so the state holds steady through evidence-free
                   stretches (hands merged on the grip).
    Groups are decoded whole-limb first, then leaf-only (see REPAIR_GROUPS)
    so elbows/knees are always judged together with their children.
    Validated on 400 clips: limb bone-length CV improves in ~3/4 of clips
    with small bounded regressions and 2D wrist jitter flat-to-better.

    Returns (xy_fixed, conf_fixed, {group_name: n_frames_swapped}).
    """
    xy = xy.copy()
    conf = conf.copy()
    T = xy.shape[0]
    scale = _torso_scale(xy)
    pen = switch_pen * scale
    stats = {}
    for name, group, bones in REPAIR_GROUPS:
        swap_map = {}
        for l, r in group:
            swap_map[l], swap_map[r] = r, l
        involved = sorted({j for bn in bones for j in bn})
        swapped_joints = sorted(swap_map)
        ok = (conf[:, involved] >= min_conf).all(axis=1)
        if ok.sum() < 10:
            stats[name] = 0
            continue

        def _len(a, b, state):
            aa = swap_map.get(a, a) if state else a
            bb = swap_map.get(b, b) if state else b
            return np.linalg.norm(xy[:, aa] - xy[:, bb], axis=1)

        # reference bone lengths from the straight assignment: the majority of
        # frames are correctly labeled, and the median survives the rest
        refs = [float(np.median(_len(a, b, 0)[ok])) for a, b in bones]
        E = np.zeros((T, 2), np.float32)                        # emission: (t, state)
        for state in (0, 1):
            dev = sum(np.abs(_len(a, b, state) - r) for (a, b), r in zip(bones, refs))
            E[ok, state] = dev[ok] * bone_w
        # labeled positions of the group's joints under each state:
        # (t, state, joint, xy)
        P = np.stack([np.stack([xy[:, j] for j in swapped_joints], 1),
                      np.stack([xy[:, swap_map[j]] for j in swapped_joints], 1)], 1)
        SW = np.array([[0, pen], [pen, 0]], np.float32)
        D = np.full((T, 2), np.inf, np.float32)
        D[0] = E[0]
        D[0, 1] += pen                                          # prior: start straight
        bp = np.zeros((T, 2), np.int8)
        for t in range(1, T):
            if ok[t] and ok[t - 1]:
                trans = np.linalg.norm(P[t][None] - P[t - 1][:, None], axis=3).sum(2) + SW
            else:
                trans = SW                                      # no evidence across the gap
            tot = D[t - 1][:, None] + trans + E[t][None, :]
            bp[t] = tot.argmin(0)
            D[t] = tot[bp[t], [0, 1]]
        s = np.zeros(T, np.int8)
        s[-1] = int(D[-1].argmin())
        for t in range(T - 1, 0, -1):
            s[t - 1] = bp[t, s[t]]
        idx = np.where(s == 1)[0]
        for l, r in group:
            xy[np.ix_(idx, [l, r])] = xy[np.ix_(idx, [r, l])]
            conf[np.ix_(idx, [l, r])] = conf[np.ix_(idx, [r, l])]
        stats[name] = int(len(idx))
    return xy, conf, stats


def write_2d_parquet(xy: np.ndarray, conf: np.ndarray, out: Path) -> None:
    """Write (T,17,2)+( T,17) back to the canonical 2D landmark parquet schema."""
    T = xy.shape[0]
    fr = np.repeat(np.arange(T), 17)
    kp = np.tile(np.arange(17), T)
    pd.DataFrame({
        "frame": fr,
        "kp_idx": kp,
        "kp_name": [COCO17_NAMES[k] for k in kp],
        "x": xy[fr, kp, 0],
        "y": xy[fr, kp, 1],
        "conf": conf[fr, kp],
    }).to_parquet(out, index=False)


def low_motion_mask(xy: np.ndarray, conf: np.ndarray,
                    min_conf: float = 0.3, quantile: float = 0.3) -> np.ndarray:
    """(T,) mask of the stillest frames (address / post-finish) by median joint speed."""
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2)          # (T-1, 17)
    ok = (conf[1:] >= min_conf) & (conf[:-1] >= min_conf)
    v = np.where(ok, v, np.nan)
    med = np.nanmedian(v, axis=1)                            # (T-1,)
    med = np.concatenate([[med[0] if med.size else 0.0], med])
    thr = np.nanquantile(med, quantile) if np.isfinite(med).any() else 0.0
    return med <= thr


def jitter_2d(xy: np.ndarray, conf: np.ndarray, joints: list[int],
              mask: np.ndarray | None = None, min_conf: float = 0.3) -> float:
    """Mean 2nd-derivative magnitude over given joints, torso-normalized (x100)."""
    scale = _torso_scale(xy)
    a = np.diff(xy[:, joints], n=2, axis=0)                  # (T-2, J, 2)
    ok = (conf[2:, joints] >= min_conf) & (conf[1:-1, joints] >= min_conf) \
         & (conf[:-2, joints] >= min_conf)
    mag = np.where(ok, np.linalg.norm(a, axis=2), np.nan)
    if mask is not None:
        mag = np.where(mask[2:, None], mag, np.nan)
    return float(np.nanmean(mag) / scale * 100.0) if np.isfinite(mag).any() else float("nan")


def ankle_asymmetry_3d(xyz: np.ndarray, still: np.ndarray) -> dict:
    """Left-minus-right ankle gaps during the still (address) frames.

    Camera coords are y-DOWN: dy > 0 means the LEFT ankle is LOWER in the
    image (right ankle rides higher -> right leg looks elevated in the
    viewer, which maps camera -y straight to world up). Gaps are fractions
    of body height (vertical extent), i.e. 0.02 = 2% of body height.
    """
    still_idx = np.where(still[:xyz.shape[0]])[0]
    if still_idx.size == 0:
        still_idx = np.arange(xyz.shape[0])
    height = float(np.median(xyz[:, :, 1].max(axis=1) - xyz[:, :, 1].min(axis=1)))
    height = max(height, 1e-6)
    dy = xyz[still_idx, L_AN, 1] - xyz[still_idx, R_AN, 1]
    dz = xyz[still_idx, L_AN, 2] - xyz[still_idx, R_AN, 2]
    return {
        "ankle_dy_frac": float(np.median(dy) / height),
        "ankle_dz_frac": float(np.median(dz) / height),
        "higher_leg": "right" if np.median(dy) > 0 else "left",
        "n_still_frames": int(still_idx.size),
    }


# ---------------------------------------------------------------------------
# Per-clip diagnostic
# ---------------------------------------------------------------------------

def diagnose_clip(parquet_2d: Path, parquet_3d: Path | None = None) -> dict:
    xy, conf = load_2d(parquet_2d)
    still = low_motion_mask(xy, conf)
    swaps = detect_swaps(xy, conf)
    wrists, ankles, hips = [L_WR, R_WR], [L_AN, R_AN], [L_HP, R_HP]
    diag = {
        "clip": parquet_2d.stem,
        "n_frames": int(xy.shape[0]),
        "swaps": swaps,
        "conf": {
            "left_wrist_mean": float(conf[:, L_WR].mean()),
            "right_wrist_mean": float(conf[:, R_WR].mean()),
            "left_wrist_frac_low": float((conf[:, L_WR] < 0.5).mean()),
            "right_wrist_frac_low": float((conf[:, R_WR] < 0.5).mean()),
        },
        "jitter_2d": {   # torso-normalized x100; "still" = address-like frames
            "wrists_all": jitter_2d(xy, conf, wrists),
            "wrists_still": jitter_2d(xy, conf, wrists, mask=still),
            "ankles_still": jitter_2d(xy, conf, ankles, mask=still),
            "hips_still": jitter_2d(xy, conf, hips, mask=still),
        },
    }
    if parquet_3d is not None and parquet_3d.exists():
        xyz = load_3d(parquet_3d)
        diag["ankle_3d"] = ankle_asymmetry_3d(xyz, still)
    return diag


# ---------------------------------------------------------------------------
# L/R-colored debug skeleton overlay
# ---------------------------------------------------------------------------

# COCO links split by side so a crossed wrist is visible at a glance
LEFT_LINKS = [(5, 7), (7, 9), (11, 13), (13, 15), (5, 11)]
RIGHT_LINKS = [(6, 8), (8, 10), (12, 14), (14, 16), (6, 12)]
CENTER_LINKS = [(5, 6), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4)]
LEFT_COLOR = (0, 140, 255)    # orange (BGR)
RIGHT_COLOR = (255, 220, 0)   # cyan-ish
CENTER_COLOR = (120, 220, 120)


def write_pose_debug_overlay(video_path: Path, landmarks_2d_parquet: Path,
                             out_mp4: Path, diag: dict | None = None):
    """MediaPipe skeleton debug video: left limbs orange, right limbs cyan,
    joints colored by confidence (green=high, red=low), wrist L/R tags,
    and a red SWAP banner on detected wrist identity-swap frames."""
    import cv2

    xy, conf = load_2d(landmarks_2d_parquet)
    swap_frames = set()
    if diag is not None:
        swap_frames = set(diag.get("swaps", {}).get("wrist", {}).get("swap_frames", []))

    cap = open_capture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # tiny sources (e.g. 160px eval clips) get upscaled so the markup is legible;
    # marker sizes then scale with the OUTPUT resolution so phone uploads and
    # eval clips read the same.
    up = max(1, int(np.ceil(480 / max(w, h)))) if max(w, h) < 480 else 1
    ow, oh = w * up, h * up
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    fscale = max(ow, oh) / 1200.0
    th = max(1, round(2 * fscale * 2))          # limb line thickness
    rj = max(2, round(4 * fscale * 2))          # joint radius (wrists get +2)

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if up > 1:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_CUBIC)
        if fi < xy.shape[0]:
            pts = xy[fi] * up
            cf = conf[fi]

            def _seg(links, color):
                for a, b in links:
                    if cf[a] >= 0.3 and cf[b] >= 0.3:
                        cv2.line(frame, tuple(pts[a].astype(int)),
                                 tuple(pts[b].astype(int)), color, th, cv2.LINE_AA)

            _seg(LEFT_LINKS, LEFT_COLOR)
            _seg(RIGHT_LINKS, RIGHT_COLOR)
            _seg(CENTER_LINKS, CENTER_COLOR)
            for j in range(17):
                if cf[j] < 0.3:
                    continue
                c = float(np.clip(cf[j], 0, 1))
                col = (0, int(220 * c), int(220 * (1 - c)))  # green->red by conf
                r = rj + 2 if j in (L_WR, R_WR) else rj
                cv2.circle(frame, tuple(pts[j].astype(int)), r, col, -1, cv2.LINE_AA)
            for j, tag in ((L_WR, "L"), (R_WR, "R")):
                if cf[j] >= 0.3:
                    cv2.putText(frame, tag, (int(pts[j, 0]) + rj + 3, int(pts[j, 1]) - rj - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fscale,
                                LEFT_COLOR if j == L_WR else RIGHT_COLOR,
                                max(1, th), cv2.LINE_AA)
            hud = f"f{fi}  wristL {cf[L_WR]:.2f}  wristR {cf[R_WR]:.2f}"
            cv2.putText(frame, hud, (8, int(34 * fscale) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fscale, (255, 255, 255),
                        max(1, th), cv2.LINE_AA)
            if fi in swap_frames:
                cv2.putText(frame, "WRIST SWAP", (8, int(80 * fscale) + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4 * fscale, (0, 0, 255),
                            max(2, th + 1), cv2.LINE_AA)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()

    # h264 for browser playback (same fallback dance as write_2d_overlay)
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_mp4)],
                           check=True, timeout=300)
            tmp.unlink()
            return
        except Exception as e:
            print(f"  [pose_debug] ffmpeg failed: {e}; keeping mp4v")
    tmp.rename(out_mp4)


# ---------------------------------------------------------------------------
# Batch aggregation over the eval cache
# ---------------------------------------------------------------------------

def run_batch(n: int, backbone: str, lifter: str, out_json: Path | None,
              seed: int = 7) -> pd.DataFrame:
    root = Path(__file__).parent.parent / "Data" / "eval_runs"
    dir2d = root / backbone
    dir3d = root / f"{lifter}_from_{backbone}"
    stems = sorted(p.stem for p in dir2d.glob("*.parquet")
                   if (dir3d / p.name).exists())
    rng = np.random.default_rng(seed)
    if n and n < len(stems):
        stems = list(rng.choice(stems, size=n, replace=False))

    rows = []
    for i, stem in enumerate(stems):
        try:
            d = diagnose_clip(dir2d / f"{stem}.parquet", dir3d / f"{stem}.parquet")
        except Exception as e:
            print(f"  [batch] {stem}: {type(e).__name__}: {e}")
            continue
        rows.append({
            "clip": stem,
            "n_frames": d["n_frames"],
            "wrist_swaps_per_100f": d["swaps"]["wrist"]["swaps_per_100f"],
            "wrist_merged_frac": d["swaps"]["wrist"]["merged_frac"],
            "ankle_swaps_per_100f": d["swaps"]["ankle"]["swaps_per_100f"],
            "wrist_conf_min_side": min(d["conf"]["left_wrist_mean"],
                                       d["conf"]["right_wrist_mean"]),
            "jit_wrists_still": d["jitter_2d"]["wrists_still"],
            "jit_ankles_still": d["jitter_2d"]["ankles_still"],
            "jit_hips_still": d["jitter_2d"]["hips_still"],
            "ankle_dy_frac": d.get("ankle_3d", {}).get("ankle_dy_frac", np.nan),
            "ankle_dz_frac": d.get("ankle_3d", {}).get("ankle_dz_frac", np.nan),
            "higher_leg": d.get("ankle_3d", {}).get("higher_leg", ""),
        })
        if (i + 1) % 50 == 0:
            print(f"  [batch] {i + 1}/{len(stems)}")
    df = pd.DataFrame(rows)

    print(f"\n=== pose diagnostics: {len(df)} clips ({backbone} -> {lifter}) ===")
    print(f"wrist swaps/100f:  median {df.wrist_swaps_per_100f.median():.2f}  "
          f"mean {df.wrist_swaps_per_100f.mean():.2f}  "
          f"clips with >=1 swap: {(df.wrist_swaps_per_100f > 0).mean() * 100:.0f}%")
    print(f"wrists merged (unattributable) frac of frames: "
          f"median {df.wrist_merged_frac.median():.2f}")
    print(f"2D jitter on STILL frames (x100/torso): "
          f"wrists {df.jit_wrists_still.median():.3f}  "
          f"ankles {df.jit_ankles_still.median():.3f}  "
          f"hips {df.jit_hips_still.median():.3f}")
    dy = df.ankle_dy_frac.dropna()
    if len(dy):
        print(f"3D ankle height gap (L-R, frac of body height, +ve = RIGHT leg higher): "
              f"median {dy.median():+.4f}  mean {dy.mean():+.4f}")
        print(f"  right leg higher in {(dy > 0).mean() * 100:.0f}% of clips; "
              f"|gap| > 1% of height in {(dy.abs() > 0.01).mean() * 100:.0f}%")
        dz = df.ankle_dz_frac.dropna()
        print(f"  ankle depth gap (L-R): median {dz.median():+.4f}; "
              f"corr(dy, dz) = {df.ankle_dy_frac.corr(df.ankle_dz_frac):.2f}")
    if out_json:
        out_json.write_text(df.to_json(orient="records", indent=1), encoding="utf-8")
        print(f"[batch] wrote {out_json}")
    return df


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="?", help="Video path (single-clip mode with debug mp4)")
    ap.add_argument("--stem", help="Cached clip stem (e.g. 1292); uses Data/videos_160/<stem>.mp4 if present")
    ap.add_argument("--backbone", default="mediapipe_lite")
    ap.add_argument("--lifter", default="golfpose3d")
    ap.add_argument("--batch", type=int, default=0, help="Diagnose N random cached clips (0=all)")
    ap.add_argument("--out", default=None, help="Output path (json for batch, dir for single)")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    if args.batch or (not args.video and not args.stem):
        out = Path(args.out) if args.out else root / "Data" / "coaching" / "pose_diag_batch.json"
        run_batch(args.batch, args.backbone, args.lifter, out)
        return

    if args.stem:
        stem = args.stem
        video = root / "Data" / "videos_160" / f"{stem}.mp4"
    else:
        video = Path(args.video).resolve()
        stem = video.stem
    p2d = root / "Data" / "eval_runs" / args.backbone / f"{stem}.parquet"
    p3d = root / "Data" / "eval_runs" / f"{args.lifter}_from_{args.backbone}" / f"{stem}.parquet"
    if not p2d.exists():
        raise FileNotFoundError(f"no cached 2D parquet: {p2d} — run pipeline.py first")

    diag = diagnose_clip(p2d, p3d if p3d.exists() else None)
    out_dir = Path(args.out) if args.out else root / "Data" / "coaching" / "pose_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_path = out_dir / f"{stem}_pose_diag.json"
    diag_path.write_text(json.dumps(diag, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in diag.items() if k != "swaps"}, indent=1))
    print("wrist swaps:", diag["swaps"]["wrist"]["n_swaps"],
          "at frames", diag["swaps"]["wrist"]["swap_frames"][:20])
    print(f"[diag] wrote {diag_path}")
    if video.exists():
        mp4 = out_dir / f"{stem}_pose_debug.mp4"
        write_pose_debug_overlay(video, p2d, mp4, diag)
        print(f"[diag] wrote {mp4}")
    else:
        print(f"[diag] no source video at {video}; skipped debug overlay")


if __name__ == "__main__":
    main()
