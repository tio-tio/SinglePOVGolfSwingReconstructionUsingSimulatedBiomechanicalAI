"""
Evaluation framework for the golf swing pose estimation comparison.

Centralizes:
  - canonical 17-keypoint COCO schema (so models with different output
    skeletons can be compared head-to-head)
  - reference-free quality metrics (bone-length stability, jitter, foot
    planting, joint-angle plausibility, detection rate)
  - reference-based metrics (PCE on the 8 GolfDB swing events)
  - operational metrics (runtime per frame, MB on disk)
  - BaseAdapter abstract class that every model wrapper inherits from

The contract: every adapter.predict(video_path) -> pandas.DataFrame with
columns [frame, kp_idx, kp_name, x, y, conf]. (x,y) are pixel coordinates
in the *input video* frame (top-left origin, y down). For models that
return normalized coords (MediaPipe), the adapter denormalizes before
returning. For models that return a different skeleton (MediaPipe-33,
SMPL-24), the adapter maps into the canonical COCO-17 before returning.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Canonical 17-keypoint COCO schema
# -------------------------------------------------------------------------

COCO17_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
COCO17_IDX = {n: i for i, n in enumerate(COCO17_NAMES)}

# MediaPipe-33 -> COCO-17 index map
MP33_TO_COCO17: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Bones used for limb-length stability. Each is a (keypoint, keypoint) pair.
BONES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),  # clavicle line
    ("left_hip", "right_hip"),            # pelvis line
    ("left_shoulder", "left_hip"),        # torso left
    ("right_shoulder", "right_hip"),      # torso right
)

# H36M-17 keypoint names (the ordering used by Human3.6M, which is the
# canonical input format for MotionBERT, GolfPose, MixSTE, PoseFormer, etc).
# COCO and H36M share label semantics but differ in (a) ordering and
# (b) which joints exist (H36M has Hip-center/Spine/Thorax/Neck which COCO
# lacks; COCO has eye/ear/nose face landmarks H36M lacks).
H36M17_NAMES: tuple[str, ...] = (
    "hip_center",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)
H36M17_IDX = {n: i for i, n in enumerate(H36M17_NAMES)}


def coco17_to_h36m17(xy_coco: np.ndarray) -> np.ndarray:
    """Convert a (..., 17, 2|3) COCO-17 array to H36M-17 ordering.

    Joints that don't exist in COCO (hip_center, spine, thorax, neck) are
    derived as midpoints. This conversion is standard practice for
    feeding COCO-trained 2D detectors (MediaPipe, YOLO, MoveNet, ViTPose)
    into H36M-trained 3D lifters (MotionBERT, MixSTE, GolfPose, PoseFormer).
    """
    L_HIP, R_HIP = COCO17_IDX["left_hip"], COCO17_IDX["right_hip"]
    L_SH,  R_SH  = COCO17_IDX["left_shoulder"], COCO17_IDX["right_shoulder"]
    NOSE         = COCO17_IDX["nose"]

    out = np.zeros(xy_coco.shape[:-2] + (17, xy_coco.shape[-1]), dtype=xy_coco.dtype)
    hip_center = (xy_coco[..., L_HIP, :] + xy_coco[..., R_HIP, :]) / 2.0
    thorax     = (xy_coco[..., L_SH,  :] + xy_coco[..., R_SH,  :]) / 2.0
    spine      = (hip_center + thorax) / 2.0
    head       = xy_coco[..., NOSE, :]
    neck       = (thorax + head) / 2.0

    out[..., 0, :]  = hip_center
    out[..., 1, :]  = xy_coco[..., R_HIP, :]
    out[..., 2, :]  = xy_coco[..., COCO17_IDX["right_knee"],  :]
    out[..., 3, :]  = xy_coco[..., COCO17_IDX["right_ankle"], :]
    out[..., 4, :]  = xy_coco[..., L_HIP, :]
    out[..., 5, :]  = xy_coco[..., COCO17_IDX["left_knee"],   :]
    out[..., 6, :]  = xy_coco[..., COCO17_IDX["left_ankle"],  :]
    out[..., 7, :]  = spine
    out[..., 8, :]  = thorax
    out[..., 9, :]  = neck
    out[..., 10, :] = head
    out[..., 11, :] = xy_coco[..., L_SH, :]
    out[..., 12, :] = xy_coco[..., COCO17_IDX["left_elbow"], :]
    out[..., 13, :] = xy_coco[..., COCO17_IDX["left_wrist"], :]
    out[..., 14, :] = xy_coco[..., R_SH, :]
    out[..., 15, :] = xy_coco[..., COCO17_IDX["right_elbow"], :]
    out[..., 16, :] = xy_coco[..., COCO17_IDX["right_wrist"], :]
    return out


def h36m17_to_coco17_subset(xyz_h36m: np.ndarray) -> np.ndarray:
    """Inverse mapping for the joints that exist in both. Used when a 3D
    lifter returns H36M-17 output and we want to score it with our
    canonical COCO-17 metrics (bone CV, jitter, joint angles). The 5 face
    keypoints have no H36M source, so we zero them — metrics that depend
    on confidence will skip them."""
    H = H36M17_IDX
    out = np.zeros(xyz_h36m.shape[:-2] + (17, xyz_h36m.shape[-1]), dtype=xyz_h36m.dtype)
    out[..., COCO17_IDX["nose"],          :] = xyz_h36m[..., H["head"],           :]
    # eyes / ears: synthesize at head, will get zeroed by callers via conf
    out[..., COCO17_IDX["left_eye"],      :] = xyz_h36m[..., H["head"],           :]
    out[..., COCO17_IDX["right_eye"],     :] = xyz_h36m[..., H["head"],           :]
    out[..., COCO17_IDX["left_ear"],      :] = xyz_h36m[..., H["head"],           :]
    out[..., COCO17_IDX["right_ear"],     :] = xyz_h36m[..., H["head"],           :]
    out[..., COCO17_IDX["left_shoulder"], :] = xyz_h36m[..., H["left_shoulder"],  :]
    out[..., COCO17_IDX["right_shoulder"],:] = xyz_h36m[..., H["right_shoulder"], :]
    out[..., COCO17_IDX["left_elbow"],    :] = xyz_h36m[..., H["left_elbow"],     :]
    out[..., COCO17_IDX["right_elbow"],   :] = xyz_h36m[..., H["right_elbow"],    :]
    out[..., COCO17_IDX["left_wrist"],    :] = xyz_h36m[..., H["left_wrist"],     :]
    out[..., COCO17_IDX["right_wrist"],   :] = xyz_h36m[..., H["right_wrist"],    :]
    out[..., COCO17_IDX["left_hip"],      :] = xyz_h36m[..., H["left_hip"],       :]
    out[..., COCO17_IDX["right_hip"],     :] = xyz_h36m[..., H["right_hip"],      :]
    out[..., COCO17_IDX["left_knee"],     :] = xyz_h36m[..., H["left_knee"],      :]
    out[..., COCO17_IDX["right_knee"],    :] = xyz_h36m[..., H["right_knee"],     :]
    out[..., COCO17_IDX["left_ankle"],    :] = xyz_h36m[..., H["left_ankle"],     :]
    out[..., COCO17_IDX["right_ankle"],   :] = xyz_h36m[..., H["right_ankle"],    :]
    return out


# 8 GolfDB swing events (in order). Index 0 = address, 7 = finish.
SWING_EVENTS: tuple[str, ...] = (
    "address",
    "toe_up",
    "mid_backswing",
    "top",
    "mid_downswing",
    "impact",
    "mid_follow_through",
    "finish",
)


# -------------------------------------------------------------------------
# Adapter base class
# -------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """One adapter run on one video."""
    landmarks: pd.DataFrame  # [frame, kp_idx, kp_name, x, y, conf]
    seconds_per_frame: float
    n_frames: int
    n_frames_detected: int  # frames where any keypoint had conf > 0


class BaseAdapter(ABC):
    """All model wrappers inherit from this. `predict()` must return canonical
    COCO-17 landmarks per frame, with (x,y) in input pixel coordinates."""

    name: str = "base"
    family: str = "2d"            # "2d" or "3d"
    native_skeleton: str = "coco17"

    @abstractmethod
    def predict(self, video_path: str | Path) -> InferenceResult:
        ...

    def predict_and_cache(self, video_path: str | Path, cache_dir: str | Path,
                          overwrite: bool = False) -> InferenceResult:
        cache_dir = Path(cache_dir) / self.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{Path(video_path).stem}.parquet"
        meta_path = cache_dir / f"{Path(video_path).stem}.meta.json"

        if cache_path.exists() and not overwrite:
            import json
            df = pd.read_parquet(cache_path)
            with open(meta_path) as f:
                meta = json.load(f)
            return InferenceResult(
                landmarks=df,
                seconds_per_frame=meta["seconds_per_frame"],
                n_frames=meta["n_frames"],
                n_frames_detected=meta["n_frames_detected"],
            )

        result = self.predict(video_path)
        result.landmarks.to_parquet(cache_path, index=False)
        import json
        with open(meta_path, "w") as f:
            json.dump({
                "model": self.name,
                "seconds_per_frame": result.seconds_per_frame,
                "n_frames": result.n_frames,
                "n_frames_detected": result.n_frames_detected,
            }, f)
        return result


# -------------------------------------------------------------------------
# Helpers: video metadata + frame iteration
# -------------------------------------------------------------------------

def video_info(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    info = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(video_path: str | Path, rgb: bool = True):
    """Yield (frame_idx, frame_array). frame is HxWxC, dtype uint8."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, frame
            idx += 1
    finally:
        cap.release()


# -------------------------------------------------------------------------
# DataFrame -> tensor helpers
# -------------------------------------------------------------------------

def landmarks_to_array(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pivot a landmarks DataFrame into two (n_frames, 17, 2|1) arrays:
    xy in pixels and conf in [0,1]. Missing keypoints are NaN."""
    n_kp = len(COCO17_NAMES)
    frames = sorted(df["frame"].unique())
    frame_to_idx = {f: i for i, f in enumerate(frames)}
    n_frames = len(frames)

    xy = np.full((n_frames, n_kp, 2), np.nan, dtype=np.float32)
    conf = np.full((n_frames, n_kp), 0.0, dtype=np.float32)

    for row in df.itertuples(index=False):
        fi = frame_to_idx[row.frame]
        ki = row.kp_idx
        xy[fi, ki, 0] = row.x
        xy[fi, ki, 1] = row.y
        conf[fi, ki] = row.conf
    return xy, conf


# -------------------------------------------------------------------------
# Reference-free quality metrics
# -------------------------------------------------------------------------

def bone_lengths(xy: np.ndarray) -> dict[str, np.ndarray]:
    """For each bone, return a (n_frames,) array of euclidean length."""
    out = {}
    for a, b in BONES:
        ai, bi = COCO17_IDX[a], COCO17_IDX[b]
        d = np.linalg.norm(xy[:, ai] - xy[:, bi], axis=1)
        out[f"{a}__{b}"] = d
    return out


def bone_length_cv(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> dict[str, float]:
    """Coefficient of variation (std/mean) of each bone length, computed
    only over frames where both endpoints have confidence >= min_conf.
    Lower CV = more physically stable model. A perfect model has CV=0."""
    n_frames, n_kp, _ = xy.shape
    out = {}
    for a, b in BONES:
        ai, bi = COCO17_IDX[a], COCO17_IDX[b]
        valid = (conf[:, ai] >= min_conf) & (conf[:, bi] >= min_conf)
        if valid.sum() < 5:
            out[f"{a}__{b}"] = np.nan
            continue
        d = np.linalg.norm(xy[valid, ai] - xy[valid, bi], axis=1)
        mean = d.mean()
        std = d.std()
        out[f"{a}__{b}"] = float(std / mean) if mean > 0 else np.nan
    return out


def jitter(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> dict[str, float]:
    """Mean frame-to-frame acceleration magnitude per keypoint (in pixels).
    Lower = smoother (under the prior that real human motion is smooth at
    the sample rate of typical phone video)."""
    n_frames, n_kp, _ = xy.shape
    out = {}
    for ki, kn in enumerate(COCO17_NAMES):
        # second derivative
        v = np.diff(xy[:, ki], axis=0)         # (n_frames-1, 2)
        a = np.diff(v, axis=0)                 # (n_frames-2, 2)
        a_mag = np.linalg.norm(a, axis=1)      # (n_frames-2,)
        valid = (conf[:-2, ki] >= min_conf) & (conf[1:-1, ki] >= min_conf) & (conf[2:, ki] >= min_conf)
        if valid.sum() < 3:
            out[kn] = np.nan
            continue
        out[kn] = float(np.nanmean(a_mag[valid]))
    return out


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at vertex `b` of the triangle a-b-c, in degrees.
    a, b, c are (n, 2) arrays."""
    ba = a - b
    bc = c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def joint_angle_plausibility(xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.3) -> dict[str, float]:
    """Fraction of frames where elbow / knee joint angles fall outside the
    plausible [0, 180] range OR are within [0, 5] / [175, 180] (which
    indicates near-degenerate joint geometry typical of failure modes)."""
    out = {}
    for side in ("left", "right"):
        # elbow: shoulder -> elbow -> wrist
        sh = xy[:, COCO17_IDX[f"{side}_shoulder"]]
        el = xy[:, COCO17_IDX[f"{side}_elbow"]]
        wr = xy[:, COCO17_IDX[f"{side}_wrist"]]
        ang = joint_angle(sh, el, wr)
        sh_c = conf[:, COCO17_IDX[f"{side}_shoulder"]]
        el_c = conf[:, COCO17_IDX[f"{side}_elbow"]]
        wr_c = conf[:, COCO17_IDX[f"{side}_wrist"]]
        valid = (sh_c >= min_conf) & (el_c >= min_conf) & (wr_c >= min_conf)
        if valid.sum() < 5:
            out[f"{side}_elbow_implausible_frac"] = np.nan
        else:
            bad = (ang < 5) | (ang > 175) | np.isnan(ang)
            out[f"{side}_elbow_implausible_frac"] = float((bad & valid).sum() / valid.sum())

        # knee: hip -> knee -> ankle
        hp = xy[:, COCO17_IDX[f"{side}_hip"]]
        kn = xy[:, COCO17_IDX[f"{side}_knee"]]
        an = xy[:, COCO17_IDX[f"{side}_ankle"]]
        ang = joint_angle(hp, kn, an)
        hp_c = conf[:, COCO17_IDX[f"{side}_hip"]]
        kn_c = conf[:, COCO17_IDX[f"{side}_knee"]]
        an_c = conf[:, COCO17_IDX[f"{side}_ankle"]]
        valid = (hp_c >= min_conf) & (kn_c >= min_conf) & (an_c >= min_conf)
        if valid.sum() < 5:
            out[f"{side}_knee_implausible_frac"] = np.nan
        else:
            bad = (ang < 5) | (ang > 175) | np.isnan(ang)
            out[f"{side}_knee_implausible_frac"] = float((bad & valid).sum() / valid.sum())
    return out


def foot_planting(xy: np.ndarray, conf: np.ndarray,
                   address_frame: int, impact_frame: int,
                   min_conf: float = 0.3) -> dict[str, float]:
    """Std dev of ankle position between address and impact frames.
    In a real golf swing the lead foot stays planted; large std => sliding."""
    out = {}
    a, i = max(0, address_frame), min(xy.shape[0] - 1, impact_frame)
    if i <= a + 1:
        return {"left_ankle_planting_std_px": np.nan, "right_ankle_planting_std_px": np.nan}
    for side in ("left", "right"):
        ki = COCO17_IDX[f"{side}_ankle"]
        seg = xy[a:i + 1, ki]
        cseg = conf[a:i + 1, ki]
        mask = cseg >= min_conf
        if mask.sum() < 3:
            out[f"{side}_ankle_planting_std_px"] = np.nan
            continue
        out[f"{side}_ankle_planting_std_px"] = float(np.linalg.norm(seg[mask].std(axis=0)))
    return out


def detection_rate(conf: np.ndarray, min_conf: float = 0.3) -> float:
    """Fraction of (frame, keypoint) cells with conf >= min_conf."""
    return float((conf >= min_conf).mean())


# -------------------------------------------------------------------------
# Reference-based metric: PCE on GolfDB swing events
# -------------------------------------------------------------------------

def derive_events_from_landmarks(xy: np.ndarray, conf: np.ndarray) -> dict[str, int]:
    """Heuristic event detector. Same logic applied to every model, so any
    difference in PCE reflects landmark quality, not detector cleverness.

    Heuristics (right-handed assumption; left-handed flips left/right wrist):
      - address: frame 0 (clip starts at address by construction)
      - top:     frame where right-wrist Y is minimum (highest point)
      - impact:  frame where right-wrist Y is maximum after top
      - finish:  last frame of clip
      - toe_up:           halfway between address and top
      - mid_backswing:    3/4 between address and top
      - mid_downswing:    halfway between top and impact
      - mid_follow_through: halfway between impact and finish

    A real detector would be smarter, but for *relative* benchmarking what
    matters is that the same algorithm runs on every model's output.
    """
    n_frames = xy.shape[0]
    rw_y = xy[:, COCO17_IDX["right_wrist"], 1].copy()
    # Mask out low-confidence frames with the median so they don't win argmin/argmax
    rw_c = conf[:, COCO17_IDX["right_wrist"]]
    if (rw_c >= 0.3).sum() < 5:
        # fall back to nose Y if right wrist mostly missing
        rw_y = xy[:, COCO17_IDX["nose"], 1].copy()
    rw_y[np.isnan(rw_y)] = np.nanmedian(rw_y)

    address = 0
    finish = n_frames - 1
    top = int(np.argmin(rw_y))
    # impact = lowest wrist (i.e. ball-strike level) AFTER top
    post_top = rw_y.copy()
    post_top[:top + 1] = -np.inf
    impact = int(np.argmax(post_top))
    if impact <= top:
        impact = min(n_frames - 1, top + max(1, (finish - top) // 3))

    return {
        "address": address,
        "toe_up": int(address + (top - address) * 0.5),
        "mid_backswing": int(address + (top - address) * 0.75),
        "top": top,
        "mid_downswing": int(top + (impact - top) * 0.5),
        "impact": impact,
        "mid_follow_through": int(impact + (finish - impact) * 0.5),
        "finish": finish,
    }


def pce(predicted_events: dict[str, int], golfdb_events: Sequence[int],
        n_frames: int, tolerances: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """Percentage of Correct Events at each tolerance. The GolfDB official
    metric: an event is "correct" if predicted frame is within ±tol of
    labeled frame.

    GolfDB events come from preprocess_videos.py: the clip starts at
    events[0]. The 8 swing events stored in `golfdb_events` are assumed
    to already be re-indexed relative to the clip (frame 0 = events[0]).
    See notebook §2 for the offsetting logic."""
    if len(golfdb_events) != 8:
        return {f"pce_at_{t}": np.nan for t in tolerances}
    out = {}
    for tol in tolerances:
        hits = 0
        for ev_name, true_frame in zip(SWING_EVENTS, golfdb_events):
            pred = predicted_events[ev_name]
            if abs(pred - true_frame) <= tol:
                hits += 1
        out[f"pce_at_{tol}"] = hits / len(SWING_EVENTS)
    return out


# -------------------------------------------------------------------------
# Top-level metric pipeline
# -------------------------------------------------------------------------

def compute_all_metrics(result: InferenceResult,
                         golfdb_events_relative: Sequence[int] | None = None
                         ) -> dict[str, float]:
    """Run every metric on one (model, clip) result. Returns a flat dict
    suitable for stacking into a leaderboard DataFrame."""
    xy, conf = landmarks_to_array(result.landmarks)
    n_frames = xy.shape[0]

    metrics: dict[str, float] = {}

    # operational
    metrics["seconds_per_frame"] = result.seconds_per_frame
    metrics["fps_inference"] = (1.0 / result.seconds_per_frame) if result.seconds_per_frame > 0 else np.nan
    metrics["n_frames"] = n_frames
    metrics["detection_rate"] = detection_rate(conf)

    # bone-length CV (we report mean across bones + worst bone)
    bcv = bone_length_cv(xy, conf)
    valid_bcv = {k: v for k, v in bcv.items() if not np.isnan(v)}
    metrics["bone_cv_mean"] = float(np.mean(list(valid_bcv.values()))) if valid_bcv else np.nan
    metrics["bone_cv_max"] = float(np.max(list(valid_bcv.values()))) if valid_bcv else np.nan

    # jitter (mean across keypoints + worst keypoint)
    j = jitter(xy, conf)
    valid_j = {k: v for k, v in j.items() if not np.isnan(v)}
    metrics["jitter_mean_px"] = float(np.mean(list(valid_j.values()))) if valid_j else np.nan
    metrics["jitter_max_px"] = float(np.max(list(valid_j.values()))) if valid_j else np.nan

    # joint plausibility (worst across joints)
    jap = joint_angle_plausibility(xy, conf)
    valid_jap = {k: v for k, v in jap.items() if not np.isnan(v)}
    metrics["implausible_frac_mean"] = float(np.mean(list(valid_jap.values()))) if valid_jap else np.nan
    metrics["implausible_frac_max"] = float(np.max(list(valid_jap.values()))) if valid_jap else np.nan

    # PCE (only if ground truth events are provided)
    if golfdb_events_relative is not None:
        events_pred = derive_events_from_landmarks(xy, conf)
        # foot planting: use address->impact from ground truth
        address_idx = int(golfdb_events_relative[0])
        impact_idx = int(golfdb_events_relative[5])
        fp = foot_planting(xy, conf, address_idx, impact_idx)
        metrics.update(fp)
        metrics.update(pce(events_pred, golfdb_events_relative, n_frames))
    else:
        # Without ground truth, planting uses 0..n/2 as a rough proxy
        fp = foot_planting(xy, conf, 0, n_frames // 2)
        metrics.update(fp)
        metrics["pce_at_1"] = np.nan
        metrics["pce_at_3"] = np.nan
        metrics["pce_at_5"] = np.nan

    return metrics
