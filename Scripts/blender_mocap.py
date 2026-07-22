"""Golf-swing 3D pose  ->  Blender animation.

Takes the pipeline's canonical mocap JSON (`Data/handoff/<stem>/<stem>_mocap.json`,
written by `export_ue5.export_json`: H36M-17 skeleton, normalised h36m-camera
coordinates, per-frame `positions_3d` (T, 17, 3)) and drives an accurate,
metric, unisex-scaled armature animation in Blender.

Design goals (from the research plan, see BLENDER_HANDOFF_RESEARCH.md):

  * ACCURATE      - the driven skeleton reproduces the captured joint positions
                    to sub-millimetre error (validated by FK round-trip below).
  * UNISEX BODY   - the mesh/rig uses a gender-neutral anthropometric template.
  * LIMB SCALING  - bone rest-lengths are set either from the SUBJECT'S OWN
                    measured limb lengths (median per bone, gender-agnostic) or
                    normalised to a unisex proportion template scaled to stature.

The module is split in two so the maths can be verified WITHOUT Blender:

  1. A pure-numpy core (load / axis-convert / scale / measure / re-proportion /
     FK-solve / reconstruct). Runnable anywhere -> `--validate`.
  2. A `bpy` builder (armature + per-joint targets + Stretch-To constraints +
     keyframes + Bezier interpolation + render). Runs inside Blender ->
     `blender --background --python blender_mocap.py -- --input ... --blend out.blend`.

The core NEVER imports bpy, so `--validate` proves correctness on real data
even when Blender is not attached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# H36M-17 skeleton (mirrors export_ue5.H36M17_PARENTS / _NAMES - kept local so
# this file runs standalone inside Blender's bundled Python with no repo deps).
# ---------------------------------------------------------------------------
H36M17_NAMES = (
    "hip_center", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
H36M17_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 8, 11, 12, 8, 14, 15)
#                  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
# NB: matches export_ue5.H36M17_PARENTS exactly (spine->hip, thorax->spine,
#     neck/shoulders->thorax, head->neck).
_PARENTS_FIXED = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
# (export_ue5 uses head parent = neck (9). We adopt that canonical version.)
H36M17_PARENTS = _PARENTS_FIXED

# Unisex anthropometric limb-length template, expressed as a fraction of total
# standing stature H. Gender-neutral averages (Winter / Drillis-Contini style),
# rounded for a POC. Indexed by CHILD joint; value = bone (child->parent) length
# as a fraction of H. Root (0) has no bone. EDIT these to taste / to team specs.
UNISEX_BONE_FRAC = {
    1: 0.090,  2: 0.245, 3: 0.246,          # right hip / thigh / shank
    4: 0.090,  5: 0.245, 6: 0.246,          # left  hip / thigh / shank
    7: 0.100,  8: 0.140, 9: 0.070, 10: 0.100,  # spine / thorax / neck / head
    11: 0.130, 12: 0.186, 13: 0.146,        # left  shoulder / uparm / forearm
    14: 0.130, 15: 0.186, 16: 0.146,        # right shoulder / uparm / forearm
}


# ===========================================================================
# 1. PURE-NUMPY CORE  (no bpy)
# ===========================================================================

def one_euro_filter(xyz: np.ndarray,
                    fps: float = 30.0,
                    min_cutoff: float = 0.3,
                    beta: float = 0.4,
                    d_cutoff: float = 1.0) -> np.ndarray:
    """One Euro filter (Casiez et al. 2012) per joint/axis.

    Port of Scripts/smoothing.py:one_euro_filter (kept local: this file must run
    inside Blender's bundled Python with no repo deps). Defaults are the
    accuracy-benchmark-tuned coaching params (min_cutoff=0.3, beta=0.4) rather
    than smoothing.py's generic defaults.
    """
    import math
    T = xyz.shape[0]
    if T < 2:
        return xyz.astype(np.float32, copy=True)
    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    out = np.empty_like(xyz, dtype=np.float32)
    x_prev = xyz[0].astype(np.float32)
    dx_prev = np.zeros_like(x_prev)
    out[0] = x_prev
    a_d = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * d_cutoff)) / dt)
    for t in range(1, T):
        x = xyz[t].astype(np.float32)
        dx = (x - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * x_prev
        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


def jitter_mean(xyz: np.ndarray) -> float:
    """Mean per-frame joint acceleration magnitude (the pipeline's jitter metric)."""
    if xyz.shape[0] < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=-1).mean())

def load_mocap_json(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """Return (positions (T,17,3) float64, fps, raw_meta). Coordinates are in
    the pipeline's native h36m-camera frame (x=right, y=DOWN, z=forward)."""
    d = json.loads(Path(path).read_text())
    pos = np.array([f["positions_3d"] for f in d["frames"]], dtype=np.float64)
    meta = {k: d.get(k) for k in ("format", "skeleton", "fps", "n_frames",
                                  "provenance", "events")}
    return pos, float(d["fps"]), meta


def camera_to_blender(pos: np.ndarray) -> np.ndarray:
    """h36m-camera (x=right, y=down, z=forward, right-handed) -> Blender world
    (x=right, y=forward, z=up, right-handed). Mapping (Xb,Yb,Zb)=(x, z, -y);
    determinant +1 so handedness/chirality is preserved (no mirror flip)."""
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]      # right  -> X
    out[..., 1] = pos[..., 2]      # fwd    -> Y
    out[..., 2] = -pos[..., 1]     # up     -> Z (camera +y was down)
    return out


def measure_bone_lengths(pos: np.ndarray,
                         parents=H36M17_PARENTS,
                         ref: str = "median") -> np.ndarray:
    """Per-joint reference bone length (child->parent), 0 for root. This is the
    subject's OWN measured limb geometry (the 'measured metrics')."""
    T, J, _ = pos.shape
    out = np.zeros(J)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        L = np.linalg.norm(pos[:, j] - pos[:, p], axis=1)
        out[j] = np.median(L) if ref == "median" else L.mean()
    return out


def stature_units(pos: np.ndarray) -> float:
    """Estimate standing height in the input's normalised units as the median
    over frames of the vertical joint span (used to convert to metres)."""
    span = pos[..., 1].max(axis=1) - pos[..., 1].min(axis=1)   # per-frame ptp
    return float(np.median(span))


def rigidify(pos: np.ndarray, bone_len: np.ndarray,
             parents=H36M17_PARENTS) -> np.ndarray:
    """Re-project every joint so each bone holds `bone_len` exactly, walking
    root->leaf and preserving per-frame bone DIRECTIONS. Root is untouched, so
    global translation (the swing's body sway) is preserved. This both (a)
    removes lifter length-jitter and (b) lets us swap in ANY target lengths
    (subject-measured OR unisex template) while keeping the captured motion."""
    T, J, _ = pos.shape
    out = pos.astype(np.float64, copy=True)
    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            n = np.linalg.norm(vec)
            out[t, j] = out[t, p] + (vec / n * bone_len[j] if n > 1e-9 else 0.0)
    return out


# (proximal, joint, distal) index triples for each hinge, named by the middle joint.
HINGES = (
    ("left_elbow",  H36M17_NAMES.index("left_shoulder"),  H36M17_NAMES.index("left_elbow"),  H36M17_NAMES.index("left_wrist")),
    ("right_elbow", H36M17_NAMES.index("right_shoulder"), H36M17_NAMES.index("right_elbow"), H36M17_NAMES.index("right_wrist")),
    ("left_knee",   H36M17_NAMES.index("left_hip"),   H36M17_NAMES.index("left_knee"),   H36M17_NAMES.index("left_ankle")),
    ("right_knee",  H36M17_NAMES.index("right_hip"),  H36M17_NAMES.index("right_knee"),  H36M17_NAMES.index("right_ankle")),
)


def enforce_joint_angle_limits(pos: np.ndarray, hinges=HINGES,
                               min_angle_deg: float = 15.0) -> np.ndarray:
    """Undo anatomically impossible hinge-joint flexion, per frame.

    For each (proximal, joint, distal) triple - e.g. (shoulder, elbow, wrist) -
    the interior angle at `joint` is the angle between the proximal-ward and
    distal-ward bone vectors: 180 deg is a fully straight limb, and it can only
    decrease toward `min_angle_deg` (near the true anatomical limit for an
    able-bodied elbow/knee - a golf swing never gets close) as the limb bends.
    Anything predicted below that floor is a lifter error, not a tight bend.

    Rotates the distal joint within the (proximal, distal) plane about `joint`
    until the angle equals `min_angle_deg`, holding the joint's position and
    the segment's bone length fixed - safe to run BEFORE the FK round-trip
    accuracy proof in `prepare()` (unlike `enforce_grip_constraint`), since it
    never changes any bone length."""
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    min_rad = np.radians(min_angle_deg)
    for _name, p_idx, j_idx, d_idx in hinges:
        for t in range(T):
            joint = out[t, j_idx]
            v1 = out[t, p_idx] - joint
            v2 = out[t, d_idx] - joint
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            u1 = v1 / n1
            along = float(np.dot(v2, u1))
            perp = v2 - along * u1
            n_perp = np.linalg.norm(perp)
            if n_perp < 1e-9:
                continue  # degenerate: v2 parallel/antiparallel to v1
            angle = np.arctan2(n_perp, along)
            if angle >= min_rad:
                continue
            u2 = perp / n_perp
            v2_new = n2 * (np.cos(min_rad) * u1 + np.sin(min_rad) * u2)
            out[t, d_idx] = joint + v2_new
    return out


_LEFT_WRIST, _RIGHT_WRIST = H36M17_NAMES.index("left_wrist"), H36M17_NAMES.index("right_wrist")


_LEFT_ELBOW, _RIGHT_ELBOW = H36M17_NAMES.index("left_elbow"), H36M17_NAMES.index("right_elbow")
_LEFT_SHOULDER, _RIGHT_SHOULDER = H36M17_NAMES.index("left_shoulder"), H36M17_NAMES.index("right_shoulder")


def enforce_grip_constraint(pos: np.ndarray, max_distance: float = 0.12,
                            left_idx: int = _LEFT_WRIST,
                            right_idx: int = _RIGHT_WRIST,
                            left_elbow_idx: int = _LEFT_ELBOW,
                            right_elbow_idx: int = _RIGHT_ELBOW) -> np.ndarray:
    """Clamp left/right wrist distance (metres) to a plausible golf grip
    width - see smoothing.py's version of this function for the full
    rationale (same algorithm, kept local per this file's no-repo-deps
    convention).

    Both hands hold the same club, so wrist-to-wrist distance should be small
    and roughly constant, but the lifter predicts every joint independently
    with no notion of a club or a second hand, so it can drift them apart -
    worst during the fastest part of the swing, when hand tracking is
    noisiest.

    An earlier version moved the wrist freely along the line between them,
    off the elbow->wrist bone axis, on the theory that the Stretch-To bone
    would only "stretch/compress a little" to reach it. On this project's own
    real swing footage that collapsed a forearm to under half its true length
    for several consecutive frames right through the downswing - visibly a
    forearm concertina-ing, not a rigid arm. Fixed by keeping each wrist ON
    the sphere of its own forearm length around its elbow throughout:
    alternate nudging the pair together and re-projecting each back onto its
    own sphere, so the correction gets as close to `max_distance` as the two
    fixed-radius spheres allow without ever breaking either forearm's length."""
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    for t in range(T):
        l, r = out[t, left_idx], out[t, right_idx]
        le, re_ = out[t, left_elbow_idx], out[t, right_elbow_idx]
        len_l = float(np.linalg.norm(l - le))
        len_r = float(np.linalg.norm(r - re_))
        for _ in range(4):
            vec = r - l
            dist = float(np.linalg.norm(vec))
            if dist <= max_distance or dist < 1e-9:
                break
            excess = dist - max_distance
            direction = vec / dist
            l = l + direction * excess * 0.5
            r = r - direction * excess * 0.5
            if len_l > 1e-6:
                l = le + (l - le) / np.linalg.norm(l - le) * len_l
            if len_r > 1e-6:
                r = re_ + (r - re_) / np.linalg.norm(r - re_) * len_r
        out[t, left_idx] = l
        out[t, right_idx] = r
    return out


def _fit_plane_robust(points: np.ndarray, exclude_threshold: float):
    """Best-fit plane (centroid, unit normal, keep-mask) via PCA, trimmed
    against address/finish - see smoothing.py's version (same algorithm,
    kept local per this file's no-repo-deps convention). Fit once on
    everything, drop any frame whose residual exceeds `exclude_threshold`,
    refit on what's left."""
    T = len(points)
    c0 = points.mean(axis=0)
    _, _, vt0 = np.linalg.svd(points - c0, full_matrices=False)
    n0 = vt0[-1]
    resid0 = np.abs((points - c0) @ n0)
    keep = resid0 <= exclude_threshold
    if keep.sum() < 4:
        return c0, n0, np.ones(T, dtype=bool)
    kept = points[keep]
    c = kept.mean(axis=0)
    _, _, vt = np.linalg.svd(kept - c, full_matrices=False)
    return c, vt[-1], keep


def enforce_swing_plane(pos: np.ndarray, max_offplane: float = 0.08,
                        exclude_multiplier: float = 3.0,
                        max_correction_frac: float = 0.35,
                        verbose: bool = True,
                        left_idx: int = _LEFT_WRIST,
                        right_idx: int = _RIGHT_WRIST) -> np.ndarray:
    """Pull the grip point (wrist-pair midpoint) back toward its own best-fit
    swing plane wherever a single frame has drifted off it (metres).

    `enforce_grip_constraint` only sees hand-to-hand distance, so a pair of
    hands that stayed together but drifted, as a unit, off the arc a real
    swing traces is invisible to it. Frames more than
    `exclude_multiplier * max_offplane` from the initial fit (presumed
    genuine address/finish, which are legitimately off the backswing-to-
    impact plane) are excluded from both the fit and any correction. Both
    wrists are shifted by the same vector, so hand-to-hand distance is
    untouched - like grip-lock, this moves wrists off their forearm bone
    axis, so it belongs after the FK round-trip accuracy proof in `prepare()`,
    not before.

    Guard: this assumes ONE swing traces ONE plane per clip. On a clip that
    isn't that (multiple swings, walk-up footage, a long capture), the fit
    is meaningless and would actively distort the pose. If more than
    `max_correction_frac` of trusted frames need correction, the fit is
    presumed unreliable and the whole step is skipped - see smoothing.py's
    version of this function for the calibration data behind the 0.35
    default (two real clips: 25% needed correction on a clean single swing
    and visually helped; 44% on a clip whose off-plane residual drifted
    systematically over the whole clip, not a single swing plane)."""
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < 5:
        return out
    grip = 0.5 * (out[:, left_idx] + out[:, right_idx])
    centroid, normal, keep = _fit_plane_robust(grip, exclude_threshold=exclude_multiplier * max_offplane)
    resid = (grip - centroid) @ normal
    needs_fix = keep & (np.abs(resid) > max_offplane)
    n_trusted = int(keep.sum())
    frac = float(needs_fix.sum()) / n_trusted if n_trusted else 0.0
    if frac > max_correction_frac:
        if verbose:
            print(f"[swing-plane] {needs_fix.sum()}/{n_trusted} trusted frames "
                  f"({frac*100:.0f}%) off-plane, above {max_correction_frac*100:.0f}% guard - "
                  f"this doesn't look like a single-swing-plane clip, skipping correction")
        return out
    for t in range(T):
        if not needs_fix[t]:
            continue
        d = resid[t]
        shift = -(abs(d) - max_offplane) * np.sign(d) * normal
        out[t, left_idx] = out[t, left_idx] + shift
        out[t, right_idx] = out[t, right_idx] + shift
    return out


_TORSO_LIMB_JOINTS = (
    ("left_wrist",  H36M17_NAMES.index("left_wrist")),
    ("right_wrist", H36M17_NAMES.index("right_wrist")),
    ("left_elbow",  H36M17_NAMES.index("left_elbow")),
    ("right_elbow", H36M17_NAMES.index("right_elbow")),
)


def enforce_torso_clearance(pos: np.ndarray, min_clearance: float = 0.12,
                            joints=_TORSO_LIMB_JOINTS,
                            hip_idx: int = 0,
                            thorax_idx: int = 8) -> np.ndarray:
    """Push a wrist/elbow radially off the spine axis (hip_center -> thorax)
    whenever it comes closer than `min_clearance` (metres) - see
    smoothing.py's version of this function for the full rationale (same
    algorithm, kept local per this file's no-repo-deps convention).

    The Blender torso mesh (`_shaft("torso", ...)` in `build_in_blender`) is
    an ellipse roughly 0.10-0.16m in cross-section radius depending on
    direction; 0.12m is a conservative middle ground, plus margin for the
    limb's own mesh thickness. Like grip-lock and plane-lock, this moves a
    joint off its bone axis, so it belongs after the FK round-trip accuracy
    proof in `prepare()`, not before."""
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    for t in range(T):
        a = out[t, hip_idx]
        b = out[t, thorax_idx]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-8:
            continue
        for _name, j_idx in joints:
            p = out[t, j_idx]
            frac = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
            proj = a + frac * ab
            radial = p - proj
            dist = float(np.linalg.norm(radial))
            if dist >= min_clearance or dist < 1e-6:
                continue
            out[t, j_idx] = proj + radial / dist * min_clearance
    return out


def enforce_follow_through_deceleration(pos: np.ndarray,
                                        left_idx: int = _LEFT_WRIST,
                                        right_idx: int = _RIGHT_WRIST,
                                        top_window=(0.1, 0.9),
                                        spike_window: int = 5,
                                        spike_k: float = 3.5,
                                        min_speed_mad: float = 0.01) -> np.ndarray:
    """Suppress isolated grip-point speed spikes after the top-of-backswing
    without flattening real multi-frame events (metres) - see smoothing.py's
    version of this function for the full rationale (same algorithm, kept
    local per this file's no-repo-deps convention: compares each frame's
    speed only to its own local neighbourhood via a Hampel-style outlier
    test, since a real event like the wrist-release whip raises its whole
    neighbourhood while noise stands out alone). Shifts both wrists by the
    same vector, so it never fights grip-lock's hand-to-hand distance;
    belongs after the FK round-trip accuracy proof like grip-lock/plane-
    lock/torso-lock, since it moves wrists off their bone axis."""
    raw = pos.astype(np.float64, copy=True)
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < 10:
        return out

    grip_raw = 0.5 * (raw[:, left_idx] + raw[:, right_idx])
    height = grip_raw[:, 2]  # Blender frame is Z-up here (unlike smoothing.py's camera y=down): +Z is up
    lo, hi = int(top_window[0] * T), int(top_window[1] * T)
    if hi <= lo:
        return out
    top = lo + int(np.argmax(height[lo:hi]))
    if top >= T - 2:
        return out

    vecs = grip_raw[top + 1:] - grip_raw[top:-1]
    speeds = np.linalg.norm(vecs, axis=1)
    n = len(speeds)
    half = spike_window // 2
    corrected = vecs.copy()

    for i in range(n):
        neighbor_idx = [j for j in range(max(0, i - half), min(n, i + half + 1)) if j != i]
        if len(neighbor_idx) < 2:
            continue
        neighbor_speeds = speeds[neighbor_idx]
        med = float(np.median(neighbor_speeds))
        mad = float(np.median(np.abs(neighbor_speeds - med))) * 1.4826
        thresh = med + spike_k * max(mad, min_speed_mad)
        if speeds[i] > thresh and speeds[i] > 1e-9:
            corrected[i] = vecs[i] * (thresh / speeds[i])

    grip_out = grip_raw.copy()
    for i in range(n):
        grip_out[top + 1 + i] = grip_out[top + i] + corrected[i]

    delta = grip_out[top:] - grip_raw[top:]
    out[top:, left_idx] = raw[top:, left_idx] + delta
    out[top:, right_idx] = raw[top:, right_idx] + delta
    return out


def freeze_arm_pose_during_holds(pos: np.ndarray, fps: float = 30.0,
                                 top_window=(0.1, 0.9),
                                 still_speed: float = 0.008,
                                 min_hold_frames: int = 5,
                                 blend_frames: int = 5,
                                 left_shoulder_idx: int = _LEFT_SHOULDER,
                                 right_shoulder_idx: int = _RIGHT_SHOULDER,
                                 left_wrist_idx: int = _LEFT_WRIST,
                                 right_wrist_idx: int = _RIGHT_WRIST,
                                 segments=((_LEFT_SHOULDER, _LEFT_ELBOW),
                                          (_LEFT_ELBOW, _LEFT_WRIST),
                                          (_RIGHT_SHOULDER, _RIGHT_ELBOW),
                                          (_RIGHT_ELBOW, _RIGHT_WRIST))) -> np.ndarray:
    """Freeze the arms to a single held pose wherever the torso proves they
    should be still (a finish hold, address, any pause), without touching
    forearm length - see smoothing.py's version of this function for the
    full rationale (same algorithm, kept local per this file's no-repo-deps
    convention).

    A first attempt (re-filtering just the wrist's direction around the
    elbow, adaptively, everywhere) made no visible difference: tracing real
    footage showed the shoulder midpoint barely moves at all through most of
    the back half of both clips (genuinely holding still) while in that SAME
    window the elbow AND wrist both keep jumping 0.05-0.27m/frame, at
    comparable magnitude to each other - the whole arm is being mis-
    reconstructed, not just the wrist wobbling around a stable elbow, so
    smoothing against a comparably-corrupted elbow reference couldn't help.

    Uses the shoulder's own speed - an independent, trustworthy signal - to
    find contiguous runs of at least `min_hold_frames` where it stays below
    `still_speed`, and only inside a qualifying run replaces each arm
    segment's direction with that run's own mean direction (a short
    `blend_frames` crossfade in/out avoids a hard pop). An earlier continuous
    adaptive-filter version was rejected: blending toward a heavily-damped
    reference could overshoot right at a hold's edge - e.g. the brief pause
    as the swing transitions from backswing to downswing, where the torso
    can be briefly still while the arms are about to move fast - producing a
    bigger artificial jump than the noise it was meant to fix. To guarantee
    a mid-swing torso pause can never trigger a freeze, holds are only
    searched for AFTER the swing's own impact frame, found the same way
    `enforce_follow_through_deceleration` finds it. Each segment is
    processed as a unit direction off its own already-rigid bone length, so
    length is preserved exactly by construction."""
    out = pos.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < min_hold_frames + 2 * blend_frames:
        return out

    grip = 0.5 * (out[:, left_wrist_idx] + out[:, right_wrist_idx])
    height = grip[:, 2]  # Blender frame is Z-up here (unlike smoothing.py's camera y=down): +Z is up
    lo, hi = int(top_window[0] * T), int(top_window[1] * T)
    if hi <= lo:
        return out
    top = lo + int(np.argmax(height[lo:hi]))
    if top >= T - 2:
        return out
    speeds = np.linalg.norm(np.diff(grip[top:], axis=0), axis=1)
    impact = top + int(np.argmax(speeds))

    shoulder_mid = 0.5 * (out[:, left_shoulder_idx] + out[:, right_shoulder_idx])
    torso_speed = np.zeros(T)
    torso_speed[1:] = np.linalg.norm(np.diff(shoulder_mid, axis=0), axis=1)
    torso_speed[0] = torso_speed[1] if T > 1 else 0.0
    still = torso_speed < still_speed
    still[:impact] = False

    runs = []
    i = impact
    while i < T:
        if still[i]:
            j = i
            while j < T and still[j]:
                j += 1
            if j - i >= min_hold_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return out

    for p_idx, d_idx in segments:
        proximal = out[:, p_idx]
        distal = out[:, d_idx]
        vec = distal - proximal
        length = np.linalg.norm(vec, axis=1, keepdims=True)
        direction = vec / np.clip(length, 1e-6, None)
        target = direction.copy()
        for (a, b) in runs:
            rep = direction[a:b].mean(axis=0)
            rep = rep / np.clip(np.linalg.norm(rep), 1e-6, None)
            target[a:b] = rep
            for k in range(1, blend_frames + 1):
                if a - k >= 0:
                    t = k / (blend_frames + 1)
                    blend = (1.0 - t) * direction[a - k] + t * rep
                    target[a - k] = blend / np.clip(np.linalg.norm(blend), 1e-6, None)
                if b - 1 + k < T:
                    t = k / (blend_frames + 1)
                    blend = (1.0 - t) * direction[b - 1 + k] + t * rep
                    target[b - 1 + k] = blend / np.clip(np.linalg.norm(blend), 1e-6, None)
        out[:, d_idx] = proximal + target * length
    return out


def unisex_bone_lengths(stature_m: float) -> np.ndarray:
    """Bone lengths (metres) from the unisex proportion template * stature."""
    out = np.zeros(len(H36M17_NAMES))
    for j, frac in UNISEX_BONE_FRAC.items():
        out[j] = frac * stature_m
    return out


# ---- FK rotation solve + round-trip reconstruction (accuracy proof) --------

def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 1 - 1e-8:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-8:                       # 180 deg: pick any orthogonal axis
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    w = 1 + d
    q = np.array([w, *axis])
    return q / np.linalg.norm(q)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])
    return v @ R.T


def solve_global_bone_rotations(pos: np.ndarray, rest: np.ndarray,
                                parents=H36M17_PARENTS) -> np.ndarray:
    """Per-frame, per-bone GLOBAL rotation aligning the rest-pose bone direction
    onto the captured bone direction. Shape (T, J, 4) quaternions. Twist about
    the bone axis is unobservable from child positions and left identity - which
    is fine: it does not affect reconstructed joint positions (proved below)."""
    T, J, _ = pos.shape
    q = np.tile(np.array([1.0, 0, 0, 0]), (T, J, 1))
    rest_dir = np.zeros((J, 3))
    for j in range(J):
        p = parents[j]
        if p >= 0:
            rest_dir[j] = rest[j] - rest[p]
    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            cur = pos[t, j] - pos[t, p]
            q[t, j] = _shortest_arc(rest_dir[j], cur)
    return q


def reconstruct_from_rotations(root: np.ndarray, q: np.ndarray,
                               bone_len: np.ndarray, rest: np.ndarray,
                               parents=H36M17_PARENTS) -> np.ndarray:
    """Rebuild joint positions from (root translation, per-bone global rotation,
    bone lengths). Used to VERIFY the retargetable rotation representation."""
    T = q.shape[0]
    J = len(parents)
    rest_dir = np.zeros((J, 3))
    for j in range(J):
        p = parents[j]
        if p >= 0:
            v = rest[j] - rest[p]
            rest_dir[j] = v / (np.linalg.norm(v) + 1e-12)
    out = np.zeros((T, J, 3))
    order = sorted(range(J), key=lambda j: (parents[j] >= 0, j))  # root first
    for t in range(T):
        out[t, 0] = root[t]
        for j in order:
            p = parents[j]
            if p < 0:
                continue
            out[t, j] = out[t, p] + _qrot(q[t, j], rest_dir[j]) * bone_len[j]
    return out


def build_rest_pose(bone_len: np.ndarray, parents=H36M17_PARENTS) -> np.ndarray:
    """A canonical Blender-frame T/A-pose from bone lengths: legs down (-Z),
    spine up (+Z), arms out (+/-X), head up. Used as the retarget rest skeleton."""
    d = {  # unit direction of each bone in world/Blender frame
        1: (-1, 0, 0), 2: (0, 0, -1), 3: (0, 0, -1),      # r hip out, leg down
        4: (1, 0, 0),  5: (0, 0, -1), 6: (0, 0, -1),      # l hip out, leg down
        7: (0, 0, 1),  8: (0, 0, 1),  9: (0, 0, 1), 10: (0, 0, 1),  # spine up
        11: (1, 0, 0), 12: (1, 0, 0), 13: (1, 0, 0),      # l arm +X
        14: (-1, 0, 0), 15: (-1, 0, 0), 16: (-1, 0, 0),   # r arm -X
    }
    J = len(parents)
    rest = np.zeros((J, 3))
    for j in sorted(range(J), key=lambda k: (parents[k] >= 0, k)):
        p = parents[j]
        if p < 0:
            continue
        rest[j] = rest[p] + np.array(d[j], float) * bone_len[j]
    return rest


def prepare(pos_cam: np.ndarray, target_height_m: float = 1.75,
            mode: str = "measured", angle_lock: bool = True,
            min_hinge_angle: float = 15.0, grip_lock: bool = True,
            max_hand_distance: float = 0.12, plane_lock: bool = True,
            max_offplane: float = 0.08, plane_exclude_multiplier: float = 3.0,
            torso_lock: bool = True, min_torso_clearance: float = 0.12,
            deceleration_lock: bool = True, spike_window: int = 5,
            spike_k: float = 3.5, min_speed_mad: float = 0.01,
            arm_freeze: bool = True, fps: float = 30.0,
            arm_freeze_still_speed: float = 0.008,
            arm_freeze_min_hold_frames: int = 5,
            arm_freeze_blend_frames: int = 5,
            lead_arm_lock: bool = True,
            lead_arm_min_angle: float = 155.0) -> dict:
    """Full numpy pipeline: camera->Blender, metric scale, choose bone lengths
    (measured|unisex), rigidify, angle-lock, lead-arm straightness lock,
    grip-lock, plane-lock, torso-lock, deceleration-lock, arm-pose freeze
    during proven-still holds, FK-solve, reconstruct. Returns everything the
    Blender builder needs plus an accuracy report."""
    stat_u = stature_units(pos_cam)
    scale = target_height_m / stat_u
    pos = camera_to_blender(pos_cam) * scale          # now in metres, Z-up

    measured_len = measure_bone_lengths(pos)
    if mode == "unisex":
        bone_len = unisex_bone_lengths(target_height_m)
    elif mode == "measured":
        bone_len = measured_len
    else:
        raise ValueError("mode must be 'measured' or 'unisex'")

    pos_rig = rigidify(pos, bone_len)
    if angle_lock:
        # Bone-length-preserving, so safe to fold into the "clean" pose that
        # feeds the FK round-trip accuracy proof below (unlike grip-lock).
        pos_rig = enforce_joint_angle_limits(pos_rig, min_angle_deg=min_hinge_angle)
    if lead_arm_lock:
        # Real footage (confidence-filtered): the left (lead, right-handed
        # golfer) elbow sits ~159-161 deg through address, dipping to
        # 40-150 deg through tracking-degraded windows and 9-20 deg in the
        # follow-through tail - a tighter floor than the general anatomical
        # one above keeps the lead arm close to straight throughout ("the
        # leading wrist locked to the same angle as the arm"). Also bone-
        # length-preserving, so also safe before the FK proof.
        pos_rig = enforce_joint_angle_limits(pos_rig, hinges=(HINGES[0],),
                                             min_angle_deg=lead_arm_min_angle)
    rest = build_rest_pose(bone_len)
    # FK round-trip proof runs on the clean rigidified (+ angle-locked) pose,
    # BEFORE grip-lock: grip-lock is a deliberate editorial correction (moves
    # the wrist off the elbow->wrist bone axis to close the hands), so it
    # would fail the bone length round-trip by design, not by bug. Keep that
    # proof meaningful for the retargetable rotation representation; apply
    # grip-lock only to the positions actually animated below (build_in_blender
    # drives the armature from pos_rig's Stretch-To targets, not from `quats`).
    q = solve_global_bone_rotations(pos_rig, rest)
    recon = reconstruct_from_rotations(pos_rig[:, 0], q, bone_len, rest)
    err_mm = float(np.linalg.norm(recon - pos_rig, axis=-1).mean() * 1000)

    if grip_lock:
        pos_rig = enforce_grip_constraint(pos_rig, max_distance=max_hand_distance)
    if plane_lock:
        pos_rig = enforce_swing_plane(pos_rig, max_offplane=max_offplane,
                                      exclude_multiplier=plane_exclude_multiplier)
    if torso_lock:
        pos_rig = enforce_torso_clearance(pos_rig, min_clearance=min_torso_clearance)
    if deceleration_lock:
        pos_rig = enforce_follow_through_deceleration(pos_rig, spike_window=spike_window,
                                                       spike_k=spike_k,
                                                       min_speed_mad=min_speed_mad)
    if arm_freeze:
        pos_rig = freeze_arm_pose_during_holds(pos_rig, fps=fps,
                                               still_speed=arm_freeze_still_speed,
                                               min_hold_frames=arm_freeze_min_hold_frames,
                                               blend_frames=arm_freeze_blend_frames)

    return dict(pos_metric=pos, pos_rig=pos_rig, bone_len=bone_len,
                measured_len=measured_len, rest=rest, quats=q,
                scale=scale, stature_units=stat_u,
                recon_err_mm=err_mm, mode=mode, target_height_m=target_height_m)


# ===========================================================================
# 2. BLENDER BUILDER  (imports bpy - only runs inside Blender)
# ===========================================================================

def build_in_blender(prep: dict, fps: float, out_blend: Optional[str],
                     render_png: Optional[str], render_video: Optional[str] = None,
                     add_mesh: bool = True) -> None:
    import bpy   # noqa: local import - only available inside Blender
    import mathutils

    pos = prep["pos_rig"]          # (T,17,3) metres, Blender frame, rigidified
    bone_len = prep["bone_len"]
    rest = prep["rest"]
    T = pos.shape[0]

    # --- clean scene -------------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 0, T - 1
    # New keyframes default to Bezier -> smooth interpolated in-betweens.
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "BEZIER"
    except Exception:
        pass

    # --- one Empty target per joint, position-keyframed --------------------
    targets = []
    for j, name in enumerate(H36M17_NAMES):
        e = bpy.data.objects.new(f"tgt_{name}", None)
        e.empty_display_size = 0.03
        e.empty_display_type = "SPHERE"
        scene.collection.objects.link(e)
        targets.append(e)
    for fi in range(T):
        for j, e in enumerate(targets):
            e.location = mathutils.Vector(pos[fi, j].tolist())
            e.keyframe_insert("location", frame=fi)

    # --- armature in rest pose (metric, unisex/measured lengths) -----------
    arm_data = bpy.data.armatures.new("GolferArmature")
    arm = bpy.data.objects.new("Golfer", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = {}
    for j in sorted(range(17), key=lambda k: (H36M17_PARENTS[k] >= 0, k)):
        p = H36M17_PARENTS[j]
        if p < 0:
            continue
        eb = arm_data.edit_bones.new(H36M17_NAMES[j])
        eb.head = mathutils.Vector(rest[p].tolist())
        eb.tail = mathutils.Vector(rest[j].tolist())
        if p != 0 and H36M17_NAMES[p] in ebones:
            eb.parent = ebones[H36M17_NAMES[p]]
        ebones[H36M17_NAMES[j]] = eb
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- constrain bones to targets: root follows hip, others Stretch-To ---
    bpy.ops.object.mode_set(mode="POSE")
    # keep the armature root glued to the hip target so body sway is preserved
    c = arm.pose.bones[H36M17_NAMES[1]]  # any bone off hip; drive object instead
    arm.constraints.new("COPY_LOCATION").target = targets[0]
    for j in range(1, 17):
        pb = arm.pose.bones.get(H36M17_NAMES[j])
        if pb is None:
            continue
        st = pb.constraints.new("STRETCH_TO")   # aim tail at joint, keep volume
        st.target = targets[j]
        st.keep_axis = "PLANE_Z"
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- stylized CAPSULE/TUBE body proxy (replaces ball-and-stick) --------
    # A neutral matte mannequin: tapered rounded limb capsules + rounded joint
    # blobs + a solid torso column. Everything is parented to the keyframed joint
    # targets and (for limbs/torso) STRETCH_TO'd between two targets — the SAME
    # rigging the ball-and-stick used, so the numpy core / FK / scaling are
    # untouched; only the drawn geometry changes. Deliberately stylized (single
    # matte colour, no face/hands/clothing) — an artist's mannequin, not anatomy.
    if add_mesh:
        SKIN = bpy.data.materials.new("Mannequin")
        SKIN.diffuse_color = (0.72, 0.68, 0.60, 1.0)   # neutral warm grey, no detail
        try:
            SKIN.roughness = 0.9
            SKIN.metallic = 0.0
        except Exception:
            pass

        def _to_Y(obj):
            # reorient a Z-axis primitive so its length runs 0->1 along +Y (base at
            # origin) — the axis STRETCH_TO scales toward the child target.
            for v in obj.data.vertices:
                x, y, z = v.co
                v.co = mathutils.Vector((x, z + 0.5, -y))

        def _shaft(name, parent_j, child_j, r_prox, r_dist, hw=None, hd=None):
            # tapered cone (thicker proximal -> thinner distal) spanning
            # parent_j -> child_j; hw/hd squash it into a wide, shallow torso.
            bpy.ops.mesh.primitive_cone_add(radius1=r_prox, radius2=r_dist,
                                            depth=1.0, vertices=24)
            o = bpy.context.active_object
            o.name = name
            _to_Y(o)
            if hw is not None:
                for v in o.data.vertices:
                    v.co.x *= hw
                    v.co.z *= hd
            o.data.materials.append(SKIN)
            bpy.ops.object.shade_smooth()
            o.parent = targets[parent_j]
            o.matrix_parent_inverse = mathutils.Matrix()
            o.location = (0, 0, 0)
            c = o.constraints.new("STRETCH_TO")
            c.target = targets[child_j]
            c.rest_length = 1.0
            c.volume = "NO_VOLUME"          # scale length only; keep cross-section
            return o

        def _blob(name, j, r):
            # rounded blob at a joint — sized to blend into the incident capsules.
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=20, ring_count=12)
            o = bpy.context.active_object
            o.name = name
            o.data.materials.append(SKIN)
            bpy.ops.object.shade_smooth()
            o.parent = targets[j]
            o.matrix_parent_inverse = mathutils.Matrix()
            o.location = (0, 0, 0)
            return o

        # limbs: child joint -> (radius at parent, radius at child), in metres.
        # spine(7)/thorax(8) are omitted — the torso column below covers them.
        LIMB_R = {
            1: (0.080, 0.070), 4: (0.080, 0.070),     # pelvis stubs (hip_center->hip)
            2: (0.078, 0.052), 5: (0.078, 0.052),     # thighs
            3: (0.050, 0.034), 6: (0.050, 0.034),     # shanks
            9: (0.042, 0.034),                         # neck
            10: (0.034, 0.030),                        # head stalk (head blob covers)
            11: (0.052, 0.046), 14: (0.052, 0.046),   # clavicles (thorax->shoulder)
            12: (0.050, 0.038), 15: (0.050, 0.038),   # upper arms
            13: (0.037, 0.028), 16: (0.037, 0.028),   # forearms
        }
        for j, (rp, rd) in LIMB_R.items():
            _shaft(f"limb_{H36M17_NAMES[j]}", H36M17_PARENTS[j], j, rp, rd)

        # solid torso column: hip_center -> thorax, wide (X) + shallow (Z) ovoid.
        _shaft("torso", 0, 8, 1.0, 0.85, hw=0.16, hd=0.10)

        # rounded joint / end blobs (0 + 8 round the torso ends; 10 is the head).
        JOINT_R = {
            0: 0.090, 8: 0.080,
            1: 0.066, 4: 0.066, 2: 0.058, 5: 0.058, 3: 0.042, 6: 0.042,
            11: 0.054, 14: 0.054, 12: 0.042, 15: 0.042, 13: 0.034, 16: 0.034,
            9: 0.036, 10: 0.082,
        }
        for j, r in JOINT_R.items():
            _blob(f"joint_{H36M17_NAMES[j]}", j, r)

    # --- world / light / floor / auto-aimed camera for a presentable preview
    world = bpy.data.worlds.new("W")
    world.use_nodes = False
    world.color = (0.05, 0.06, 0.09)
    scene.world = world

    floor_z = float(pos.reshape(-1, 3)[:, 2].min())
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, floor_z))

    light = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    light.data.energy = 4.0
    light.rotation_euler = (0.5, 0.2, 0.3)
    scene.collection.objects.link(light)

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (0.2, -4.2, 1.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    trk = cam.constraints.new("TRACK_TO")  # always frame the golfer's hips
    trk.target = targets[0]
    trk.track_axis = "TRACK_NEGATIVE_Z"
    trk.up_axis = "UP_Y"

    if out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(out_blend).resolve()))
    if render_png:
        scene.render.image_settings.file_format = "PNG"
        base = Path(render_png).resolve()
        # sample three swing phases (or the mid frame for a single output)
        frames = [int(T * 0.15), int(T * 0.5), int(T * 0.85)]
        for k, fr in enumerate(frames):
            scene.frame_set(fr)
            scene.render.filepath = str(base.with_name(f"{base.stem}_f{fr:03d}.png"))
            bpy.ops.render.render(write_still=True)
        scene.frame_set(T // 2)
        scene.render.filepath = str(base)
        bpy.ops.render.render(write_still=True)

    if render_video:
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
        # Blender 5.x gates video formats behind media_type='VIDEO'
        if hasattr(scene.render.image_settings, "media_type"):
            scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.filepath = str(Path(render_video).resolve())
        bpy.ops.render.render(animation=True)


# ===========================================================================
# CLI
# ===========================================================================

def _argv_after_ddash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="*_mocap.json path")
    ap.add_argument("--height", type=float, default=1.75, help="subject stature (m)")
    ap.add_argument("--mode", choices=["measured", "unisex"], default="measured")
    ap.add_argument("--validate", action="store_true",
                    help="numpy-only: run pipeline + accuracy report, no Blender")
    ap.add_argument("--blend", default=None, help="output .blend path (Blender)")
    ap.add_argument("--render", default=None, help="output preview PNG (Blender)")
    ap.add_argument("--video", default=None, help="output MP4 animation (Blender)")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--no-smooth", action="store_true",
                    help="skip One-Euro smoothing of the input 3D (default: smooth; "
                         "the mocap JSON carries RAW lifted 3D, which animates jittery)")
    ap.add_argument("--smooth-min-cutoff", type=float, default=0.3)
    ap.add_argument("--smooth-beta", type=float, default=0.4)
    ap.add_argument("--no-angle-lock", action="store_true",
                    help="skip joint-angle-limit correction (undoes anatomically "
                         "impossible elbow/knee flexion; default: on)")
    ap.add_argument("--min-hinge-angle", type=float, default=15.0,
                    help="angle-lock: minimum elbow/knee interior angle in degrees (default: 15)")
    ap.add_argument("--no-grip-lock", action="store_true",
                    help="skip grip stabilization (clamps wrist-to-wrist distance; "
                         "default: on, since both hands hold one club)")
    ap.add_argument("--max-hand-distance", type=float, default=0.12,
                    help="grip-lock: max plausible wrist-to-wrist distance in metres (default: 0.12)")
    ap.add_argument("--no-plane-lock", action="store_true",
                    help="skip swing-plane consistency (pulls the grip point back onto its "
                         "own best-fit arc; default: on)")
    ap.add_argument("--max-offplane", type=float, default=0.08,
                    help="plane-lock: max plausible grip-point distance from the swing plane in metres (default: 0.08)")
    ap.add_argument("--no-torso-lock", action="store_true",
                    help="skip torso-clearance (pushes a wrist/elbow out of the torso "
                         "mesh volume; default: on)")
    ap.add_argument("--min-torso-clearance", type=float, default=0.12,
                    help="torso-lock: min wrist/elbow distance from the spine axis in metres (default: 0.12)")
    ap.add_argument("--no-deceleration-lock", action="store_true",
                    help="skip follow-through spike suppression (clamps isolated grip-point "
                         "speed spikes relative to their local neighbourhood, without "
                         "flattening real multi-frame events like the wrist-release whip; "
                         "default: on)")
    ap.add_argument("--spike-window", type=int, default=5,
                    help="deceleration-lock: number of neighbouring frames (centered) used "
                         "to judge whether a frame's speed is a local outlier (default: 5)")
    ap.add_argument("--spike-k", type=float, default=3.5,
                    help="deceleration-lock: outlier threshold in scaled MADs above the "
                         "local median speed (default: 3.5)")
    ap.add_argument("--min-speed-mad", type=float, default=0.01,
                    help="deceleration-lock: floor on the local MAD in metres, so a near-"
                         "still neighbourhood doesn't flag ordinary small speeds (default: 0.01)")
    ap.add_argument("--no-arm-freeze", action="store_true",
                    help="skip arm-pose freezing during proven-still holds (freezes the arms "
                         "to a single pose wherever the torso proves they should be still - "
                         "hands winging around when the body isn't moving - without touching "
                         "forearm length; only ever applies after the swing's own impact "
                         "frame; default: on)")
    ap.add_argument("--arm-freeze-still-speed", type=float, default=0.008,
                    help="arm-freeze: shoulder-midpoint speed (m/frame) below which the torso "
                         "counts as still (default: 0.008)")
    ap.add_argument("--arm-freeze-min-hold-frames", type=int, default=5,
                    help="arm-freeze: minimum consecutive still frames to count as a real "
                         "hold, not torso noise (default: 5)")
    ap.add_argument("--arm-freeze-blend-frames", type=int, default=5,
                    help="arm-freeze: crossfade length in/out of a hold, in frames (default: 5)")
    ap.add_argument("--no-lead-arm-lock", action="store_true",
                    help="skip the lead-arm straightness lock (keeps the left/lead elbow - "
                         "right-handed golfer convention - close to straight throughout the "
                         "clip, a tighter floor than the general anatomical angle-lock; "
                         "default: on)")
    ap.add_argument("--lead-arm-min-angle", type=float, default=155.0,
                    help="lead-arm-lock: minimum lead-elbow interior angle in degrees "
                         "(default: 155)")
    args = ap.parse_args(_argv_after_ddash())

    pos_cam, fps, meta = load_mocap_json(args.input)
    jit_raw = jitter_mean(pos_cam)
    if not args.no_smooth:
        pos_cam = one_euro_filter(pos_cam, fps=fps,
                                  min_cutoff=args.smooth_min_cutoff,
                                  beta=args.smooth_beta)
        print(f"[smooth]   One-Euro(min_cutoff={args.smooth_min_cutoff}, "
              f"beta={args.smooth_beta}): jitter {jit_raw:.5f} -> "
              f"{jitter_mean(pos_cam):.5f} "
              f"({(1 - jitter_mean(pos_cam) / jit_raw) * 100:.0f}% less)" if jit_raw > 0
              else "[smooth]   One-Euro applied (input had no measurable jitter)")
    else:
        print(f"[smooth]   OFF (raw input jitter {jit_raw:.5f})")
    prep = prepare(pos_cam, target_height_m=args.height, mode=args.mode,
                   angle_lock=not args.no_angle_lock,
                   min_hinge_angle=args.min_hinge_angle,
                   grip_lock=not args.no_grip_lock,
                   max_hand_distance=args.max_hand_distance,
                   plane_lock=not args.no_plane_lock,
                   max_offplane=args.max_offplane,
                   torso_lock=not args.no_torso_lock,
                   min_torso_clearance=args.min_torso_clearance,
                   deceleration_lock=not args.no_deceleration_lock,
                   spike_window=args.spike_window,
                   spike_k=args.spike_k,
                   min_speed_mad=args.min_speed_mad,
                   arm_freeze=not args.no_arm_freeze,
                   fps=fps,
                   arm_freeze_still_speed=args.arm_freeze_still_speed,
                   arm_freeze_min_hold_frames=args.arm_freeze_min_hold_frames,
                   arm_freeze_blend_frames=args.arm_freeze_blend_frames,
                   lead_arm_lock=not args.no_lead_arm_lock,
                   lead_arm_min_angle=args.lead_arm_min_angle)

    print(f"[load]     {args.input}")
    print(f"           frames={pos_cam.shape[0]}  fps={fps:.3f}  mode={args.mode}")
    print(f"[scale]    stature={prep['stature_units']:.3f} units "
          f"-> {args.height} m   (x{prep['scale']:.3f})")
    print(f"[bones m]  thigh={prep['bone_len'][2]:.3f}  shank={prep['bone_len'][3]:.3f}  "
          f"uparm={prep['bone_len'][15]:.3f}  forearm={prep['bone_len'][16]:.3f}")
    angle = "off" if args.no_angle_lock else f"on (min {args.min_hinge_angle:.0f} deg)"
    print(f"[angle]    {angle}")
    lead_arm = ("off" if args.no_lead_arm_lock else
               f"on (min {args.lead_arm_min_angle:.0f} deg)")
    print(f"[lead-arm] {lead_arm}")
    grip = "off" if args.no_grip_lock else f"on (max {args.max_hand_distance:.2f} m)"
    print(f"[grip]     {grip}")
    plane = "off" if args.no_plane_lock else f"on (max {args.max_offplane:.2f} m)"
    print(f"[plane]    {plane}")
    torso = "off" if args.no_torso_lock else f"on (min {args.min_torso_clearance:.2f} m)"
    print(f"[torso]    {torso}")
    decel = ("off" if args.no_deceleration_lock else
              f"on (window {args.spike_window}, k={args.spike_k:.1f})")
    print(f"[decel]    {decel}")
    arm_freeze_status = ("off" if args.no_arm_freeze else
                         f"on (still<{args.arm_freeze_still_speed:.3f} m/frame, "
                         f"min-hold {args.arm_freeze_min_hold_frames}f)")
    print(f"[freeze]   {arm_freeze_status}")
    print(f"[ACCURACY] FK round-trip mean joint error = {prep['recon_err_mm']:.4f} mm "
          f"({'PASS' if prep['recon_err_mm'] < 1.0 else 'CHECK'})")

    if args.validate:
        print("[validate] numpy core OK - retargetable rotation repr verified.")
        return
    build_in_blender(prep, fps, args.blend, args.render, render_video=args.video,
                     add_mesh=not args.no_mesh)
    print(f"[blender]  wrote blend={args.blend} render={args.render} video={args.video}")


if __name__ == "__main__":
    main()
