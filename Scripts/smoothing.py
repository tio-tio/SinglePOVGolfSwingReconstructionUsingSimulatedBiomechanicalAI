"""Temporal smoothing for 3D pose sequences before Unreal Engine handoff.

Operates on (T, 17, 3) H36M-17 arrays (the lifter output loaded by
export_ue5.load_3d_parquet_as_h36m). Insert one call to `smooth_sequence`
between lifting and export to de-jitter the CSV/JSON/BVH and the HTML preview
at once.

Golf constraint: impact is 2-3 frames of very fast wrist/club motion, so the
default filter is the One Euro filter, which backs off its smoothing as joint
speed rises and therefore preserves the impact snap. Savitzky-Golay is offered
as an offline alternative that preserves derivatives well on slower clips.

Pipeline order inside `smooth_sequence`:
  1. interpolate_gaps  - linear-fill short low-confidence runs (needs conf)
  2. temporal filter   - one_euro_filter | savgol_smooth
  3. enforce_bone_lengths (optional) - rigidify the skeleton frame-to-frame
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_ue5 import H36M17_PARENTS


# ---------------------------------------------------------------------------
# 1. Gap handling
# ---------------------------------------------------------------------------

def interpolate_gaps(xyz: np.ndarray,
                     conf: np.ndarray | None,
                     min_conf: float = 0.3,
                     max_gap: int = 5) -> np.ndarray:
    """Linearly interpolate joint positions across short low-confidence runs.

    A "gap" is a maximal run of consecutive frames where conf < min_conf for a
    given joint. Gaps no longer than `max_gap` and bounded by valid frames on
    both sides are filled by linear interpolation; longer gaps and gaps at the
    sequence ends are left untouched (no reliable anchor to interpolate from).

    If `conf` is None this is a no-op (the caller has no per-joint confidence).
    """
    if conf is None:
        return xyz.astype(np.float32, copy=True)

    out = xyz.astype(np.float32, copy=True)
    T, J, _ = out.shape
    for j in range(J):
        valid = conf[:, j] >= min_conf
        if valid.all() or valid.sum() < 2:
            continue
        t = 0
        while t < T:
            if valid[t]:
                t += 1
                continue
            start = t
            while t < T and not valid[t]:
                t += 1
            end = t  # first valid frame after the gap (or T)
            gap_len = end - start
            if start == 0 or end == T or gap_len > max_gap:
                continue  # unbounded or too long -> leave as-is
            lo, hi = start - 1, end
            for k in range(start, end):
                w = (k - lo) / (hi - lo)
                out[k, j] = (1.0 - w) * out[lo, j] + w * out[hi, j]
    return out


# ---------------------------------------------------------------------------
# 2a. One Euro filter (adaptive low-pass, default)
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_filter(xyz: np.ndarray,
                    fps: float = 30.0,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    d_cutoff: float = 1.0) -> np.ndarray:
    """One Euro filter (Casiez et al. 2012) applied per joint and axis.

    Adaptive low-pass: at low speeds the cutoff approaches `min_cutoff`
    (heavy smoothing, kills tremor); as speed rises the cutoff grows by
    `beta * |velocity|` (light smoothing, low lag). Tuning:
      - lower min_cutoff -> smoother slow phases (more lag)
      - higher beta      -> snappier fast phases (less lag at impact)
    """
    T = xyz.shape[0]
    if T < 2:
        return xyz.astype(np.float32, copy=True)

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    out = np.empty_like(xyz, dtype=np.float32)
    x_prev = xyz[0].astype(np.float32)
    dx_prev = np.zeros_like(x_prev)
    out[0] = x_prev
    a_d = _alpha(d_cutoff, dt)
    for t in range(1, T):
        x = xyz[t].astype(np.float32)
        dx = (x - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        # vectorized alpha per joint/axis
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * x_prev
        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


# ---------------------------------------------------------------------------
# 2b. Savitzky-Golay (offline alternative)
# ---------------------------------------------------------------------------

def savgol_smooth(xyz: np.ndarray,
                  window: int = 7,
                  polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing along the time axis.

    Window is forced odd and clamped to the sequence length; polyorder is
    clamped below the window. Offline (uses future frames) and preserves
    peaks/derivatives better than a moving average.
    """
    from scipy.signal import savgol_filter

    T = xyz.shape[0]
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win > T:
        win = T if T % 2 == 1 else T - 1
    if win < 3:
        return xyz.astype(np.float32, copy=True)
    poly = min(int(polyorder), win - 1)
    return savgol_filter(xyz, window_length=win, polyorder=poly,
                         axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Bone-length stabilization
# ---------------------------------------------------------------------------

def enforce_bone_lengths(xyz: np.ndarray,
                         parents: tuple[int, ...] = H36M17_PARENTS,
                         ref: str = "median") -> np.ndarray:
    """Rigidify the skeleton: re-project each joint so every bone keeps a
    constant length across all frames.

    For each bone (child -> parent) a reference length is taken as the
    median (or mean) of its per-frame lengths. Then, walking the skeleton
    root->leaf, each child is placed along its current bone direction at the
    reference length from the already-corrected parent. The root joint is
    untouched, so global translation is preserved.

    Relies on H36M17_PARENTS having parent index < child index (true for the
    H36M-17 ordering), so a single forward pass processes parents first.
    """
    T, J, _ = xyz.shape
    out = xyz.astype(np.float32, copy=True)

    # reference length per joint (0 for root)
    ref_len = np.zeros(J, dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        lengths = np.linalg.norm(xyz[:, j] - xyz[:, p], axis=1)
        ref_len[j] = np.median(lengths) if ref == "median" else lengths.mean()

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                # degenerate frame: keep parent position
                out[t, j] = out[t, p]
            else:
                out[t, j] = out[t, p] + vec / norm * ref_len[j]
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def smooth_sequence(xyz: np.ndarray,
                    conf: np.ndarray | None = None,
                    method: str = "oneeuro",
                    fps: float = 30.0,
                    bone_lock: bool = True,
                    min_conf: float = 0.3,
                    max_gap: int = 5,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    window: int = 7,
                    polyorder: int = 2) -> np.ndarray:
    """Smooth a (T, 17, 3) H36M-17 sequence.

    method: "none" | "oneeuro" | "savgol". Gap interpolation runs first when
    `conf` is provided; bone-length stabilization runs last when `bone_lock`.
    """
    if xyz.ndim != 3 or xyz.shape[1:] != (17, 3):
        raise ValueError(f"expected (T,17,3) H36M array, got {xyz.shape}")

    out = interpolate_gaps(xyz, conf, min_conf=min_conf, max_gap=max_gap)

    if method == "none":
        pass
    elif method == "oneeuro":
        out = one_euro_filter(out, fps=fps, min_cutoff=min_cutoff, beta=beta)
    elif method == "savgol":
        out = savgol_smooth(out, window=window, polyorder=polyorder)
    else:
        raise ValueError(f"unknown smoothing method '{method}'")

    if bone_lock:
        out = enforce_bone_lengths(out)
    return out
