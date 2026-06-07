"""Quantify smoothing: jitter (mean frame-to-frame acceleration) on raw vs
smoothed 3D output, per clip and per joint.

Point it at one or more cached 3D landmark parquet files (the lifter outputs
under Data/eval_runs/<lifter>/<clip>.parquet) and it prints the jitter
reduction so you can tune --smooth-min-cutoff / --smooth-beta against numbers.

Usage:
  python eval_smoothing.py Data/eval_runs/motionbert_lite_from_mediapipe_heavy/*.parquet
  python eval_smoothing.py <parquet> --method savgol --window 9 --no-bone-lock
  python eval_smoothing.py <parquet> --fps 120 --per-joint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_ue5 import H36M17_NAMES, load_3d_parquet_as_h36m
from smoothing import smooth_sequence


def per_joint_jitter(xyz: np.ndarray) -> np.ndarray:
    """Mean acceleration magnitude per joint -> (17,)."""
    if xyz.shape[0] < 3:
        return np.full(xyz.shape[1], np.nan, dtype=np.float32)
    a = np.diff(xyz, n=2, axis=0)          # (T-2, 17, 3)
    return np.linalg.norm(a, axis=2).mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquets", nargs="+", help="3D landmark parquet file(s)")
    p.add_argument("--method", choices=["oneeuro", "savgol"], default="oneeuro")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--no-bone-lock", action="store_true")
    p.add_argument("--per-joint", action="store_true",
                   help="Also print per-joint jitter for each clip")
    args = p.parse_args()

    raw_means, smo_means = [], []
    for pq in args.parquets:
        pq = Path(pq)
        xyz = load_3d_parquet_as_h36m(pq)
        smoothed = smooth_sequence(
            xyz, conf=None, method=args.method, fps=args.fps,
            bone_lock=not args.no_bone_lock,
            min_cutoff=args.min_cutoff, beta=args.beta, window=args.window,
        )
        raw_pj = per_joint_jitter(xyz)
        smo_pj = per_joint_jitter(smoothed)
        raw_m, smo_m = float(raw_pj.mean()), float(smo_pj.mean())
        raw_means.append(raw_m)
        smo_means.append(smo_m)
        pct = (1.0 - smo_m / raw_m) * 100.0 if raw_m else 0.0
        print(f"{pq.name:<45} raw {raw_m:.5f}  ->  smoothed {smo_m:.5f}  ({pct:+.1f}%)")
        if args.per_joint:
            for ji, name in enumerate(H36M17_NAMES):
                jp = (1.0 - smo_pj[ji] / raw_pj[ji]) * 100.0 if raw_pj[ji] else 0.0
                print(f"    {name:<14} {raw_pj[ji]:.5f} -> {smo_pj[ji]:.5f} ({jp:+.1f}%)")

    if len(args.parquets) > 1:
        rm, sm = np.mean(raw_means), np.mean(smo_means)
        pct = (1.0 - sm / rm) * 100.0 if rm else 0.0
        print(f"\n{'OVERALL':<45} raw {rm:.5f}  ->  smoothed {sm:.5f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
