"""Aggregate cached per-clip inference into a metrics dataframe.

Reads every cached parquet under Data/eval_runs/<model>/<id>.parquet,
joins to the GolfDB labels for ground-truth swing events, computes the
full metric battery, and writes Data/all_metrics.parquet.

Run after run_benchmark.py has populated the cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import InferenceResult, compute_all_metrics

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
CACHE_DIR    = DATA_DIR / "eval_runs"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


def events_clip_local(events_row) -> np.ndarray:
    ev = np.asarray(events_row)
    return ev[1:9] - ev[0]


def main():
    df = pd.read_pickle(GOLFDB_PKL)
    df_indexed = df.set_index("id")

    if not CACHE_DIR.exists():
        print("[compute_metrics] no cache dir, run benchmark first")
        return

    models = sorted([p.name for p in CACHE_DIR.iterdir() if p.is_dir()])
    print(f"[compute_metrics] models with cache: {models}")

    all_rows = []
    for mname in models:
        mdir = CACHE_DIR / mname
        clip_files = sorted(mdir.glob("*.parquet"))
        if not clip_files:
            print(f"  {mname}: 0 cached clips, skip")
            continue
        print(f"  {mname}: {len(clip_files)} cached clips")

        for cache_path in tqdm(clip_files, desc=mname, ncols=80, leave=False):
            cid = int(cache_path.stem)
            meta_path = mdir / f"{cid}.meta.json"
            if not meta_path.exists():
                continue
            try:
                lm_df = pd.read_parquet(cache_path)
                with open(meta_path) as f:
                    meta_j = json.load(f)
                if cid not in df_indexed.index:
                    continue
                gt_events = events_clip_local(df_indexed.loc[cid, "events"])
                res = InferenceResult(
                    landmarks=lm_df,
                    seconds_per_frame=meta_j["seconds_per_frame"],
                    n_frames=meta_j["n_frames"],
                    n_frames_detected=meta_j["n_frames_detected"],
                )
                row = compute_all_metrics(res, golfdb_events_relative=gt_events)
                row["model"] = mname
                row["clip_id"] = cid
                row["view"] = df_indexed.loc[cid, "view"]
                row["slow"] = int(df_indexed.loc[cid, "slow"])
                row["sex"] = df_indexed.loc[cid, "sex"]
                row["club"] = df_indexed.loc[cid, "club"]
                all_rows.append(row)
            except Exception as e:
                print(f"    clip {cid}: {repr(e)[:120]}")

    out = pd.DataFrame(all_rows)
    out_path = DATA_DIR / "all_metrics.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[compute_metrics] wrote {out_path} : {len(out):,} rows, {out['model'].nunique()} models")
    print("\n=== leaderboard (median per model) ===")
    print(out.groupby("model").agg({
        "fps_inference":           "median",
        "detection_rate":          "mean",
        "bone_cv_mean":            "median",
        "jitter_mean_px":          "median",
        "implausible_frac_mean":   "median",
        "left_ankle_planting_std_px":"median",
        "pce_at_5":                "mean",
        "pce_at_3":                "mean",
        "pce_at_1":                "mean",
    }).round(3).sort_values("pce_at_5", ascending=False).to_string())


if __name__ == "__main__":
    main()
