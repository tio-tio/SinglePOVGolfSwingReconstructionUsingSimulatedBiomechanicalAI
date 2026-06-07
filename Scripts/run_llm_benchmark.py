"""Run the LLM event-detector adapter on a stratified subset of GolfDB.

LLM calls cost money, so we don't sweep the full 1,400 corpus. Instead we
sample a small but representative subset (default 30 clips) stratified
by view x slow-mo. Output goes alongside the other adapters in
Data/eval_runs/<llm_model>/<clip>.parquet.

After running, compute_metrics.py will pick it up automatically. We also
print PCE@5 right here so you don't have to wait for the full metric
aggregation.

Usage:
    python run_llm_benchmark.py                       # 30 clips, Claude Sonnet 4.5
    python run_llm_benchmark.py --n 10                # cheaper smoke test
    python run_llm_benchmark.py --provider openai --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from adapters.llm_event_adapter import LLMEventAdapter
from eval_utils import SWING_EVENTS

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


def pick_stratified_clips(n: int, seed: int = 42) -> list[int]:
    """Sample n clip IDs stratified by view x slow."""
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    available = sorted([cid for cid in df.index
                        if (VIDEO_DIR / f"{cid}.mp4").exists()])
    df = df.loc[available]
    df["strat"] = df["view"].astype(str) + "_" + df["slow"].astype(str)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=seed)
    _, test_idx = next(sss.split(df, df["strat"]))
    return df.iloc[test_idx].index.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="Number of clips to sample")
    p.add_argument("--provider", default="anthropic",
                   choices=["anthropic", "openai"])
    p.add_argument("--model", default="claude-sonnet-4-5",
                   help="API model name (default: claude-sonnet-4-5)")
    p.add_argument("--grid-frames", type=int, default=24)
    p.add_argument("--cell-size", type=int, default=200)
    p.add_argument("--clips", nargs="+", type=int, default=None,
                   help="Specific clip IDs (overrides --n / stratification)")
    args = p.parse_args()

    # Pre-flight check: API key
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it via: export ANTHROPIC_API_KEY='sk-ant-...'  (bash)")
        print("        or: $env:ANTHROPIC_API_KEY = 'sk-ant-...'  (PowerShell)")
        sys.exit(1)
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    if args.clips:
        clip_ids = args.clips
    else:
        clip_ids = pick_stratified_clips(args.n)
    print(f"[llm-bench] {len(clip_ids)} clips, provider={args.provider}, model={args.model}")
    print(f"[llm-bench] grid={args.grid_frames} frames x {args.cell_size}px cells")

    adapter = LLMEventAdapter(
        provider=args.provider, model=args.model,
        n_grid_frames=args.grid_frames, cell_size=args.cell_size,
    )

    df_labels = pd.read_pickle(GOLFDB_PKL).set_index("id")
    n_ok = 0
    pce5_hits = pce3_hits = pce1_hits = 0
    pce5_total = pce3_total = pce1_total = 0
    cache_dir = CACHE_DIR / adapter.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    for cid in tqdm(clip_ids, desc="LLM"):
        video_path = VIDEO_DIR / f"{cid}.mp4"
        if not video_path.exists():
            continue
        try:
            res = adapter.predict_and_cache(video_path, CACHE_DIR, overwrite=True)
        except Exception as e:
            print(f"  clip {cid}: error - {e}")
            continue
        # Pull predicted events back out of the cached parquet
        event_rows = res.landmarks[res.landmarks["frame"] == -1]
        if event_rows.empty:
            continue
        events_pred = {r.kp_name.replace("EVENT::", ""): int(r.x)
                       for r in event_rows.itertuples()}
        gt = np.asarray(df_labels.loc[cid, "events"])
        gt_local = gt[1:9] - gt[0]
        # Score
        for i, ev_name in enumerate(SWING_EVENTS):
            pred = events_pred.get(ev_name, None)
            if pred is None: continue
            err = abs(pred - int(gt_local[i]))
            pce5_total += 1
            pce3_total += 1
            pce1_total += 1
            if err <= 5: pce5_hits += 1
            if err <= 3: pce3_hits += 1
            if err <= 1: pce1_hits += 1
        n_ok += 1

    print(f"\n[llm-bench] processed {n_ok}/{len(clip_ids)} clips")
    if pce5_total:
        print(f"  PCE@5: {pce5_hits / pce5_total:.3f}")
        print(f"  PCE@3: {pce3_hits / pce3_total:.3f}")
        print(f"  PCE@1: {pce1_hits / pce1_total:.3f}")
    print(f"\n[llm-bench] cache: {cache_dir}")


if __name__ == "__main__":
    main()
