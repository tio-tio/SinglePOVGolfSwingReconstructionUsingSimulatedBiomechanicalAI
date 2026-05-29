"""Run a list of pose models on the full GolfDB corpus and cache landmarks.

Usage:
    python run_benchmark.py --models yolov8n_pose yolov8m_pose vitpose_base
    python run_benchmark.py --models mediapipe_heavy mediapipe_lite movenet_thunder movenet_lightning
    python run_benchmark.py --models all
    python run_benchmark.py --models <m1> <m2> --limit 50  # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from adapters import ADAPTER_REGISTRY

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "Data"
VIDEO_DIR    = DATA_DIR / "videos_160"
CACHE_DIR    = DATA_DIR / "eval_runs"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names (or 'all')")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N clips (smoke testing)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if cached")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle clip order (so partial results are stratified)")
    args = p.parse_args()

    if args.models == ["all"]:
        models = list(ADAPTER_REGISTRY.keys())
    else:
        models = args.models

    df = pd.read_pickle(GOLFDB_PKL)
    all_ids = sorted([cid for cid in df["id"].tolist()
                      if (VIDEO_DIR / f"{cid}.mp4").exists()])
    if args.shuffle:
        import random
        random.Random(42).shuffle(all_ids)
    if args.limit:
        all_ids = all_ids[: args.limit]

    print(f"[runner] {len(all_ids)} clips, {len(models)} models")
    overall_start = time.perf_counter()

    for mname in models:
        if mname not in ADAPTER_REGISTRY:
            print(f"[runner] !! unknown model {mname}, skipping")
            continue
        print(f"\n[runner] === {mname} ===", flush=True)
        t0 = time.perf_counter()
        try:
            adapter = ADAPTER_REGISTRY[mname]()
        except Exception as e:
            print(f"[runner] !! failed to instantiate {mname}: {e}")
            continue
        bar = tqdm(all_ids, desc=mname, ncols=80)
        n_done = 0
        for cid in bar:
            try:
                adapter.predict_and_cache(VIDEO_DIR / f"{cid}.mp4",
                                          CACHE_DIR, overwrite=args.overwrite)
                n_done += 1
            except Exception as e:
                print(f"  [{mname}] clip {cid} failed: {repr(e)[:150]}")
        elapsed = time.perf_counter() - t0
        print(f"[runner] {mname} : {n_done}/{len(all_ids)} clips in {elapsed/60:.1f} min", flush=True)

        # Free GPU memory between models
        try:
            del adapter
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    total = (time.perf_counter() - overall_start) / 60
    print(f"\n[runner] all done in {total:.1f} min")


if __name__ == "__main__":
    main()
