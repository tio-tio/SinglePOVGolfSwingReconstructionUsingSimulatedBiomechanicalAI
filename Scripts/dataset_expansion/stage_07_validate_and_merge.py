"""Stage 7 — merge MotionCaddie-Extra clips into the combined dataset.

Produces Data/motioncaddie_combined.pkl that has the same schema as
GolfDB's golfDB.pkl but with an extra `source` column to tell golf-db
rows from motioncaddie-extra rows.

Any downstream tool (run_benchmark, compute_metrics, pipeline) can now
take the combined pkl as its dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import CLIPS_VALIDATED_PKL, COMBINED_PKL, CLIPS_DIR

PROJECT_ROOT = Path(__file__).parent.parent.parent
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"


def main():
    print("[merge] loading GolfDB ...")
    golfdb = pd.read_pickle(GOLFDB_PKL)
    print(f"  {len(golfdb)} GolfDB rows")
    golfdb["source"] = "golfdb"
    golfdb["labeler"] = "human"

    extra = pd.read_pickle(CLIPS_VALIDATED_PKL) if CLIPS_VALIDATED_PKL.exists() else pd.DataFrame()
    print(f"  {len(extra)} motioncaddie-extra rows")

    if len(extra) == 0:
        print("[merge] no extra clips to add; output mirrors GolfDB")
        combined = golfdb.copy()
    else:
        combined = pd.concat([golfdb, extra], ignore_index=True)

    # Final sanity check: each motioncaddie row's video must exist on disk
    if len(extra):
        for _, row in extra.iterrows():
            video_path = CLIPS_DIR / f"{row['id']}.mp4"
            if not video_path.exists():
                print(f"  WARN: {row['id']}.mp4 missing")

    combined.to_pickle(COMBINED_PKL)
    print(f"[merge] wrote {COMBINED_PKL}")
    print(f"  total: {len(combined)} clips")
    print(f"  by source: {dict(combined['source'].value_counts())}")
    print(f"  by labeler: {dict(combined['labeler'].value_counts())}")


if __name__ == "__main__":
    main()
