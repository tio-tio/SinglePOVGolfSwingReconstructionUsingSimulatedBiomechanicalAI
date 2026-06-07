"""Batch the Codex CLI as the LLM event detector.

For each clip in a stratified subset:
  1. Generate a labeled grid PNG (24 frames sampled evenly)
  2. Invoke `codex exec` with the analysis prompt + image path
  3. Parse the JSON output
  4. Score against GolfDB ground truth (PCE@5/3/1)
  5. Cache the predicted events under Data/eval_runs/llm_codex/<clip>.parquet

Why Codex (vs direct API): Codex CLI is already auth'd on this machine and
runs GPT-5.2 with vision. No additional API key setup needed.

Usage:
    python run_codex_llm_benchmark.py                  # 30 stratified clips
    python run_codex_llm_benchmark.py --n 10           # smoke test
    python run_codex_llm_benchmark.py --clips 0 173 7  # specific clips
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from adapters.llm_event_adapter import LLMEventAdapter
from eval_utils import (
    SWING_EVENTS,
    COCO17_NAMES,
    iter_frames,
    video_info,
    InferenceResult,
)

PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR    = PROJECT_ROOT / "Data" / "videos_160"
CACHE_DIR    = PROJECT_ROOT / "Data" / "eval_runs"
STAGING_DIR  = PROJECT_ROOT / "Data" / "llm_staging"
GOLFDB_PKL   = PROJECT_ROOT / "golfdb" / "golfDB.pkl"

MODEL_NAME = "llm_codex_gpt5"   # cache subdir

# Resolve the codex executable explicitly. On Windows, `codex` from npm has
# both a shell script (no extension) and a `.cmd` wrapper for cmd.exe.
# subprocess.run needs the .cmd version (or shell=True).
import shutil as _shutil
CODEX_BIN = _shutil.which("codex.cmd") or _shutil.which("codex") or "codex"

PROMPT = """You are a golf swing analyst. The attached PNG image contains 24 frames sampled evenly from a single golf swing video, arranged in a 6-column grid. Each frame is labeled in the top-left with its frame number (e.g. "f=42").

Identify the frame number in the grid where each of these 8 canonical swing events occurs:

  1. address          - golfer set up over the ball, club resting behind it, body still
  2. toe_up           - club shaft rotated to horizontal during backswing
  3. mid_backswing    - halfway up; lead arm roughly parallel to the ground
  4. top              - highest point of backswing; wrists at maximum height
  5. mid_downswing    - halfway down; lead arm parallel to the ground, descending
  6. impact           - clubhead strikes the ball
  7. mid_follow_through - halfway through follow-through
  8. finish           - end of the swing; weight on lead foot, club over the shoulder

Constraints:
  - Return the frame number of the closest labeled frame visible in the grid.
  - Events MUST appear in monotonic order: address < toe_up < mid_backswing < top < mid_downswing < impact < mid_follow_through < finish.
  - Respond with ONLY the JSON object that matches the output schema. No prose, no markdown fences.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "address":          {"type": "integer", "minimum": 0},
        "toe_up":           {"type": "integer", "minimum": 0},
        "mid_backswing":    {"type": "integer", "minimum": 0},
        "top":              {"type": "integer", "minimum": 0},
        "mid_downswing":    {"type": "integer", "minimum": 0},
        "impact":           {"type": "integer", "minimum": 0},
        "mid_follow_through": {"type": "integer", "minimum": 0},
        "finish":           {"type": "integer", "minimum": 0},
    },
    "required": ["address","toe_up","mid_backswing","top","mid_downswing",
                  "impact","mid_follow_through","finish"],
}


def pick_stratified_clips(n: int, seed: int = 42) -> list[int]:
    df = pd.read_pickle(GOLFDB_PKL).set_index("id")
    available = sorted([cid for cid in df.index
                        if (VIDEO_DIR / f"{cid}.mp4").exists()])
    df = df.loc[available]
    df["strat"] = df["view"].astype(str) + "_" + df["slow"].astype(str)
    n_classes = df["strat"].nunique()
    if n < n_classes:
        # Too few clips to stratify — random sample
        return df.sample(n=n, random_state=seed).index.tolist()
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=seed)
    _, test_idx = next(sss.split(df, df["strat"]))
    return df.iloc[test_idx].index.tolist()


def build_grid_for_clip(clip_id: int) -> Path:
    """Generate the grid PNG and write it to STAGING_DIR."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STAGING_DIR / f"grid_{clip_id}.png"
    if out_path.exists():
        return out_path
    video_path = VIDEO_DIR / f"{clip_id}.mp4"
    frames = [f for _, f in iter_frames(video_path, rgb=True)]

    # Use the static helpers on LLMEventAdapter without instantiating
    class _Stub: pass
    stub = _Stub()
    stub.n_grid_frames = 24
    stub.grid_cols = 6
    stub.cell_size = 200
    indices = LLMEventAdapter._sample_frame_indices(stub, len(frames))
    sampled = [frames[i] for i in indices]
    png = LLMEventAdapter._build_grid_image(stub, sampled, indices)
    out_path.write_bytes(png)
    return out_path


_SCHEMA_PATH = STAGING_DIR / "_codex_output_schema.json"


def _ensure_schema_on_disk():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    if not _SCHEMA_PATH.exists():
        _SCHEMA_PATH.write_text(json.dumps(OUTPUT_SCHEMA))


def call_codex(image_path: Path, timeout_sec: int = 180) -> dict | None:
    """Run `codex exec` with image attached + JSON schema enforced."""
    _ensure_schema_on_disk()
    out_file = STAGING_DIR / f"_codex_out_{image_path.stem}.txt"
    try:
        cp = subprocess.run(
            [CODEX_BIN, "exec",
             "--sandbox", "read-only",
             "-i", str(image_path),
             "--output-schema", str(_SCHEMA_PATH),
             "-o", str(out_file),
             PROMPT],
            timeout=timeout_sec,
            capture_output=True,
            text=True,
            shell=False,
        )
        if cp.returncode != 0:
            print(f"  codex exec returned {cp.returncode}: {cp.stderr[:300]}")
            return None
        text = out_file.read_text() if out_file.exists() else cp.stdout
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            print(f"  no JSON in codex output: {text[:200]}")
            return None
        events = json.loads(m.group())
        return {k: int(events[k]) for k in SWING_EVENTS if k in events}
    except subprocess.TimeoutExpired:
        print(f"  codex timed out after {timeout_sec}s")
        return None
    except Exception as e:
        print(f"  codex error: {repr(e)[:200]}")
        return None


def cache_predictions(clip_id: int, events: dict, sec_per_frame: float,
                       n_frames: int):
    """Write predictions in the standard parquet schema, with event rows
    using frame=-1 and kp_name='EVENT::<name>'."""
    cache_subdir = CACHE_DIR / MODEL_NAME
    cache_subdir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Zero-conf landmark stubs so the schema matches
    for fi in range(n_frames):
        for ki in range(17):
            rows.append({"frame": fi, "kp_idx": ki, "kp_name": COCO17_NAMES[ki],
                          "x": 0.0, "y": 0.0, "conf": 0.0})
    for ev_name in SWING_EVENTS:
        rows.append({"frame": -1, "kp_idx": -1,
                      "kp_name": f"EVENT::{ev_name}",
                      "x": float(events[ev_name]), "y": 0.0, "conf": 1.0})
    df = pd.DataFrame(rows)
    df.to_parquet(cache_subdir / f"{clip_id}.parquet", index=False)
    meta = {
        "model": MODEL_NAME,
        "seconds_per_frame": sec_per_frame,
        "n_frames": n_frames,
        "n_frames_detected": n_frames,
    }
    (cache_subdir / f"{clip_id}.meta.json").write_text(json.dumps(meta))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--clips", nargs="+", type=int, default=None)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    df_labels = pd.read_pickle(GOLFDB_PKL).set_index("id")
    clip_ids = args.clips or pick_stratified_clips(args.n)
    print(f"[codex-llm] {len(clip_ids)} clips")

    pce_hits = {1: 0, 3: 0, 5: 0}
    pce_total = 0
    n_ok = 0
    t_all0 = time.perf_counter()
    per_clip_results = []

    for cid in tqdm(clip_ids, ncols=80):
        if cid not in df_labels.index:
            continue
        video_path = VIDEO_DIR / f"{cid}.mp4"
        if not video_path.exists():
            continue
        # Build grid
        png_path = build_grid_for_clip(cid)
        # Call Codex
        t0 = time.perf_counter()
        events = call_codex(png_path, timeout_sec=args.timeout)
        elapsed = time.perf_counter() - t0
        if events is None or set(events.keys()) != set(SWING_EVENTS):
            print(f"  clip {cid}: no valid response")
            continue
        # Score
        gt = np.asarray(df_labels.loc[cid, "events"])
        gt_local = gt[1:9] - gt[0]
        errs = {}
        for i, ev_name in enumerate(SWING_EVENTS):
            err = abs(events[ev_name] - int(gt_local[i]))
            errs[ev_name] = err
            pce_total += 1
            for tol in (1, 3, 5):
                if err <= tol: pce_hits[tol] += 1
        n_ok += 1
        per_clip_pce5 = sum(1 for e in errs.values() if e <= 5) / 8
        per_clip_results.append({
            "clip_id": cid, "elapsed_sec": elapsed,
            "pce5": per_clip_pce5,
            "errs": errs,
        })
        # Cache (predictions only — synthetic landmark rows)
        info = video_info(video_path)
        cache_predictions(cid, events, elapsed / info["n_frames"], info["n_frames"])

    t_all = time.perf_counter() - t_all0

    print(f"\n[codex-llm] processed {n_ok}/{len(clip_ids)} clips in {t_all/60:.1f} min")
    if pce_total:
        print(f"  PCE@5: {pce_hits[5] / pce_total:.3f}")
        print(f"  PCE@3: {pce_hits[3] / pce_total:.3f}")
        print(f"  PCE@1: {pce_hits[1] / pce_total:.3f}")
        print(f"  median elapsed: {np.median([r['elapsed_sec'] for r in per_clip_results]):.1f}s/clip")
    print(f"\n[codex-llm] cache: {CACHE_DIR / MODEL_NAME}")

    # Save per-clip details for downstream analysis
    summary_path = STAGING_DIR / "codex_run_summary.json"
    summary_path.write_text(json.dumps({
        "n_ok": n_ok,
        "n_total": len(clip_ids),
        "pce_at_5": pce_hits[5] / pce_total if pce_total else 0,
        "pce_at_3": pce_hits[3] / pce_total if pce_total else 0,
        "pce_at_1": pce_hits[1] / pce_total if pce_total else 0,
        "per_clip": per_clip_results,
    }, indent=2))
    print(f"[codex-llm] summary: {summary_path}")


if __name__ == "__main__":
    main()
