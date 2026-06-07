"""Stage 5 — label each extracted clip with the 8 GolfDB swing events via Codex.

Same prompt + JSON schema as run_codex_llm_benchmark.py. Cache per-clip
JSON so re-runs are cheap.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import CLIPS_DIR, EXTRA_ROOT, LABELS_DIR, SCHEMA_PATH, read_candidates, write_candidates
from adapters.llm_event_adapter import LLMEventAdapter


CODEX_BIN = shutil.which("codex.cmd") or shutil.which("codex") or "codex"

STAGING_DIR = EXTRA_ROOT / "_label_staging"

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
    "properties": {k: {"type": "integer", "minimum": 0}
                   for k in ["address","toe_up","mid_backswing","top",
                              "mid_downswing","impact","mid_follow_through","finish"]},
    "required": ["address","toe_up","mid_backswing","top",
                  "mid_downswing","impact","mid_follow_through","finish"],
}


def ensure_schema():
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SCHEMA_PATH.exists():
        SCHEMA_PATH.write_text(json.dumps(OUTPUT_SCHEMA))


def build_grid_png(clip_path: Path) -> Path:
    """Create the 24-frame grid PNG for one clip; returns path."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = STAGING_DIR / f"grid_{clip_path.stem}.png"
    if grid_path.exists():
        return grid_path
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < 24:
        return None
    n = 24
    indices = [int(round(i * (len(frames) - 1) / (n - 1))) for i in range(n)]
    sampled = [frames[i] for i in indices]

    class _Stub: pass
    stub = _Stub()
    stub.n_grid_frames = 24
    stub.grid_cols = 6
    stub.cell_size = 200
    png = LLMEventAdapter._build_grid_image(stub, sampled, indices)
    grid_path.write_bytes(png)
    return grid_path


def call_codex(grid_path: Path, timeout: int = 180) -> dict | None:
    ensure_schema()
    out_file = STAGING_DIR / f"_codex_out_{grid_path.stem}.txt"
    try:
        cp = subprocess.run(
            [CODEX_BIN, "exec",
             "--sandbox", "read-only",
             "-i", str(grid_path),
             "--output-schema", str(SCHEMA_PATH),
             "-o", str(out_file),
             PROMPT],
            timeout=timeout, capture_output=True, text=True, shell=False,
        )
        if cp.returncode != 0:
            return None
        text = out_file.read_text() if out_file.exists() else cp.stdout
        m = re.search(r"\{[\s\S]*\}", text)
        if not m: return None
        return json.loads(m.group())
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--timeout", type=int, default=180)
    args = p.parse_args()

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    records = read_candidates()

    todo: list[tuple[Path, dict, dict]] = []
    for rec in records:
        for c in rec.get("clips", []):
            cid = c.get("clip_id")
            if not cid: continue
            label_path = LABELS_DIR / f"{cid}.json"
            clip_path = CLIPS_DIR / f"{cid}.mp4"
            if not clip_path.exists(): continue
            if label_path.exists():
                rec.setdefault("labels", {})[cid] = json.loads(label_path.read_text())
                continue
            todo.append((clip_path, rec, c))

    if args.max_clips:
        todo = todo[: args.max_clips]
    print(f"[label] {len(todo)} clips to label")

    n_ok = 0
    for clip_path, rec, c in tqdm(todo, ncols=80):
        cid = c["clip_id"]
        grid_path = build_grid_png(clip_path)
        if grid_path is None:
            continue
        t0 = time.perf_counter()
        events = call_codex(grid_path, timeout=args.timeout)
        elapsed = time.perf_counter() - t0
        if not events:
            print(f"  {cid}: no response")
            continue
        label = {**events, "elapsed_sec": elapsed, "labeler": "codex_gpt5"}
        (LABELS_DIR / f"{cid}.json").write_text(json.dumps(label, indent=1))
        rec.setdefault("labels", {})[cid] = label
        n_ok += 1

    write_candidates(records)
    print(f"[label] success: {n_ok}/{len(todo)}")


if __name__ == "__main__":
    main()
