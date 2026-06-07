"""Shared paths + helpers for the dataset_expansion pipeline."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTRA_ROOT   = PROJECT_ROOT / "Data" / "youtube_extra"
RAW_DIR      = EXTRA_ROOT / "raw"
CLIPS_DIR    = EXTRA_ROOT / "videos_160"
LABELS_DIR   = EXTRA_ROOT / "labels_codex"
SCHEMA_PATH  = EXTRA_ROOT / "_codex_schema.json"

# Master candidates file accumulated across stages — each line is a JSON dict
# {"youtube_id": str, "url": str, "query": str, "title": str, "duration": int,
#  "downloaded": bool, "swings": [{"start_f": int, "end_f": int, "fps": float, ...}],
#  "clips": [{"clip_id": str, ...}],
#  "labels": {clip_id: {address: int, ...}}}
CANDIDATES_JSONL = EXTRA_ROOT / "candidates.jsonl"

CLIPS_PKL    = EXTRA_ROOT / "clips.pkl"
CLIPS_VALIDATED_PKL = EXTRA_ROOT / "clips_validated.pkl"
COMBINED_PKL = PROJECT_ROOT / "Data" / "motioncaddie_combined.pkl"


def read_candidates() -> list[dict]:
    if not CANDIDATES_JSONL.exists():
        return []
    out = []
    for line in CANDIDATES_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_candidates(records: list[dict]) -> None:
    EXTRA_ROOT.mkdir(parents=True, exist_ok=True)
    CANDIDATES_JSONL.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def by_youtube_id(records: list[dict]) -> dict[str, dict]:
    return {r["youtube_id"]: r for r in records}
