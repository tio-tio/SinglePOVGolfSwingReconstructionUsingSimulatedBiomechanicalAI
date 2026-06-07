"""Stage 1 — search YouTube for golf-swing candidates via yt-dlp.

For each search query, take the top N results. Filter by duration
(10-300s), language (en), and basic title keywords. Output goes to
Data/youtube_extra/candidates.jsonl.

Resumable: re-running skips video IDs we already have.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    CANDIDATES_JSONL,
    EXTRA_ROOT,
    by_youtube_id,
    read_candidates,
    write_candidates,
)

# Curated query list aimed at: slow-motion preferred, face-on or DTL angle,
# single golfer, varied skill level
DEFAULT_QUERIES = [
    "pga tour iron shot slow motion",
    "pga tour driver swing face on",
    "pga tour driver down the line",
    "lpga slow motion swing",
    "rory mcilroy swing slow motion",
    "scottie scheffler driver swing",
    "tiger woods iron slow motion",
    "jon rahm swing slow motion",
    "viktor hovland iron swing",
    "amateur golf swing slow motion",
    "high handicap golf swing",
    "my golf swing slow motion 2025",
    "home video golf swing slow motion",
    "trackman golf swing slow motion",
    "iron 7 swing slow motion",
    "driver swing slow motion face on",
    "down the line golf swing slow motion",
    "back view golf swing",
    "amateur golfer practice swing",
    "weekend golfer swing",
]


def search_query(yt, query: str, max_results: int) -> list[dict]:
    """Run yt-dlp search for one query, return candidate dicts."""
    search_url = f"ytsearch{max_results}:{query}"
    info = yt.extract_info(search_url, download=False)
    out = []
    for entry in info.get("entries", []) or []:
        if not entry:
            continue
        if entry.get("duration") is None:
            continue
        out.append({
            "youtube_id": entry["id"],
            "url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}",
            "query": query,
            "title": entry.get("title", "").strip(),
            "duration": int(entry["duration"]),
            "channel": entry.get("channel", ""),
            "view_count": entry.get("view_count"),
            "downloaded": False,
            "swings": [],
            "clips": [],
            "labels": {},
        })
    return out


def passes_basic_filter(rec: dict) -> bool:
    if not (10 <= rec["duration"] <= 300):
        return False
    title = rec["title"].lower()
    # Reject obvious lesson/talking-head/compilation noise
    BAD = ["compilation", "best of", "podcast", "review", "ranking", "vlog"]
    if any(b in title for b in BAD):
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    p.add_argument("--per-query", type=int, default=5,
                   help="Top-N results per query (default 5)")
    p.add_argument("--target", type=int, default=100,
                   help="Stop early once we have this many candidates")
    args = p.parse_args()

    import yt_dlp
    yt = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                            "skip_download": True, "extract_flat": "in_playlist"})

    existing = by_youtube_id(read_candidates())
    print(f"[search] {len(existing)} existing candidates")

    candidates = list(existing.values())
    seen = set(existing.keys())

    for q in args.queries:
        print(f"[search] {q!r}")
        try:
            hits = search_query(yt, q, args.per_query)
        except Exception as e:
            print(f"  error: {type(e).__name__}: {str(e)[:120]}")
            continue
        kept = 0
        for h in hits:
            if h["youtube_id"] in seen:
                continue
            if not passes_basic_filter(h):
                continue
            candidates.append(h)
            seen.add(h["youtube_id"])
            kept += 1
        print(f"  +{kept} new")
        if len(candidates) >= args.target:
            print(f"[search] reached target {args.target}")
            break

    write_candidates(candidates)
    print(f"[search] total candidates: {len(candidates)}")
    print(f"[search] wrote {CANDIDATES_JSONL}")


if __name__ == "__main__":
    main()
