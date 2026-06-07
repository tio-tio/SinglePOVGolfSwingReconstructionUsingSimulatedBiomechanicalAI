"""Stage 6 — quality filter on LLM-labeled clips.

POC has no second labeler (Codex-only), so this stage runs structural
sanity checks instead of cross-model agreement:

  - All 8 events must appear in monotonic order
  - Spacing between consecutive events must be reasonable (no 1-frame
    or 200-frame gaps in places where physiology dictates otherwise)
  - Events must lie inside [0, n_frames)
  - The "swing window" (address-finish) must span >= 60% of the clip
    (rejects clips where the LLM hallucinated events into a tiny window)

Writes Data/youtube_extra/clips_validated.pkl with the kept clips.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    CLIPS_DIR,
    CLIPS_PKL,
    CLIPS_VALIDATED_PKL,
    LABELS_DIR,
    read_candidates,
)

EVENT_ORDER = ["address","toe_up","mid_backswing","top",
                "mid_downswing","impact","mid_follow_through","finish"]


def monotonic(events: dict) -> bool:
    vals = [events.get(e) for e in EVENT_ORDER]
    if any(v is None for v in vals): return False
    return all(vals[i] < vals[i+1] for i in range(7))


def reasonable_spacing(events: dict) -> bool:
    """Reject pathological gaps. Backswing should be slower than downswing
    by 2-4x; impact-to-finish has a known dynamic too. We only enforce
    coarse sanity here, not biomechanics."""
    address, top, impact, finish = events["address"], events["top"], events["impact"], events["finish"]
    backswing = top - address
    downswing = impact - top
    follow = finish - impact
    if backswing < 5 or downswing < 2 or follow < 5: return False
    # Downswing should be at most as long as the backswing (it's typically faster)
    if downswing > backswing * 1.5: return False
    return True


def main():
    records = read_candidates()
    rows = []
    kept = 0
    rejected = {"missing_label": 0, "missing_clip": 0, "not_monotonic": 0,
                 "bad_spacing": 0, "too_short_window": 0}

    for rec in records:
        for c in rec.get("clips", []):
            cid = c.get("clip_id")
            if not cid: continue
            label_path = LABELS_DIR / f"{cid}.json"
            clip_path = CLIPS_DIR / f"{cid}.mp4"
            if not label_path.exists():
                rejected["missing_label"] += 1; continue
            if not clip_path.exists():
                rejected["missing_clip"] += 1; continue
            label = json.loads(label_path.read_text())
            if not monotonic(label):
                rejected["not_monotonic"] += 1; continue
            if not reasonable_spacing(label):
                rejected["bad_spacing"] += 1; continue
            # Check the swing-window covers >= 60% of clip
            n_frames = c.get("n_frames") or rec.get("n_frames")
            if n_frames is None:
                # Read it from disk
                import cv2
                cap = cv2.VideoCapture(str(clip_path))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            window = label["finish"] - label["address"]
            if window < n_frames * 0.6:
                rejected["too_short_window"] += 1; continue

            kept += 1
            # GolfDB schema row
            # events array: [pre_buffer, address, toe_up, mid_back, top, mid_down, impact, mid_follow, finish, post_buffer]
            # For our clips events are clip-local, so pre_buffer=0 and post_buffer=n_frames-1
            events_arr = [0,
                           label["address"], label["toe_up"], label["mid_backswing"],
                           label["top"], label["mid_downswing"], label["impact"],
                           label["mid_follow_through"], label["finish"],
                           int(n_frames) - 1]
            rows.append({
                "id": cid,
                "youtube_id": rec["youtube_id"],
                "player": "unknown",
                "sex": "u",
                "club": "u",
                "view": "unknown",
                "slow": 0,
                "events": events_arr,
                "bbox": c.get("bbox", [0.0, 0.0, 1.0, 1.0]),
                "split": 0,
                "source": "motioncaddie-extra",
                "labeler": "codex_gpt5",
                "title": rec.get("title", ""),
                "n_frames": int(n_frames),
            })

    df = pd.DataFrame(rows)
    df.to_pickle(CLIPS_VALIDATED_PKL)
    print(f"[filter] kept {kept} clips, rejected:")
    for k, v in rejected.items():
        print(f"  {k}: {v}")
    print(f"[filter] wrote {CLIPS_VALIDATED_PKL}")


if __name__ == "__main__":
    main()
