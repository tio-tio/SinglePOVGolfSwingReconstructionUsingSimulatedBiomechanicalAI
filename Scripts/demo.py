"""Motion Caddie — full demo: video in → coaching scorecard out.

Closes the loop end-to-end:
    raw .mp4  →  MediaPipe Lite (2D)  →  GolfPose MixSTE (3D)
              →  trained 1D-CNN events (raw 3D)  +  One-Euro smoothing (measure)
              →  biomechanical indicators  →  coaching-lite scorecard
              →  grounded Claude read  +  browser 3D replay

Process isolation: each heavy stage runs as its own subprocess (mirrors the
benchmark) so the mediapipe + torch co-import never segfaults.

Usage:
    # raw video (runs the whole pipeline)
    python demo.py path/to/swing.mp4

    # GolfDB clip by id (3D already cached — instant)
    python demo.py --golfdb-clip 1292
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = sys.executable
SCRIPTS = Path(__file__).parent
PROJECT_ROOT = SCRIPTS.parent
CACHE_3D_DIR = PROJECT_ROOT / "Data" / "eval_runs" / "golfpose3d_from_mediapipe_lite"
DEMO_ROOT = PROJECT_ROOT / "Data" / "demo"


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n=== {desc} ===")
    cp = subprocess.run(cmd, cwd=str(SCRIPTS))
    if cp.returncode != 0:
        print(f"  !! step failed (exit {cp.returncode}): {desc}")
        return False
    return True


def demo_golfdb_clip(clip_id: int):
    """Fast path: 3D is cached for all GolfDB clips."""
    import pandas as pd
    out_dir = DEMO_ROOT / f"clip{clip_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_3d = CACHE_3D_DIR / f"{clip_id}.parquet"
    if not parquet_3d.exists():
        sys.exit(f"no cached 3D for clip {clip_id} — use a raw mp4 path instead")

    # metadata for the scorecard header
    gdb = pd.read_pickle(PROJECT_ROOT / "golfdb" / "golfDB.pkl").set_index("id")
    meta = {"clip_id": clip_id, "player": str(gdb.loc[clip_id, "player"]),
            "club": str(gdb.loc[clip_id, "club"]), "view": str(gdb.loc[clip_id, "view"])}
    meta_path = out_dir / "_meta.json"; meta_path.write_text(json.dumps(meta))

    ok = run([PY, "scorecard_step.py", "--parquet", str(parquet_3d),
              "--out-dir", str(out_dir), "--stem", f"clip{clip_id}",
              "--meta-json", str(meta_path)],
             "Event detection + coaching scorecard")
    if ok:
        run([PY, "coaching_explain.py", "--scorecard",
             str(out_dir / f"clip{clip_id}_scorecard.json")],
            "LLM explanation (gated f_strict_grounding)")
        _print_bundle(out_dir)


def demo_raw_video(video: Path, backbone="mediapipe_lite", lifter="golfpose3d"):
    stem = video.stem
    out_dir = DEMO_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: full pipeline (2D -> 3D -> smoothing -> 3D exports + overlay + browser replay)
    ok = run([PY, "pipeline.py", str(video),
              "--backbone", backbone, "--lifter", lifter,
              "--out-dir", str(out_dir)],
             "Pipeline: 2D -> 3D -> smoothing -> 3D exports + browser replay")
    if not ok:
        return

    # Step 2: the lifter cached the 3D parquet here — the L/R-repair path writes it
    # under the _lrfix backbone name, so prefer that variant (mirrors the cloud
    # handler's lookup in deploy/processing/processing_handler.py).
    cache_root = CACHE_3D_DIR.parent
    candidates = [cache_root / f"{lifter}_from_{backbone}_lrfix" / f"{stem}.parquet",
                  cache_root / f"{lifter}_from_{backbone}" / f"{stem}.parquet"]
    parquet_3d = next((p for p in candidates if p.exists()), None)
    if parquet_3d is None:
        print(f"  !! expected 3D cache not found: {candidates[1]}")
        return

    # Step 3: detector + scorecard (separate process, torch only)
    ok = run([PY, "scorecard_step.py", "--parquet", str(parquet_3d),
              "--out-dir", str(out_dir), "--stem", stem],
             "Event detection + coaching scorecard")
    if ok:
        run([PY, "ball_step.py", str(video),
             "--landmarks", str(out_dir / f"{stem}_landmarks_2d.csv"),
             "--scorecard", str(out_dir / f"{stem}_scorecard.json"),
             "--replay", str(out_dir / f"{stem}_replay_3d.json"),
             "--overlay", str(out_dir / f"{stem}_overlay.mp4"),
             "--out", str(out_dir / f"{stem}_ball_3d.json")],
            "Ball tracking + flight fit")
        run([PY, "coaching_explain.py", "--scorecard",
             str(out_dir / f"{stem}_scorecard.json")],
            "LLM explanation (gated f_strict_grounding)")
        _print_bundle(out_dir)


def _print_bundle(out_dir: Path):
    print(f"\n{'='*56}\n  DEMO BUNDLE: {out_dir}\n{'='*56}")
    for f in sorted(out_dir.iterdir()):
        if f.name.startswith("_"):
            continue
        kb = f.stat().st_size // 1024
        print(f"  {f.name:42} {kb:>6} KB")
    sc = out_dir / f"{out_dir.name}_scorecard.json"
    if sc.exists():
        d = json.loads(sc.read_text())
        print(f"\n  SUMMARY: {d['summary']}")
        for fb in d["feedback"]:
            print(f"    • {fb['message']}")
        if d.get("llm_explanation"):
            g = d.get("llm_grounding", {})
            print(f"\n  EXPLANATION (grounded={g.get('grounded')}):\n    " +
                  d["llm_explanation"].replace("\n", "\n    "))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", help="Path to a raw .mp4 swing")
    p.add_argument("--golfdb-clip", type=int, default=None,
                   help="Demo on a cached GolfDB clip id (instant)")
    args = p.parse_args()

    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    if args.golfdb_clip is not None:
        demo_golfdb_clip(args.golfdb_clip)
    elif args.video:
        demo_raw_video(Path(args.video).resolve())
    else:
        p.error("give a video path or --golfdb-clip <id>")


if __name__ == "__main__":
    main()
