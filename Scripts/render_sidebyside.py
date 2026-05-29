"""Build a side-by-side comparison MP4 for each curated clip.

For each clip, tiles the 7 per-model overlay videos into a single 4x2 grid
(plus a label tile) so the team can watch all models on the same swing
simultaneously. Output: Data/overlays/sidebyside/<clip_id>_compare.mp4.

Run after render_overlays.py has produced the per-model MP4s.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
OVERLAYS_DIR = PROJECT_ROOT / "Data" / "overlays"
OUT_DIR      = OVERLAYS_DIR / "sidebyside"

ALL_MODELS = [
    "mediapipe_heavy", "mediapipe_lite",
    "movenet_thunder", "movenet_lightning",
    "yolov8n_pose", "yolov8m_pose",
    "vitpose_base",
]


def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def render_grid_for_clip(clip_id: int, label: str = "") -> Path | None:
    """Open each model's overlay MP4 in parallel, stitch into a 4x2 grid
    per frame, write a single grid MP4. h264 conversion at the end."""
    caps = []
    for m in ALL_MODELS:
        p = OVERLAYS_DIR / m / f"{clip_id}_{m}.mp4"
        if not p.exists():
            return None
        caps.append((m, cv2.VideoCapture(str(p))))

    fps = caps[0][1].get(cv2.CAP_PROP_FPS) or 30.0
    tile_w = int(caps[0][1].get(cv2.CAP_PROP_FRAME_WIDTH))
    tile_h = int(caps[0][1].get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 4 columns x 2 rows (8 cells, 7 models + 1 label panel)
    cols, rows = 4, 2
    grid_w = tile_w * cols
    grid_h = tile_h * rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"{clip_id}_compare_tmp.mp4"
    final = OUT_DIR / f"{clip_id}_compare.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (grid_w, grid_h))

    n_frames_min = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for _, c in caps)

    for fi in range(n_frames_min):
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, (m, cap) in enumerate(caps):
            ok, frame = cap.read()
            if not ok: continue
            r, c = divmod(i, cols)
            canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = frame
        # 8th cell (bottom-right): clip label
        i = len(ALL_MODELS)
        r, c = divmod(i, cols)
        y0, x0 = r * tile_h, c * tile_w
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w, y0 + tile_h), (40, 40, 40), -1)
        text_lines = [
            f"clip {clip_id}",
            label,
            f"frame {fi+1}/{n_frames_min}",
        ]
        for k, line in enumerate(text_lines):
            cv2.putText(canvas, line, (x0 + 5, y0 + 15 + k * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        writer.write(canvas)

    for _, cap in caps:
        cap.release()
    writer.release()

    # h264 convert
    ffp = ffmpeg_path()
    if ffp:
        try:
            subprocess.run(
                [ffp, "-y", "-i", str(tmp), "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", "-loglevel", "error", str(final)],
                check=True, timeout=120,
            )
            tmp.unlink()
            return final
        except Exception as e:
            print(f"  ffmpeg failed for clip {clip_id}: {e}")
            tmp.rename(final)
            return final
    tmp.rename(final)
    return final


def main():
    manifest_path = OVERLAYS_DIR / "manifest.json"
    if not manifest_path.exists():
        print("[sidebyside] no manifest; run render_overlays.py first")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[sidebyside] rendering {len(manifest)} side-by-side grids ({len(ALL_MODELS)} models per clip)")
    for entry in manifest:
        cid = entry["clip_id"]
        label = f"{entry['selection_label']}  view={entry['view']}  slow={entry['slow_mo']}  club={entry['club']}"
        out = render_grid_for_clip(cid, label=label)
        if out:
            print(f"  ok  {out.relative_to(PROJECT_ROOT)}  ({out.stat().st_size // 1024} KB)")
        else:
            print(f"  --  clip {cid} missing inputs")


if __name__ == "__main__":
    main()
