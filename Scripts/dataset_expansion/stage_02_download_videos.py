"""Stage 2 — download candidate videos via yt-dlp.

Downloads ~480p mp4 (small, fast) into Data/youtube_extra/raw/.
Resumable: skips videos already on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    RAW_DIR,
    by_youtube_id,
    read_candidates,
    write_candidates,
)


def download_one(yt, rec: dict) -> bool:
    out_template = str(RAW_DIR / "%(id)s.%(ext)s")
    yt.params["outtmpl"] = {"default": out_template}
    try:
        info = yt.extract_info(rec["url"], download=True)
        rec["downloaded"] = True
        # Save the info dict next to the video
        info_path = RAW_DIR / f"{rec['youtube_id']}.info.json"
        info_path.write_text(json.dumps({
            "id": info.get("id"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "fps": info.get("fps"),
            "width": info.get("width"),
            "height": info.get("height"),
            "channel": info.get("channel"),
            "upload_date": info.get("upload_date"),
        }))
        return True
    except Exception as e:
        print(f"  {rec['youtube_id']}: {type(e).__name__}: {str(e)[:140]}")
        rec["download_error"] = str(e)[:200]
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=100, help="Max videos to download")
    p.add_argument("--throttle-sec", type=float, default=2.0,
                   help="Sleep between downloads (be nice to YouTube)")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = read_candidates()
    print(f"[download] {len(records)} candidates")

    import yt_dlp
    # ffmpeg from imageio-ffmpeg (bundled) so we don't depend on a system install
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_dir = str(Path(ffmpeg_path).parent)
    except Exception:
        ffmpeg_dir = None

    yt_opts = {
        "quiet": True,
        "no_warnings": True,
        # Prefer a single mp4 file (no merge needed); else allow merge as fallback.
        "format": "b[height<=480][ext=mp4]/bv*[height<=480][ext=mp4]+ba[ext=m4a]/bv*[height<=480]+ba/b[height<=480]",
        "merge_output_format": "mp4",
        "retries": 2,
        "fragment_retries": 2,
        "skip_unavailable_fragments": True,
    }
    if ffmpeg_dir:
        yt_opts["ffmpeg_location"] = ffmpeg_dir
    yt = yt_dlp.YoutubeDL(yt_opts)

    n_done = 0
    n_new = 0
    for rec in records:
        video_path = RAW_DIR / f"{rec['youtube_id']}.mp4"
        if video_path.exists():
            rec["downloaded"] = True
            n_done += 1
            continue
        if rec.get("download_error"):
            continue
        if n_new >= args.max:
            break
        print(f"[download] {rec['youtube_id']}  {rec['title'][:60]}")
        if download_one(yt, rec):
            n_new += 1
            n_done += 1
        time.sleep(args.throttle_sec)

    write_candidates(records)
    print(f"[download] total downloaded: {n_done}, new this run: {n_new}")


if __name__ == "__main__":
    main()
