"""Stage 4 — extract one 160x160 mp4 per detected swing.

Crops around the golfer's bounding box (computed from MediaPipe landmarks
during swing detection), resizes longest side to 160, pads to square with
ImageNet means — matches GolfDB's `preprocess_videos.py` exactly so the
new clips are drop-in compatible with the existing pipeline.

Each clip gets a unique clip_id of the form `mc_<youtube_id>_<peak_f>`
which makes the source video and swing instant recoverable from the ID.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from common import CLIPS_DIR, RAW_DIR, read_candidates, write_candidates


def ffmpeg_path():
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def extract_clip(video_path: Path, start_f: int, end_f: int, out_path: Path) -> dict | None:
    """Crop the bbox around the golfer, resize to 160x160 padded square,
    write h264 mp4. Returns metadata dict on success."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # Two-pass: first pass finds the golfer's bbox via simple background
    # difference + landmark sampling. To keep this lightweight we use the
    # MediaPipe-Lite-derived bbox computed during swing detection if we have
    # it (cached); else estimate from the first frame via a quick MediaPipe
    # detection pass.
    bbox_frames = []
    frame_buf = []
    for fi in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok: break
        frame_buf.append(frame)
    cap.release()
    if not frame_buf:
        return None

    # Quick bbox estimation: use the mean-pose landmarks from MediaPipe on the
    # middle frame
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    model_path = Path(__file__).parent.parent.parent / "Models" / "pose_landmarker_lite.task"
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.3,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
    try:
        # Sample 5 frames evenly across the clip for bbox calc
        sample_idx = np.linspace(0, len(frame_buf) - 1, 5).astype(int)
        bbox_xy = []
        for si in sample_idx:
            rgb = cv2.cvtColor(frame_buf[si], cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
            res = landmarker.detect(mp_img)
            if not res.pose_landmarks: continue
            xs = [lm.x for lm in res.pose_landmarks[0]]
            ys = [lm.y for lm in res.pose_landmarks[0]]
            bbox_xy.append((min(xs), min(ys), max(xs), max(ys)))
        if not bbox_xy:
            return None
        xs1, ys1, xs2, ys2 = zip(*bbox_xy)
        # Loose bbox covering all sampled poses, 20% margin
        x1 = max(0, min(xs1) - 0.1)
        y1 = max(0, min(ys1) - 0.1)
        x2 = min(1, max(xs2) + 0.1)
        y2 = min(1, max(ys2) + 0.1)
    finally:
        landmarker.close()

    # Convert normalized bbox to pixel coords in the source video
    px1 = int(x1 * w0); py1 = int(y1 * h0)
    pw  = int((x2 - x1) * w0); ph = int((y2 - y1) * h0)
    if pw < 50 or ph < 50:
        return None

    # Crop + resize each frame to 160x160 with ImageNet mean padding
    dim = 160
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (dim, dim))
    pad_color = (int(0.406 * 255), int(0.456 * 255), int(0.485 * 255))  # ImageNet means in BGR

    for frame in frame_buf:
        crop = frame[py1:py1 + ph, px1:px1 + pw]
        if crop.size == 0: continue
        ch, cw = crop.shape[:2]
        scale = dim / max(ch, cw)
        new_w = int(cw * scale); new_h = int(ch * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        dw = dim - new_w; dh = dim - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=pad_color)
        writer.write(padded)
    writer.release()

    # h264 conversion for playback
    ffp = ffmpeg_path()
    if ffp:
        try:
            subprocess.run([ffp, "-y", "-i", str(tmp), "-vcodec", "libx264",
                             "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_path)],
                             check=True, timeout=60)
            tmp.unlink()
        except Exception:
            tmp.rename(out_path)
    else:
        tmp.rename(out_path)

    return {
        "fps": fps,
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "n_frames": len(frame_buf),
        "source_start_f": start_f,
        "source_end_f": end_f,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-clips", type=int, default=None)
    args = p.parse_args()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    records = read_candidates()
    total_new = 0

    for rec in records:
        if not rec.get("swings"):
            continue
        if rec.get("clips_extracted"):
            continue
        video_path = RAW_DIR / f"{rec['youtube_id']}.mp4"
        if not video_path.exists():
            continue

        out_clips = []
        for iv in rec["swings"]:
            clip_id = f"mc_{rec['youtube_id']}_{iv['peak_f']}"
            out_path = CLIPS_DIR / f"{clip_id}.mp4"
            if out_path.exists():
                out_clips.append({"clip_id": clip_id, "skipped": True})
                continue
            meta = extract_clip(video_path, iv["start_f"], iv["end_f"], out_path)
            if meta is None:
                continue
            meta["clip_id"] = clip_id
            out_clips.append(meta)
            total_new += 1
            if args.max_clips and total_new >= args.max_clips:
                break
        rec["clips"] = out_clips
        rec["clips_extracted"] = True
        print(f"  {rec['youtube_id']}: {len(out_clips)} clips written")
        if args.max_clips and total_new >= args.max_clips:
            break

    write_candidates(records)
    print(f"[extract] total new clips: {total_new}")


if __name__ == "__main__":
    main()
