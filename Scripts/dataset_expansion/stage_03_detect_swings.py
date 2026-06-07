"""Stage 3 — find swing intervals within each downloaded video.

We run MediaPipe Pose Lite at every Nth frame, track right-wrist Y, and
detect intervals where the wrist trajectory shows the classic swing
signature: stable -> rising -> peak -> rapid descent -> stable.

Annotates each candidate record with a list of swing intervals (start_f,
end_f, fps). Fast filter: skips intervals where the body wasn't detected
in >50% of frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import RAW_DIR, read_candidates, write_candidates


def extract_wrist_y_trajectory(video_path: Path, sample_every: int = 2) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run MediaPipe on every Nth frame, return wrist_y, confidence, fps, total_frames."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(__file__).parent.parent.parent / "Models" / "pose_landmarker_lite.task"
    if not model_path.exists():
        # MediaPipe Heavy is the only one downloaded already; Lite is small enough
        # to fetch on the fly
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    wrist_y = np.full(n_total, np.nan, dtype=np.float32)
    wrist_conf = np.zeros(n_total, dtype=np.float32)

    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if fi % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int((fi / fps) * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb))
                res = landmarker.detect_for_video(mp_img, ts_ms)
                if res.pose_landmarks:
                    lm = res.pose_landmarks[0][16]  # MediaPipe 33: 16 = right wrist
                    wrist_y[fi] = lm.y
                    wrist_conf[fi] = lm.visibility
            fi += 1
    finally:
        cap.release()
        landmarker.close()

    return wrist_y, wrist_conf, fps, n_total


def detect_swing_intervals(wrist_y: np.ndarray, wrist_conf: np.ndarray,
                            min_swing_frames: int = 30,
                            max_swing_frames: int = 600,
                            min_drop: float = 0.15) -> list[dict]:
    """Find intervals where the wrist Y-trajectory shows: rise (backswing),
    peak (top), rapid drop (downswing/impact), then leveling (follow-through).

    Heuristic: locate frames where wrist Y reaches a local minimum
    (highest point, MediaPipe Y is normalized with 0=top, 1=bottom),
    then bracket with the surrounding rise + drop.
    """
    # Interpolate gaps from detection misses
    valid = wrist_conf >= 0.3
    if valid.sum() < 30:
        return []
    y = wrist_y.copy()
    # Fill NaNs by linear interp
    idx = np.arange(len(y))
    y = np.interp(idx, idx[valid], y[valid])

    # Smooth lightly for peak finding
    kernel = max(5, int(len(y) * 0.005) | 1)  # odd
    pad = kernel // 2
    y_smooth = np.convolve(np.pad(y, pad, mode="edge"), np.ones(kernel) / kernel, mode="valid")

    intervals = []
    i = 0
    while i < len(y_smooth):
        # Find the start of a rise (wrist starts moving up — y decreasing)
        # Then find the peak (local min in y_smooth), then the drop
        if i + min_swing_frames >= len(y_smooth):
            break
        window = y_smooth[i:min(i + max_swing_frames, len(y_smooth))]
        if len(window) < min_swing_frames:
            break
        peak_local = int(np.argmin(window))
        if peak_local < 5 or peak_local >= len(window) - 5:
            i += 30  # nothing here, jump
            continue
        peak_idx = i + peak_local
        # Verify the drop after the peak
        drop = y_smooth[min(peak_idx + 20, len(y_smooth) - 1)] - y_smooth[peak_idx]
        if drop < min_drop:
            i = peak_idx + 10
            continue
        # Bracket: search backwards for a rise start
        start_f = max(0, peak_idx - 80)
        end_f = min(len(y_smooth) - 1, peak_idx + 80)
        if end_f - start_f < min_swing_frames:
            i = end_f + 1
            continue
        intervals.append({
            "start_f": int(start_f),
            "peak_f": int(peak_idx),
            "end_f": int(end_f),
            "peak_drop": float(drop),
            "valid_frac": float((wrist_conf[start_f:end_f] >= 0.3).mean()),
        })
        # Advance past this interval to avoid overlap
        i = end_f + 30

    return intervals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-videos", type=int, default=None)
    args = p.parse_args()

    records = read_candidates()
    todo = [r for r in records
            if r.get("downloaded") and not r.get("swings_detected")
            and (RAW_DIR / f"{r['youtube_id']}.mp4").exists()]
    if args.max_videos:
        todo = todo[: args.max_videos]
    print(f"[swings] {len(todo)} videos to scan")

    for rec in tqdm(todo, ncols=80):
        video_path = RAW_DIR / f"{rec['youtube_id']}.mp4"
        try:
            wrist_y, wrist_conf, fps, n_total = extract_wrist_y_trajectory(video_path)
            intervals = detect_swing_intervals(wrist_y, wrist_conf)
            # Keep only intervals with high detection rate
            good = [iv for iv in intervals if iv["valid_frac"] >= 0.5]
            rec["fps"] = fps
            rec["n_frames"] = n_total
            rec["swings"] = good
            rec["swings_detected"] = True
            rec["swings_all_n"] = len(intervals)
            rec["swings_good_n"] = len(good)
        except Exception as e:
            print(f"  {rec['youtube_id']}: {type(e).__name__}: {str(e)[:120]}")
            rec["swing_error"] = str(e)[:200]

    write_candidates(records)
    n_total_swings = sum(len(r.get("swings", [])) for r in records)
    n_videos_with_swings = sum(1 for r in records if r.get("swings"))
    print(f"[swings] {n_videos_with_swings} videos contributed {n_total_swings} candidate swings")


if __name__ == "__main__":
    main()
