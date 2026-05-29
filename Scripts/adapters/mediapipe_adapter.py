"""MediaPipe Pose Landmarker adapter.

Outputs MediaPipe's 33-keypoint skeleton; we project to COCO-17 via
MP33_TO_COCO17 in eval_utils. Runs in VIDEO mode for temporal smoothing.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    MP33_TO_COCO17,
    iter_frames,
    video_info,
)


_MP_MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


class MediaPipeAdapter(BaseAdapter):
    family = "2d"
    native_skeleton = "mediapipe33"

    def __init__(self, variant: str = "heavy", model_dir: str | Path = "../Models"):
        assert variant in _MP_MODEL_URLS, f"unknown variant {variant}"
        self.variant = variant
        self.name = f"mediapipe_{variant}"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f"pose_landmarker_{variant}.task"
        if not self.model_path.exists():
            print(f"  downloading mediapipe {variant} model...")
            urllib.request.urlretrieve(_MP_MODEL_URLS[variant], self.model_path)

    def _make_landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def predict(self, video_path):
        import mediapipe as mp

        info = video_info(video_path)
        fps = info["fps"] or 30.0

        landmarker = self._make_landmarker()
        try:
            rows: list[dict] = []
            n_detected = 0
            t0 = time.perf_counter()
            for fi, frame_rgb in iter_frames(video_path, rgb=True):
                h, w = frame_rgb.shape[:2]
                ts_ms = int((fi / fps) * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=np.asarray(frame_rgb))
                res = landmarker.detect_for_video(mp_img, ts_ms)
                if res.pose_landmarks:
                    lms = res.pose_landmarks[0]
                    detected_this_frame = False
                    for coco_name, mp_idx in MP33_TO_COCO17.items():
                        lm = lms[mp_idx]
                        rows.append({
                            "frame": fi,
                            "kp_idx": COCO17_IDX[coco_name],
                            "kp_name": coco_name,
                            "x": float(lm.x) * w,
                            "y": float(lm.y) * h,
                            "conf": float(lm.visibility),
                        })
                        if lm.visibility >= 0.3:
                            detected_this_frame = True
                    if detected_this_frame:
                        n_detected += 1
                else:
                    # emit zero-conf placeholders so frame indexing stays dense
                    for coco_name in MP33_TO_COCO17:
                        rows.append({
                            "frame": fi,
                            "kp_idx": COCO17_IDX[coco_name],
                            "kp_name": coco_name,
                            "x": 0.0, "y": 0.0, "conf": 0.0,
                        })
            t1 = time.perf_counter()
        finally:
            landmarker.close()

        df = pd.DataFrame(rows)
        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, info["n_frames"]),
            n_frames=info["n_frames"],
            n_frames_detected=n_detected,
        )
