"""TensorFlow Hub MoveNet adapter (Lightning + Thunder).

MoveNet on Windows runs CPU-only because tensorflow-gpu is Linux/Mac only on
modern TF versions. That's fine — MoveNet was designed to be small, and
Thunder still hits ~50 FPS on a decent CPU. We include it as the reference
CPU-only single-person detector that Google ships for sports.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    iter_frames,
    video_info,
)


_MOVENET_URLS = {
    "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
    "thunder":   "https://tfhub.dev/google/movenet/singlepose/thunder/4",
}
_MOVENET_INPUT_SIZE = {"lightning": 192, "thunder": 256}


class MoveNetAdapter(BaseAdapter):
    family = "2d"
    native_skeleton = "coco17"

    def __init__(self, variant: str = "thunder"):
        import tensorflow_hub as hub
        assert variant in _MOVENET_URLS
        self.variant = variant
        self.name = f"movenet_{variant}"
        self.input_size = _MOVENET_INPUT_SIZE[variant]
        self.module = hub.load(_MOVENET_URLS[variant])
        self.infer = self.module.signatures["serving_default"]

    def _predict_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf
        h, w = frame_rgb.shape[:2]
        # resize keeping aspect, pad with zeros to square input_size
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame_rgb, (new_w, new_h))
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        pad_top = (self.input_size - new_h) // 2
        pad_left = (self.input_size - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        inp = tf.cast(padded[np.newaxis, ...], dtype=tf.int32)
        out = self.infer(inp)["output_0"].numpy()[0, 0]  # (17, 3): y, x, conf (normalized)
        # convert back to original-frame pixel coords
        # MoveNet returns normalized y/x in [0,1] within the padded input
        out_yx = out[:, :2] * self.input_size
        out_yx[:, 0] = (out_yx[:, 0] - pad_top) / scale
        out_yx[:, 1] = (out_yx[:, 1] - pad_left) / scale
        # back to (x, y, conf)
        return np.stack([out_yx[:, 1], out_yx[:, 0]], axis=1), out[:, 2]

    def predict(self, video_path):
        info = video_info(video_path)
        rows: list[dict] = []
        n_detected = 0
        t0 = time.perf_counter()
        for fi, frame in iter_frames(video_path, rgb=True):
            xy, conf = self._predict_frame(frame)
            detected = False
            for ki in range(17):
                rows.append({
                    "frame": fi,
                    "kp_idx": ki,
                    "kp_name": COCO17_NAMES[ki],
                    "x": float(xy[ki, 0]),
                    "y": float(xy[ki, 1]),
                    "conf": float(conf[ki]),
                })
                if conf[ki] >= 0.3:
                    detected = True
            if detected:
                n_detected += 1
        t1 = time.perf_counter()

        df = pd.DataFrame(rows)
        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, info["n_frames"]),
            n_frames=info["n_frames"],
            n_frames_detected=n_detected,
        )
