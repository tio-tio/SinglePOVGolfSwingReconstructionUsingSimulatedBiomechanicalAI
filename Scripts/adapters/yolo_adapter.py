"""Ultralytics YOLOv8-Pose adapter. Native COCO-17 output, no remapping needed."""

from __future__ import annotations

import time
from pathlib import Path

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


class YoloPoseAdapter(BaseAdapter):
    family = "2d"
    native_skeleton = "coco17"

    # Process frames in fixed-size GPU batches. Passing all frames of a
    # 700-frame slow-mo clip in one .predict() call exhausts VRAM on
    # consumer GPUs and causes thrashing — empirically a 64-frame batch
    # is a good speed/memory trade on a 4080.
    CHUNK_SIZE = 64

    def __init__(self, variant: str = "yolov8n-pose", device: str = "cuda"):
        from ultralytics import YOLO
        self.variant = variant
        self.name = variant.replace("-", "_")
        self.device = device
        self.model = YOLO(f"{variant}.pt")  # auto-downloads on first call
        # warm up
        self.model.predict(np.zeros((160, 160, 3), dtype=np.uint8),
                           device=device, verbose=False)

    def predict(self, video_path):
        import torch
        info = video_info(video_path)
        rows: list[dict] = []
        n_detected = 0
        t0 = time.perf_counter()

        frames = [f for _, f in iter_frames(video_path, rgb=True)]

        fi = 0
        for chunk_start in range(0, len(frames), self.CHUNK_SIZE):
            chunk = frames[chunk_start:chunk_start + self.CHUNK_SIZE]
            results = self.model.predict(chunk, device=self.device, verbose=False,
                                          conf=0.1, iou=0.5)
            for r in results:
                kpts_obj = r.keypoints
                if kpts_obj is None or kpts_obj.xy is None or kpts_obj.xy.shape[0] == 0:
                    for kn in COCO17_NAMES:
                        rows.append({"frame": fi, "kp_idx": COCO17_IDX[kn],
                                     "kp_name": kn, "x": 0.0, "y": 0.0, "conf": 0.0})
                    fi += 1
                    continue
                confs = kpts_obj.conf
                if confs is None:
                    best = 0
                else:
                    per_det = confs.mean(dim=1).cpu().numpy() if confs.ndim == 2 else confs.cpu().numpy()
                    best = int(np.argmax(per_det))
                xy = kpts_obj.xy[best].cpu().numpy()
                cf = kpts_obj.conf[best].cpu().numpy() if kpts_obj.conf is not None else np.ones(17)
                detected_this_frame = False
                for ki in range(17):
                    rows.append({
                        "frame": fi,
                        "kp_idx": ki,
                        "kp_name": COCO17_NAMES[ki],
                        "x": float(xy[ki, 0]),
                        "y": float(xy[ki, 1]),
                        "conf": float(cf[ki]),
                    })
                    if cf[ki] >= 0.3:
                        detected_this_frame = True
                if detected_this_frame:
                    n_detected += 1
                fi += 1
            # Free per-chunk GPU tensors before next chunk
            del results
            if self.device == "cuda":
                torch.cuda.empty_cache()

        t1 = time.perf_counter()
        df = pd.DataFrame(rows)
        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, info["n_frames"]),
            n_frames=info["n_frames"],
            n_frames_detected=n_detected,
        )
