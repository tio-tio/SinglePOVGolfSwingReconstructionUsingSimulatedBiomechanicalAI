"""ViTPose adapter via HuggingFace transformers.

ViTPose is a transformer-based 2D pose estimator that tops several keypoint
benchmarks. The HF transformers integration ships its own image processor
that expects per-person bounding boxes. Because GolfDB videos are already
cropped to the golfer (preprocess_videos.py crops by the labeled bbox before
resizing to 160px), we use the full frame as the bounding box.

Native output is COCO-17 in the same ordering as our canonical schema, so
no remapping is needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    iter_frames,
    video_info,
)


class ViTPoseAdapter(BaseAdapter):
    family = "2d"
    native_skeleton = "coco17"

    _MODEL_IDS = {
        "small":  "usyd-community/vitpose-plus-small",
        "base":   "usyd-community/vitpose-base-simple",
        "huge":   "usyd-community/vitpose-plus-huge",
    }

    def __init__(self, variant: str = "base", device: str = "cuda", batch_size: int = 16):
        from transformers import AutoImageProcessor, VitPoseForPoseEstimation
        self.variant = variant
        self.name = f"vitpose_{variant}"
        self.device = device
        self.batch_size = batch_size

        model_id = self._MODEL_IDS[variant]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        # fp32 because the postprocess (scipy.ndimage.gaussian_filter on heatmaps)
        # doesn't accept fp16. ViTPose-Base at fp32 still fits comfortably in 16 GB.
        self.model = VitPoseForPoseEstimation.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device).eval()

    @torch.no_grad()
    def predict(self, video_path):
        info = video_info(video_path)
        h, w = info["height"], info["width"]
        # Use full frame as bbox (golfdb videos are pre-cropped)
        # transformers ViTPose expects [x, y, w, h] in pixel coords
        bbox = [[0.0, 0.0, float(w), float(h)]]

        frames = [f for _, f in iter_frames(video_path, rgb=True)]

        rows: list[dict] = []
        n_detected = 0
        t0 = time.perf_counter()

        for batch_start in range(0, len(frames), self.batch_size):
            batch = frames[batch_start:batch_start + self.batch_size]
            # ViTPose processor needs one boxes-list per image
            boxes_batch = [bbox for _ in batch]
            inputs = self.processor(batch, boxes=boxes_batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
            outputs = self.model(pixel_values=pixel_values)
            results = self.processor.post_process_pose_estimation(
                outputs, boxes=boxes_batch
            )
            for i, res_per_img in enumerate(results):
                fi = batch_start + i
                if not res_per_img:
                    for ki in range(17):
                        rows.append({
                            "frame": fi, "kp_idx": ki,
                            "kp_name": COCO17_NAMES[ki],
                            "x": 0.0, "y": 0.0, "conf": 0.0,
                        })
                    continue
                # one detection per image (since one bbox supplied)
                d = res_per_img[0]
                kpts = d["keypoints"].cpu().numpy()  # (17, 2)
                scores = d["scores"].cpu().numpy()    # (17,)
                detected = False
                for ki in range(17):
                    rows.append({
                        "frame": fi, "kp_idx": ki,
                        "kp_name": COCO17_NAMES[ki],
                        "x": float(kpts[ki, 0]),
                        "y": float(kpts[ki, 1]),
                        "conf": float(scores[ki]),
                    })
                    if scores[ki] >= 0.3:
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
