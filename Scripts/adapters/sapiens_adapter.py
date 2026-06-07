"""Meta Sapiens-Pose adapter — foundation model for human pose.

Sapiens-Pose-1B outputs the GOLIATH-308 keypoint set (17 body + 71 face +
42 hands + 6 feet + 172 dense body landmarks). We extract the first 17
body keypoints, which match COCO-17 ordering, for direct apples-to-apples
comparison with our other 2D models.

Sapiens was trained on 1024x768 (portrait) inputs. GolfDB clips are 160x160
squares — we resize+pad to the expected aspect to feed the model, then map
predictions back to original pixel coordinates.

Reference: Khirodkar et al., "Sapiens: Foundation for Human Vision Models"
(ECCV 2024).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    iter_frames,
    video_info,
)

# GOLIATH skeleton ordering — first 17 keypoints match COCO-17 body order:
#   nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr,
#   l_hip, r_hip, l_kn, r_kn, l_an, r_an
GOLIATH_TO_COCO17 = list(range(17))

# Sapiens preprocessing — ImageNet normalization, BGR conventionally
_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

_SAPIENS_INPUT_H = 1024
_SAPIENS_INPUT_W = 768


class SapiensAdapter(BaseAdapter):
    family = "2d"
    native_skeleton = "goliath308"

    def __init__(self, variant: str = "1b", device: str = "cuda",
                 fp16: bool = True, batch_size: int = 4):
        self.variant = variant
        self.name = f"sapiens_pose_{variant}"
        self.device = device
        self.fp16 = fp16
        self.batch_size = batch_size

        from huggingface_hub import hf_hub_download
        repo_id = f"facebook/sapiens-pose-{variant}-torchscript"
        # Filename pattern observed: sapiens_<size>_goliath_best_goliath_AP_<score>_torchscript.pt2
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        ts_files = [f for f in files if f.endswith(".pt2")]
        if not ts_files:
            raise FileNotFoundError(f"No torchscript file found in {repo_id}")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ts_files[0],
                                     cache_dir=str(Path.home() / ".cache" / "sapiens"))
        print(f"  [sapiens] loading {ts_files[0]}")
        # Load torchscript model. NB: don't call optimize_for_inference here —
        # it injects MKLDNN ops which break CUDA execution.
        self.model = torch.jit.load(ckpt_path, map_location="cpu")
        if self.fp16:
            self.model = self.model.half()
        self.model = self.model.to(device).eval()
        print(f"  [sapiens] loaded")

    @staticmethod
    def _preprocess(frame_rgb: np.ndarray) -> tuple[torch.Tensor, float, int, int]:
        """Resize+pad to (1024, 768) preserving aspect ratio. Returns the
        normalized tensor and the scale/pad metadata needed to map outputs
        back to original pixel coords."""
        h, w = frame_rgb.shape[:2]
        target_h, target_w = _SAPIENS_INPUT_H, _SAPIENS_INPUT_W
        # Compute scale (fit longest side)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to target size, centered
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_l = pad_w // 2
        pad_t = pad_h // 2
        padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
        padded[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized.astype(np.float32)
        # ImageNet normalize (Sapiens uses BGR-order mean/std but operates on RGB
        # tensors — the official preprocessing handles this in the model wrapper).
        # We follow the reference: subtract mean, divide by std, channels-first.
        padded = (padded - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(padded.transpose(2, 0, 1))  # (C, H, W)
        return tensor, scale, pad_l, pad_t

    @staticmethod
    def _heatmaps_to_keypoints(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """heatmaps: (K, H, W) — argmax per channel + max value as confidence."""
        K, H, W = heatmaps.shape
        flat = heatmaps.view(K, -1)
        max_vals, max_idx = flat.max(dim=-1)
        ys = (max_idx // W).float()
        xs = (max_idx % W).float()
        xy = torch.stack([xs, ys], dim=-1).cpu().numpy()
        conf = max_vals.cpu().numpy()
        return xy, conf

    @torch.no_grad()
    def _predict_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (coco17_xy[17, 2], coco17_conf[17]) in original pixel coords."""
        tensor, scale, pad_l, pad_t = self._preprocess(frame_rgb)
        tensor = tensor.unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half()

        out = self.model(tensor)  # (1, K, H_out, W_out)
        heatmaps = out[0].float()
        K, H_out, W_out = heatmaps.shape

        # Heatmap coords in the output stride
        xy_hm, conf = self._heatmaps_to_keypoints(heatmaps)

        # Map heatmap coords -> input image coords (heatmaps are typically 1/4 input res)
        stride_x = _SAPIENS_INPUT_W / W_out
        stride_y = _SAPIENS_INPUT_H / H_out
        xy_in = xy_hm.copy()
        xy_in[:, 0] *= stride_x
        xy_in[:, 1] *= stride_y

        # Undo pad + scale -> original pixel coords
        xy_orig = xy_in.copy()
        xy_orig[:, 0] = (xy_orig[:, 0] - pad_l) / scale
        xy_orig[:, 1] = (xy_orig[:, 1] - pad_t) / scale

        # Take the first 17 keypoints (body, matches COCO-17 ordering)
        xy_coco = xy_orig[GOLIATH_TO_COCO17]
        conf_coco = conf[GOLIATH_TO_COCO17]
        return xy_coco, conf_coco

    @torch.no_grad()
    def _predict_batch(self, frames_rgb: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run inference on a batch of frames; returns list of (xy, conf) tuples."""
        # Preprocess each frame to tensor + remember its scale/pad metadata
        prepped = [self._preprocess(f) for f in frames_rgb]
        tensors = torch.stack([p[0] for p in prepped], dim=0).to(self.device)
        if self.fp16:
            tensors = tensors.half()
        out = self.model(tensors)  # (B, K, H_out, W_out)
        heatmaps_batch = out.float()
        results = []
        for i, (_, scale, pad_l, pad_t) in enumerate(prepped):
            heatmaps = heatmaps_batch[i]
            xy_hm, conf = self._heatmaps_to_keypoints(heatmaps)
            K, H_out, W_out = heatmaps.shape
            stride_x = _SAPIENS_INPUT_W / W_out
            stride_y = _SAPIENS_INPUT_H / H_out
            xy_in = xy_hm.copy()
            xy_in[:, 0] *= stride_x
            xy_in[:, 1] *= stride_y
            xy_orig = xy_in.copy()
            xy_orig[:, 0] = (xy_orig[:, 0] - pad_l) / scale
            xy_orig[:, 1] = (xy_orig[:, 1] - pad_t) / scale
            results.append((xy_orig[GOLIATH_TO_COCO17], conf[GOLIATH_TO_COCO17]))
        return results

    def predict(self, video_path):
        info = video_info(video_path)
        rows: list[dict] = []
        n_detected = 0
        t0 = time.perf_counter()

        # Collect frames then batch
        frames = [f for _, f in iter_frames(video_path, rgb=True)]
        fi = 0
        for chunk_start in range(0, len(frames), self.batch_size):
            chunk = frames[chunk_start:chunk_start + self.batch_size]
            batch_results = self._predict_batch(chunk)
            for (xy, conf) in batch_results:
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
                fi += 1

        t1 = time.perf_counter()
        df = pd.DataFrame(rows)
        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, info["n_frames"]),
            n_frames=info["n_frames"],
            n_frames_detected=n_detected,
        )
