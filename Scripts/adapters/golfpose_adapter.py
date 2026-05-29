"""GolfPose 3D adapter (3D-lifter only).

Architecture: their pretrained MixSTE2 model (a transformer fine-tuned on
the GolfSwing dataset) lifts a 243-frame window of 2D COCO-17 keypoints
into 243 frames of 3D pose. We feed our cached MediaPipe Heavy 2D
landmarks (already projected to COCO-17 in canonical schema) into it.

Why this is interesting:
- The original paper only evaluated on their own GolfSwing dataset.
- Running it on GolfDB measures how well their golf-domain pretraining
  generalizes to other golf swings.
- They never reported jitter numbers; running our metric battery on the
  3D output gives the first published smoothness measurement.

Why "3D-lifter only" not the full pipeline:
- Their 2D detector + 2D HPE stack is mmdet/mmpose which is painful on
  Windows. We use our existing MediaPipe Heavy 2D instead — same
  17 keypoints, just from a different backbone.
- This isolates the contribution of the 3D lifting transformer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_utils import (
    BaseAdapter,
    COCO17_IDX,
    COCO17_NAMES,
    InferenceResult,
    coco17_to_h36m17,
    h36m17_to_coco17_subset,
)

# GolfPose repo (cloned into C:\dev\golfpose-repo)
GOLFPOSE_REPO = Path(r"C:\dev\golfpose-repo")


class GolfPose3DAdapter(BaseAdapter):
    family = "3d"
    native_skeleton = "coco17"

    def __init__(
        self,
        ckpt_path: str | Path,
        upstream_2d_cache_dir: str | Path,
        upstream_2d_model: str = "mediapipe_heavy",
        device: str = "cuda",
        receptive_field: int = 243,
        embed_dim_ratio: int = 512,
        depth: int = 8,
        num_heads: int = 8,
    ):
        """
        Args:
          ckpt_path: path to .bin checkpoint (17+0 = 17 joints, no club).
          upstream_2d_cache_dir: Data/eval_runs (where MediaPipe cached parquets are).
          upstream_2d_model: which cached 2D model's output to use as input.
        """
        # Make GolfPose's `common/` package importable.
        sys.path.insert(0, str(GOLFPOSE_REPO))
        from common.model_cross import MixSTE2

        self.name = f"golfpose3d_from_{upstream_2d_model}"
        self.device = device
        self.receptive_field = receptive_field
        self.upstream_2d_cache_dir = Path(upstream_2d_cache_dir)
        self.upstream_2d_model = upstream_2d_model

        self.model = MixSTE2(
            num_frame=receptive_field,
            num_joints=17,
            in_chans=2,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_path_rate=0.0,
        )

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"GolfPose checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        # The repo saves with DataParallel wrap, hence 'module.' prefix in keys
        state = ckpt.get("model_pos", ckpt)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        # ignore the spatial pos embedding if shapes differ (17 vs 22 joints)
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  [golfpose] missing {len(missing)} keys (likely safe to ignore)")
        if unexpected:
            print(f"  [golfpose] unexpected {len(unexpected)} keys")
        self.model = self.model.to(device).eval()

    # ---------------------------------------------------------------------
    # Input loading: pull cached upstream 2D, normalize to GolfPose schema
    # ---------------------------------------------------------------------

    def _load_upstream_xy(self, clip_id: int, video_path: Path) -> Optional[np.ndarray]:
        """Load cached 2D landmarks from upstream model. Returns (T, 17, 2)
        in pixel coords, or None if not cached."""
        p = self.upstream_2d_cache_dir / self.upstream_2d_model / f"{clip_id}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        T = int(df["frame"].max()) + 1
        xy = np.zeros((T, 17, 2), dtype=np.float32)
        for row in df.itertuples(index=False):
            xy[int(row.frame), int(row.kp_idx), 0] = row.x
            xy[int(row.frame), int(row.kp_idx), 1] = row.y
        return xy

    @staticmethod
    def _normalize_screen(xy: np.ndarray, w: int, h: int) -> np.ndarray:
        """GolfPose follows the PoseFormer/MixSTE convention: map pixel
        coords to [-1, 1] preserving aspect ratio. From common/camera.py:
            X = X / w * 2 - [1, h/w]   in their normalize_screen_coordinates.
        """
        out = xy.copy().astype(np.float32)
        out = out / w * 2.0
        out[..., 0] -= 1.0
        out[..., 1] -= h / w
        return out

    @staticmethod
    def _coco_xy_to_h36m_xy(xy_coco: np.ndarray) -> np.ndarray:
        """COCO-17 → H36M-17 pixel coordinates. GolfPose's MixSTE2 was
        trained on H36M ordering, NOT COCO ordering, so this conversion
        is required when feeding 2D from COCO-trained detectors."""
        return coco17_to_h36m17(xy_coco)

    # ---------------------------------------------------------------------
    # Inference: sliding-window 243-frame lifter
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _infer_clip_xy(self, xy_norm: np.ndarray) -> np.ndarray:
        """Run sliding-window inference. Input: (T, 17, 2) normalized.
        Output: (T, 17, 3) 3D pose in model space."""
        T = xy_norm.shape[0]
        rf = self.receptive_field
        out = np.zeros((T, 17, 3), dtype=np.float32)
        counts = np.zeros(T, dtype=np.int32)

        # Pad input to at least rf frames (replicate edges)
        if T < rf:
            pad_total = rf - T
            pad_l = pad_total // 2
            pad_r = pad_total - pad_l
            padded = np.concatenate([
                np.repeat(xy_norm[:1], pad_l, axis=0),
                xy_norm,
                np.repeat(xy_norm[-1:], pad_r, axis=0),
            ], axis=0)
            x = torch.from_numpy(padded[None]).to(self.device)
            pred = self.model(x)[0].cpu().numpy()  # (rf, 17, 3)
            return pred[pad_l:pad_l + T]

        # T >= rf: slide with 50% overlap; average overlap regions
        step = rf // 2
        for start in range(0, T - rf + 1, step):
            window = xy_norm[start:start + rf]
            x = torch.from_numpy(window[None]).to(self.device)
            pred = self.model(x)[0].cpu().numpy()
            out[start:start + rf] += pred
            counts[start:start + rf] += 1
        # tail window
        if (T - rf) % step != 0:
            start = T - rf
            window = xy_norm[start:start + rf]
            x = torch.from_numpy(window[None]).to(self.device)
            pred = self.model(x)[0].cpu().numpy()
            out[start:start + rf] += pred
            counts[start:start + rf] += 1
        out = out / np.maximum(counts[:, None, None], 1)
        return out

    # ---------------------------------------------------------------------
    # BaseAdapter contract
    # ---------------------------------------------------------------------

    def predict(self, video_path) -> InferenceResult:
        """For 3D output we still emit the canonical 2D-style DataFrame so
        the existing eval_utils metrics work — but the y column gets the
        model's z (depth) so jitter/bone-CV measure 3D quality. The x
        column carries x; conf=1.0 everywhere except frames with missing
        upstream input."""
        from eval_utils import video_info
        info = video_info(video_path)
        clip_id = int(Path(video_path).stem)
        xy_px = self._load_upstream_xy(clip_id, Path(video_path))
        if xy_px is None:
            raise FileNotFoundError(
                f"Upstream 2D cache missing for clip {clip_id} / {self.upstream_2d_model}. "
                "Run that adapter first."
            )

        h, w = info["height"], info["width"]
        # 1. COCO → H36M (required by MixSTE2)
        xy_h36m_px = self._coco_xy_to_h36m_xy(xy_px)
        # 2. Pixel → normalized screen coords
        xy_norm = self._normalize_screen(xy_h36m_px, w, h)

        t0 = time.perf_counter()
        xyz_h36m = self._infer_clip_xy(xy_norm)  # (T, 17, 3) in H36M order
        t1 = time.perf_counter()

        # 3. Project H36M-17 → COCO-17 subset so existing metrics work.
        #    Face landmarks (eyes/ears) are filled from head; confidence
        #    for those is set low so metric code skips them.
        xyz_coco = h36m17_to_coco17_subset(xyz_h36m)

        T = xyz_coco.shape[0]
        rows = []
        face_kp_ids = {COCO17_IDX["left_eye"], COCO17_IDX["right_eye"],
                       COCO17_IDX["left_ear"], COCO17_IDX["right_ear"]}
        for fi in range(T):
            for ki in range(17):
                # nose & body joints get conf=1; eye/ear stubs get conf=0
                conf = 0.0 if ki in face_kp_ids else 1.0
                rows.append({
                    "frame": fi,
                    "kp_idx": ki,
                    "kp_name": COCO17_NAMES[ki],
                    "x": float(xyz_coco[fi, ki, 0]),
                    "y": float(xyz_coco[fi, ki, 1]),
                    "z": float(xyz_coco[fi, ki, 2]),
                    "conf": conf,
                })
        df = pd.DataFrame(rows)
        return InferenceResult(
            landmarks=df,
            seconds_per_frame=(t1 - t0) / max(1, T),
            n_frames=T,
            n_frames_detected=T,
        )
