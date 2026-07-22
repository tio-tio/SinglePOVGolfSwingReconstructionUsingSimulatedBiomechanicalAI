"""MotionBERT 2D-to-3D pose lifter adapter.

MotionBERT (ICCV 2023) is a generic 2D→3D pose lifter pretrained on AMASS +
Human3.6M. We use it the same way as GolfPose: feed cached MediaPipe Heavy
2D landmarks (converted COCO-17 → H36M-17), get 3D output per frame.

The head-to-head: GolfPose (golf-fine-tuned) vs MotionBERT (generic SOTA)
on GolfDB clips. Same input, same metric battery, different 3D lifters.

Weights via Hugging Face Hub (walterzhu/MotionBERT) — no manual download.
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


class MotionBERTAdapter(BaseAdapter):
    """Wraps MotionBERT's DSTformer for 2D→3D lifting.

    Pipeline per clip:
      cached 2D (COCO-17, px) -> H36M-17 conversion -> screen normalization
        -> sliding 243-frame window inference -> H36M-17 3D output
        -> project to COCO-17 subset for unified metrics
    """

    family = "3d"
    native_skeleton = "coco17"  # output canonicalized

    def __init__(
        self,
        upstream_2d_cache_dir: str | Path,
        upstream_2d_model: str = "mediapipe_heavy",
        device: str = "cuda",
        variant: str = "lite",  # "lite" (61MB) or "full" (162MB)
        receptive_field: int = 243,
        weights_dir: Optional[str | Path] = None,
    ):
        self.name = f"motionbert_{variant}_from_{upstream_2d_model}"
        self.device = device
        self.receptive_field = receptive_field
        self.upstream_2d_cache_dir = Path(upstream_2d_cache_dir)
        self.upstream_2d_model = upstream_2d_model

        self.model = self._load_motionbert(variant, weights_dir)
        self.model = self.model.to(device).eval()

    # ---------------------------------------------------------------------
    # Model construction + weight loading
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_motionbert(variant: str, weights_dir: Optional[str | Path]):
        """Build DSTformer with the right hyperparams + load pretrained
        weights from Hugging Face Hub. The model class lives in the
        MotionBERT repo — we vendor only the network definition since
        their full repo has training infrastructure we don't need."""
        # The DSTformer class is defined in MotionBERT/lib/model/DSTformer.py.
        # We expect the user to have cloned the repo to C:\dev\MotionBERT.
        motionbert_repo = Path(r"C:\dev\MotionBERT")
        if not motionbert_repo.exists():
            raise FileNotFoundError(
                f"Clone MotionBERT first: git clone https://github.com/Walter0807/MotionBERT.git {motionbert_repo}"
            )
        sys.path.insert(0, str(motionbert_repo))
        from lib.model.DSTformer import DSTformer

        # Architecture hyperparams differ between Lite and Full:
        #   - Lite (MB_lite.yaml): dim_feat=256, mlp_ratio=4
        #   - Full (MB_pretrain.yaml): dim_feat=512, mlp_ratio=2
        # Both share depth=5, num_heads=8, dim_rep=512, att_fuse=True.
        if variant == "lite":
            model = DSTformer(
                dim_in=3, dim_out=3,
                dim_feat=256, dim_rep=512,
                depth=5, num_heads=8, mlp_ratio=4,
                num_joints=17, maxlen=243, att_fuse=True,
            )
        else:  # full
            model = DSTformer(
                dim_in=3, dim_out=3,
                dim_feat=512, dim_rep=512,
                depth=5, num_heads=8, mlp_ratio=2,
                num_joints=17, maxlen=243, att_fuse=True,
            )

        # Download checkpoint from HF Hub. The official walterzhu/MotionBERT
        # repo has the 3D-pose-finetuned weights nested under checkpoint/.
        from huggingface_hub import hf_hub_download
        fname = ("checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
                 if variant == "lite"
                 else "checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin")
        ckpt_path = hf_hub_download(repo_id="walterzhu/MotionBERT",
                                     filename=fname,
                                     cache_dir=str(weights_dir or Path.home() / ".cache" / "motionbert"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_pos", ckpt)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"  [motionbert] missing {len(missing)} keys (likely safe)")
        if unexpected:
            print(f"  [motionbert] unexpected {len(unexpected)} keys")
        return model

    # ---------------------------------------------------------------------
    # 2D input loading
    # ---------------------------------------------------------------------

    def _load_upstream_xy(self, clip_id: str) -> Optional[np.ndarray]:
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

    # ---------------------------------------------------------------------
    # MotionBERT-specific preprocessing
    # ---------------------------------------------------------------------

    @staticmethod
    def _prepare_input(xy_h36m_px: np.ndarray, w: int, h: int) -> np.ndarray:
        """MotionBERT's pretrained input format per their repo's
        `tools/preprocess.py`: normalize to [-1,1] using image width as
        scale, and append a third channel = confidence (1.0 if dense, else 0).
        Output shape: (T, 17, 3)."""
        out = np.zeros(xy_h36m_px.shape[:-1] + (3,), dtype=np.float32)
        out[..., :2] = xy_h36m_px / w * 2.0
        out[..., 0] -= 1.0
        out[..., 1] -= h / w
        out[..., 2] = 1.0  # confidence channel (dense input)
        return out

    # ---------------------------------------------------------------------
    # Sliding-window inference
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _infer(self, x_t173: np.ndarray) -> np.ndarray:
        """Sliding window over a (T, 17, 3) array → (T, 17, 3) output."""
        T = x_t173.shape[0]
        rf = self.receptive_field

        if T <= rf:
            pad_l = (rf - T) // 2
            pad_r = rf - T - pad_l
            padded = np.concatenate([
                np.repeat(x_t173[:1], pad_l, axis=0), x_t173,
                np.repeat(x_t173[-1:], pad_r, axis=0),
            ]) if T < rf else x_t173
            x = torch.from_numpy(padded[None]).to(self.device)
            pred = self.model(x)[0].cpu().numpy()
            return pred[pad_l:pad_l + T] if T < rf else pred

        out = np.zeros((T, 17, 3), dtype=np.float32)
        cnts = np.zeros(T, dtype=np.int32)
        step = rf // 2
        positions = list(range(0, T - rf + 1, step))
        if positions[-1] != T - rf: positions.append(T - rf)
        for start in positions:
            window = x_t173[start:start + rf]
            x = torch.from_numpy(window[None]).to(self.device)
            pred = self.model(x)[0].cpu().numpy()
            out[start:start + rf] += pred
            cnts[start:start + rf] += 1
        return out / np.maximum(cnts[:, None, None], 1)

    # ---------------------------------------------------------------------
    # BaseAdapter contract
    # ---------------------------------------------------------------------

    def predict(self, video_path) -> InferenceResult:
        from eval_utils import video_info
        info = video_info(video_path)
        clip_id = Path(video_path).stem
        xy_coco_px = self._load_upstream_xy(clip_id)
        if xy_coco_px is None:
            raise FileNotFoundError(
                f"Upstream 2D cache missing for clip {clip_id} / {self.upstream_2d_model}."
            )

        # COCO-17 → H36M-17 → prepare 3-channel input
        xy_h36m_px = coco17_to_h36m17(xy_coco_px)
        x_in = self._prepare_input(xy_h36m_px, info["width"], info["height"])

        t0 = time.perf_counter()
        xyz_h36m = self._infer(x_in)
        t1 = time.perf_counter()

        # H36M-17 → COCO-17 subset for unified metrics
        xyz_coco = h36m17_to_coco17_subset(xyz_h36m)

        T = xyz_coco.shape[0]
        rows = []
        face_kp_ids = {COCO17_IDX["left_eye"], COCO17_IDX["right_eye"],
                       COCO17_IDX["left_ear"], COCO17_IDX["right_ear"]}
        for fi in range(T):
            for ki in range(17):
                conf = 0.0 if ki in face_kp_ids else 1.0
                rows.append({
                    "frame": fi, "kp_idx": ki, "kp_name": COCO17_NAMES[ki],
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
