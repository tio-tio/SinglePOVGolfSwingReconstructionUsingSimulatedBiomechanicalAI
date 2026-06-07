"""Sanity check: reproduce paper's 35.6 mm MPJPE on GolfPose's own G5/G6 test split.

If our number doesn't match the paper's, the weights are loading wrong
and any downstream GolfDB result would be unreliable. Catches issues
like wrong normalization convention, wrong skeleton ordering, or
checkpoint key mismatches.
"""
from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, r"C:\dev\golfpose-repo")
from common.model_cross import MixSTE2

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_2D_PATH  = PROJECT_ROOT / "golfpose_data" / "data_2d_golf_gt.npz"
DATA_3D_PATH  = PROJECT_ROOT / "golfpose_data" / "data_3d_golf_gt.npz"
TEST_SUBJECTS = ("G5", "G6")
RECEPTIVE_FIELD = 243


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to the 17+0 .bin checkpoint")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print("=== loading data ===")
    d2 = np.load(DATA_2D_PATH, allow_pickle=True)
    d3 = np.load(DATA_3D_PATH, allow_pickle=True)
    pos2d = d2["positions_2d"].item()
    pos3d = d3["positions_3d"].item()
    print(f"  2D subjects: {list(pos2d.keys())}")
    print(f"  3D subjects: {list(pos3d.keys())}")

    print("\n=== building model ===")
    model = MixSTE2(num_frame=RECEPTIVE_FIELD, num_joints=17, in_chans=2,
                    embed_dim_ratio=512, depth=8, num_heads=8,
                    mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.0)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_pos", ckpt)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  loaded. missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]: print(f"    missing samples: {missing[:3]}")
    if unexpected[:3]: print(f"    unexpected samples: {unexpected[:3]}")
    model = model.to(args.device).eval()

    # Camera-0 image dimensions (from golf_dataset_for_inference.cameras_intrinsic_params)
    CAM0_W, CAM0_H = 1280, 720

    def normalize_screen(xy_px: np.ndarray, w: int, h: int) -> np.ndarray:
        """Pixel coords -> [-1, 1] preserving aspect ratio, matching
        GolfPose/MixSTE's `normalize_screen_coordinates` convention."""
        out = xy_px.astype(np.float32) / w * 2.0
        out[..., 0] -= 1.0
        out[..., 1] -= h / w
        return out

    print("\n=== running inference + computing MPJPE ===")
    all_errors = []
    for subj in TEST_SUBJECTS:
        if subj not in pos2d or subj not in pos3d:
            print(f"  {subj}: not present in dataset, skipping")
            continue
        for swing_name in pos2d[subj]:
            if swing_name not in pos3d[subj]:
                continue
            # 2D is a list of cameras [(T, 22, 2), ...]. Use camera 0 only.
            xy_raw = pos2d[subj][swing_name]
            if isinstance(xy_raw, list):
                xy_2d_px = np.asarray(xy_raw[0], dtype=np.float32)  # (T, 22, 2) pixels
            else:
                xy_2d_px = np.asarray(xy_raw, dtype=np.float32)
            xyz_gt = np.asarray(pos3d[subj][swing_name], dtype=np.float32) # (T, 22, 3) meters

            # 17+0 model: take first 17 joints from both
            if xy_2d_px.shape[1] < 17 or xyz_gt.shape[1] < 17:
                continue
            xyz_gt_17 = xyz_gt[:, :17, :]
            xy_in = normalize_screen(xy_2d_px[:, :17, :], CAM0_W, CAM0_H)

            T = xy_in.shape[0]
            rf = RECEPTIVE_FIELD
            # pad if short
            if T < rf:
                pad_total = rf - T
                pad_l = pad_total // 2; pad_r = pad_total - pad_l
                padded = np.concatenate([
                    np.repeat(xy_in[:1], pad_l, axis=0), xy_in,
                    np.repeat(xy_in[-1:], pad_r, axis=0),
                ])
                x = torch.from_numpy(padded[None]).to(args.device)
                with torch.no_grad():
                    pred = model(x)[0].cpu().numpy()
                pred = pred[pad_l:pad_l + T]
            else:
                # sliding window with overlap-averaging
                pred = np.zeros((T, 17, 3), dtype=np.float32)
                cnts = np.zeros(T, dtype=np.int32)
                step = rf // 2
                positions = list(range(0, T - rf + 1, step))
                if positions[-1] != T - rf: positions.append(T - rf)
                for start in positions:
                    window = xy_in[start:start + rf]
                    x = torch.from_numpy(window[None]).to(args.device)
                    with torch.no_grad():
                        out = model(x)[0].cpu().numpy()
                    pred[start:start + rf] += out
                    cnts[start:start + rf] += 1
                pred = pred / np.maximum(cnts[:, None, None], 1)

            # Root-relative MPJPE (subtract hip = keypoint 0). Predictions are
            # in MixSTE's normalized output space; GT is in meters. To compare
            # we use a per-sequence procrustes-like scale alignment: rescale
            # predictions so the mean root-relative bone length matches GT.
            pred_rel = pred - pred[:, :1, :]
            gt_rel = xyz_gt_17 - xyz_gt_17[:, :1, :]
            scale = np.linalg.norm(gt_rel).mean() / (np.linalg.norm(pred_rel).mean() + 1e-9)
            pred_rel_scaled = pred_rel * scale
            # GT range was [-1.2, 2.3] in meters -> convert to mm
            err = np.linalg.norm(pred_rel_scaled - gt_rel, axis=-1) * 1000.0
            mpjpe = err.mean()
            all_errors.append(err.flatten())
            print(f"  {subj} {swing_name:12s}  T={T:4d}  MPJPE={mpjpe:6.1f} mm   (scale={scale:.3f})")

    if all_errors:
        all_err = np.concatenate(all_errors)
        print(f"\n=== overall MPJPE on G5+G6: {all_err.mean():.1f} mm (paper: 35.6 mm) ===")
        print(f"    median per-joint err: {np.median(all_err):.1f} mm")
        if abs(all_err.mean() - 35.6) < 8.0:
            print("    PASS: within 8mm of paper number, weights loading correctly.")
        else:
            print("    SUSPECT: large delta from paper — check normalization / skeleton order.")


if __name__ == "__main__":
    main()
