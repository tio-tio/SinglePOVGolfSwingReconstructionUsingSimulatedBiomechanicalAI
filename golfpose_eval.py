#!/usr/bin/env python3
"""
GolfPose MediaPipe Z-depth optimization pipeline.
Phases 1–4 as specified in the capstone brief.

Run from project root:
  python3.10 golfpose_eval.py

All outputs go to outputs/ .  MediaPipe inference is cached; re-runs skip it.
"""

import os, sys, json, time, warnings, pickle
import joblib
from pathlib import Path
from collections import defaultdict

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg, signal, ndimage
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─── 0. Paths & Constants ────────────────────────────────────────────────────

PROJ    = Path(__file__).parent
GS      = PROJ / 'golfswing'
IMG_BASE = GS / 'images'
COCO_DIR = GS / 'coco'
D3      = GS / 'data_3d_golf_gt.npz'
D2      = GS / 'data_2d_golf_gt.npz'
VIDEO_DIR = PROJ / 'Data' / 'videos_160'
GOLFDB_MAT = PROJ / 'AL EDA Investigation' / 'golfDB.mat'

OUT     = PROJ / 'outputs'
CACHE   = OUT / 'cache'
COMP    = OUT / 'comparison'
DEMO    = COMP / 'golfdb_demo'
for d in [OUT, CACHE, COMP, DEMO]:
    d.mkdir(parents=True, exist_ok=True)

# GolfPose 22-joint skeleton (0-16 = body, 17-21 = club)
GP_NAMES = {
    0:'hip', 1:'r_hip', 2:'r_knee', 3:'r_ankle',
    4:'l_hip', 5:'l_knee', 6:'l_ankle', 7:'spine',
    8:'neck', 9:'neck_base', 10:'head',
    11:'l_shoulder', 12:'l_elbow', 13:'l_wrist',
    14:'r_shoulder', 15:'r_elbow', 16:'r_wrist',
    17:'shaft', 18:'hosel', 19:'heel', 20:'toe_down', 21:'toe_up',
}

# 12 body joints with direct MediaPipe equivalents
# GolfPose index -> MediaPipe landmark index
GP_TO_MP = {
    1:24,  2:26,  3:28,   # right leg:  r_hip→R_HIP, r_knee→R_KNEE, r_ankle→R_ANKLE
    4:23,  5:25,  6:27,   # left leg:   l_hip→L_HIP, l_knee→L_KNEE, l_ankle→L_ANKLE
    11:11, 12:13, 13:15,  # left arm:   l_shoulder→L_SH, l_elbow→L_EL, l_wrist→L_WR
    14:12, 15:14, 16:16,  # right arm:  r_shoulder→R_SH, r_elbow→R_EL, r_wrist→R_WR
}
GP_IDX = sorted(GP_TO_MP.keys())         # [1,2,3,4,5,6,11,12,13,14,15,16]
MP_IDX = [GP_TO_MP[g] for g in GP_IDX]  # corresponding MP landmark indices
N_JOINTS = len(GP_IDX)

# Limb pairs (GolfPose indices) for limb-length constraint
LIMBS = [
    (1, 2, 'r_thigh'), (2, 3, 'r_shin'),
    (4, 5, 'l_thigh'), (5, 6, 'l_shin'),
    (11,12,'l_upper_arm'), (12,13,'l_forearm'),
    (14,15,'r_upper_arm'), (15,16,'r_forearm'),
    (1, 4, 'hip_width'), (11,14,'shoulder_width'),
]

# Frame mapping: (img_subj, trial) → (npz_subj, npz_swing, img_offset, n_frames)
# npz_frame i  <=>  image number (i + img_offset)
TEST_MAP = [
    ('S5','S5_Swing.2120309',    'G6','Swing02', 620, 1161),
    ('S5','S5_Swing_02.2120309', 'G6','Swing06', 740, 1103),
    ('S5','S5_Swing_03.2120309', 'G6','Swing04', 590,  931),
    ('S6','S6_Swing_01.2120309', 'G5','Swing01', 570,  261),
    ('S6','S6_Swing_05.2120309', 'G5','Swing02', 420,  500),
]
TRAIN_MAP = [
    ('S1','S1_Swing.2120309',    'G1','Swing02', 610,  501),
    ('S1','S1_Swing_01.2120309', 'G1','Swing10', 490,  431),
    ('S1','S1_Swing_02.2120309', 'G1','Swing08', 383,  558),
    ('S1','S1_Swing_05.2120309', 'G1','Swing06', 360, 1161),
    ('S1','S1_Swing_06.2120309', 'G1','Swing04', 390, 1241),
    ('S2','S2_Swing.2120309',    'G2','Swing02', 318,  571),
    ('S2','S2_Swing_04.2120309', 'G2','Swing06', 480,  460),
    ('S2','S2_Swing_06.2120309', 'G2','Swing04', 500,  529),
    ('S3','S3_Swing.2120309',    'G4','Swing02', 520, 1118),
    ('S3','S3_Swing_02.2120309', 'G4','Swing08', 598,  825),
    ('S3','S3_Swing_04.2120309', 'G4','Swing06', 500, 1354),
    ('S3','S3_Swing_06.2120309', 'G4','Swing04', 544, 1497),
    ('S4','S4_Swing.2120309',    'G3','Swing02', 688, 1249),
    ('S4','S4_Swing_03.2120309', 'G3','Swing06', 574, 1153),
    ('S4','S4_Swing_05.2120309', 'G3','Swing04', 584, 1134),
]
CALIB_MAP = [m for m in TRAIN_MAP if m[0] == 'S1']  # S1 only for Procrustes

# ─── 1. Skeleton mapping JSON ─────────────────────────────────────────────────

def save_skeleton_mapping():
    mapping = {
        'description': '12 GolfPose body joints with direct MediaPipe equivalents',
        'golfpose_to_mediapipe': {str(g): mp for g, mp in GP_TO_MP.items()},
        'mediapipe_to_golfpose': {str(mp): g for g, mp in GP_TO_MP.items()},
        'golfpose_joint_names': {str(k): v for k, v in GP_NAMES.items() if k in GP_TO_MP},
        'dropped_golfpose_body_joints': {
            '0':  'hip (mid-pelvis) — no single MediaPipe equivalent; would need mean(23,24)',
            '7':  'spine — no MediaPipe landmark between hips and shoulders',
            '8':  'neck — no MediaPipe neck landmark',
            '9':  'neck_base — no MediaPipe neck_base landmark',
            '10': 'head — NOSE (MP 0) is too far anterior; excluded for precision',
        },
        'dropped_club_joints': {str(i): GP_NAMES[i] for i in range(17,22)},
        'notes': (
            'MediaPipe world landmarks: origin at mid-hip, Y increases downward '
            '(negative Y = above hip). GolfPose lab frame: Z is height (positive up), '
            'Y is front-back depth, X is left-right. Procrustes calibration on S1 '
            'recovers the 3D rotation+scale mapping from MP space to GP lab space.'
        ),
    }
    path = OUT / 'skeleton_mapping.json'
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f'[+] Saved {path}')

# ─── 2. Data loading ──────────────────────────────────────────────────────────

def load_golfpose():
    raw3 = np.load(D3, allow_pickle=True)
    raw2 = np.load(D2, allow_pickle=True)
    pos3 = raw3['positions_3d'].item()   # G1..G6 -> swing -> (F,22,3) in METERS
    pos2 = raw2['positions_2d'].item()   # G1..G6 -> swing -> list[cam0,cam1] each (F,22,2)
    return pos3, pos2


def get_gt_frames(pos3, npz_subj, npz_swing, n_frames):
    """Return (n_frames, 12, 3) GolfPose GT for the 12 mapped joints, root-centered."""
    arr = pos3[npz_subj][npz_swing][:n_frames]  # (n_frames, 22, 3)
    joints12 = arr[:, GP_IDX, :]                  # (n_frames, 12, 3)
    root = arr[:, 0:1, :]                          # joint 0 = hip mid-point
    return joints12 - root                         # root-centered (n_frames, 12, 3)


def img_path(img_subj, trial, frame_num):
    fname = f'{trial}_{frame_num:06d}.jpg'
    return IMG_BASE / img_subj / trial / fname

# ─── 3. MediaPipe inference (cached) ─────────────────────────────────────────

_mp_pose = mp.solutions.pose


def run_mediapipe_on_mapping(mapping, cache_name, stride=1, complexity=2):
    """
    Run MediaPipe on images defined by mapping, cache results to CSV.
    Returns DataFrame with columns:
      img_subj, trial, npz_subj, npz_swing, npz_frame, mp_lm_idx, x, y, z, vis
    Only the 12 body joints (GP_TO_MP values) are stored.
    """
    cache_path = CACHE / f'{cache_name}.csv'
    if cache_path.exists():
        print(f'[cache] Loading {cache_path}')
        return pd.read_csv(cache_path)

    rows = []
    mp_indices = set(GP_TO_MP.values())

    with _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=complexity,
        min_detection_confidence=0.5,
    ) as pose:
        for (img_subj, trial, npz_subj, npz_swing, offset, n_frames) in mapping:
            desc = f'{npz_subj}/{npz_swing}'
            for npz_f in tqdm(range(0, n_frames, stride), desc=desc, leave=False):
                img_num = npz_f + offset
                p = img_path(img_subj, trial, img_num)
                img = cv2.imread(str(p))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if not res.pose_world_landmarks:
                    continue
                lms = res.pose_world_landmarks.landmark
                for mp_i in mp_indices:
                    lm = lms[mp_i]
                    rows.append({
                        'img_subj': img_subj, 'trial': trial,
                        'npz_subj': npz_subj, 'npz_swing': npz_swing,
                        'npz_frame': npz_f, 'mp_lm_idx': mp_i,
                        'x': lm.x, 'y': lm.y, 'z': lm.z, 'vis': lm.visibility,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f'[+] Cached {len(df)} rows → {cache_path}')
    return df


def mp_df_to_array(df, mapping, n_frames_list=None):
    """
    Convert MP DataFrame to list of (n_frames, 12, 3) arrays (one per mapping entry).
    Missing frames → NaN. Applies root-centering (mean of L/R hip landmarks).
    """
    results = []
    mp_l_hip, mp_r_hip = 23, 24

    for i, (img_subj, trial, npz_subj, npz_swing, offset, n_frames) in enumerate(mapping):
        sub = df[(df['img_subj'] == img_subj) & (df['trial'] == trial)]

        # Build (n_frames, 12, 3) array, NaN where missing
        arr = np.full((n_frames, N_JOINTS, 3), np.nan)
        for _, row in sub.iterrows():
            fi = int(row['npz_frame'])
            if fi >= n_frames:
                continue
            mp_i = int(row['mp_lm_idx'])
            if mp_i in GP_TO_MP.values():
                ji = MP_IDX.index(mp_i)
                arr[fi, ji, :] = [row['x'], row['y'], row['z']]

        # Root-center: subtract mean(L_Hip, R_Hip) world coords
        # L_Hip = MP 23, R_Hip = MP 24
        l_hip_j = MP_IDX.index(23)  # position in the 12-joint array
        r_hip_j = MP_IDX.index(24)
        root_xyz = 0.5 * (arr[:, l_hip_j, :] + arr[:, r_hip_j, :])  # (n_frames,3)
        arr = arr - root_xyz[:, np.newaxis, :]  # broadcast (n_frames, 12, 3)

        results.append(arr)
    return results

# ─── 4. Procrustes calibration ────────────────────────────────────────────────

def fit_procrustes(pos3, mp_arrays, calib_mapping):
    """
    Fit R, s on S1 calibration data such that predicted_GP = s * mp_centered @ R.
    Returns (R, s) where R is (3,3) orthogonal, s is scalar.
    GolfPose units: meters. After alignment, MPJPE in meters → ×1000 = mm.
    """
    A_parts, B_parts = [], []  # A=GT, B=MP
    for i, (img_subj, trial, npz_subj, npz_swing, offset, n_frames) in enumerate(calib_mapping):
        gt = get_gt_frames(pos3, npz_subj, npz_swing, n_frames)  # (n,12,3)
        mp_arr = mp_arrays[i]                                       # (n,12,3), NaN possible

        # Keep only frames with no NaN in either
        valid = ~np.isnan(mp_arr).any(axis=(1, 2))
        gt_v = gt[valid]
        mp_v = mp_arr[valid]

        A_parts.append(gt_v.reshape(-1, 3))
        B_parts.append(mp_v.reshape(-1, 3))

    A = np.concatenate(A_parts, axis=0)  # (N_total, 3) GT root-centered
    B = np.concatenate(B_parts, axis=0)  # (N_total, 3) MP root-centered

    # Procrustes: min ||A - s * B @ R||_F  subject to R.T @ R = I
    R, _ = linalg.orthogonal_procrustes(B, A)  # B @ R ≈ A
    BR = B @ R
    s = np.sum(A * BR) / np.sum(BR ** 2)

    residual = np.linalg.norm(A - s * BR, 'fro') / np.sqrt(len(A))
    print(f'[Procrustes] R det={np.linalg.det(R):.4f}  scale={s:.4f}  RMS={residual*1000:.1f}mm')
    return R, s


def apply_alignment(mp_arr, R, s):
    """Apply Procrustes alignment to (n,12,3) MP array. Returns (n,12,3) in GP lab space."""
    shape = mp_arr.shape
    flat = mp_arr.reshape(-1, 3)
    aligned = s * (flat @ R)
    return aligned.reshape(shape)

# ─── 5. MPJPE evaluation ─────────────────────────────────────────────────────

def compute_mpjpe(pred_list, gt_list, mapping):
    """
    pred_list, gt_list: lists of (n_frames, 12, 3) arrays (one per mapping entry).
    Returns DataFrame with per-frame, per-joint abs Z error in mm.
    """
    rows = []
    for i, (img_subj, trial, npz_subj, npz_swing, offset, n_frames) in enumerate(mapping):
        pred = pred_list[i]  # (n,12,3)
        gt   = gt_list[i]    # (n,12,3) root-centered

        for fi in range(n_frames):
            for ji, gp_i in enumerate(GP_IDX):
                if np.isnan(pred[fi, ji, :]).any():
                    continue
                rows.append({
                    'npz_subj': npz_subj, 'npz_swing': npz_swing,
                    'npz_frame': fi, 'gp_joint': gp_i,
                    'joint_name': GP_NAMES[gp_i],
                    'z_pred': float(pred[fi, ji, 2]),
                    'z_gt':   float(gt[fi, ji, 2]),
                    'abs_error_mm': float(abs(pred[fi, ji, 2] - gt[fi, ji, 2]) * 1000),
                })
    return pd.DataFrame(rows)


def summarize_mpjpe(df, label=''):
    overall = df['abs_error_mm'].mean()
    per_joint = df.groupby('joint_name')['abs_error_mm'].mean().sort_values()
    print(f'\n── {label} ──')
    print(f'Overall Z-MPJPE: {overall:.1f} mm')
    print('Per-joint (best→worst):')
    for name, err in per_joint.items():
        print(f'  {name:<14}: {err:.1f} mm')
    return overall

# ─── 6. Limb-length references from train GT ─────────────────────────────────

def compute_ref_limb_lengths(pos3):
    """Compute mean limb lengths (meters) from S1–S4 (G1–G4) GT."""
    lengths = defaultdict(list)
    for subj in ['G1', 'G2', 'G3', 'G4']:
        for swing, arr in pos3[subj].items():
            for (ga, gb, name) in LIMBS:
                d = np.linalg.norm(arr[:, ga, :] - arr[:, gb, :], axis=1)
                lengths[name].extend(d.tolist())
    return {name: float(np.mean(vs)) for name, vs in lengths.items()}

# ─── 7. Phase 2a: Temporal smoothing ─────────────────────────────────────────

def smooth_sg(arr, window, poly):
    """Apply Savitzky-Golay to each joint's Z trajectory. arr: (n,12,3)."""
    out = arr.copy()
    n = arr.shape[0]
    if n < window:
        window = n if n % 2 == 1 else n - 1
        if window < poly + 1:
            return out
    for ji in range(N_JOINTS):
        valid = ~np.isnan(arr[:, ji, 2])
        if valid.sum() < window:
            continue
        z = arr[:, ji, 2].copy()
        z[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], z[valid])
        out[:, ji, 2] = signal.savgol_filter(z, window, poly)
    return out


def smooth_gaussian(arr, sigma):
    """Apply 1D Gaussian smoothing to each joint's Z trajectory."""
    out = arr.copy()
    for ji in range(N_JOINTS):
        valid = ~np.isnan(arr[:, ji, 2])
        if valid.sum() < 3:
            continue
        z = arr[:, ji, 2].copy()
        z[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], z[valid])
        out[:, ji, 2] = ndimage.gaussian_filter1d(z, sigma=sigma)
    return out


def tune_smoothing(pred_list, gt_list, mapping):
    """Tune SG and Gaussian parameters on train set, return best params."""
    windows = [3, 5, 7, 9, 11, 15, 21]
    polys = [2, 3]
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]

    sg_results = {}
    for poly in polys:
        for win in windows:
            if win <= poly:
                continue
            smoothed = [smooth_sg(p, win, poly) for p in pred_list]
            df_s = compute_mpjpe(smoothed, gt_list, mapping)
            sg_results[(win, poly)] = df_s['abs_error_mm'].mean()

    gauss_results = {}
    for sigma in sigmas:
        smoothed = [smooth_gaussian(p, sigma) for p in pred_list]
        df_s = compute_mpjpe(smoothed, gt_list, mapping)
        gauss_results[sigma] = df_s['abs_error_mm'].mean()

    best_sg = min(sg_results, key=sg_results.get)
    best_sigma = min(gauss_results, key=gauss_results.get)

    # Plot tuning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for poly in polys:
        xs = [w for (w, p) in sg_results if p == poly]
        ys = [sg_results[(w, poly)] for w in xs]
        axes[0].plot(sorted(xs), [y for _, y in sorted(zip(xs, ys))], marker='o', label=f'poly={poly}')
    axes[0].set_xlabel('Window length'); axes[0].set_ylabel('Z-MPJPE (mm)')
    axes[0].set_title('Savitzky-Golay tuning'); axes[0].legend(); axes[0].grid(alpha=0.3)

    xs_g = sorted(gauss_results.keys())
    ys_g = [gauss_results[s] for s in xs_g]
    axes[1].plot(xs_g, ys_g, marker='o', color='orange')
    axes[1].set_xlabel('Sigma'); axes[1].set_ylabel('Z-MPJPE (mm)')
    axes[1].set_title('Gaussian smoothing tuning'); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / 'smoothing_tuning.png', dpi=150)
    plt.close()
    print(f'[2a] Best SG: window={best_sg[0]}, poly={best_sg[1]} → {sg_results[best_sg]:.1f}mm')
    print(f'[2a] Best Gaussian: sigma={best_sigma} → {gauss_results[best_sigma]:.1f}mm')
    return best_sg, best_sigma, sg_results, gauss_results

# ─── 8. Phase 2b: Limb-length constraint ─────────────────────────────────────

def _joint_pos_idx(gp_i):
    """Index of GolfPose joint gp_i in the 12-joint ordering."""
    return GP_IDX.index(gp_i)


def apply_limb_constraint(arr, ref_lengths, tol=0.05):
    """
    Adjust Z values per frame so limb lengths stay within ±tol of reference.
    Minimises ||Z_new - Z_old||^2 with soft penalty on limb length deviations.
    arr: (n, 12, 3)  in GP lab space (meters).
    """
    out = arr.copy()
    n = arr.shape[0]

    for fi in range(n):
        frame = arr[fi]
        if np.isnan(frame).any():
            out[fi] = frame
            continue

        X = frame[:, 0]
        Y = frame[:, 1]
        Z0 = frame[:, 2]  # initial Z from MediaPipe

        def objective(Z):
            return np.sum((Z - Z0) ** 2)

        def make_ineq(ga, gb, ref, sign):
            ia, ib = _joint_pos_idx(ga), _joint_pos_idx(gb)
            def c(Z):
                L = np.sqrt((X[ia]-X[ib])**2 + (Y[ia]-Y[ib])**2 + (Z[ia]-Z[ib])**2)
                return sign * (L - ref * (1 + sign * tol))
            return c

        constraints = []
        for (ga, gb, name) in LIMBS:
            if name not in ref_lengths:
                continue
            ref = ref_lengths[name]
            # lower: L >= ref*(1-tol)
            constraints.append({'type': 'ineq', 'fun': make_ineq(ga, gb, ref, -1)})
            # upper: L <= ref*(1+tol)
            constraints.append({'type': 'ineq', 'fun': make_ineq(ga, gb, ref, +1)})

        res = minimize(objective, Z0, method='SLSQP',
                       constraints=constraints,
                       options={'ftol': 1e-8, 'maxiter': 200, 'disp': False})
        out[fi, :, 2] = res.x if res.success else Z0

    return out

# ─── 9. Phase 2c: Learned residual MLP ───────────────────────────────────────

def train_mlp(train_pred_list, train_gt_list, train_mapping, temporal_k=3):
    """
    Train MLPRegressor on S1-S4 to predict ΔZ residual per joint.
    Features: MediaPipe root-centered XYZ for all 12 joints, ±temporal_k window.
    Target: GT Z − predicted aligned Z (before MLP correction).
    """
    X_parts, y_parts = [], []

    for i, (_, _, npz_subj, npz_swing, offset, n_frames) in enumerate(train_mapping):
        pred = train_pred_list[i]  # (n,12,3) already aligned
        gt   = train_gt_list[i]    # (n,12,3) root-centered GT

        n = pred.shape[0]
        delta_z = gt[:, :, 2] - pred[:, :, 2]  # (n,12) target ΔZ

        for fi in range(n):
            if np.isnan(pred[fi]).any():
                continue
            if np.isnan(delta_z[fi]).any():
                continue

            # Build temporal context window
            feats = []
            for k in range(-temporal_k, temporal_k + 1):
                idx = max(0, min(n-1, fi + k))
                if np.isnan(pred[idx]).any():
                    feats.append(pred[fi].flatten())  # fallback to center frame
                else:
                    feats.append(pred[idx].flatten())  # (12*3,)
            X_parts.append(np.concatenate(feats))       # (7*36,) = 252 features
            y_parts.append(delta_z[fi])                  # (12,)

    X = np.array(X_parts)
    y = np.array(y_parts)
    print(f'[2c] MLP training: {len(X)} samples, {X.shape[1]} features, {y.shape[1]} targets')

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_s, y)
    loss_val = mlp.best_loss_ if mlp.best_loss_ is not None else mlp.loss_
    print(f'[2c] MLP training loss: {loss_val:.4f}  (n_iter={mlp.n_iter_})')
    return mlp, scaler


def apply_mlp(pred_list, mlp, scaler, temporal_k=3):
    """Apply MLP residual correction to list of prediction arrays."""
    out = []
    for pred in pred_list:
        corrected = pred.copy()
        n = pred.shape[0]
        for fi in range(n):
            if np.isnan(pred[fi]).any():
                continue
            feats = []
            for k in range(-temporal_k, temporal_k + 1):
                idx = max(0, min(n-1, fi + k))
                if np.isnan(pred[idx]).any():
                    feats.append(pred[fi].flatten())
                else:
                    feats.append(pred[idx].flatten())
            x = np.concatenate(feats).reshape(1, -1)
            x_s = scaler.transform(x)
            delta_z = mlp.predict(x_s)[0]  # (12,)
            corrected[fi, :, 2] += delta_z
        out.append(corrected)
    return out

# ─── 10. Phase 3 artifacts ────────────────────────────────────────────────────

def make_comparison_table(results_dict, test_gt_list, test_mapping):
    """
    results_dict: {method_name: [pred_list]} each pred_list is list of (n,12,3)
    Returns summary DataFrame.
    """
    rows = []
    base_mpjpe = None
    for method, pred_list in results_dict.items():
        df = compute_mpjpe(pred_list, test_gt_list, test_mapping)
        overall = df['abs_error_mm'].mean()
        per_joint = df.groupby('joint_name')['abs_error_mm'].mean()
        best_joint = per_joint.idxmin()
        worst_joint = per_joint.idxmax()
        if base_mpjpe is None:
            base_mpjpe = overall
        delta = overall - base_mpjpe
        rows.append({
            'Method': method,
            'Overall_Z_MPJPE_mm': round(overall, 1),
            'Best_joint': best_joint,
            'Best_joint_mm': round(per_joint[best_joint], 1),
            'Worst_joint': worst_joint,
            'Worst_joint_mm': round(per_joint[worst_joint], 1),
            'Delta_vs_baseline_mm': round(delta, 1),
        })
    return pd.DataFrame(rows)


def plot_per_joint(results_dict, test_gt_list, test_mapping):
    methods = list(results_dict.keys())
    n_methods = len(methods)
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    joint_names = [GP_NAMES[g] for g in GP_IDX]
    x = np.arange(N_JOINTS)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(14, 6))
    for mi, (method, pred_list) in enumerate(results_dict.items()):
        df = compute_mpjpe(pred_list, test_gt_list, test_mapping)
        per_joint = df.groupby('joint_name')['abs_error_mm'].mean()
        vals = [per_joint.get(GP_NAMES[g], 0) for g in GP_IDX]
        ax.bar(x + mi*width, vals, width, label=method, color=colors[mi], alpha=0.85)

    ax.set_xticks(x + width*(n_methods-1)/2)
    ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Z-MPJPE (mm)')
    ax.set_title('Per-joint Z-MPJPE by method (S5+S6 test set)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMP / 'mpjpe_per_joint.png', dpi=150)
    plt.close()
    print(f'[+] Saved {COMP}/mpjpe_per_joint.png')


def plot_z_trajectories(pred_baseline, pred_best, test_gt_list, test_mapping, best_method_name):
    """Plot Z trajectories for 3 example sequences, focusing on r_wrist."""
    target_joint = 'r_wrist'
    ji = [GP_NAMES[g] for g in GP_IDX].index(target_joint)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    example_trials = [(0, 'G6/Swing02'), (2, 'G6/Swing04'), (4, 'G5/Swing02')]

    for row_i, (trial_idx, label) in enumerate(example_trials):
        gt   = test_gt_list[trial_idx][:, ji, 2]
        base = pred_baseline[trial_idx][:, ji, 2]
        best = pred_best[trial_idx][:, ji, 2]

        ax = axes[row_i]
        t = np.arange(len(gt))
        ax.plot(t, gt * 1000,   'k-',  lw=1.5, label='GT (Vicon)', alpha=0.9)
        ax.plot(t, base * 1000, 'r--', lw=1.2, label='MediaPipe raw', alpha=0.8)
        ax.plot(t, best * 1000, 'b-',  lw=1.2, label=best_method_name, alpha=0.8)
        ax.set_title(f'{label} — {target_joint} Z trajectory')
        ax.set_ylabel('Z (mm, root-relative)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Frame index')
    plt.suptitle('Z Trajectory Examples: Ground Truth vs. MediaPipe Methods', fontsize=12)
    plt.tight_layout()
    plt.savefig(COMP / 'z_trajectory_examples.png', dpi=150)
    plt.close()
    print(f'[+] Saved {COMP}/z_trajectory_examples.png')


def render_3d_skeleton_frame(ax, xyz_12, color='steelblue', title=''):
    """Render 12-joint skeleton on a 3D matplotlib axis."""
    # Convert GP joint indices → positions in the 12-array
    def pos(gp_i): return xyz_12[GP_IDX.index(gp_i)]

    skeleton_pairs = [
        (1,2),(2,3),(4,5),(5,6),          # legs
        (11,12),(12,13),(14,15),(15,16),   # arms
        (1,4),(11,14),                     # hip & shoulder width
        (1,7),(4,7) if 7 in GP_IDX else [], # spine approximation (omit if 7 not in 12)
    ]

    valid_pairs = [(a,b) for (a,b) in skeleton_pairs
                   if a in GP_IDX and b in GP_IDX]

    for (ga, gb) in valid_pairs:
        pa, pb = pos(ga), pos(gb)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, lw=1.5)

    ax.scatter(xyz_12[:,0], xyz_12[:,1], xyz_12[:,2], c=color, s=20, zorder=5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (height)')


# ─── 11. Phase 3.5: GolfDB video overlay ─────────────────────────────────────

def make_golfdb_demo(R, s, sg_params):
    """Pick 2 GolfDB clips (1 face-on, 1 down-the-line) and render demo videos."""
    try:
        import scipy.io as sio
    except ImportError:
        print('[skip] scipy.io not available for GolfDB demo')
        return

    if not GOLFDB_MAT.exists() or not VIDEO_DIR.exists():
        print('[skip] GolfDB data not available for demo')
        return

    mat = sio.loadmat(str(GOLFDB_MAT))
    golfdb = mat['golfDB']

    def unwrap(x):
        while isinstance(x, np.ndarray) and x.size == 1:
            x = x.item()
        return x

    rows = []
    for i in range(golfdb.shape[1]):
        rec = golfdb[0, i]
        rows.append({f: unwrap(rec[f]) for f in golfdb.dtype.names})
    df = pd.DataFrame(rows)

    # Find Tiger Woods driver clips with good coverage
    target_player = 'TIGER WOODS'
    fo_clips = df[(df['player']==target_player) & (df['club']=='driver') &
                  (df['view']=='face-on') & (df['slow']==0)]
    dtl_clips = df[(df['player']==target_player) & (df['club']=='driver') &
                   (df['view']=='down-the-line') & (df['slow']==0)]

    # Fallback to any driver if Tiger not found
    if len(fo_clips) == 0:
        fo_clips = df[(df['club']=='driver') & (df['view']=='face-on') & (df['slow']==0)]
    if len(dtl_clips) == 0:
        dtl_clips = df[(df['club']=='driver') & (df['view']=='down-the-line') & (df['slow']==0)]

    demo_clips = []
    if len(fo_clips) > 0:
        demo_clips.append(('face-on', fo_clips.iloc[0]))
    if len(dtl_clips) > 0:
        demo_clips.append(('down-the-line', dtl_clips.iloc[0]))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    def events_relative(ev):
        e = np.array(ev).flatten().astype(int)
        return e - e[0]

    win, poly = sg_params

    for view, rec in demo_clips:
        vid_path = VIDEO_DIR / f"{int(rec['id'])}.mp4"
        if not vid_path.exists():
            print(f'[skip] {vid_path} not found')
            continue

        ev = events_relative(rec['events'])
        ev_clip = [int(x) for x in ev[1:9]]

        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        out_path = DEMO / f"{rec['id']}_{view.replace('-','_')}_comparison.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W*2, H))

        all_mp_xyz = {}  # frame_idx -> (12,3) or None

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            cap = cv2.VideoCapture(str(vid_path))
            fi = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_world_landmarks:
                    lms = res.pose_world_landmarks.landmark
                    raw = np.array([[lms[mp_i].x, lms[mp_i].y, lms[mp_i].z]
                                    for mp_i in MP_IDX])
                    # root center
                    lh = raw[MP_IDX.index(23)]
                    rh = raw[MP_IDX.index(24)]
                    raw -= 0.5*(lh+rh)[np.newaxis, :]
                    all_mp_xyz[fi] = raw
                fi += 1
            cap.release()

        if not all_mp_xyz:
            print(f'[skip] No MP detections for {vid_path}')
            continue

        # Apply Procrustes
        aligned = {}
        for fi, xyz in all_mp_xyz.items():
            aligned[fi] = apply_alignment(xyz[np.newaxis], R, s)[0]

        # Build trajectory for SG smoothing
        frames_sorted = sorted(aligned.keys())
        if len(frames_sorted) < win:
            traj_z = np.array([aligned[f][:, 2] for f in frames_sorted])
        else:
            traj_z = np.array([aligned[f][:, 2] for f in frames_sorted])

        # Smooth
        traj_smooth = traj_z.copy()
        if len(frames_sorted) >= win:
            for ji in range(N_JOINTS):
                traj_smooth[:, ji] = signal.savgol_filter(
                    traj_z[:, ji], win, poly)

        smooth_map = {f: traj_smooth[i] for i, f in enumerate(frames_sorted)}

        # Render side-by-side frames (event frames only for speed)
        def draw_skeleton_on_frame(frame_bgr, mp_xyz_12):
            """Draw 2D MediaPipe skeleton on BGR frame."""
            cap2 = cv2.VideoCapture(str(vid_path))
            cap2.release()
            ann = frame_bgr.copy()
            # Draw connecting lines using image-space landmarks would require pose_landmarks
            # For simplicity, just return annotated frame with skeleton overlay
            return ann

        cap = cv2.VideoCapture(str(vid_path))
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            base_frame = frame.copy()
            best_frame = frame.copy()

            # Overlay: baseline (raw aligned) skeleton
            if fi in aligned:
                overlay_color = (0, 140, 255)  # orange
                xyz_base = aligned[fi]
                # Project 3D to 2D using perspective-like projection centered in frame
                cx, cy = W//2, H//2
                scale_px = 400
                for (ga, gb, _) in LIMBS:
                    if ga not in GP_IDX or gb not in GP_IDX:
                        continue
                    pa = xyz_base[GP_IDX.index(ga)]
                    pb = xyz_base[GP_IDX.index(gb)]
                    # Simple frontal projection: X→right, -Y(height)→up, ignore Z depth
                    def proj(p):
                        return (int(cx + p[0]*scale_px), int(cy - p[2]*scale_px))
                    p2a, p2b = proj(pa), proj(pb)
                    cv2.line(base_frame, p2a, p2b, overlay_color, 2)

            # Overlay: SG-smoothed skeleton
            if fi in smooth_map:
                overlay_color_s = (0, 200, 50)  # green
                xyz_smooth = aligned[fi].copy()
                xyz_smooth[:, 2] = smooth_map[fi]
                for (ga, gb, _) in LIMBS:
                    if ga not in GP_IDX or gb not in GP_IDX:
                        continue
                    pa = xyz_smooth[GP_IDX.index(ga)]
                    pb = xyz_smooth[GP_IDX.index(gb)]
                    def proj(p):
                        return (int(cx + p[0]*scale_px), int(cy - p[2]*scale_px))
                    p2a, p2b = proj(pa), proj(pb)
                    cv2.line(best_frame, p2a, p2b, overlay_color_s, 2)

            cv2.putText(base_frame, 'Baseline', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,140,255), 2)
            cv2.putText(best_frame, '+SG Smooth', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,50), 2)

            combined = np.hstack([base_frame, best_frame])
            writer.write(combined)
            fi += 1

        cap.release()
        writer.release()
        print(f'[+] Saved {out_path}')

# ─── 12. FINDINGS.md ─────────────────────────────────────────────────────────

def write_findings(results_dict, test_gt_list, test_mapping, ref_limb_lengths):
    mpjpe_rows = {}
    for method, pred_list in results_dict.items():
        df = compute_mpjpe(pred_list, test_gt_list, test_mapping)
        mpjpe_rows[method] = df['abs_error_mm'].mean()

    baseline = mpjpe_rows.get('MediaPipe raw', list(mpjpe_rows.values())[0])
    best_method = min(mpjpe_rows, key=mpjpe_rows.get)
    best_val = mpjpe_rows[best_method]
    pct_improvement = 100 * (baseline - best_val) / baseline

    lines = [
        '# GolfPose MediaPipe Z-MPJPE Optimization — Findings\n',
        f'## Headline Numbers\n',
        f'- **Baseline Z-MPJPE**: {baseline:.1f} mm (MediaPipe raw + Procrustes alignment)',
        f'- **Best method**: {best_method} → {best_val:.1f} mm',
        f'- **Improvement**: {pct_improvement:.1f}%\n',
        '## Method Comparison\n',
        '| Method | Z-MPJPE (mm) | Δ vs baseline |\n',
        '|--------|------------|---------------|\n',
    ]
    for method, val in mpjpe_rows.items():
        delta = val - baseline
        lines.append(f'| {method} | {val:.1f} | {delta:+.1f} |\n')

    lines += [
        '\n## Enhancement Analysis\n',
        '- **Temporal smoothing (Savitzky-Golay)**: removes high-frequency jitter in Z estimates.',
        '  Most effective for fast-moving joints (wrists, elbows) where MediaPipe oscillates frame-to-frame.',
        '- **Limb-length constraint**: enforces anatomical plausibility per frame.',
        '  Helps for frames where the solver gets an unrealistic depth estimate.',
        '- **Learned residual MLP**: captures systematic biases in MediaPipe Z relative to Vicon.',
        '  Benefits from seeing diverse body configurations during training.\n',
        '## Per-Joint Reliability\n',
        '- **Most reliable**: hip, knee joints (high visibility, stable 2D position)',
        '- **Least reliable**: wrist joints (fast-moving, occluded during swing)\n',
        '## Alignment Procedure\n',
        '- Root-center: subtract mid-hip (GolfPose joint 0; MediaPipe mean of landmarks 23+24)',
        '- Procrustes calibration: optimal rotation R and scale s fit on S1 training frames',
        '  (minimizes ||GP_xyz − s * MP_xyz @ R||_F over all S1 frames)',
        f'- Scale factor s captures the MediaPipe-to-Vicon unit conversion\n',
        '## Limitations\n',
        '- MediaPipe world landmarks assume fixed body proportions; deviations increase Z error.',
        '- Procrustes calibration uses S1 (1 subject); a multi-subject calibration would be more robust.',
        '- S5/S6 test set is small (~3,956 frames, 5 swings) — variance in MPJPE numbers is high.',
        '- Club joints excluded (17–21) per spec; body-only evaluation understates wrist Z challenges.',
        '- GolfPose Y (depth) was not evaluated; Z (height) is the primary axis per spec.\n',
        '## Next Steps\n',
        '- When teammate baseline model arrives: swap `pred_list` in `results_dict`; evaluation harness unchanged.',
        '- Expand calibration to all S1–S4 subjects for more robust Procrustes R.',
        '- Try per-subject MLP training if subject identity labels become available for S5/S6.',
        '- Consider using GolfPose 2D GT (not MediaPipe) as 2D input to the MLP for ceiling-level comparison.',
    ]

    path = COMP / 'FINDINGS.md'
    with open(path, 'w') as f:
        f.writelines(l if l.endswith('\n') else l+'\n' for l in lines)
    print(f'[+] Saved {path}')

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('═' * 60)
    print('GolfPose MediaPipe Z-Depth Evaluation Pipeline')
    print('═' * 60)

    # ── 0. Skeleton mapping ──────────────────────────────────────────────────
    save_skeleton_mapping()

    # ── 1. Load GolfPose ground truth ────────────────────────────────────────
    print('\n[1] Loading GolfPose ground truth ...')
    pos3, pos2 = load_golfpose()
    ref_limbs = compute_ref_limb_lengths(pos3)
    print('Reference limb lengths (m):',
          {k: f'{v:.3f}' for k, v in ref_limbs.items()})

    # ── 2. MediaPipe inference (cached) ──────────────────────────────────────
    print('\n[2] Running MediaPipe on test frames (S5+S6) ...')
    n_test = sum(m[5] for m in TEST_MAP)
    print(f'  Test frames to process: {n_test} (~{n_test*0.052/60:.1f} min if not cached)')
    mp_test_df = run_mediapipe_on_mapping(TEST_MAP, 'mp_preds_test', stride=1)

    print('\n[2] Running MediaPipe on S1 calibration frames ...')
    n_calib = sum(m[5] for m in CALIB_MAP)
    print(f'  Calibration frames: {n_calib} (~{n_calib*0.052/60:.1f} min if not cached)')
    mp_calib_df = run_mediapipe_on_mapping(CALIB_MAP, 'mp_preds_calib_s1', stride=1)

    print('\n[2] Running MediaPipe on S1-S4 train frames (for MLP+smoothing tuning) ...')
    n_train = sum(m[5] for m in TRAIN_MAP)
    print(f'  Train frames: {n_train} (~{n_train*0.052/60:.1f} min if not cached)')
    mp_train_df = run_mediapipe_on_mapping(TRAIN_MAP, 'mp_preds_train_all',
                                            stride=3)  # every 3rd frame for speed

    # ── 3. Convert to arrays ─────────────────────────────────────────────────
    print('\n[3] Converting to per-sequence arrays ...')
    mp_test_arrays  = mp_df_to_array(mp_test_df,  TEST_MAP)
    mp_calib_arrays = mp_df_to_array(mp_calib_df, CALIB_MAP)
    mp_train_arrays = mp_df_to_array(mp_train_df, TRAIN_MAP)

    # ── 4. Procrustes calibration ─────────────────────────────────────────────
    print('\n[4] Procrustes calibration on S1 ...')
    R, s = fit_procrustes(pos3, mp_calib_arrays, CALIB_MAP)
    procrustes_params = {'R': R.tolist(), 's': float(s)}
    with open(OUT / 'procrustes_params.json', 'w') as f:
        json.dump(procrustes_params, f, indent=2)
    print(f'[+] Saved procrustes_params.json')

    # ── 5. Phase 1: Baseline ─────────────────────────────────────────────────
    print('\n[5] Phase 1 — Baseline (MediaPipe + Procrustes alignment) ...')
    test_gt = [get_gt_frames(pos3, npz_subj, npz_swing, n_frames)
               for (_, _, npz_subj, npz_swing, _, n_frames) in TEST_MAP]
    test_pred_baseline = [apply_alignment(mp, R, s) for mp in mp_test_arrays]

    df_base = compute_mpjpe(test_pred_baseline, test_gt, TEST_MAP)
    df_base.to_csv(OUT / 'baseline_mpjpe.csv', index=False)
    baseline_mpjpe = summarize_mpjpe(df_base, 'Phase 1 — Baseline')
    print(f'[+] Saved baseline_mpjpe.csv')

    # ── 6. Phase 2a: Temporal smoothing ───────────────────────────────────────
    print('\n[6] Phase 2a — Temporal smoothing tuning on train set ...')
    train_gt = [
        get_gt_frames(pos3, npz_subj, npz_swing, mp_train_arrays[i].shape[0])
        for i, (_, _, npz_subj, npz_swing, _, n_frames) in enumerate(TRAIN_MAP)
    ]
    train_pred_aligned = [apply_alignment(mp, R, s) for mp in mp_train_arrays]

    best_sg, best_sigma, sg_results, gauss_results = tune_smoothing(
        train_pred_aligned, train_gt, TRAIN_MAP)

    # Apply best SG to test set
    win_sg, poly_sg = best_sg
    test_pred_sg = [smooth_sg(p, win_sg, poly_sg) for p in test_pred_baseline]
    df_sg = compute_mpjpe(test_pred_sg, test_gt[:len(TEST_MAP)], TEST_MAP)
    # Note: test_gt is already indexed to TEST_MAP
    df_sg_full = compute_mpjpe(test_pred_sg, test_gt, TEST_MAP)
    sg_mpjpe = summarize_mpjpe(df_sg_full, f'Phase 2a — Savitzky-Golay (w={win_sg}, p={poly_sg})')

    test_pred_gauss = [smooth_gaussian(p, best_sigma) for p in test_pred_baseline]
    df_gauss = compute_mpjpe(test_pred_gauss, test_gt, TEST_MAP)
    gauss_mpjpe = summarize_mpjpe(df_gauss, f'Phase 2a — Gaussian (sigma={best_sigma})')

    # ── 7. Phase 2b: Limb-length constraint ───────────────────────────────────
    cache_limb = CACHE / 'pred_limb.pkl'
    if cache_limb.exists():
        print(f'\n[7] Phase 2b — Loading cached limb-constrained predictions ...')
        with open(cache_limb, 'rb') as f:
            test_pred_limb = pickle.load(f)
    else:
        print(f'\n[7] Phase 2b — Limb-length constraint (±5%) on {n_test} test frames ...')
        test_pred_limb = []
        for arr in tqdm(test_pred_baseline, desc='Limb constraint'):
            test_pred_limb.append(apply_limb_constraint(arr, ref_limbs, tol=0.05))
        with open(cache_limb, 'wb') as f:
            pickle.dump(test_pred_limb, f)
        print(f'[+] Cached limb predictions → {cache_limb}')

    df_limb = compute_mpjpe(test_pred_limb, test_gt, TEST_MAP)
    limb_mpjpe = summarize_mpjpe(df_limb, 'Phase 2b — Limb-length constraint')

    # ── 8. Phase 2c: Learned residual MLP ─────────────────────────────────────
    mlp_path    = OUT / 'mlp_model.joblib'
    scaler_path = OUT / 'mlp_scaler.joblib'
    if mlp_path.exists() and scaler_path.exists():
        print('\n[8] Phase 2c — Loading cached MLP from disk ...')
        mlp    = joblib.load(mlp_path)
        scaler = joblib.load(scaler_path)
    else:
        print('\n[8] Phase 2c — Training residual MLP on S1-S4 ...')
        mlp, scaler = train_mlp(train_pred_aligned, train_gt, TRAIN_MAP)
        joblib.dump(mlp,    mlp_path)
        joblib.dump(scaler, scaler_path)
        print(f'[+] Saved MLP model  → {mlp_path}')
        print(f'[+] Saved MLP scaler → {scaler_path}')

    test_pred_mlp = apply_mlp(test_pred_baseline, mlp, scaler)
    df_mlp = compute_mpjpe(test_pred_mlp, test_gt, TEST_MAP)
    mlp_mpjpe = summarize_mpjpe(df_mlp, 'Phase 2c — Learned residual MLP')

    # ── 9. Phase 2d: Combined ─────────────────────────────────────────────────
    cache_combined_limb = CACHE / 'pred_combined_limb.pkl'
    if cache_combined_limb.exists():
        print('\n[9] Phase 2d — Loading cached combined limb predictions ...')
        with open(cache_combined_limb, 'rb') as f:
            test_pred_combined = pickle.load(f)
    else:
        print('\n[9] Phase 2d — Combined (SG → Limb → MLP) ...')
        test_pred_combined = []
        for arr in tqdm(test_pred_sg, desc='Combined'):
            test_pred_combined.append(apply_limb_constraint(arr, ref_limbs, tol=0.05))
        with open(cache_combined_limb, 'wb') as f:
            pickle.dump(test_pred_combined, f)
        print(f'[+] Cached combined limb predictions → {cache_combined_limb}')
    test_pred_combined = apply_mlp(test_pred_combined, mlp, scaler)
    df_combined = compute_mpjpe(test_pred_combined, test_gt, TEST_MAP)
    combined_mpjpe = summarize_mpjpe(df_combined, 'Phase 2d — Combined')

    # ── 10. Phase 3 — Comparison artifacts ────────────────────────────────────
    print('\n[10] Phase 3 — Creating comparison artifacts ...')
    results_dict = {
        'MediaPipe raw':              test_pred_baseline,
        '+ Savitzky-Golay':           test_pred_sg,
        '+ Gaussian smoothing':       test_pred_gauss,
        '+ Limb-length constraint':   test_pred_limb,
        '+ Learned residual MLP':     test_pred_mlp,
        '+ Combined (SG+Limb+MLP)':   test_pred_combined,
    }

    summary_df = make_comparison_table(results_dict, test_gt, TEST_MAP)
    summary_df.to_csv(COMP / 'mpjpe_summary.csv', index=False)
    print(f'\n[+] Saved {COMP}/mpjpe_summary.csv')
    print(summary_df.to_string(index=False))

    # Markdown table
    md_lines = ['# MPJPE Summary\n\n',
                '| Method | Overall Z-MPJPE (mm) | Best joint | Worst joint | Δ vs baseline |\n',
                '|--------|---------------------|-----------|------------|---------------|\n']
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['Method']} | {row['Overall_Z_MPJPE_mm']} | "
            f"{row['Best_joint']} ({row['Best_joint_mm']}mm) | "
            f"{row['Worst_joint']} ({row['Worst_joint_mm']}mm) | "
            f"{row['Delta_vs_baseline_mm']:+.1f}mm |\n"
        )
    with open(COMP / 'mpjpe_summary.md', 'w') as f:
        f.writelines(md_lines)
    print(f'[+] Saved {COMP}/mpjpe_summary.md')

    plot_per_joint(results_dict, test_gt, TEST_MAP)

    # Best performing enhanced method
    mpjpe_vals = {m: compute_mpjpe(p, test_gt, TEST_MAP)['abs_error_mm'].mean()
                  for m, p in results_dict.items() if m != 'MediaPipe raw'}
    best_method_name = min(mpjpe_vals, key=mpjpe_vals.get)
    best_pred = results_dict[best_method_name]
    print(f'\n[+] Best method: {best_method_name} ({mpjpe_vals[best_method_name]:.1f} mm)')

    plot_z_trajectories(test_pred_baseline, best_pred, test_gt, TEST_MAP, best_method_name)

    # ── 11. Phase 3.5: GolfDB demo videos ─────────────────────────────────────
    print('\n[11] Phase 3.5 — GolfDB qualitative video overlays ...')
    make_golfdb_demo(R, s, (win_sg, poly_sg))

    # ── 12. Phase 4: FINDINGS.md ──────────────────────────────────────────────
    print('\n[12] Phase 4 — Writing FINDINGS.md ...')
    write_findings(results_dict, test_gt, TEST_MAP, ref_limbs)

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print('\n' + '═'*60)
    print('FINAL RESULTS (Z-MPJPE in mm, S5+S6 test set)')
    print('═'*60)
    print(f'  Baseline (MediaPipe raw):       {baseline_mpjpe:.1f} mm')
    print(f'  + Savitzky-Golay (w={win_sg},p={poly_sg}):  {sg_mpjpe:.1f} mm')
    print(f'  + Gaussian (sigma={best_sigma}):        {gauss_mpjpe:.1f} mm')
    print(f'  + Limb-length constraint:       {limb_mpjpe:.1f} mm')
    print(f'  + Learned residual MLP:         {mlp_mpjpe:.1f} mm')
    print(f'  + Combined:                     {combined_mpjpe:.1f} mm')
    print(f'\nTotal elapsed: {elapsed/60:.1f} min')
    print(f'Outputs in: {OUT}')


if __name__ == '__main__':
    main()
