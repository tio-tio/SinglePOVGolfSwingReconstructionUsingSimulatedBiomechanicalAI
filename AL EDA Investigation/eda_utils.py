"""
Shared helpers for GolfDB EDA notebooks.
Import this from any notebook in AL EDA Investigation/.
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import cv2
import mediapipe as mp

# ── Constants ─────────────────────────────────────────────────────────────────

SWING_EVENT_NAMES = [
    'Address', 'Toe-up', 'Mid-backswing', 'Top',
    'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish',
]

ALL_EVENT_NAMES = [
    'clip_start', 'address', 'toe_up', 'mid_backswing', 'top',
    'mid_downswing', 'impact', 'mid_follow_through', 'finish', 'clip_end',
]

KEY_JOINTS = {
    11: 'L_Shoulder', 12: 'R_Shoulder',
    13: 'L_Elbow',    14: 'R_Elbow',
    15: 'L_Wrist',    16: 'R_Wrist',
    23: 'L_Hip',      24: 'R_Hip',
    25: 'L_Knee',     26: 'R_Knee',
}

# (proximal_idx, distal_idx, limb_name)
LIMB_PAIRS = [
    (11, 13, 'L_UpperArm'),
    (13, 15, 'L_Forearm'),
    (12, 14, 'R_UpperArm'),
    (14, 16, 'R_Forearm'),
    (23, 25, 'L_Thigh'),
    (24, 26, 'R_Thigh'),
    (11, 12, 'Shoulder_Width'),
    (23, 24, 'Hip_Width'),
]

# ── Data loading ──────────────────────────────────────────────────────────────

def unwrap(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return x


def load_golfdb(mat_path):
    """Load golfDB.mat and return a flat DataFrame (1400 rows)."""
    mat = sio.loadmat(mat_path)
    golfdb = mat['golfDB']
    rows = [
        {field: unwrap(golfdb[0, i][field]) for field in golfdb.dtype.names}
        for i in range(golfdb.shape[1])
    ]
    return pd.DataFrame(rows)

# ── Event helpers ─────────────────────────────────────────────────────────────

def events_to_relative(events_raw):
    """10-element int array, shifted so clip_start = 0."""
    e = np.array(events_raw).flatten().astype(int)
    return e - e[0]


def events_to_dict(events_raw):
    """Return {all_event_name: relative_frame} for all 10 events."""
    return dict(zip(ALL_EVENT_NAMES, events_to_relative(events_raw).tolist()))


def swing_event_frames(events_raw):
    """8 relative frame indices for the named swing events (no clip boundaries)."""
    return events_to_relative(events_raw)[1:9]


def tempo_stats(events_raw):
    """
    Return (backswing_frames, downswing_frames, tempo_ratio) where
    tempo_ratio = backswing / downswing.
    Events: Address=idx1, Top=idx4, Impact=idx6 in the raw 10-element array.
    """
    e = np.array(events_raw).flatten().astype(int)
    address = e[1]
    top     = e[4]
    impact  = e[6]
    back  = int(top - address)
    down  = int(impact - top)
    ratio = back / down if down > 0 else np.nan
    return back, down, ratio

# ── Video helpers ─────────────────────────────────────────────────────────────

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_swing_frames(video_path, events_raw):
    """Return list of 8 RGB ndarrays at each swing event frame."""
    frames_idx = swing_event_frames(events_raw)
    cap = cv2.VideoCapture(video_path)
    out = []
    for idx in frames_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        out.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
            else np.zeros((160, 120, 3), dtype=np.uint8)
        )
    cap.release()
    return out

# ── Bounding-box helpers ──────────────────────────────────────────────────────

def parse_bbox(bbox_raw):
    """
    Return (x_min, y_min, x_max, y_max) as floats from the MATLAB bbox field.
    GolfDB stores bbox as [[x_min, y_min, x_max, y_max]] in normalised [0,1] coords.
    """
    b = np.array(bbox_raw).flatten()
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def bbox_stats(bbox_raw):
    """Return dict with width, height, area, aspect_ratio, cx, cy."""
    x0, y0, x1, y1 = parse_bbox(bbox_raw)
    w, h = x1 - x0, y1 - y0
    return {
        'width':        w,
        'height':       h,
        'area':         w * h,
        'aspect_ratio': w / h if h > 0 else np.nan,
        'cx':           (x0 + x1) / 2,
        'cy':           (y0 + y1) / 2,
    }

# ── MediaPipe extraction with caching ────────────────────────────────────────

_mp_pose = mp.solutions.pose


def extract_world_landmarks_at_events(record, video_dir, cache_dir=None):
    """
    Run MediaPipe (world landmarks) at each of the 8 swing-event frames.
    If cache_dir is set, saves per-record CSVs so reruns skip inference.
    Returns a DataFrame with columns:
      record_id, player, club, view, slow,
      event_idx, event_name, joint_idx, x, y, z, visibility
    """
    record_id = int(record['id'])

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'landmarks_{record_id}.csv')
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)

    video_path = os.path.join(video_dir, f'{record_id}.mp4')
    ev_frames  = swing_event_frames(record['events'])

    rows = []
    cap  = cv2.VideoCapture(video_path)

    with _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        for event_idx, frame_num in enumerate(ev_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
            ret, frame = cap.read()
            if not ret:
                continue
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if not results.pose_world_landmarks:
                continue
            for joint_idx, lm in enumerate(results.pose_world_landmarks.landmark):
                rows.append({
                    'record_id':  record_id,
                    'player':     record['player'],
                    'club':       record['club'],
                    'view':       record['view'],
                    'slow':       int(record['slow']),
                    'event_idx':  event_idx,
                    'event_name': SWING_EVENT_NAMES[event_idx],
                    'joint_idx':  joint_idx,
                    'x': lm.x, 'y': lm.y, 'z': lm.z,
                    'visibility': lm.visibility,
                })

    cap.release()
    result_df = pd.DataFrame(rows)

    if cache_dir and not result_df.empty:
        result_df.to_csv(cache_path, index=False)

    return result_df
