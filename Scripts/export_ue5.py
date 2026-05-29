"""Export cached 3D landmark parquet files into UE5-ingestible formats.

Three formats:

1. **CSV** (rows = frames × joints) — the safest, dumbest, most ingestible.
   Open in Excel, import as DataTable in UE5, drive a Control Rig via
   Blueprints. Recommended for Team 2's first cut.

2. **JSON** — same data as CSV plus skeleton metadata, FPS, and a slot
   for GolfDB swing-event frame indices. Used as the canonical
   intermediate format for our own visualizations + any custom tooling.

3. **BVH** (BioVision Hierarchy) — proper industry mocap format. UE5
   imports BVH via the "Mocap Plugin" or third-party plugins like
   "BVH Importer for Unreal". Our writer is best-effort: we emit
   positional channels per joint (a permissive subset of BVH spec
   that most importers accept) so we don't have to solve the inverse
   kinematics rotation pipeline tonight. Document for Team 2 that this
   is a position-channel BVH, not a rotation-channel BVH.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import (
    COCO17_IDX,
    COCO17_NAMES,
    H36M17_IDX,
    H36M17_NAMES,
    coco17_to_h36m17,
    h36m17_to_coco17_subset,
)

# H36M-17 skeleton hierarchy (parent index per joint, -1 = root).
# Indices follow H36M17_NAMES ordering.
H36M17_PARENTS: tuple[int, ...] = (
    -1,  # 0  hip_center (root)
    0,   # 1  right_hip
    1,   # 2  right_knee
    2,   # 3  right_ankle
    0,   # 4  left_hip
    4,   # 5  left_knee
    5,   # 6  left_ankle
    0,   # 7  spine
    7,   # 8  thorax
    8,   # 9  neck
    9,   # 10 head
    8,   # 11 left_shoulder
    11,  # 12 left_elbow
    12,  # 13 left_wrist
    8,   # 14 right_shoulder
    14,  # 15 right_elbow
    15,  # 16 right_wrist
)


# ============================================================================
# Loading helpers
# ============================================================================

def load_3d_parquet_as_h36m(parquet_path: str | Path) -> np.ndarray:
    """Load a cached 3D landmark parquet and return as (T, 17, 3) in H36M-17
    ordering. The parquet was written in COCO-17 schema for metric
    compatibility (face landmarks zeroed), so we convert back."""
    df = pd.read_parquet(parquet_path)
    T = int(df["frame"].max()) + 1
    xyz_coco = np.zeros((T, 17, 3), dtype=np.float32)
    for row in df.itertuples(index=False):
        xyz_coco[int(row.frame), int(row.kp_idx), 0] = row.x
        xyz_coco[int(row.frame), int(row.kp_idx), 1] = row.y
        if "z" in df.columns:
            xyz_coco[int(row.frame), int(row.kp_idx), 2] = row.z
    return coco17_to_h36m17(xyz_coco)


def load_2d_parquet_as_h36m(parquet_path: str | Path) -> np.ndarray:
    """Load a 2D landmark parquet and project to H36M-17 ordering."""
    df = pd.read_parquet(parquet_path)
    T = int(df["frame"].max()) + 1
    xy_coco = np.zeros((T, 17, 2), dtype=np.float32)
    for row in df.itertuples(index=False):
        xy_coco[int(row.frame), int(row.kp_idx), 0] = row.x
        xy_coco[int(row.frame), int(row.kp_idx), 1] = row.y
    return coco17_to_h36m17(xy_coco)


# ============================================================================
# CSV exporter
# ============================================================================

def export_csv(xyz_h36m: np.ndarray,
                out_path: str | Path,
                fps: float = 30.0,
                events_frame_indices: Optional[Sequence[int]] = None,
                event_names: Optional[Sequence[str]] = None) -> Path:
    """One row per (frame, joint). Columns: frame_idx, time_s, joint_idx,
    joint_name, x, y, z, event_label (or empty).

    UE5 ingestion: import as a CSV Data Table with the JointPoseRow struct
    defined on the UE5 side. A small Blueprint reads each frame's rows
    and applies (x,y,z) to a Control Rig.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = xyz_h36m.shape[0]

    # Build event-label lookup (frame_idx -> event name)
    event_lookup: dict[int, str] = {}
    if events_frame_indices is not None and event_names is not None:
        for ev_name, ev_frame in zip(event_names, events_frame_indices):
            event_lookup[int(ev_frame)] = ev_name

    rows = []
    for fi in range(T):
        time_s = fi / fps
        for ji in range(17):
            rows.append({
                "frame_idx": fi,
                "time_s": round(time_s, 6),
                "joint_idx": ji,
                "joint_name": H36M17_NAMES[ji],
                "x": float(xyz_h36m[fi, ji, 0]),
                "y": float(xyz_h36m[fi, ji, 1]),
                "z": float(xyz_h36m[fi, ji, 2]),
                "event_label": event_lookup.get(fi, ""),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# ============================================================================
# JSON exporter (canonical intermediate format)
# ============================================================================

def export_json(xyz_h36m: np.ndarray,
                 out_path: str | Path,
                 fps: float = 30.0,
                 source_video: Optional[str] = None,
                 backbone_2d: Optional[str] = None,
                 lifter_3d: Optional[str] = None,
                 events_frame_indices: Optional[Sequence[int]] = None,
                 event_names: Optional[Sequence[str]] = None,
                 axis_convention: str = "h36m-camera",
                 ) -> Path:
    """Self-describing JSON containing skeleton hierarchy + per-frame
    positions + event labels + provenance. Recommended as the canonical
    intermediate format — any downstream renderer (UE5, Blender,
    matplotlib, Three.js) can ingest this.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = xyz_h36m.shape[0]

    payload = {
        "format": "golfpose-mocap-v1",
        "skeleton": {
            "name": "h36m-17",
            "joint_names": list(H36M17_NAMES),
            "joint_parents": list(H36M17_PARENTS),
            "axis_convention": axis_convention,  # see README - typically x=right, y=down, z=forward in camera frame
        },
        "fps": fps,
        "n_frames": T,
        "n_joints": 17,
        "provenance": {
            "source_video": source_video,
            "backbone_2d": backbone_2d,
            "lifter_3d": lifter_3d,
        },
        "events": [
            {"name": en, "frame_idx": int(ef)}
            for en, ef in zip(event_names or [], events_frame_indices or [])
        ],
        "frames": [
            {
                "frame_idx": fi,
                "time_s": round(fi / fps, 6),
                "positions_3d": xyz_h36m[fi].round(6).tolist(),  # (17, 3)
            }
            for fi in range(T)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    return out_path


# ============================================================================
# BVH exporter (positional channels — permissive subset)
# ============================================================================

def export_bvh(xyz_h36m: np.ndarray,
                out_path: str | Path,
                fps: float = 30.0,
                use_positional_channels: bool = True,
                ) -> Path:
    """BVH writer using POSITIONAL CHANNELS instead of the more common
    rotational channels. The BVH spec technically only allows position on
    the root joint, but most modern importers (UE5 Mocap Plugin, Blender's
    BVH importer, MotionBuilder) accept positional channels on all joints
    when present. This sidesteps the IK/FK rotation problem entirely:
    the file directly stores per-joint world-space positions.

    If the importer rejects this, swap to a rotation-channel BVH via the
    `bvh-converter` library — but that requires solving local frame
    rotations from positions, which is its own can of worms (we'd need
    `scipy.spatial.transform.Rotation` + careful handling of the
    parent-frame transformations).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = xyz_h36m.shape[0]
    rest_pose = xyz_h36m[0]  # use frame 0 as T-pose
    parents = H36M17_PARENTS

    # ---- HIERARCHY ----
    hier_lines: list[str] = []
    indent = 0

    def add(line: str):
        hier_lines.append("  " * indent + line)

    # Recursive writer
    children: dict[int, list[int]] = {i: [] for i in range(17)}
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)

    def write_joint(j: int):
        nonlocal indent
        is_root = parents[j] == -1
        name = H36M17_NAMES[j]
        offset = rest_pose[j] if is_root else (rest_pose[j] - rest_pose[parents[j]])
        if is_root:
            add(f"ROOT {name}")
        else:
            add(f"JOINT {name}")
        add("{")
        indent += 1
        add(f"OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
        if use_positional_channels:
            add("CHANNELS 3 Xposition Yposition Zposition")
        else:
            if is_root:
                add("CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
            else:
                add("CHANNELS 3 Zrotation Xrotation Yrotation")
        kids = children[j]
        if not kids:
            add("End Site")
            add("{")
            indent += 1
            # extend the bone a little for End Site
            if not is_root:
                dir_vec = rest_pose[j] - rest_pose[parents[j]]
                norm = np.linalg.norm(dir_vec) + 1e-9
                end_off = dir_vec / norm * (norm * 0.5)
            else:
                end_off = np.array([0.0, 0.1, 0.0])
            add(f"OFFSET {end_off[0]:.6f} {end_off[1]:.6f} {end_off[2]:.6f}")
            indent -= 1
            add("}")
        for c in kids:
            write_joint(c)
        indent -= 1
        add("}")

    add("HIERARCHY")
    write_joint(0)

    # ---- MOTION ----
    motion_lines: list[str] = ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
    if use_positional_channels:
        # 3 positional channels per joint × 17 joints
        for fi in range(T):
            row = []
            for j in range(17):
                p = xyz_h36m[fi, j]
                row.extend([f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])
            motion_lines.append(" ".join(row))
    else:
        raise NotImplementedError(
            "Rotational-channel BVH not implemented. Use positional channels for now."
        )

    out_path.write_text("\n".join(hier_lines + motion_lines) + "\n")
    return out_path


# ============================================================================
# Main entry point
# ============================================================================

def export_all(parquet_path: str | Path,
               out_dir: str | Path,
               clip_name: str,
               fps: float = 30.0,
               source_video: Optional[str] = None,
               backbone_2d: Optional[str] = None,
               lifter_3d: Optional[str] = None,
               events_frame_indices: Optional[Sequence[int]] = None,
               event_names: Optional[Sequence[str]] = None,
               ) -> dict[str, Path]:
    """Convenience: write all three formats for one (clip, model) parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz_h36m = load_3d_parquet_as_h36m(parquet_path)
    return {
        "csv":  export_csv(xyz_h36m, out_dir / f"{clip_name}.csv", fps,
                            events_frame_indices, event_names),
        "json": export_json(xyz_h36m, out_dir / f"{clip_name}.json", fps,
                             source_video=source_video,
                             backbone_2d=backbone_2d, lifter_3d=lifter_3d,
                             events_frame_indices=events_frame_indices,
                             event_names=event_names),
        "bvh":  export_bvh(xyz_h36m, out_dir / f"{clip_name}.bvh", fps),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="3D landmarks parquet to export")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--name", required=True, help="Output filename stem")
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()
    paths = export_all(args.parquet, args.out, args.name, fps=args.fps)
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
