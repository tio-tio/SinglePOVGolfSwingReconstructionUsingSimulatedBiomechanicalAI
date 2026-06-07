"""End-to-end golf swing 3D mocap pipeline for Unreal Engine handoff.

Run on a single phone video (or any mp4) and produce a folder containing:

  <stem>_landmarks_2d.csv     Per-frame COCO-17 2D landmarks (pixel coords)
  <stem>_landmarks_3d.csv     Per-frame H36M-17 3D positions (lifter output)
  <stem>_animation.bvh        BioVision Hierarchy mocap file (UE5/Blender/MotionBuilder)
  <stem>_mocap.json           Canonical JSON with skeleton + frames + provenance
  <stem>_overlay.mp4          Visualization: 2D landmarks drawn on the original video
  <stem>_preview_3d.html      Standalone 3D viewer (Three.js, opens in browser)

Usage:

  python pipeline.py path/to/swing.mp4
  python pipeline.py path/to/swing.mp4 --backbone vitpose_base --lifter motionbert_full
  python pipeline.py path/to/swing.mp4 --out-dir Data/handoff/swing_123 --no-overlay
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import COCO17_NAMES, COCO17_IDX, video_info, coco17_to_h36m17
from adapters import ADAPTER_REGISTRY
from export_ue5 import (
    H36M17_NAMES, H36M17_PARENTS,
    load_3d_parquet_as_h36m, load_2d_parquet_as_h36m,
    export_csv, export_json, export_bvh,
)
from smoothing import smooth_sequence


def _mean_acceleration(xyz: np.ndarray) -> float:
    """Mean frame-to-frame acceleration magnitude over all joints (jitter proxy,
    same second-derivative notion as eval_utils.jitter but on the H36M array)."""
    if xyz.shape[0] < 3:
        return float("nan")
    a = np.diff(xyz, n=2, axis=0)              # (T-2, 17, 3)
    return float(np.linalg.norm(a, axis=2).mean())


def _load_3d_conf_as_h36m(parquet_3d: Path, n_frames: int) -> np.ndarray | None:
    """Read the conf column from a 3D parquet and project to H36M-17 ordering.
    Returns (T, 17) or None if the parquet has no conf column."""
    df = pd.read_parquet(parquet_3d)
    if "conf" not in df.columns:
        return None
    conf_coco = np.zeros((n_frames, 17, 1), dtype=np.float32)
    for row in df.itertuples(index=False):
        conf_coco[int(row.frame), int(row.kp_idx), 0] = float(row.conf)
    return coco17_to_h36m17(conf_coco)[..., 0]


# COCO links for overlay rendering
COCO_LINKS = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
              (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]


def run_2d(video_path: Path, backbone: str, cache_dir: Path) -> Path:
    """Run a 2D pose adapter on the video, cache landmarks parquet, return path."""
    adapter = ADAPTER_REGISTRY[backbone]()
    cache_subdir = cache_dir / backbone
    cache_subdir.mkdir(parents=True, exist_ok=True)
    parquet_path = cache_subdir / f"{video_path.stem}.parquet"
    print(f"  [2d:{backbone}] running ...")
    t0 = time.perf_counter()
    res = adapter.predict_and_cache(video_path, cache_dir, overwrite=False)
    print(f"  [2d:{backbone}] {res.n_frames} frames, {res.n_frames_detected} detected, "
          f"{(time.perf_counter()-t0):.1f}s wall, {res.seconds_per_frame*1000:.1f} ms/frame")
    return parquet_path


def run_3d_lift(video_path: Path, lifter: str, upstream_2d: str, cache_dir: Path) -> Path:
    """Run a 3D-lifter adapter on top of an upstream 2D model's cache."""
    # The adapter name in the registry already encodes the upstream binding.
    full_name = f"{lifter}_from_{upstream_2d}"
    if full_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"No registered adapter '{full_name}'. Available 3D lifters:\n  " +
            "\n  ".join(n for n in ADAPTER_REGISTRY if n.startswith(("motionbert_", "golfpose")))
        )
    adapter = ADAPTER_REGISTRY[full_name]()
    print(f"  [3d:{full_name}] running ...")
    t0 = time.perf_counter()
    res = adapter.predict_and_cache(video_path, cache_dir, overwrite=False)
    print(f"  [3d:{full_name}] {res.n_frames} frames, "
          f"{(time.perf_counter()-t0):.1f}s wall, {res.seconds_per_frame*1000:.2f} ms/frame")
    return cache_dir / full_name / f"{video_path.stem}.parquet"


# ---------------------------------------------------------------------------
# 2D overlay video (matches coworker's pipeline)
# ---------------------------------------------------------------------------

def write_2d_overlay(video_path: Path, landmarks_2d_parquet: Path, out_mp4: Path):
    df = pd.read_parquet(landmarks_2d_parquet)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    by_frame = {fi: g for fi, g in df.groupby("frame")}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fi in by_frame:
            kp = {int(r.kp_idx): (int(r.x), int(r.y), float(r.conf))
                  for r in by_frame[fi].itertuples()}
            for a, b in COCO_LINKS:
                if a in kp and b in kp and kp[a][2] >= 0.3 and kp[b][2] >= 0.3:
                    cv2.line(frame, (kp[a][0], kp[a][1]), (kp[b][0], kp[b][1]),
                              (50, 220, 50), 2, cv2.LINE_AA)
            for ki, (x, y, c) in kp.items():
                if c >= 0.3:
                    cv2.circle(frame, (x, y), 3, (0, 0, 220), -1, cv2.LINE_AA)
        writer.write(frame)
        fi += 1
    cap.release(); writer.release()

    # h264 conversion for browser playback
    import shutil, subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        try:
            subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-loglevel", "error", str(out_mp4)],
                            check=True, timeout=180)
            tmp.unlink()
            return
        except Exception as e:
            print(f"  [overlay] ffmpeg failed: {e}; keeping mp4v")
    tmp.rename(out_mp4)


# ---------------------------------------------------------------------------
# 3D preview as standalone HTML (Three.js)
# ---------------------------------------------------------------------------

_HTML_VIEWER = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{TITLE}</title>
<style>body{margin:0;font-family:system-ui;background:#222;color:#ddd}
#hud{position:absolute;top:8px;left:8px;background:#0008;padding:8px;border-radius:4px;z-index:1}
#hud button{margin-right:4px}</style>
</head><body>
<div id="hud">
  <button onclick="paused=!paused">⏯</button>
  <span id="frame">0 / 0</span>
  &nbsp;<span id="event"></span>
</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.166.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.166.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = {DATA_JSON};
const PARENTS = {PARENTS_JSON};
const NAMES = {NAMES_JSON};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);
const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100);
camera.position.set(2, 2, 3);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const ctrl = new OrbitControls(camera, renderer.domElement);

const grid = new THREE.GridHelper(4, 16, 0x444444, 0x333333);
scene.add(grid);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(2, 4, 2); scene.add(dl);

const jointGeo = new THREE.SphereGeometry(0.025, 12, 12);
const jointMat = new THREE.MeshStandardMaterial({color: 0xff5555});
const joints = NAMES.map(_ => {
  const m = new THREE.Mesh(jointGeo, jointMat);
  scene.add(m); return m;
});

const lineMat = new THREE.LineBasicMaterial({color: 0x55ff55, linewidth: 3});
const lines = [];
for (let i = 0; i < PARENTS.length; i++) {
  if (PARENTS[i] < 0) continue;
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const ln = new THREE.Line(geo, lineMat); scene.add(ln);
  lines.push({child: i, parent: PARENTS[i], geo});
}

let fi = 0, paused = false; window.paused = false;
const T = DATA.frames.length;
const fps = DATA.fps || 30;
let last = performance.now();

function setFrame(f) {
  const frame = DATA.frames[f];
  for (let j = 0; j < 17; j++) {
    const p = frame.positions_3d[j];
    joints[j].position.set(p[0], -p[1], p[2]);  // flip Y for screen up
  }
  for (const ln of lines) {
    const a = joints[ln.parent].position, b = joints[ln.child].position;
    ln.geo.setFromPoints([a, b]); ln.geo.attributes.position.needsUpdate = true;
  }
  document.getElementById('frame').textContent = `${f+1} / ${T}`;
  const ev = (DATA.events || []).find(e => e.frame_idx === f);
  document.getElementById('event').textContent = ev ? `[${ev.name}]` : '';
}

function tick() {
  const now = performance.now();
  if (!window.paused && now - last >= 1000 / fps) {
    fi = (fi + 1) % T; setFrame(fi); last = now;
  }
  ctrl.update(); renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
setFrame(0); tick();
addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script></body></html>
"""

def write_3d_preview_html(xyz_h36m: np.ndarray, out_html: Path, fps: float,
                           events_frame_indices=None, event_names=None,
                           title: str = "Golf swing 3D preview"):
    import json as _json
    data = {
        "fps": fps,
        "frames": [{"frame_idx": fi,
                     "positions_3d": xyz_h36m[fi].round(5).tolist()}
                   for fi in range(xyz_h36m.shape[0])],
        "events": [{"name": en, "frame_idx": int(ef)}
                   for en, ef in zip(event_names or [], events_frame_indices or [])],
    }
    html = (_HTML_VIEWER
            .replace("{TITLE}", title)
            .replace("{DATA_JSON}", _json.dumps(data))
            .replace("{PARENTS_JSON}", _json.dumps(list(H36M17_PARENTS)))
            .replace("{NAMES_JSON}", _json.dumps(list(H36M17_NAMES))))
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to phone-video MP4 (or any video)")
    p.add_argument("--out-dir", default=None,
                   help="Output dir. Default: Data/handoff/<video_stem>/")
    p.add_argument("--backbone", default="mediapipe_heavy",
                   help="2D pose backbone. Choices in adapters/__init__.py.")
    p.add_argument("--lifter", default="motionbert_full",
                   help="3D lifter: motionbert_full (default, best PCE), motionbert_lite (cheaper), golfpose3d")
    p.add_argument("--cache-dir", default=None,
                   help="Where to cache adapter outputs. Default: <project>/Data/eval_runs")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip the 2D overlay MP4 (faster)")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip the 3D HTML preview")
    p.add_argument("--fps", type=float, default=None,
                   help="Override video FPS (e.g. slow-mo clips with fake fps metadata)")
    p.add_argument("--events", type=str, default=None,
                   help="Optional CSV of 8 comma-separated GolfDB-style swing event frame indices")
    p.add_argument("--smooth", choices=["none", "oneeuro", "savgol"], default="oneeuro",
                   help="Temporal smoothing of 3D output before UE export (default: oneeuro)")
    p.add_argument("--smooth-min-cutoff", type=float, default=1.0,
                   help="One Euro min_cutoff: lower = smoother slow phases (more lag)")
    p.add_argument("--smooth-beta", type=float, default=0.3,
                   help="One Euro beta: higher = snappier fast phases / impact (less lag)")
    p.add_argument("--smooth-window", type=int, default=7,
                   help="Savgol window length (odd; clamped to clip length)")
    p.add_argument("--no-bone-lock", action="store_true",
                   help="Disable bone-length stabilization (rigid skeleton pass)")
    args = p.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    project_root = Path(__file__).parent.parent
    cache_dir = Path(args.cache_dir) if args.cache_dir else project_root / "Data" / "eval_runs"
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "Data" / "handoff" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    info = video_info(video_path)
    fps = args.fps or info["fps"] or 30.0
    print(f"[pipeline] {video_path.name}  {info['n_frames']} frames @ {fps:.1f} fps")

    events_frame_indices = None
    event_names = None
    if args.events:
        events_frame_indices = [int(x) for x in args.events.split(",")]
        event_names = ["address","toe_up","mid_backswing","top","mid_downswing",
                       "impact","mid_follow_through","finish"][:len(events_frame_indices)]

    # --- 1. 2D pose ---
    parquet_2d = run_2d(video_path, args.backbone, cache_dir)
    # Copy as the deliverable CSV (in raw COCO-17 form for transparency)
    df2 = pd.read_parquet(parquet_2d)
    csv2d_path = out_dir / f"{video_path.stem}_landmarks_2d.csv"
    df2.to_csv(csv2d_path, index=False)
    print(f"[pipeline] wrote {csv2d_path.name}")

    # --- 2. 3D lift ---
    parquet_3d = run_3d_lift(video_path, args.lifter, args.backbone, cache_dir)
    xyz_h36m = load_3d_parquet_as_h36m(parquet_3d)
    print(f"[pipeline] 3D output: shape {xyz_h36m.shape}, "
          f"range [{xyz_h36m.min():.3f}, {xyz_h36m.max():.3f}]")

    # --- 2b. Smoothing (de-jitter before UE handoff) ---
    if args.smooth != "none" or not args.no_bone_lock:
        conf_h36m = _load_3d_conf_as_h36m(parquet_3d, xyz_h36m.shape[0])
        jitter_before = _mean_acceleration(xyz_h36m)
        xyz_h36m = smooth_sequence(
            xyz_h36m, conf=conf_h36m, method=args.smooth, fps=fps,
            bone_lock=not args.no_bone_lock,
            min_cutoff=args.smooth_min_cutoff, beta=args.smooth_beta,
            window=args.smooth_window,
        )
        jitter_after = _mean_acceleration(xyz_h36m)
        bone = "off" if args.no_bone_lock else "on"
        pct = (1.0 - jitter_after / jitter_before) * 100.0 if jitter_before else 0.0
        print(f"[pipeline] smoothing: {args.smooth} (bone-lock {bone})  "
              f"jitter {jitter_before:.5f} -> {jitter_after:.5f}  ({pct:+.1f}%)")

    # --- 3. UE5-format exports ---
    full_lifter_name = f"{args.lifter}_from_{args.backbone}"
    paths = {}
    paths["csv_3d"]  = export_csv(xyz_h36m, out_dir / f"{video_path.stem}_landmarks_3d.csv",
                                    fps=fps,
                                    events_frame_indices=events_frame_indices,
                                    event_names=event_names)
    paths["json"]    = export_json(xyz_h36m, out_dir / f"{video_path.stem}_mocap.json",
                                    fps=fps, source_video=str(video_path),
                                    backbone_2d=args.backbone, lifter_3d=full_lifter_name,
                                    events_frame_indices=events_frame_indices,
                                    event_names=event_names)
    paths["bvh"]     = export_bvh(xyz_h36m, out_dir / f"{video_path.stem}_animation.bvh", fps=fps)
    for k, v in paths.items():
        print(f"[pipeline] wrote {v.name}  ({v.stat().st_size//1024} KB)  [{k}]")

    # --- 4. Visualizations ---
    if not args.no_overlay:
        overlay_path = out_dir / f"{video_path.stem}_overlay.mp4"
        write_2d_overlay(video_path, parquet_2d, overlay_path)
        print(f"[pipeline] wrote {overlay_path.name}")
    if not args.no_preview:
        html_path = out_dir / f"{video_path.stem}_preview_3d.html"
        write_3d_preview_html(xyz_h36m, html_path, fps=fps,
                               events_frame_indices=events_frame_indices,
                               event_names=event_names,
                               title=f"{video_path.stem} - {full_lifter_name}")
        print(f"[pipeline] wrote {html_path.name}")

    print(f"\n[pipeline] DONE. All files in: {out_dir}")
    print("\nFor Team 2 UE5 ingestion:")
    print(f"  - Recommended:   {paths['csv_3d'].name}  (open in Excel; import as DataTable in UE5)")
    print(f"  - Canonical:     {paths['json'].name}  (richest metadata)")
    print(f"  - Mocap standard: {paths['bvh'].name}  (drag into UE5 Mocap Plugin or Blender)")


if __name__ == "__main__":
    main()
