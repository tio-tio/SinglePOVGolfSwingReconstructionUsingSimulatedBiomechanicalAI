"""Pipeline step: ball tracking + flight fit + visual artifacts.

Runs after the scorecard step (needs events.impact) and after pipeline.py
(needs landmarks CSV, replay JSON, overlay.mp4). Produces:
  <stem>_ball_3d.json   — track + fit + trajectories (ball_track.analyze output
                          plus `trajectory_replay` in the replay viewer frame)
  overlay.mp4 (updated) — tracked ball drawn as a dot with a fading trail

Absence of a confident track is not an error: the artifact is still written
with quality:"simulated" and the overlay is left untouched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ball_track import analyze

L_ANKLE, R_ANKLE = 6, 3          # replay joint order (see web_artifacts joint_names)
GOLFER_M = 1.75


def trajectory_replay(ball: dict, replay_path: Path) -> list | None:
    """Map the camera-frame ball trajectory into the replay viewer's frame.

    Replay and camera share the h36m-camera axis convention (x right, y DOWN,
    z depth); the replay is additionally leveled by <=20 deg — small enough to
    skip for the ball arc (v1). Anchor: mid-ankle at impact sits at the tee;
    scale: golfer standing height in replay units / GOLFER_M.
    """
    cam = ball.get("trajectory_cam_m")
    if not cam:
        return None
    replay = json.loads(replay_path.read_text())
    frames = replay.get("frames")
    if not frames:
        return None
    F = np.array(frames, dtype=float)               # [T, J, 3]
    impact = min(int(ball["impact_frame"]), len(F) - 1)
    anchor = (F[impact, L_ANKLE] + F[impact, R_ANKLE]) / 2
    spans = F[:, :, 1].max(axis=1) - F[:, :, 1].min(axis=1)
    units_per_m = float(np.percentile(spans, 95)) / GOLFER_M
    out = []
    for t, x, y, z in cam:
        p = anchor + np.array([x, y, z]) * units_per_m
        out.append([round(float(t), 3), round(float(p[0]), 4),
                    round(float(min(p[1], 0.0)), 4), round(float(p[2]), 4)])
    return out


def redraw_overlay(overlay_mp4: Path, track: list, fps: float) -> bool:
    """Re-encode overlay.mp4 with the tracked ball (dot + fading trail)."""
    cap = cv2.VideoCapture(str(overlay_mp4))
    if not cap.isOpened():
        return False
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps
    by_frame = {int(f): (x, y) for f, x, y in track}
    tmp = overlay_mp4.with_suffix(".ball.tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        trail = [(f, xy) for f, xy in by_frame.items() if fi - 12 <= f <= fi]
        for f, (x, y) in trail:
            age = fi - f
            alpha = max(0.15, 1.0 - age / 12.0)
            col = (0, int(200 * alpha + 55), int(255 * alpha))
            rad = 6 if age == 0 else max(2, 4 - age // 4)
            cv2.circle(frame, (int(round(x)), int(round(y))), rad, col,
                       -1 if age == 0 else 1, cv2.LINE_AA)
        if fi in by_frame:
            x, y = by_frame[fi]
            cv2.circle(frame, (int(round(x)), int(round(y))), 9, (255, 255, 255), 1,
                       cv2.LINE_AA)
        writer.write(frame)
        fi += 1
    writer.release()
    cap.release()
    # H.264 for browser playback (same recipe as the pipeline's overlay writer)
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        final = overlay_mp4.with_suffix(".ball.mp4")
        subprocess.run([ff, "-y", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                        "-loglevel", "error", str(final)], check=True, timeout=300)
        tmp.unlink()
        final.replace(overlay_mp4)
        return True
    except Exception as e:                           # noqa: BLE001 — non-fatal step
        print(f"[ball_step] overlay H.264 re-encode failed ({e}); overlay unchanged")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--scorecard")
    ap.add_argument("--replay", help="replay_3d.json (for trajectory_replay)")
    ap.add_argument("--overlay", help="overlay.mp4 to redraw with the ball")
    ap.add_argument("--out", required=True, help="output ball_3d.json path")
    ap.add_argument("--club")
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--golfer-height-m", type=float, default=GOLFER_M)
    a = ap.parse_args()

    res = analyze(a.video, a.landmarks, a.scorecard, a.club, a.speed_scale,
                  a.golfer_height_m)
    if a.replay and Path(a.replay).exists():
        tr = trajectory_replay(res, Path(a.replay))
        if tr:
            res["trajectory_replay"] = tr
            # inject into the replay JSON so 3D viewers can animate the ball
            # without fetching a second artifact
            try:
                rp = Path(a.replay)
                replay = json.loads(rp.read_text())
                replay["ball"] = {"quality": res.get("quality"),
                                  "impact_frame": res.get("impact_frame"),
                                  "trajectory": tr}
                rp.write_text(json.dumps(replay))
                print(f"[ball_step] ball trajectory injected into {rp.name}")
            except (OSError, json.JSONDecodeError) as e:
                print(f"[ball_step] replay injection skipped: {e}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[ball_step] quality={res.get('quality')} "
          f"n={res.get('n_track_points', 0)} -> {a.out}")

    track = res.get("track_2d") or []
    if a.overlay and track and Path(a.overlay).exists():
        if redraw_overlay(Path(a.overlay), track, float(res.get("fps") or 30.0)):
            print(f"[ball_step] overlay updated with ball trail: {a.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
