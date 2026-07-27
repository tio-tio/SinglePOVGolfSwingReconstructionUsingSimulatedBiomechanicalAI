"""Offline tests for video_orientation — synthesises its own clips, no network.

    .venv/Scripts/python.exe Scripts/test_video_orientation.py

Fixtures mimic what a phone actually writes: the pixels are stored rotated a
quarter turn at a time, plus a display matrix that tells the player how to put
them back. So the ground truth is known exactly — every fixture must come back
looking like the base clip — which pins the rotation DIRECTION, not just that
the two code paths agree with each other. (Direction is the easy thing to get
backwards: a 90-degree fix applied the wrong way lands 180 degrees out and
still looks self-consistent.)

The other load-bearing assertion is EQUIVALENCE: normalize_video() (the ingest
bake) and open_capture() (the reader safety net) must hand back the same
pixels, or a clip that goes through one path on the server and the other on a
laptop lands in a different orientation than the metrics were tuned on.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_orientation import normalize_video, open_capture, probe_rotation

_PASS = _FAIL = 0
# fixtures are re-encoded twice (build + bake), so compare tolerantly; a wrong
# rotation is off by tens of levels, far above this floor
_TOL = 8.0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {extra}"))


def _ffmpeg():
    from video_orientation import _ffmpeg_exe
    return _ffmpeg_exe()


def _first_frame(path, auto=False):
    cap = open_capture(path) if auto else cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _diff(a, b):
    """Mean abs pixel difference, or None if the shapes disagree."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.mean(np.abs(a.astype(int) - b.astype(int))))


# pixels stored k quarter-turns clockwise, and the matching display rotation
# that undoes it. k=0 is the untagged desktop case.
_TURNS = {
    0: ([], "0"),
    1: (["-vf", "transpose=1"], "90"),
    2: (["-vf", "transpose=1,transpose=1"], "180"),
    3: (["-vf", "transpose=2"], "270"),
}


def _build_fixtures(ff, work: Path) -> tuple[Path, dict[int, Path]]:
    base = work / "base.mp4"
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=640x360:rate=30:duration=1",
         "-pix_fmt", "yuv420p", str(base)], check=True, capture_output=True)
    out = {}
    for k, (vf, tag) in _TURNS.items():
        stored = work / f"stored{k}.mp4"
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", str(base), *vf,
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(stored)],
            check=True, capture_output=True)
        if k == 0:
            out[k] = stored
            continue
        tagged = work / f"phone{k}.mp4"
        # -display_rotation writes the matrix without touching pixels (-c copy),
        # exactly how a phone tags a portrait recording
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-display_rotation", tag,
             "-i", str(stored), "-c", "copy", str(tagged)],
            check=True, capture_output=True)
        out[k] = tagged
    return base, out


def main() -> int:
    ff = _ffmpeg()
    if ff is None:
        print("SKIP: no ffmpeg available to synthesise fixtures")
        return 0

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        base, fx = _build_fixtures(ff, work)
        base_frame = _first_frame(base)
        bh, bw = base_frame.shape[:2]

        print("probe_rotation")
        check("untagged clip probes as 0", probe_rotation(fx[0]) == 0,
              f"got {probe_rotation(fx[0])}")
        for k in (1, 2, 3):
            r = probe_rotation(fx[k])
            check(f"{k*90}deg-stored clip probes a quarter turn",
                  r in (90, 180, 270), f"got {r}")

        print("open_capture: corrected pixels match the known original")
        for k, path in fx.items():
            frame = _first_frame(path, auto=True)
            d = _diff(frame, base_frame)
            check(f"{k*90}deg-stored clip reads back as the base clip",
                  d is not None and d < _TOL,
                  f"shape={None if frame is None else frame.shape[:2]} "
                  f"(base {(bh, bw)}) mean_abs_diff={d}")

        print("normalize_video: baked pixels match the known original")
        for k, path in fx.items():
            dst = work / f"norm{k}.mp4"
            out, reported = normalize_video(path, dst)
            if k == 0:
                check("untagged clip passes through untouched, no transcode",
                      out == path and reported == 0 and not dst.exists(),
                      f"out={out.name} reported={reported} made_file={dst.exists()}")
                continue
            check(f"{k*90}deg: reports the source rotation", reported == probe_rotation(path))
            check(f"{k*90}deg: output carries NO residual rotation",
                  probe_rotation(out) == 0, f"got {probe_rotation(out)}")
            d = _diff(_first_frame(out), base_frame)
            check(f"{k*90}deg: baked clip reads back as the base clip",
                  d is not None and d < _TOL, f"mean_abs_diff={d}")

        print("equivalence: bake == auto-rotate (what lets the two paths compose)")
        for k in (1, 2, 3):
            d = _diff(_first_frame(work / f"norm{k}.mp4"), _first_frame(fx[k], auto=True))
            check(f"{k*90}deg: baked pixels match auto-rotated pixels",
                  d is not None and d < _TOL, f"mean_abs_diff={d}")

        print("idempotence: a normalised clip must not rotate a second time")
        for k in (1, 2, 3):
            once = work / f"norm{k}.mp4"
            out, reported = normalize_video(once, work / f"twice{k}.mp4")
            check(f"{k*90}deg: re-normalising is a no-op",
                  out == once and reported == 0, f"out={out.name} reported={reported}")
            d = _diff(_first_frame(once, auto=True), base_frame)
            check(f"{k*90}deg: open_capture leaves a baked clip alone",
                  d is not None and d < _TOL, f"mean_abs_diff={d}")

    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
