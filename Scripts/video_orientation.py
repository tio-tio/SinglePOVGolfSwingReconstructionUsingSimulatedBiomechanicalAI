"""Camera-rotation normalisation for phone uploads.

Phones record in sensor orientation and tag the stream with a display-matrix
rotation (90/180/270); the stored pixels are sideways and the player is
expected to undo it. OpenCV's FFmpeg backend leaves CAP_PROP_ORIENTATION_AUTO
OFF by default, so a plain cv2.VideoCapture hands back the sideways frames and
reports the sideways CAP_PROP_FRAME_WIDTH/HEIGHT. That broke phone uploads
three ways at once:

  * MediaPipe saw a golfer lying on their side, so 2D detection degraded,
  * every "vertical" assumption downstream (spine angle, stance grounding,
    the height-relative jitter scale, the ball's gravity fit) was measured
    against the wrong axis,
  * cv2.VideoWriter writes overlay.mp4 with no display matrix at all, so the
    browser could not undo the rotation either -- the sideways player.

Two layers, both keyed off the same probe:

  normalize_video()  bakes the rotation into the pixels ONCE, at ingest (the
                     deployed path: processing_handler calls it before the
                     pipeline runs), so every consumer downstream is
                     orientation-free by construction -- including scripts
                     written later that know nothing about this module.
  open_capture()     turns ORIENTATION_AUTO on for direct cv2 readers -- the
                     safety net for anyone running pipeline.py straight at a
                     phone file without going through ingest.

The two produce pixel-identical frames (asserted in test_video_orientation.py),
so composing them is safe: a normalised file probes as rotation 0, which makes
open_capture a no-op rather than a second rotation.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2

__all__ = ["probe_rotation", "open_capture", "normalize_video"]

# Re-encode quality for the bake. Near-visually-lossless: the ball tracker
# works off small, low-contrast pixel signals, so this is deliberately well
# above the usual "good enough for streaming" crf.
_CRF = "18"
_PRESET = "veryfast"


def _ffmpeg_exe() -> str | None:
    """System ffmpeg, else the static binary imageio-ffmpeg vendors (which is
    what ships in the processing container)."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        return None


def _rotation_from_ffprobe(video: Path) -> int | None:
    """Fallback probe for backends that don't expose the display matrix.

    Only consulted when the cv2 probe can't answer -- imageio-ffmpeg vendors
    ffmpeg but NOT ffprobe, so in the container this returns None and the cv2
    reading (FFmpeg backend there) stands on its own.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        res = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-of", "json", str(video)],
            capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return None
        stream = (json.loads(res.stdout).get("streams") or [{}])[0]
        # modern ffmpeg reports it as display-matrix side data; older builds
        # only set the legacy "rotate" tag
        for side in stream.get("side_data_list") or []:
            if "rotation" in side:
                return int(round(float(side["rotation"]))) % 360
        tag = (stream.get("tags") or {}).get("rotate")
        return int(round(float(tag))) % 360 if tag is not None else None
    except Exception:
        return None


def probe_rotation(video: str | Path) -> int:
    """Display-matrix rotation in degrees, normalised to one of 0/90/180/270.

    This is the angle OpenCV's own auto-rotation applies, so it is also the
    angle normalize_video has to bake in for the two paths to agree.
    """
    video = Path(video)
    rot: float | None = None
    try:
        cap = cv2.VideoCapture(str(video))
        if cap.isOpened():
            rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        cap.release()
    except Exception:
        rot = None
    if rot is None or rot != rot:                       # unopenable, or NaN
        rot = _rotation_from_ffprobe(video)
    if not rot:                                          # None, 0, or 0.0
        return 0
    # cameras write -90 as readily as 270; snap to the nearest quarter turn so
    # an off-by-a-fraction tag can never fall through as "no rotation"
    return int(round(float(rot) / 90.0) * 90) % 360


def open_capture(video: str | Path) -> cv2.VideoCapture:
    """cv2.VideoCapture that honours the display matrix.

    Frames come back upright and CAP_PROP_FRAME_WIDTH/HEIGHT are swapped to
    match, so callers that size a VideoWriter from the same handle stay
    consistent. The flag has to be set AFTER open -- passing it through the
    open-params overload is rejected by the FFmpeg backend.

    A no-op on files with no rotation tag, which includes everything this repo
    writes itself and anything already through normalize_video().
    """
    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        # ignore the return: backends without the property (MSMF, GStreamer)
        # simply keep their existing behaviour rather than failing the open
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def normalize_video(src: str | Path, dst: str | Path | None = None) -> tuple[Path, int]:
    """Bake any camera rotation into the pixels. Returns (video, rotation).

    `rotation` is what the display matrix claimed (0/90/180/270). The returned
    path is a NEW upright file when a bake was both needed and possible, and
    `src` unchanged otherwise -- so the caller can always just use the path it
    gets back, and open_capture() stays the safety net when the bake was
    skipped.

    Rotation-free uploads (every desktop file, and phone clips already
    transcoded by a share sheet) cost one metadata probe and no transcode,
    which is the case that has to stay free.
    """
    src = Path(src)
    rotation = probe_rotation(src)
    if rotation == 0:
        return src, 0

    exe = _ffmpeg_exe()
    if exe is None:
        print(f"[orient] {src.name}: {rotation}deg rotation but no ffmpeg — "
              f"leaving it to ORIENTATION_AUTO", flush=True)
        return src, rotation

    dst = Path(dst) if dst else src.with_name(f"{src.stem}_upright.mp4")
    # ffmpeg applies the display matrix on transcode by default (-autorotate)
    # and writes no matrix of its own, so a plain re-encode IS the bake. The
    # flag is left implicit for compatibility with older builds; the verify
    # step below is what actually guarantees we got what we asked for.
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", _PRESET, "-crf", _CRF,
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-an",                       # audio is never read; skip the work
           str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[orient] {src.name}: bake failed (rc={res.returncode}) "
              f"{res.stderr[-400:]} — leaving it to ORIENTATION_AUTO", flush=True)
        dst.unlink(missing_ok=True)
        return src, rotation

    if not _verify_upright(src, dst, rotation):
        print(f"[orient] {src.name}: bake did not land upright — "
              f"leaving it to ORIENTATION_AUTO", flush=True)
        dst.unlink(missing_ok=True)
        return src, rotation

    print(f"[orient] {src.name}: baked {rotation}deg camera rotation into pixels "
          f"-> {dst.name}", flush=True)
    return dst, rotation


def _raw_frame_shape(video: Path) -> tuple[int, int] | None:
    """(h, w) of the first decoded frame, ignoring any display matrix."""
    try:
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        return frame.shape[:2] if ok else None
    except Exception:
        return None


def _verify_upright(src: Path, dst: Path, rotation: int) -> bool:
    """Guard against a silently wrong bake (an ffmpeg whose autorotate default
    ever flips, or a stream it declined to rotate).

    Checks the output against dimensions derived from the SOURCE's stored
    frame and the tagged rotation -- deliberately not against open_capture(),
    so this still catches a bad bake on a backend where ORIENTATION_AUTO is
    itself unavailable (precisely the case where the bake is load-bearing).
    """
    if probe_rotation(dst) != 0:            # residual matrix would double-rotate
        return False
    src_shape = _raw_frame_shape(src)
    dst_shape = _raw_frame_shape(dst)
    if src_shape is None or dst_shape is None:
        return False
    sh, sw = src_shape
    want = (sw, sh) if rotation in (90, 270) else (sh, sw)
    return dst_shape == want
