"""Regression tests for ball_track.

Part 1 (synthetic, always runs) projects a KNOWN flight through a camera model
derived independently of ball_track's own projector, then checks the fitter
recovers it. Deriving the geometry twice is the point: the production code once
had the camera-pitch sign inverted, which inflated every fitted launch angle by
twice the camera pitch (a 26 deg "launch" on a 12 deg drive, 80 yd apexes and
250+ yd carries for everyone) and no self-consistent test could see it.

Part 2 runs the two validation clips plus a negative clip that contains no
usable ball flight and must be REJECTED rather than fitted.

Run: python test_ball_track.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

import ball_track as bt
from ball_flight import simulate_flight

YD_M = 0.9144
F = bt.DEFAULT_F_PX
W, H = 720, 1280
CX, CY = W / 2, H / 2


# --------------------------------------------------------------------------
# synthetic ground truth
# --------------------------------------------------------------------------

def project_truth(speed, launch, azim, spin, pitch_deg, d_m=4.0, hcam_m=1.4,
                  fps=30.0, n_frames=20, W=W, H=H):
    """Camera-level frame: Xw right, Uw up, Fw forward. Camera pitched UP by t:
        Z_cam = Uw sin t + Fw cos t
        Y_cam = -Uw cos t + Fw sin t
    so the horizon (Uw=0, Fw->inf) lands on row cy + f*tan(t).

    The camera's focal length in PIXELS follows the frame's short side: one lens
    and one field of view, sampled onto whatever grid the phone recorded.
    """
    F = bt.DEFAULT_F_PX * min(W, H) / bt.REF_SHORT_SIDE
    CX, CY = W / 2, H / 2
    t = math.radians(pitch_deg)
    r = simulate_flight(speed, launch, spin, 0.0, 0.0, trajectory_points=400)
    traj = np.array(r["trajectory"]) * YD_M          # [downrange, height, side]
    tt = np.linspace(0, r["flight_time_s"], len(traj))
    a = math.radians(azim)
    Xw = traj[:, 0] * math.sin(a) + traj[:, 2] * math.cos(a)
    Uw = -hcam_m + traj[:, 1]
    Fw = d_m + traj[:, 0] * math.cos(a) - traj[:, 2] * math.sin(a)
    Zc = Uw * math.sin(t) + Fw * math.cos(t)
    Yc = -Uw * math.cos(t) + Fw * math.sin(t)
    px, py = CX + F * Xw / Zc, CY + F * Yc / Zc
    obs = []
    for k in range(n_frames):
        ti = k / fps
        j = min(max(int(np.searchsorted(tt, ti)), 1), len(tt) - 1)
        w = (ti - tt[j - 1]) / (tt[j] - tt[j - 1])
        obs.append((k, float(px[j - 1] * (1 - w) + px[j] * w),
                    float(py[j - 1] * (1 - w) + py[j] * w)))
    tee_Z = -hcam_m * math.sin(t) + d_m * math.cos(t)
    tee_Y = hcam_m * math.cos(t) + d_m * math.sin(t)
    origin = (CX, CY + F * tee_Y / tee_Z)
    return obs, origin, tee_Z, CY + F * math.tan(t), r


def _pose_for(tee_Z, W=W, H=H):
    """Pose whose 95th-pct span sets the fitter's depth anchor to tee_Z."""
    h_px = (bt.DEFAULT_F_PX * min(W, H) / bt.REF_SHORT_SIDE) * 1.75 / tee_Z
    return {i: [(300.0, 100.0), (300.0, 100.0 + h_px)] for i in range(30)}


def synthetic_case(pitch_deg, speed=150.0, launch=12.0, azim=10.0, spin=2800.0,
                   noise_px=0.0, seed=1, W=W, H=H):
    obs, origin, tee_Z, horizon, truth = project_truth(
        speed, launch, azim, spin, pitch_deg, W=W, H=H)
    if noise_px:
        rng = np.random.default_rng(seed)
        obs = [(f, x + rng.normal(0, noise_px), y + rng.normal(0, noise_px))
               for f, x, y in obs]
    fit = bt.FlightFitter(obs, 0, 30.0, origin, _pose_for(tee_Z, W, H), W, H,
                          horizon_y=horizon, backspin_rpm=spin)
    v, _ = fit.fit_free()
    _, r = fit._project(v[0], v[1], v[2], v[3])
    return {"speed": v[0], "launch": v[1], "azim": v[2], "carry": r["carry_yd"],
            "apex": r["apex_yd"], "truth_carry": truth["carry_yd"],
            "resid": fit.pixel_loss(v)}


def test_synthetic(failures):
    print("\n-- synthetic round-trip (truth: 150 mph, 12.0 deg launch, 10.0 deg azimuth)")
    got = {}
    for pitch in (0.0, -12.0, -6.0, 6.0, 12.0):
        g = synthetic_case(pitch)
        got[pitch] = g
        checks = [
            ("launch", abs(g["launch"] - 12.0) <= 1.5, f"{g['launch']:.1f}"),
            ("azimuth", abs(g["azim"] - 10.0) <= 1.5, f"{g['azim']:.1f}"),
            ("speed", abs(g["speed"] - 150.0) <= 12.0, f"{g['speed']:.1f}"),
            ("carry", abs(g["carry"] - g["truth_carry"]) <= 20.0, f"{g['carry']:.0f}"),
        ]
        bad = [(k, val) for k, ok, val in checks if not ok]
        print(f"   camera pitch {pitch:+5.1f} deg -> launch {g['launch']:5.1f} "
              f"azim {g['azim']:5.1f} speed {g['speed']:6.1f} carry {g['carry']:4.0f} "
              f"{'PASS' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            failures.append(f"synthetic pitch {pitch}: {bad}")

    # The recovered flight must not depend on how the camera was tilted.
    spread = max(g["launch"] for g in got.values()) - min(g["launch"] for g in got.values())
    ok = spread <= 2.0
    print(f"   launch spread across camera pitches: {spread:.2f} deg "
          f"{'PASS' if ok else 'FAIL (pitch is leaking into launch angle)'}")
    if not ok:
        failures.append(f"pitch-invariance: launch spread {spread:.2f} deg")

    g = synthetic_case(-6.0, noise_px=0.7)
    ok = abs(g["launch"] - 12.0) <= 2.5
    print(f"   with 0.7 px tracking noise -> launch {g['launch']:.1f} "
          f"carry {g['carry']:.0f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"noisy synthetic: launch {g['launch']:.1f}")

    # A flat, driving flight must not come back as a lofted one.
    g = synthetic_case(-6.0, speed=140.0, launch=7.0, spin=2400.0)
    ok = abs(g["launch"] - 7.0) <= 2.0 and g["apex"] <= 25
    print(f"   flat 7.0 deg launch -> {g['launch']:.1f} deg, apex {g['apex']:.0f} yd "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"flat flight: launch {g['launch']:.1f} apex {g['apex']:.0f}")


def test_resolutions(failures):
    """The same flight filmed at different resolutions must fit the same.

    Every clip in this file's Part 2 is 720x1280, and so was every clip the
    fitter was tuned on -- but real uploads are 1080x1920. While the focal
    length was hardcoded at the 720p value, a 1080p clip fitted ~4.6 deg steep
    and ~15% short, and its reprojection residual was inflated past the
    rejection gate, so genuine flights were thrown away as "not a ball flight"
    and the results page fell back to the generic tier arcs.
    """
    print("\n-- resolution invariance (truth: 150 mph, 12.0 deg launch)")
    for W_, H_ in ((720, 1280), (1080, 1920), (1440, 2560), (2160, 3840),
                   (1920, 1080)):
        g = synthetic_case(-6.0, W=W_, H=H_)
        ok = (abs(g["launch"] - 12.0) <= 1.0
              and abs(g["carry"] - g["truth_carry"]) <= 8.0
              and g["resid"] <= bt.MAX_FIT_RESIDUAL_PX)
        print(f"   {W_:5d}x{H_:<5d} f_px={bt.default_f_px(W_, H_):6.0f} -> "
              f"launch {g['launch']:5.1f}  carry {g['carry']:5.1f} "
              f"(truth {g['truth_carry']:.0f})  resid {g['resid']:4.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"resolution {W_}x{H_}: launch {g['launch']:.1f} "
                            f"carry {g['carry']:.1f} resid {g['resid']:.2f}")


# --------------------------------------------------------------------------
# real clips
# --------------------------------------------------------------------------

DEMO = Path(__file__).parent.parent / "Data" / "demo"
CLIPS = {
    "IMG_3434": {
        "video": r"C:\Users\Banjo\Downloads\IMG_3434.MOV",
        "quality": ("measured", "partial"),
        "min_pts": 10,
        "launch_deg": (8, 20), "azimuth_deg": (15, 30), "carry_yd": (150, 260),
    },
    "IMG_8107": {
        "video": r"C:\Users\Banjo\Downloads\IMG_8107.MOV",
        "quality": ("measured", "partial"),
        "min_pts": 8,
        "launch_deg": (5, 14), "azimuth_deg": (12, 25), "carry_yd": (150, 240),
    },
    # No usable ball flight (dark bay, ball lost immediately). Previously this
    # produced a confident 30 deg launch / 62 yd apex / 204 yd carry from a
    # stationary object whose x never moved.
    "IMG_8110": {
        "video": r"C:\Users\Banjo\Downloads\IMG_8110.MOV",
        "quality": ("simulated",),
    },
}


def test_clips(failures):
    print("\n-- validation clips")
    for stem, exp in CLIPS.items():
        d = DEMO / stem
        if not Path(exp["video"]).exists() or not d.exists():
            print(f"   {stem}: SKIP (video or demo bundle missing)")
            continue
        res = bt.analyze(exp["video"], d / f"{stem}_landmarks_2d.csv",
                         d / f"{stem}_scorecard.json")
        q = res.get("quality")
        fit = res.get("fit") or {}
        checks = [("quality", q in exp["quality"], q)]
        if "min_pts" in exp:
            checks.append(("n_pts", res.get("n_track_points", 0) >= exp["min_pts"],
                           res.get("n_track_points")))
        for key, label in (("launch_deg", "launch_deg"), ("azimuth_deg", "azimuth_deg"),
                           ("speed_mph", "ball_speed_mph")):
            if key in exp:
                lo, hi = exp[key]
                val = fit.get(label)
                checks.append((key, val is not None and lo <= val <= hi, val))
        if "carry_yd" in exp:
            lo, hi = exp["carry_yd"]
            val = (res.get("flight") or {}).get("carry_yd")
            checks.append(("carry_yd", val is not None and lo <= val <= hi, val))
        bad = [(k, v) for k, ok, v in checks if not ok]
        print(f"   {stem}: {'PASS' if not bad else 'FAIL ' + str(bad)}  "
              f"(quality={q}, n={res.get('n_track_points')}, fit={fit})")
        if res.get("reason"):
            print(f"       reason: {res['reason']}")
        if bad:
            failures.append(f"{stem}: {bad}")


def main() -> int:
    failures: list[str] = []
    test_synthetic(failures)
    test_resolutions(failures)
    test_clips(failures)
    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S):"))
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
