"""Regression test for ball_track on the two validation clips.

Run: python test_ball_track.py
Asserts quality tiers and physically-sane ranges (not exact values — the
tracker is deterministic but thresholds may be tuned; ranges catch breakage).
"""
import sys
from pathlib import Path

from ball_track import analyze

DEMO = Path(__file__).parent.parent / "Data" / "demo"
CLIPS = {
    "IMG_3434": {
        "video": r"C:\Users\Banjo\Downloads\IMG_3434.MOV",
        "quality": "measured",
        "min_pts": 15,
        "launch_deg": (20, 33), "azimuth_deg": (15, 30),
        "speed_mph": (140, 190), "carry_yd": (220, 300),
    },
    "IMG_8107": {
        "video": r"C:\Users\Banjo\Downloads\IMG_8107.MOV",
        "quality": ("partial", "measured"),
        "min_pts": 8,
        "launch_deg": (5, 14), "azimuth_deg": (12, 25),
    },
}


def main() -> int:
    failures = []
    for stem, exp in CLIPS.items():
        d = DEMO / stem
        res = analyze(exp["video"], d / f"{stem}_landmarks_2d.csv",
                      d / f"{stem}_scorecard.json")
        q = res.get("quality")
        ok_q = q in exp["quality"] if isinstance(exp["quality"], tuple) else q == exp["quality"]
        checks = [("quality", ok_q, q)]
        checks.append(("n_pts", res.get("n_track_points", 0) >= exp["min_pts"],
                       res.get("n_track_points")))
        fit = res.get("fit", {})
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
        bad = [(k, v) for k, okc, v in checks if not okc]
        status = "PASS" if not bad else f"FAIL {bad}"
        print(f"{stem}: {status}  (quality={q}, n={res.get('n_track_points')}, fit={fit})")
        if bad:
            failures.append(stem)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
