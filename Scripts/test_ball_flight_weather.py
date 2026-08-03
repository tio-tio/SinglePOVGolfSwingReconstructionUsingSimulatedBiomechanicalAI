"""Offline tests for the playing-conditions physics (chat v2 weather feature):
air_density, wind in simulate_flight, conditions in estimate_for_club, and the
chat tool's condition passthrough. Sanity-checked against golf rules of thumb.
Run:
    .venv/Scripts/python.exe Scripts/test_ball_flight_weather.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ball_flight as BF

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {extra}"))


DRV = dict(ball_speed_mph=167.0, launch_angle_deg=10.9, backspin_rpm=2686)

print("\n[1] air_density")
check("defaults reproduce the historic constant", abs(BF.air_density() - 1.225) < 0.001,
      str(BF.air_density()))
check("hot air is thinner", BF.air_density(temp_c=35) < BF.air_density(temp_c=5))
check("altitude thins the air (Denver ~ -15%)",
      0.80 < BF.air_density(elevation_m=1600) / BF.air_density() < 0.90,
      str(BF.air_density(elevation_m=1600)))
check("humid air is (slightly) thinner than dry",
      BF.air_density(temp_c=30, humidity_pct=100) < BF.air_density(temp_c=30, humidity_pct=0))
check("explicit pressure wins over elevation",
      abs(BF.air_density(pressure_hpa=1013.25, elevation_m=3000) - BF.air_density()) < 0.001)

print("\n[2] wind in simulate_flight")
base = BF.simulate_flight(**DRV)
tail = BF.simulate_flight(**DRV, wind=(10, 0))
head = BF.simulate_flight(**DRV, wind=(10, 180))
cross = BF.simulate_flight(**DRV, wind=(15, 90))
check("baseline is the known tour-driver carry (~288)", 275 < base["carry_yd"] < 300,
      str(base["carry_yd"]))
check("tailwind adds carry", tail["carry_yd"] > base["carry_yd"] + 5,
      f"{base['carry_yd']:.0f} -> {tail['carry_yd']:.0f}")
check("headwind costs carry", head["carry_yd"] < base["carry_yd"] - 10,
      f"{base['carry_yd']:.0f} -> {head['carry_yd']:.0f}")
check("headwind hurts more than tailwind helps (rule of thumb)",
      (base["carry_yd"] - head["carry_yd"]) > (tail["carry_yd"] - base["carry_yd"]),
      f"head -{base['carry_yd'] - head['carry_yd']:.0f} vs tail +{tail['carry_yd'] - base['carry_yd']:.0f}")
check("left-to-right wind pushes the ball right (+side)",
      cross["side_yd"] > base["side_yd"] + 3, str(cross["side_yd"]))
check("no wind kwarg == old behavior exactly",
      BF.simulate_flight(**DRV)["carry_yd"] == base["carry_yd"])

print("\n[3] density in simulate_flight")
thin = BF.simulate_flight(**DRV, rho=BF.air_density(elevation_m=1600))
hot = BF.simulate_flight(**DRV, rho=BF.air_density(temp_c=35))
cold = BF.simulate_flight(**DRV, rho=BF.air_density(temp_c=0))
check("altitude adds carry (~4-8%)",
      1.03 < thin["carry_yd"] / base["carry_yd"] < 1.10, str(thin["carry_yd"]))
check("hot day carries farther than a freezing one", hot["carry_yd"] > cold["carry_yd"] + 3,
      f"{cold['carry_yd']:.0f} vs {hot['carry_yd']:.0f}")

print("\n[4] estimate_for_club with conditions")
std = BF.estimate_for_club("driver", "tour")
windy = BF.estimate_for_club("driver", "tour",
                             conditions={"wind_mph": 15, "wind_dir_deg": 180, "temp_c": 10})
check("standard estimate untouched (no conditions key)", "conditions_applied" not in std)
check("conditions force phys_q", windy["engine"] == "phys_q")
check("conditions_applied echoes the inputs + density",
      windy["conditions_applied"]["wind_mph"] == 15
      and "air_density_kgm3" in windy["conditions_applied"], str(windy.get("conditions_applied")))
check("headwind estimate carries shorter", windy["carry_yd"] < std["carry_yd"] - 10,
      f"{std['carry_yd']} -> {windy['carry_yd']}")
check("note says measured carries never change",
      "MEASURED" in windy["conditions_applied"]["note"])
bogus = BF.estimate_for_club("driver", "tour", conditions={"wind_mph": 400, "temp_c": "hot"})
check("conditions are clamped/validated", bogus["conditions_applied"]["wind_mph"] == 40.0
      and "temp_c" not in bogus["conditions_applied"], str(bogus.get("conditions_applied")))
check("clean_conditions drops junk", BF.clean_conditions({"x": 1, "wind_mph": True}) == {})

print("\n[5] chat tool passes conditions through")
import json
import coaching_chat as C
ROOT = Path(__file__).parent.parent
ctx = C.SwingContext.from_files(ROOT / "Data" / "demo" / "0" / "0_scorecard.json")
plain = C.dispatch_tool(ctx, "estimate_ball_flight", {})
windy = C.dispatch_tool(ctx, "estimate_ball_flight", {"wind_mph": 15, "wind_dir_deg": 180})
check("tool: windy carry shorter than plain", windy["carry_yd"] < plain["carry_yd"],
      f"{plain['carry_yd']} vs {windy['carry_yd']}")
check("tool: conditions_applied surfaced", windy.get("conditions_applied", {}).get("wind_mph") == 15)


def use(name, inp, tid="t1"):
    return ([{"type": "tool_use", "id": tid, "name": name, "input": inp}], "tool_use")


def say(text):
    return ([{"type": "text", "text": text}], "end_turn")


# quoting BOTH numbers requires fetching both in the same turn — the verifier
# (rightly) flags a baseline carry the model never fetched
backend = C.ScriptedBackend([
    ([{"type": "tool_use", "id": "t1", "name": "estimate_ball_flight", "input": {}},
      {"type": "tool_use", "id": "t2", "name": "estimate_ball_flight",
       "input": {"wind_mph": 15, "wind_dir_deg": 180}}], "tool_use"),
    say(f"Into a 15 mph headwind, the same swing simulates about {windy['carry_yd']} yards "
        f"of carry instead of {plain['carry_yd']} — a simulated estimate, not a measurement."),
])
res = C.Conversation(ctx, backend).ask("How far would this go into a 15 mph headwind?")
g = C.verify_chat_grounding(ctx, res)
check("conditions-adjusted answer is grounded", g["grounded"], str(g["violations"]))

print(f"\n{'=' * 50}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 50}")
sys.exit(1 if _FAIL else 0)
