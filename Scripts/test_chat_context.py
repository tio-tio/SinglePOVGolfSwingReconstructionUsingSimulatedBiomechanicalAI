"""Offline tests for the chat-v2 context/elicitation tools (get_conditions,
save_context, get_capture_quality) and the Open-Meteo backfill. Run:
    .venv/Scripts/python.exe Scripts/test_chat_context.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C

ROOT = Path(__file__).parent.parent
SC = json.loads((ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json").read_text(encoding="utf-8"))
DIAG = json.loads((ROOT / "Data" / "demo" / "IMG_8110" / "IMG_8110_pose_diag.json").read_text(encoding="utf-8"))

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {extra}"))


CUR = "c" * 32
META = {"uploaded_at": "2026-07-30T09:10:00+00:00", "notes": "working on tempo",
        "weather": {"temp_c": 21.5, "wind_mph": 12.0, "wind_dir_met_deg": 250,
                    "humidity_pct": 60, "source": "open-meteo"}}
SAVED = {}


def make_ctx(meta=META, diag=DIAG, saver="ok"):
    def fetcher(jid):
        return {"scorecard": SC, "meta": meta, "diag": diag}
    def save_ok(jid, fields):
        SAVED.update({jid: fields})
        return {**(meta or {}), **fields}
    lib = [{"job_id": CUR, "date": "2026-07-30T09:10:00+00:00", "name": "me.mp4", "ready": True}]
    return C.SwingContext(a=SC, kb=C.v2.load_kb(), library=lib, current_id=CUR,
                          fetcher=fetcher,
                          saver=(save_ok if saver == "ok" else (lambda j, f: None)))


def use(name, inp, tid="t1"):
    return ([{"type": "tool_use", "id": tid, "name": name, "input": inp}], "tool_use")


def say(text):
    return ([{"type": "text", "text": text}], "end_turn")


print("\n[1] get_conditions")
r = C._t_get_conditions(make_ctx(), {})
check("available with recorded weather", r["available"] and r["weather"]["temp_c"] == 21.5, str(r)[:200])
check("wind direction caveat surfaced", "FROM" in r.get("weather_note", ""))
check("missing club triggers the elicit hint", "club" in r.get("missing", []) and "follow-up" in r.get("elicit_hint", ""))
r2 = C._t_get_conditions(make_ctx(meta=None), {})
check("no meta -> ask-the-golfer note", r2["available"] is False and "save_context" in r2["note"])
r3 = C._t_get_conditions(C.SwingContext(a=SC, kb=C.v2.load_kb()), {})
check("demo clip (no fetcher) -> unavailable", r3["available"] is False)

print("\n[2] save_context")
ctx = make_ctx()
r = C._t_save_context(ctx, {"club": "9 iron", "notes": "windy day", "evil": "x", "setting": "course"})
check("saves whitelisted fields", r["saved"] and r["stored"] == {"club": "9 iron", "notes": "windy day", "setting": "course"}, str(r))
check("saver received the fields", SAVED.get(CUR, {}).get("club") == "9 iron")
r = C._t_get_conditions(ctx, {})
check("conversation view updated after save", r.get("club") == "9 iron", str(r)[:200])
r = C._t_save_context(ctx, {})
check("nothing to save refuses", r["saved"] is False)
r = C._t_save_context(make_ctx(saver="fail"), {"club": "driver"})
check("saver failure degrades gracefully", r["saved"] is False and "THIS conversation" in r["note"])
r = C._t_save_context(C.SwingContext(a=SC, kb=C.v2.load_kb()), {"club": "driver"})
check("no saver -> can't save", r["saved"] is False)

print("\n[3] get_capture_quality")
r = C._t_get_capture_quality(make_ctx(), {})
check("available with diag", r["available"] and isinstance(r["left_right_swaps"], int), str(r)[:200])
check("filming guidance present + about the video", "waist height" in r["filming_guidance"]
      and "not about the golf swing" in r["filming_guidance"])
check("wrist confidence averaged", 0 <= (r["wrist_confidence_mean"] or 0) <= 1, str(r.get("wrist_confidence_mean")))
r2 = C._t_get_capture_quality(make_ctx(diag=None), {})
check("no diag -> unavailable", r2["available"] is False)

print("\n[4] gating + verifier")
check("context tools are library-gated",
      all(t in C._LIBRARY_TOOLS for t in ("get_conditions", "save_context", "get_capture_quality")))
ctx = make_ctx()
backend = C.ScriptedBackend([
    use("get_conditions", {}),
    say("It was about 21.5 degrees with 12 mph wind when you uploaded that swing — "
        "did it play as a headwind or tailwind from where you were hitting?"),
])
res = C.Conversation(ctx, backend).ask("Was it windy on Thursday?")
g = C.verify_chat_grounding(ctx, res)
check("weather narration + follow-up question is grounded", g["grounded"], str(g["violations"]))

ctx = make_ctx()
backend = C.ScriptedBackend([
    use("get_capture_quality", {}),
    say("A few left/right swaps were repaired in your video. For a cleaner capture, film "
        "from waist height on a tripod, down-the-line, with your whole body in frame."),
])
res = C.Conversation(ctx, backend).ask("Why is my arm metric low confidence?")
g = C.verify_chat_grounding(ctx, res)
check("filming guidance does not trip the prescriptive check", g["grounded"], str(g["violations"]))

print("\n[5] Open-Meteo backfill (stubbed HTTP)")
sys.path.insert(0, str(ROOT / "deploy" / "processing"))
import urllib.request as _ur
import processing_handler as P

HOURS = {"time": ["2026-07-30T08:00", "2026-07-30T09:00", "2026-07-30T10:00"],
         "temperature_2m": [18.0, 21.5, 23.0], "relative_humidity_2m": [70, 60, 55],
         "pressure_msl": [1015.0, 1014.5, 1014.0], "wind_speed_10m": [8.0, 12.0, 14.0],
         "wind_direction_10m": [240, 250, 260]}


class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


calls = []
_orig_urlopen = _ur.urlopen
_ur.urlopen = lambda url, timeout=None: (calls.append(str(url)),
    FakeResp(json.dumps({"hourly": HOURS, "elevation": 120.0}).encode()))[1]
try:
    wx = P._fetch_weather({"lat": "43.65", "lon": "-79.38",
                           "uploaded_at": "2026-07-30T09:10:00+00:00"})
    check("picks the upload hour", wx and wx["temp_c"] == 21.5 and wx["wind_mph"] == 12.0, str(wx))
    check("meteorological direction + elevation kept",
          wx["wind_dir_met_deg"] == 250 and wx["elevation_m"] == 120.0)
    check("source tagged", wx["source"] == "open-meteo")
    check("request is keyless open-meteo", "open-meteo.com" in calls[0] and "key" not in calls[0])
    check("no location -> no lookup", P._fetch_weather({"uploaded_at": "2026-07-30T09:10:00+00:00"}) is None)
    check("bad date -> no lookup", P._fetch_weather({"lat": "1", "lon": "1", "uploaded_at": "junk"}) is None)
    _ur.urlopen = lambda url, timeout=None: (_ for _ in ()).throw(OSError("offline"))
    check("network failure -> silently absent",
          P._fetch_weather({"lat": "1", "lon": "1", "uploaded_at": "2026-07-30T09:10:00+00:00"}) is None)
finally:
    _ur.urlopen = _orig_urlopen

print(f"\n{'=' * 50}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 50}")
sys.exit(1 if _FAIL else 0)
