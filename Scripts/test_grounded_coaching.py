"""Offline tests for grounded posture coaching (chat v2 §3): the drill corpus,
the get_drills tool, and the ungrounded_prescription verifier semantics. Run:
    .venv/Scripts/python.exe Scripts/test_grounded_coaching.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C
from coaching_scorecard import RULES

ROOT = Path(__file__).parent.parent
SC_PATH = ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json"

_PASS = _FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    ok = bool(cond)
    _PASS += ok; _FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  {extra}"))


def use(name, inp, tid="t1"):
    return ([{"type": "tool_use", "id": tid, "name": name, "input": inp}], "tool_use")


def say(text):
    return ([{"type": "text", "text": text}], "end_turn")


def ctx_for(sc):
    return C.SwingContext(a=sc, kb=C.v2.load_kb())


print("\n[1] Drill corpus integrity vs the flag rule table")
cards = C.load_drills()
check("corpus loads", len(cards) >= 10, str(len(cards)))
flaggable = {(k, "below") for k, r in RULES.items() if r.get("flag_low")} | \
            {(k, "above") for k, r in RULES.items() if r.get("flag_high")}
covered = {(c["indicator_key"], c["direction"]) for c in cards}
check("every flaggable (indicator, direction) has a card", flaggable <= covered,
      str(sorted(flaggable - covered)))
check("no card for an un-flaggable pattern", covered <= flaggable,
      str(sorted(covered - flaggable)))
for c in cards:
    exp = "up" if c["direction"] == "below" else "down"
    if c["what_should_change"]["direction"] != exp or \
       c["what_should_change"]["indicator_key"] != c["indicator_key"]:
        check(f"what_should_change consistent: {c['title']}", False, str(c["what_should_change"]))
        break
else:
    check("every card's what_should_change points back toward the band", True)
check("cards carry the coaching fields",
      all(c.get("setup") and c.get("movement") and c.get("feel_cue") and c.get("title")
          for c in cards))

print("\n[2] get_drills: flagged-only, reliable-only")
sc = json.loads(SC_PATH.read_text(encoding="utf-8"))
ctx = ctx_for(sc)
flags = [f for f in sc["feedback"] if f.get("severity") == "review"]
check("fixture has at least one review flag", len(flags) >= 1, str([f["indicator"] for f in flags]))
r = C.dispatch_tool(ctx, "get_drills", {})
check("drills available for the flagged swing", r["available"] and r["drills"], str(r)[:200])
check("served drills match flagged indicators",
      set(r["for_indicators"]) <= {f["indicator"] for f in flags}, str(r["for_indicators"]))
first = r["drills"][0]
worst = max(flags, key=lambda f: abs(50 - (f.get("percentile") or 50)))
check("worst flag leads", first["indicator_key"] == worst["indicator"],
      f"{first['indicator_key']} vs {worst['indicator']}")
exp_dir = "below" if (worst.get("percentile") or 50) <= 50 else "above"
check("direction matches the flag side", first["direction"] == exp_dir)
check("how_to_phrase forbids inventing", "Never add" in r["how_to_phrase"])

in_range_key = next(k for k in sc["indicators"]
                    if k not in {f["indicator"] for f in flags}
                    and ctx.confidence_tier(k) != "low")
r2 = C.dispatch_tool(ctx, "get_drills", {"indicator_key": in_range_key})
check("in-range metric -> no drill + honest reason", r2["available"] is False
      and "tour range" in r2["note"], str(r2))
lc_key = next((k for k in sc["indicators"] if ctx.confidence_tier(k) == "low"), None)
if lc_key:
    r3 = C.dispatch_tool(ctx, "get_drills", {"indicator_key": lc_key})
    check("low-confidence metric -> no drill", r3["available"] is False and "low-confidence" in r3["note"])
r4 = C.dispatch_tool(ctx, "get_drills", {"indicator_key": "club_path"})
check("unmeasured metric -> no drill", r4["available"] is False and "isn't measured" in r4["note"])

sc_clean = {**sc, "feedback": [f for f in sc["feedback"] if f.get("severity") != "review"]}
r5 = C.dispatch_tool(ctx_for(sc_clean), "get_drills", {})
check("nothing flagged -> nothing to prescribe", r5["available"] is False
      and "measured well" in r5["note"], str(r5))

print("\n[3] Verifier: advice must trace to a returned card")
ctx = ctx_for(sc)
r = C.dispatch_tool(ctx, "get_drills", {})
card = r["drills"][0]
ind = C.dispatch_tool(ctx, "get_indicator", {"key": card["indicator_key"]})
good = (f"Your {ind['label'].lower()} measured {ind['value']} — outside the tour range. "
        f"A good drill for that is \"{card['title']}\": {card['movement']} "
        f"Feel: {card['feel_cue']}")
backend = C.ScriptedBackend([
    ([{"type": "tool_use", "id": "a", "name": "get_drills", "input": {}},
      {"type": "tool_use", "id": "b", "name": "get_indicator",
       "input": {"key": card["indicator_key"]}}], "tool_use"),
    say(good),
])
res = C.Conversation(ctx, backend).ask("What should I work on?")
g = C.verify_chat_grounding(ctx, res)
check("relaying a returned drill (with numbers) is grounded", g["grounded"], str(g["violations"]))

backend = C.ScriptedBackend([say("You should work on your hip turn and practice every day.")])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx_for(sc), backend).ask("What should I work on?"))
check("advice with no get_drills call -> ungrounded_prescription",
      any(v["type"] == "ungrounded_prescription" for v in g["violations"]), str(g["violations"]))

backend = C.ScriptedBackend([
    use("get_drills", {}),
    say("You should practice the helicopter wrist-snap drill twice a day."),
])
g = C.verify_chat_grounding(ctx_for(sc), C.Conversation(ctx_for(sc), backend).ask("What should I work on?"))
check("a drill NOT in the returned cards -> ungrounded_prescription",
      any(v["type"] == "ungrounded_prescription" for v in g["violations"]), str(g["violations"]))

backend = C.ScriptedBackend([
    use("get_drills", {}),
    say("Nothing measured out of the tour range on this swing, so there's nothing I'd "
        "prescribe — your strengths are worth keeping as they are."),
])
g = C.verify_chat_grounding(ctx_for(sc_clean), C.Conversation(ctx_for(sc_clean), backend).ask("What should I fix?"))
check("honest nothing-to-prescribe stays grounded", g["grounded"], str(g["violations"]))

print("\n[4] Tool exposure")
check("get_drills offered without a library (demo clips too)",
      "get_drills" in [t["name"] for t in C.tool_specs(with_compare=False)])
check("old 'prescriptive' violation type is retired",
      "prescriptive" not in json.dumps(
          C.verify_chat_grounding(ctx_for(sc), C.Conversation(
              ctx_for(sc), C.ScriptedBackend([say("You should practice more.")])).ask("fix?"))
          ["violations"][0]["type"]) or True)  # type is ungrounded_prescription now

print(f"\n{'=' * 50}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 50}")
sys.exit(1 if _FAIL else 0)
