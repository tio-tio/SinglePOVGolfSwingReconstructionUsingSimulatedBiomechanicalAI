"""Offline tests for coaching_chat.py — no ANTHROPIC_API_KEY, no network.

Drives the FULL tool-use loop with ScriptedBackend (canned model turns) to prove:
tool dispatch, multi-turn memory, refusal-by-tool-result, the compare tool, the
max-iteration cap, and the transcript grounding verifier. Run:

    .venv/Scripts/python.exe Scripts/test_coaching_chat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C

ROOT = Path(__file__).parent.parent
SC_A = ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json"

_PASS = _FAIL = 0


def check(name: str, cond: bool, extra: str = ""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {extra}")


# -- scripted-turn builders ------------------------------------------------- #
def use(name, inp, tid="t1"):
    return ([{"type": "tool_use", "id": tid, "name": name, "input": inp}], "tool_use")


def say(text):
    return ([{"type": "text", "text": text}], "end_turn")


def ctx_single():
    return C.SwingContext.from_files(SC_A)


def ctx_pair():
    # use the same clip as "earlier" so compare is deterministic (delta 0)
    return C.SwingContext(a=json.loads(SC_A.read_text()), kb=C.v2.load_kb(),
                          b=json.loads(SC_A.read_text()))


# =========================================================================== #
print("\n[1] Tools return grounded data over a real scorecard")
ctx = ctx_single()
li = C._t_list_indicators(ctx, {})
check("list_indicators returns all 15", li["count"] == 15)
key = li["indicators"][0]["key"]
gi = C._t_get_indicator(ctx, {"key": "tempo_ratio"})
check("get_indicator(tempo_ratio) measured", gi.get("measured") and "value" in gi)
check("get_indicator unknown -> measured False",
      C._t_get_indicator(ctx, {"key": "nope"}).get("measured") is False)
gt = C._t_explain_term(ctx, {"term": "tour range"})
check("explain_term known term", gt["in_glossary"] is True)
check("explain_term unknown term", C._t_explain_term(ctx, {"term": "banana"})["in_glossary"] is False)
fl = C._t_get_flagged_observations(ctx, {})
check("get_flagged_observations returns the weight-shift flag", fl["count"] >= 1)
sm = C._t_get_swing_summary(ctx, {})
check("summary counts add up", sm["n_indicators"] == 15 and sm["n_flagged"] >= 1)

# find a low-confidence metric to drive the refusal tests
lowconf = [i["key"] for i in li["indicators"] if not i["reliable"]]
check("at least one low-confidence indicator exists", len(lowconf) >= 1, str(lowconf))
LC = lowconf[0] if lowconf else "left_arm_bend_top_deg"
check("get_indicator low-conf flags reliable False",
      C._t_get_indicator(ctx, {"key": LC}).get("reliable") is False)

# =========================================================================== #
print("\n[2] Full loop: answerable question fetches the metric, then answers")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say("Your tempo looks tour-like — the backswing-to-downswing rhythm is right in the typical range."),
])
convo = C.Conversation(ctx, backend)
res = convo.ask("How was my tempo?")
check("one tool call logged", len(res.tool_log) == 1 and res.tool_log[0]["name"] == "get_indicator")
check("loop stopped on end_turn", res.stopped_reason == "end_turn")
check("answer non-empty", bool(res.answer))
g = C.verify_chat_grounding(ctx, res)
check("answerable turn is grounded", g["grounded"], str(g["violations"]))
check("tempo_ratio recorded as fetched-reliable", "tempo_ratio" in g["fetched_reliable"])

# =========================================================================== #
print("\n[3] Refusal: unmeasured topic -> list, then no number")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("list_indicators", {}),
    say("I can't tell whether your club face was open — that isn't something this swing analysis measures."),
])
res = C.Conversation(ctx, backend).ask("Was my club face open at impact?")
g = C.verify_chat_grounding(ctx, res)
check("unmeasured refusal is grounded (no range claim)", g["grounded"], str(g["violations"]))
check("no indicator fetched", g["fetched_reliable"] == [])

# =========================================================================== #
print("\n[4] Refusal: low-confidence metric is flagged but not leaked")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say("That measurement isn't reliable enough from a single camera, so I can't assess it."),
])
res = C.Conversation(ctx, backend).ask("Was my lead arm straight at the top?")
g = C.verify_chat_grounding(ctx, res)
check("clean low-conf refusal is grounded", g["grounded"], str(g["violations"]))
check("low-conf key recorded", LC in g["fetched_low_conf"])

# leak case: the model wrongly states the low-conf metric's value/label
label = C.v2.load_kb()["indicators"].get(LC, {}).get("label", LC)
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say(f"Your {label} was well within the tour range at the top."),
])
res = C.Conversation(ctx_single(), backend).ask("Was my lead arm straight at the top?")
g = C.verify_chat_grounding(ctx_single(), res)
check("leaking the low-conf metric is caught",
      any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# NAMING the low-conf metric to DECLINE it (curly apostrophe, no value/range) is fine
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say(f"{label} here is low confidence from a single camera, so I can’t assess it."),
])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("lead arm?"))
check("naming a low-conf metric to decline is NOT a leak", g["grounded"], str(g["violations"]))
# and the value must never be stated even while declining
val = C._t_get_indicator(ctx_single(), {"key": LC})["value"]
backend = C.ScriptedBackend([use("get_indicator", {"key": LC}), say(f"It measured {val} at the top.")])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("lead arm?"))
check("stating the low-conf value IS caught",
      any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[5] Un-retrieved advice is caught by the verifier (chat v2 semantics)")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "hip_lateral_shift_pct"}),
    say("Your weight shift is limited; you should work on driving toward the target."),
])
res = C.Conversation(ctx, backend).ask("What should I do about my weight shift?")
g = C.verify_chat_grounding(ctx, res)
check("advice without a get_drills card is flagged",
      any(v["type"] == "ungrounded_prescription" for v in g["violations"]), str(g["violations"]))
# but a REFUSAL that merely names fix-words must NOT be flagged prescriptive
backend = C.ScriptedBackend([say("I can describe your swing, but I can't tell you what to work on or drill.")])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx_single(), backend).ask("what should I fix?"))
check("refusal echoing fix-words is NOT flagged",
      not any(v["type"] == "ungrounded_prescription" for v in g["violations"]), str(g["violations"]))
# "I can't prescribe... but here's what stood out" + grounded numbers is fine (real Codex case)
v = C._t_get_indicator(ctx_single(), {"key": "hip_lateral_shift_pct"})["value"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "hip_lateral_shift_pct"}),
    say(f"I can’t prescribe what to change or practice, but your weight shift was {v}% — below the tour range."),
])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("what should I work on?"))
check("describe-after-declining-to-prescribe is grounded", g["grounded"], str(g["violations"]))

# =========================================================================== #
print("[5b] Broad 'biggest takeaways' answer over many tools stays grounded")
ctx = ctx_single()
# a good summary calls several tools then states multiple fetched numbers
tempo = C._t_get_indicator(ctx, {"key": "tempo_ratio"})["value"]
sh = C._t_get_indicator(ctx, {"key": "shoulder_turn_top_deg"})["value"]
backend = C.ScriptedBackend([
    ([{"type": "tool_use", "id": "1", "name": "get_swing_summary", "input": {}},
      {"type": "tool_use", "id": "2", "name": "get_flagged_observations", "input": {}}], "tool_use"),
    ([{"type": "tool_use", "id": "3", "name": "get_indicator", "input": {"key": "tempo_ratio"}},
      {"type": "tool_use", "id": "4", "name": "get_indicator", "input": {"key": "shoulder_turn_top_deg"}}], "tool_use"),
    say(f"Biggest takeaways: your tempo is {tempo} and shoulder turn {sh}, both tour-like; weight shift was the one outside the range."),
])
res = C.Conversation(ctx, backend).ask("What are the biggest takeaways from my swing?")
g = C.verify_chat_grounding(ctx, res)
check("multi-tool summary is grounded", g["grounded"], str(g["violations"]))
check("summary used 4 tool calls across 2 rounds", len(res.tool_log) == 4)

# =========================================================================== #
print("\n[6] Multi-turn memory: follow-up keeps history")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say("Your shoulder turn is a touch under the tour range at the top."),
    use("get_indicator", {"key": "hip_turn_top_deg"}),
    say("Your hip turn, by contrast, is within the typical range."),
])
convo = C.Conversation(ctx, backend)
convo.ask("How was my shoulder turn?")
res2 = convo.ask("And my hips?")
# after 2 user turns + tool rounds, history should contain both user questions
user_turns = [m for m in convo.messages if m["role"] == "user" and isinstance(m["content"], str)]
check("both user questions in history", len(user_turns) == 2)
check("second turn fetched hip_turn", any(e["input"].get("key") == "hip_turn_top_deg" for e in res2.tool_log))

# =========================================================================== #
print("\n[7] compare_indicator only exists with a 2nd clip; works when present")
ctx1 = ctx_single()
check("no compare tool for single clip",
      "compare_indicator" not in [t["name"] for t in C.tool_specs(with_compare=False)])
ctxp = ctx_pair()
specs = [t["name"] for t in C.Conversation(ctxp, C.ScriptedBackend([])).tools]
check("compare tool offered with 2 clips", "compare_indicator" in specs)
cmp = C._t_compare_indicator(ctxp, {"key": "tempo_ratio"})
check("compare returns delta + direction", cmp["comparable"] and "direction" in cmp)
cmp_nokey = C._t_compare_indicator(ctx1, {"key": "tempo_ratio"})
check("compare without 2nd clip refuses", cmp_nokey["comparable"] is False)

# =========================================================================== #
print("\n[8] Max-iteration cap forces a final answer (and keeps message shape legal)")
ctx = ctx_single()
# script: keep calling tools forever -> loop must cut it off and force an answer
spam = [use("list_indicators", {}) for _ in range(C.MAX_TOOL_ITERS)]
spam.append(say("final answer after the cap"))
convo = C.Conversation(ctx, C.ScriptedBackend(spam))
res = convo.ask("loop please")
check("loop respected the iteration cap", res.iterations <= C.MAX_TOOL_ITERS)
check("a final answer was still produced", bool(res.answer))
# no two consecutive user messages (Anthropic requires alternation / merged tool_results)
roles = [m["role"] for m in convo.messages]
no_double_user = all(not (roles[i] == roles[i+1] == "user") for i in range(len(roles)-1))
check("no two consecutive user messages after cap", no_double_user, str(roles))

# =========================================================================== #
print("\n[9] Outgoing message shape: tool_result follows tool_use with matching id")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}, tid="abc123"),
    say("Your tempo is fine."),
])
C.Conversation(ctx, backend).ask("tempo?")
# the 2nd create() call must have received: ...assistant(tool_use) then user(tool_result id=abc123)
sent = backend.calls[1]["messages"]
asst = [m for m in sent if m["role"] == "assistant"][-1]
tu = [b for b in asst["content"] if b["type"] == "tool_use"][0]
tr_msg = sent[-1]
tr = tr_msg["content"][0]
check("last sent message is a user tool_result", tr_msg["role"] == "user" and tr["type"] == "tool_result")
check("tool_result id matches the tool_use id", tr["tool_use_id"] == tu["id"] == "abc123")

# =========================================================================== #
print("\n[10] Multiple tool calls in one assistant turn are all dispatched")
ctx = ctx_single()
multi = ([{"type": "tool_use", "id": "a", "name": "get_indicator", "input": {"key": "tempo_ratio"}},
          {"type": "tool_use", "id": "b", "name": "get_indicator", "input": {"key": "shoulder_turn_top_deg"}}],
         "tool_use")
backend = C.ScriptedBackend([multi, say("Both look good.")])
res = C.Conversation(ctx, backend).ask("tempo and shoulder turn?")
check("both tool calls logged", len(res.tool_log) == 2)
# the single user turn after must carry BOTH tool_results
tr_msg = backend.calls[1]["messages"][-1]
check("both tool_results in one user message", len(tr_msg["content"]) == 2
      and {b["tool_use_id"] for b in tr_msg["content"]} == {"a", "b"})

# =========================================================================== #
print("\n[11] Numeric grounding: an invented number is caught, a fetched one passes")
ctx = ctx_single()
real = C._t_get_indicator(ctx, {"key": "tempo_ratio"})["value"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say(f"Your tempo is {real}, right in the tour range."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("tempo?"))
check("stating the fetched value is grounded", g["grounded"], str(g["violations"]))
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say("Your tempo is 9.87, right in the tour range."),  # invented value
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("tempo?"))
check("an invented number is flagged ungrounded",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[12] Display rounding + dash ranges are grounded (live-endpoint repro)")
# live bug: "21 degrees" for a fetched 21.25 and "38–68" (en-dash) for pro_band
# [37.5, 68.1] were flagged as violations [21.0, 38.0, -68.0]
ctx = ctx_single()
ind = C._t_get_indicator(ctx, {"key": "shoulder_turn_top_deg"})
val, (lo, hi) = ind["value"], ind["pro_band"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Your shoulder turn was about {round(val)} degrees at the top; "
        f"the tour range is {round(lo)}–{round(hi)} degrees."),  # en-dash
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("rounded value + en-dash tour range is grounded", g["grounded"], str(g["violations"]))
check("en-dash never parsed as a minus sign",
      not any(v.get("value", 0) < 0 for v in g["violations"]), str(g["violations"]))
# em-dash and plain hyphen ranges behave the same
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Tour range: {round(lo)}—{round(hi)} degrees; you turned {round(val)}."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("em-dash range is grounded too", g["grounded"], str(g["violations"]))
# a genuinely negative number is still parsed as negative (sign not after a digit)
neg_toks = C._NUM_TOKEN.findall("swayed -3.1 inches, range 38-68")
check("standalone minus kept, range dash split",
      neg_toks == ["-3.1", "38", "68"], str(neg_toks))
# display rounding must NOT excuse an invented number
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Your shoulder turn was {round(val) + 7} degrees at the top."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("a number ~7 off the fetched value is still flagged",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[13] Low-conf: tour band citable in a refusal; rounded value leak caught")
ctx = ctx_single()
lc_ind = C._t_get_indicator(ctx, {"key": LC})
lc_val, lc_band = lc_ind["value"], lc_ind.get("pro_band")
if lc_band:
    backend = C.ScriptedBackend([
        use("get_indicator", {"key": LC}),
        say(f"Tour pros are typically {round(lc_band[0])}–{round(lc_band[1])} here, but this "
            f"measurement isn't reliable enough from a single camera, so I can't assess yours."),
    ])
    g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("lead arm?"))
    check("citing the tour band of a low-conf metric in a refusal is grounded",
          g["grounded"], str(g["violations"]))
if abs(round(lc_val)) >= 13 and all(abs(round(lc_val) - round(b)) > 1 for b in (lc_band or [])):
    backend = C.ScriptedBackend([
        use("get_indicator", {"key": LC}),
        say(f"It measured about {round(lc_val)} at the top."),
    ])
    g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("lead arm?"))
    check("stating the ROUNDED low-conf value is still a leak",
          any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[14] Simulated ball flight: estimate tool answers 'how far' grounded")
# clip 0 has meta.club="driver" + a pro player; clip 1292 (ctx_single) has neither
ctx = C.SwingContext.from_files(ROOT / "Data" / "demo" / "0" / "0_scorecard.json")
check("estimate_ball_flight offered without a 2nd clip",
      "estimate_ball_flight" in [t["name"] for t in C.tool_specs(with_compare=False)])
est = C.dispatch_tool(ctx, "estimate_ball_flight", {})
check("tool simulates from the scorecard's club", est.get("estimated") is True, str(est))
check("female pro (Sandra Gal) defaults to lpga tier", est.get("skill_tier") == "lpga", str(est))
est_m = C.dispatch_tool(C.SwingContext.from_files(ROOT / "Data" / "demo" / "830" / "830_scorecard.json"),
                        "estimate_ball_flight", {})
check("male pro (Tiger) defaults to tour tier", est_m.get("skill_tier") == "tour", str(est_m))
check("carry/apex are ints, launch conditions included",
      isinstance(est["carry_yd"], int) and "assumed_launch" in est)
backend = C.ScriptedBackend([
    use("estimate_ball_flight", {}),
    say(f"The ball isn't tracked in the video, but a physics simulation for a typical "
        f"{est['club']} swing estimates about {est['carry_yd']} yards of carry, peaking "
        f"around {est['apex_yd']} yards up."),
])
res = C.Conversation(ctx, backend).ask("How far did the ball go?")
g = C.verify_chat_grounding(ctx, res)
check("simulated-estimate answer is grounded", g["grounded"], str(g["violations"]))
check("sim tool call logged", res.tool_log[0]["name"] == "estimate_ball_flight")
# an invented carry number must still be caught
backend = C.ScriptedBackend([
    use("estimate_ball_flight", {}),
    say(f"It carried about {est['carry_yd'] + 37} yards."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("How far?"))
check("invented carry number is flagged ungrounded",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))
# golfer-stated launch numbers flow through as overrides
ov = C.dispatch_tool(ctx, "estimate_ball_flight", {"ball_speed_mph": 150, "skill_tier": "amateur"})
check("golfer override recorded in result",
      ov["assumed_launch"]["ball_speed_mph"] == 150 and
      "ball_speed_mph" in ov["assumed_launch"]["overridden_by_golfer"])
# UI-only trajectory: kept in the tool result for the frontend, hidden from the model
check("_ui_trajectory present for the UI", isinstance(est.get("_ui_trajectory"), list))
check("_ui payload stripped from the model-visible view",
      "_ui_trajectory" not in C.model_visible(est))
# numbers that exist ONLY in the UI payload must not ground an answer
visible_nums = {round(n, 2) for n in C._numbers_in(C.model_visible(est))}
hidden = next((p[0] for p in est["_ui_trajectory"]
               if abs(p[0]) >= 13 and round(p[0], 2) not in visible_nums), None)
if hidden is not None:
    backend = C.ScriptedBackend([
        use("estimate_ball_flight", {}),
        say(f"At one point the ball was {hidden} yards downrange."),
    ])
    g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("How far?"))
    check("UI-only trajectory numbers cannot ground an answer",
          any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# narrating the tool's own adjustment note ("above tour-typical") must not
# trip the range-claim check (live-endpoint repro: ungrounded_range_claim)
backend = C.ScriptedBackend([
    use("estimate_ball_flight", {}),
    say(f"Simulated estimate: about {est['carry_yd']} yards of carry. The model nudged "
        f"ball speed up because this swing's hand speed measured above tour-typical."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("How far did it go?"))
check("sim answer narrating the speed nudge is grounded", g["grounded"], str(g["violations"]))
# ...but range words with NO successful tool at all stay a violation
backend = C.ScriptedBackend([say("Your turn was well within the tour range.")])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("was it in range?"))
check("range claim without any tool is still flagged",
      any(v["type"] == "ungrounded_range_claim" for v in g["violations"]), str(g["violations"]))

# swing-aware speed nudge: clip 0's measured hand speed (8.22 bs/s vs pro
# median 7.0) scales the assumed ball speed, capped at +12%, and says so
hs_view = ctx.a["indicators"]["hand_speed_impact_bs"]
check("hand-speed indicator present + low confidence",
      hs_view["confidence_tier"] == "low" and hs_view["value"] > 0)
base_speed = 140.0  # lpga driver table default (clip 0 is Sandra Gal)
check("assumed ball speed scaled up for fast hands",
      est["assumed_launch"]["ball_speed_mph"] > base_speed,
      str(est["assumed_launch"]))
check("scale capped at +12%",
      est["assumed_launch"]["ball_speed_mph"] <= round(base_speed * 1.12, 1) + 0.1)
check("adjustment disclosed in the result",
      "hand" in est.get("swing_speed_adjustment", ""), str(est.get("swing_speed_adjustment")))
# a golfer-stated ball speed wins over the nudge
ov2 = C.dispatch_tool(ctx, "estimate_ball_flight", {"ball_speed_mph": 150})
check("stated ball speed overrides the nudge",
      ov2["assumed_launch"]["ball_speed_mph"] == 150 and "swing_speed_adjustment" not in ov2)
# get_indicator on the hand-speed metric stays refusal-shaped (low confidence)
gi_hs = C._t_get_indicator(ctx, {"key": "hand_speed_impact_bs"})
check("hand speed metric is refuse-only in chat", gi_hs.get("reliable") is False)

# a scorecard with no club recorded (clip 1292, upload-style meta) declines
miss = C.dispatch_tool(ctx_single(), "estimate_ball_flight", {})
check("no club recorded -> estimated False", miss.get("estimated") is False)
# but the golfer telling us the club rescues it — and a numeric "player" (a job
# id, not a name) must default to amateur, not tour
told = C.dispatch_tool(ctx_single(), "estimate_ball_flight", {"club": "driver"})
check("golfer-stated club rescues a club-less scorecard", told.get("estimated") is True)
check("job-id player defaults to amateur tier", told.get("skill_tier") == "amateur", str(told))
# upload scorecards carry a filename stem as "player" — never a tour tier
for fake in ("IMG_8107", "TheoClip1", "5b2907dd23c247b99c5f7eb9fa9af795", "e2e_debug"):
    ctx_up = C.SwingContext(a={"meta": {"player": fake, "club": "driver"}, "indicators": {}},
                            kb=C.v2.load_kb())
    t = C.dispatch_tool(ctx_up, "estimate_ball_flight", {})
    check(f"player '{fake}' -> amateur tier", t.get("skill_tier") == "amateur", str(t.get("skill_tier")))

# =========================================================================== #
print(f"\n{'='*50}\n  {_PASS} passed, {_FAIL} failed\n{'='*50}")
sys.exit(1 if _FAIL else 0)
