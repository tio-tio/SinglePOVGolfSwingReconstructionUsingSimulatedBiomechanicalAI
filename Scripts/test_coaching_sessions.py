"""Offline tests for the chat-v2 session tools (coaching_sessions.py + the
list_sessions/load_swing/compare_swings/compare_sessions tools in
coaching_chat.py). ScriptedBackend, no key, no network. Run:

    .venv/Scripts/python.exe Scripts/test_coaching_sessions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C
import coaching_sessions as CS
from session_meta import cluster_sessions

ROOT = Path(__file__).parent.parent
SC = {n: json.loads((ROOT / "Data" / "demo" / n / f"{n}_scorecard.json").read_text(encoding="utf-8"))
      for n in ("1292", "0", "830")}

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


# a fake team library: three jobs across two sessions (Tue morning x2, Thu)
CUR, TUE2, THU = "c" * 32, "a" * 32, "b" * 32
LIBRARY = [
    {"job_id": CUR, "date": "2026-07-30T09:10:00+00:00", "name": "thu-swing.mp4",
     "ready": True, "club": "driver"},
    {"job_id": TUE2, "date": "2026-07-28T10:05:00+00:00", "name": "tue-2.mp4", "ready": True},
    {"job_id": THU, "date": "2026-07-28T09:30:00+00:00", "name": "tue-1.mp4", "ready": True,
     "notes": "working on tempo"},
]
BUNDLES = {CUR: {"scorecard": SC["1292"]}, TUE2: {"scorecard": SC["0"]},
           THU: {"scorecard": SC["830"]}}


def ctx_lib(budget_used=0):
    ctx = C.SwingContext(a=SC["1292"], kb=C.v2.load_kb(), library=[dict(j) for j in LIBRARY],
                         current_id=CUR, fetcher=lambda jid: BUNDLES.get(jid))
    ctx.fetches = budget_used
    return ctx


# =========================================================================== #
print("\n[1] tool_specs gating")
names_plain = [t["name"] for t in C.tool_specs(with_compare=False)]
check("no session tools without a library", not set(names_plain) & set(C._LIBRARY_TOOLS))
names_lib = [t["name"] for t in C.tool_specs(with_compare=False, with_library=True)]
check("all four session tools with a library", set(C._LIBRARY_TOOLS) <= set(names_lib))
convo = C.Conversation(ctx_lib(), C.ScriptedBackend([]))
check("Conversation offers session tools when library present",
      "compare_sessions" in [t["name"] for t in convo.tools])
convo0 = C.Conversation(C.SwingContext(a=SC["1292"], kb=C.v2.load_kb()), C.ScriptedBackend([]))
check("Conversation hides session tools without library",
      "list_sessions" not in [t["name"] for t in convo0.tools])

# =========================================================================== #
print("\n[2] list_sessions clusters the library")
res = C._t_list_sessions(ctx_lib(), {})
check("available with sessions", res["available"] and res["n_sessions"] == 2, str(res)[:200])
newest = res["sessions"][0]
check("newest session first (Thu)", newest["date"] == "2026-07-30")
check("current swing marked", any(sw.get("is_current") for sw in newest["swings"]))
tue = res["sessions"][1]
check("Tue session has both swings + notes surfaced",
      tue["n_swings"] == 2 and any(sw.get("notes") for sw in tue["swings"]), str(tue))
res0 = C._t_list_sessions(C.SwingContext(a=SC["1292"], kb=C.v2.load_kb()), {})
check("no library -> available False", res0["available"] is False)

# =========================================================================== #
print("\n[3] load_swing + budget + get_indicator(swing_id)")
ctx = ctx_lib()
r = C._t_load_swing(ctx, {"swing_id": TUE2})
check("loads a library swing", r["loaded"] and r["name"] == "tue-2.mp4", str(r)[:200])
check("summary counts present", r["n_indicators"] > 0 and "flagged_keys" in r)
r2 = C._t_load_swing(ctx, {"swing_id": "f" * 32})
check("unknown swing refuses cleanly", r2["loaded"] is False)
gi = C._t_get_indicator(ctx, {"key": "tempo_ratio", "swing_id": TUE2})
check("get_indicator reads the loaded swing", gi.get("measured") and gi.get("swing_id") == TUE2)
check("loaded value differs from current (different clip)",
      gi["value"] != C._t_get_indicator(ctx, {"key": "tempo_ratio"})["value"])
gi_cur = C._t_get_indicator(ctx, {"key": "tempo_ratio", "swing_id": CUR})
check("current swing's id resolves to the active scorecard",
      gi_cur.get("measured") and "swing_id" not in gi_cur)
ctx_poor = ctx_lib(budget_used=C.FETCH_BUDGET)
r3 = C._t_load_swing(ctx_poor, {"swing_id": TUE2})
check("fetch budget exhausted -> clean refusal", r3["loaded"] is False)
check("current swing needs no budget",
      C._t_load_swing(ctx_poor, {"swing_id": CUR})["loaded"] is True)

# =========================================================================== #
print("\n[4] compare_swings: auto-ordered by date, delta vs tour band")
ctx = ctx_lib()
r = C._t_compare_swings(ctx, {"swing_id_a": CUR, "swing_id_b": THU})  # newer first on purpose
check("compared", r.get("compared") is True, str(r)[:200])
check("auto-ordered: earlier is the Tue swing", r["earlier_swing"]["swing_id"] == THU)
check("later is current", r["later_swing"]["swing_id"] == CUR)
rows = {row["key"]: row for row in r["indicators"] if row.get("comparable")}
check("comparable rows carry values + verdict",
      rows and all("change_vs_tour" in row and "earlier_value" in row for row in rows.values()))
lc_rows = [row for row in r["indicators"] if not row.get("comparable")]
check("low-confidence metrics excluded from comparison values",
      all("earlier_value" not in row for row in lc_rows))
check("same swing twice refuses", C._t_compare_swings(ctx, {"swing_id_a": CUR, "swing_id_b": CUR})["compared"] is False)
check("bad id refuses", C._t_compare_swings(ctx, {"swing_id_a": CUR, "swing_id_b": "nope"})["compared"] is False)

# =========================================================================== #
print("\n[5] compare_sessions: medians + session framing")
ctx = ctx_lib()
ls = C._t_list_sessions(ctx, {})
sid_new, sid_old = ls["sessions"][0]["session_id"], ls["sessions"][1]["session_id"]
r = C._t_compare_sessions(ctx, {"session_id_a": sid_new, "session_id_b": sid_old})
check("compared", r.get("compared") is True, str(r)[:300])
check("earlier session is Tue", r["earlier_session"]["date"] == "2026-07-28")
check("sampled counts reported", r["earlier_session"]["n_sampled"] == 2)
check("phrasing mentions medians", "MEDIAN" in r["how_to_phrase"])
check("unknown session refuses",
      C._t_compare_sessions(ctx, {"session_id_a": sid_new, "session_id_b": "s-zzzzzzzz"})["compared"] is False)

print("\n[5b] session_rollup unit checks")
roll = CS.session_rollup([SC["0"], SC["830"]])
k = "tempo_ratio"
vals = sorted([SC["0"]["indicators"][k]["value"], SC["830"]["indicators"][k]["value"]])
check("median of two = midpoint", abs(roll["indicators"][k]["value"] - sum(vals) / 2) < 0.01)
check("n_swings recorded", roll["indicators"][k]["n_swings"] == 2)
low_in_one = [k2 for k2, v in SC["0"]["indicators"].items() if v.get("confidence_tier") == "low"]
if low_in_one:
    check("any-low poisons the aggregate tier",
          roll["indicators"][low_in_one[0]]["confidence_tier"] == "low")

# =========================================================================== #
print("\n[6] verifier: session narration is grounded")
ctx = ctx_lib()
backend = C.ScriptedBackend([
    use("list_sessions", {}),
    say("You have two sessions: July 30 and July 28. The July 28 one has 2 swings."),
])
res = C.Conversation(ctx, backend).ask("What sessions do I have?")
g = C.verify_chat_grounding(ctx, res)
check("date narration (\"July 30\", \"28\") is grounded", g["grounded"], str(g["violations"]))

ctx = ctx_lib()
r_cmp = C._t_compare_swings(ctx, {"swing_id_a": CUR, "swing_id_b": THU})
some = next(row for row in r_cmp["indicators"] if row.get("comparable"))
lbl = ctx.kb["indicators"].get(some["key"], {}).get("label", some["key"])
backend = C.ScriptedBackend([
    use("compare_swings", {"swing_id_a": CUR, "swing_id_b": THU}),
    say(f"Compared with July 28, your {lbl.lower()} went from {some['earlier_value']} to "
        f"{some['later_value']} — {some['change_vs_tour']} relative to the tour range."),
])
res = C.Conversation(ctx_lib(), backend).ask("How does this compare to Tuesday?")
g = C.verify_chat_grounding(ctx_lib(), res)
check("compare narration with range words is grounded", g["grounded"], str(g["violations"]))

backend = C.ScriptedBackend([
    use("compare_swings", {"swing_id_a": CUR, "swing_id_b": THU}),
    say("Your tempo went from 9.87 to 4.56 since Tuesday."),   # invented numbers
])
g = C.verify_chat_grounding(ctx_lib(), C.Conversation(ctx_lib(), backend).ask("compare?"))
check("invented comparison numbers still flagged",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[7] full loop: list -> compare -> answer within iteration budget")
ctx = ctx_lib()
backend = C.ScriptedBackend([
    use("list_sessions", {}),
    use("compare_sessions", {"session_id_a": sid_old, "session_id_b": sid_new}, tid="t2"),
    say("Your later session moved several metrics toward the tour range — nice trend."),
])
res = C.Conversation(ctx, backend).ask("Was Thursday better than Tuesday?")
check("two tool calls logged", [e["name"] for e in res.tool_log] == ["list_sessions", "compare_sessions"])
check("ends within the cap", res.stopped_reason == "end_turn" and res.iterations <= C.MAX_TOOL_ITERS)
g = C.verify_chat_grounding(ctx, res)
check("qualitative trend answer is grounded", g["grounded"], str(g["violations"]))

# =========================================================================== #
print("\n[8] chat_handler wiring (offline)")
sys.path.insert(0, str(ROOT / "deploy"))
import chat_handler as H


def scripted(*turns):
    return lambda: C.ScriptedBackend(list(turns))


out = H.chat_once(ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json",
                  "What sessions do I have?",
                  backend_factory=scripted(use("list_sessions", {}),
                                           say("You have two sessions on file.")),
                  library=[dict(j) for j in LIBRARY], current_id=CUR,
                  fetcher=lambda jid: BUNDLES.get(jid))
check("chat_once threads the library through", out["tools_used"] == ["list_sessions"]
      and out["grounded"], str(out.get("violations")))
out2 = H.chat_once(ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json",
                   "How was my tempo?",
                   backend_factory=scripted(use("get_indicator", {"key": "tempo_ratio"}),
                                            say("Tempo looks tour-like.")))
check("no library -> classic per-swing turn still works", out2["grounded"])
check("jobs_library() off without env", H.jobs_library() is None or H.JOBS_API_BASE)

print(f"\n{'=' * 50}\n  {_PASS} passed, {_FAIL} failed\n{'=' * 50}")
sys.exit(1 if _FAIL else 0)
