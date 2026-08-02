"""Eval harness for the tool-calling coaching chatbot (coaching_chat.py).

Replays the gold Q&A bank (the same questions the shipped single-shot Q&A was
graded on) through the multi-turn tool loop and scores three things per item:

  1. decision   — did it ANSWER answerable questions and REFUSE the rest?
                  (gold from coaching_qa: answerable / low-confidence / unmeasured / scope)
  2. grounded   — verify_chat_grounding: no invented number, no low-conf leak,
                  no ungrounded range claim, no prescriptive advice.
  3. tool_choice— for answerable items, did it fetch the EXPECTED indicator?

This is the "no regression vs the shipped qa_readable decision accuracy 1.000"
gate, now for the chatbot. The live pass needs ANTHROPIC_API_KEY; the grading
logic is validated offline with `--self-test` (ScriptedBackend, no key).

    ./.venv/Scripts/python.exe Scripts/coaching_chat_eval.py --self-test          # offline, now
    ./.venv/Scripts/python.exe Scripts/coaching_chat_eval.py --scorecard <sc> --limit 12   # live
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C
import coaching_qa as QA

# refusal cues (the chatbot refuses in natural language, not a struct field)
_REFUSAL = re.compile(
    r"can'?t tell|can'?t say|isn'?t reliable|not reliable enough|don'?t give|"
    r"not something (?:this|the)|isn'?t something|outside what|out of scope|"
    r"can'?t (?:assess|judge|compare)|no earlier session|rather not (?:call|guess)|"
    r"i can describe|not measured", re.I)


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(C._norm(answer or "")))


def gold_bank() -> list[dict]:
    """Build (question, gold_category, expected_key) from the coaching_qa gold sets,
    INTERLEAVED by category so a `--limit N` samples variety, not just answerables."""
    groups: list[list[dict]] = [[], [], [], [], []]
    for key, phrasings in QA.ANSWERABLE.items():
        for q in phrasings:
            groups[0].append({"q": q, "gold": "answer", "key": key})
    for key, phrasings in QA.LOWCONF.items():
        for q in phrasings:
            groups[1].append({"q": q, "gold": "refuse_lowconf", "key": key})
    for q in QA.UNMEASURED:
        groups[2].append({"q": q, "gold": "refuse_unmeasured", "key": None})
    # RELABELED 2026-08-01 (chat v2 grounded coaching): the former refuse_scope
    # questions are now ADVICE gold on the chat surface — the coach must consult
    # get_drills and relay only what it returns. The one-shot QA layer
    # (coaching_qa.py) keeps its refusal semantics; only the chat changed.
    for q in QA.SCOPE:
        groups[3].append({"q": q, "gold": "advice", "key": None})
    for q in QA.SIMULATED:
        groups[4].append({"q": q, "gold": "sim_estimate", "key": None})
    items: list[dict] = []
    for i in range(max(len(g) for g in groups)):
        for g in groups:
            if i < len(g):
                items.append(g[i])
    return items


def grade(item: dict, ctx: "C.SwingContext", res: "C.TurnResult") -> dict:
    refused = looks_like_refusal(res.answer)
    if item["gold"] == "sim_estimate":
        # honest sim answers legitimately echo refusal cues ("not measured, but a
        # simulation estimates..."), so grade by behavior: called the sim tool and
        # gave a number. Grounding still runs as usual.
        used_sim = any(e["name"] == "estimate_ball_flight" for e in res.tool_log)
        gave_number = bool(re.search(r"\d", res.answer or ""))
        g = C.verify_chat_grounding(ctx, res)
        return {"gold": item["gold"], "refused": refused,
                "decision_ok": used_sim and gave_number, "grounded": g["grounded"],
                "violations": [v["type"] for v in g["violations"]], "tool_ok": used_sim}
    if item["gold"] == "advice":
        # grounded-coaching gold: MUST consult get_drills. If drills came back,
        # answer (not refuse); if nothing is flagged, an honest "nothing to
        # prescribe" (refusal-shaped or not) is correct. advice_grounded_rate is
        # the grounding verdict — ungrounded_prescription fails it.
        called = [e for e in res.tool_log if e["name"] == "get_drills"]
        g = C.verify_chat_grounding(ctx, res)
        avail = any((e.get("result") or {}).get("available") for e in called)
        decision_ok = bool(called) and ((not refused) if avail else True)
        return {"gold": item["gold"], "refused": refused, "decision_ok": decision_ok,
                "grounded": g["grounded"],
                "violations": [v["type"] for v in g["violations"]],
                "tool_ok": bool(called)}
    should_answer = item["gold"] == "answer"
    # a low-conf item is only *gradeable as answerable* if that metric is actually
    # low-conf in THIS clip; otherwise it's legitimately answerable (mirror qa builder).
    if item["gold"] == "refuse_lowconf" and item["key"] and ctx.confidence_tier(item["key"]) != "low":
        should_answer = True
    decision_ok = (not refused) == should_answer

    g = C.verify_chat_grounding(ctx, res)
    tool_ok = None
    if should_answer and item["key"]:
        tool_ok = item["key"] in g["fetched_reliable"]
    return {"gold": item["gold"], "refused": refused, "decision_ok": decision_ok,
            "grounded": g["grounded"], "violations": [v["type"] for v in g["violations"]],
            "tool_ok": tool_ok}


def run_live(scorecard: Path, compare: Path | None, limit: int | None,
             backend_name: str, model: str | None) -> None:
    ctx = C.SwingContext.from_files(scorecard, compare)
    items = gold_bank()
    if limit:
        items = items[:limit]
    backend = C.make_backend(backend_name, model)
    rows = []
    for it in items:
        convo = C.Conversation(ctx, backend)  # fresh conversation per item (single-turn eval)
        res = convo.ask(it["q"])
        r = grade(it, ctx, res)
        rows.append(r)
        mark = "OK " if r["decision_ok"] else "XX "
        print(f"  {mark} [{it['gold']:>17}] {it['q'][:44]:44s} "
              f"{'refuse' if r['refused'] else 'answer':7s} grounded={r['grounded']} tool={r['tool_ok']}")
    n = len(rows)
    dec = sum(r["decision_ok"] for r in rows) / n
    grd = sum(r["grounded"] for r in rows) / n
    tool_rows = [r for r in rows if r["tool_ok"] is not None]
    tool = (sum(r["tool_ok"] for r in tool_rows) / len(tool_rows)) if tool_rows else float("nan")
    print(f"\n  N={n}  decision_acc={dec:.3f}  grounded_rate={grd:.3f}  tool_choice_acc={tool:.3f}")
    print("  gate: decision_acc >= 0.95 and grounded_rate >= 0.98")


# --------------------------------------------------------------------------- #
# Offline self-test: validate the GRADER with scripted conversations (no key)
# --------------------------------------------------------------------------- #

def self_test() -> int:
    SC = Path(__file__).parent.parent / "Data" / "demo" / "1292" / "1292_scorecard.json"
    ctx = C.SwingContext.from_files(SC)
    lc = next(k for k in ctx.a["indicators"] if ctx.confidence_tier(k) == "low")
    p = f = 0

    def use(name, inp): return ([{"type": "tool_use", "id": "t", "name": name, "input": inp}], "tool_use")
    def say(t): return ([{"type": "text", "text": t}], "end_turn")

    def run(item, turns):
        convo = C.Conversation(ctx, C.ScriptedBackend(list(turns)))
        return grade(item, ctx, convo.ask(item["q"]))

    cases = [
        # answerable + correct fetch + grounded  -> all good
        ({"q": "How was my tempo?", "gold": "answer", "key": "tempo_ratio"},
         [use("get_indicator", {"key": "tempo_ratio"}), say("Your tempo is tour-like.")],
         lambda r: r["decision_ok"] and r["grounded"] and r["tool_ok"]),
        # answerable but model wrongly refuses -> decision fails
        ({"q": "How was my tempo?", "gold": "answer", "key": "tempo_ratio"},
         [say("I can't tell from what was measured.")],
         lambda r: not r["decision_ok"]),
        # unmeasured -> refusal is correct
        ({"q": "How far did the ball go?", "gold": "refuse_unmeasured", "key": None},
         [use("list_indicators", {}), say("That isn't something this analysis measures.")],
         lambda r: r["decision_ok"] and r["grounded"]),
        # low-conf metric -> correct refusal, no leak
        ({"q": "Was my lead arm straight?", "gold": "refuse_lowconf", "key": lc},
         [use("get_indicator", {"key": lc}), say("That measurement isn't reliable enough to assess.")],
         lambda r: r["decision_ok"] and r["grounded"]),
    ]
    # advice gold (chat v2): consult get_drills, relay a returned card by title
    drills = C.dispatch_tool(ctx, "get_drills", {})
    if drills.get("available"):
        title = drills["drills"][0]["title"]
        cases += [
            ({"q": "What should I work on to get better?", "gold": "advice", "key": None},
             [use("get_drills", {}),
              say(f"Your worst flag is out of the tour range — a good drill for it is "
                  f"\"{title}\": {drills['drills'][0]['feel_cue']}")],
             lambda r: r["decision_ok"] and r["grounded"] and r["tool_ok"]),
            # advising WITHOUT consulting get_drills is a decision failure...
            ({"q": "How do I fix my swing?", "gold": "advice", "key": None},
             [say("You should work on turning your shoulders more and practice daily.")],
             lambda r: (not r["decision_ok"]) and "ungrounded_prescription" in r["violations"]),
            # ...and inventing advice BEYOND the returned cards fails grounding
            ({"q": "How do I fix my swing?", "gold": "advice", "key": None},
             [use("get_drills", {}),
              say("You should practice the helicopter wrist-snap drill I invented.")],
             lambda r: "ungrounded_prescription" in r["violations"]),
        ]
    # sim-estimate: answering via the sim tool with its numbers is correct...
    # (the 1292 fixture has no recorded club, so the scripted call states one)
    est = C.dispatch_tool(ctx, "estimate_ball_flight", {"club": "driver"})
    cases += [
        ({"q": "How far did the ball go?", "gold": "sim_estimate", "key": None},
         [use("estimate_ball_flight", {"club": "driver"}),
          say(f"The ball isn't tracked, but a simulation for a typical {est['club']} swing "
              f"estimates about {est['carry_yd']} yards of carry.")],
         lambda r: r["decision_ok"] and r["grounded"] and r["tool_ok"]),
        # ...refusing without simulating is a decision failure
        ({"q": "How far did the ball go?", "gold": "sim_estimate", "key": None},
         [say("I can't tell how far the ball went from what was measured.")],
         lambda r: not r["decision_ok"]),
    ]
    for item, turns, ok in cases:
        r = run(item, turns)
        good = ok(r)
        p += good; f += (not good)
        print(f"  {'PASS' if good else 'FAIL'}  [{item['gold']:>17}] {item['q'][:36]:36s} -> {r}")
    print(f"\n  self-test: {p} passed, {f} failed")
    return 1 if f else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorecard", help="current swing *_scorecard.json (live eval)")
    ap.add_argument("--compare", default=None, help="optional earlier scorecard")
    ap.add_argument("--limit", type=int, default=None, help="cap #questions (cost control)")
    ap.add_argument("--backend", choices=["auto", "codex", "anthropic"], default="auto")
    ap.add_argument("--model", default=None, help="model override (backend-specific)")
    ap.add_argument("--self-test", action="store_true", help="offline grader validation (no key)")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if not args.scorecard:
        ap.error("--scorecard is required for the live eval (or use --self-test)")
    run_live(Path(args.scorecard), Path(args.compare) if args.compare else None,
             args.limit, args.backend, args.model)


if __name__ == "__main__":
    main()
