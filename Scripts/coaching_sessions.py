"""Session-level comparison for the coaching chat (chat v2, CHAT_V2_PLAN.md §1).

Deterministic code the session chat tools call into — the model only narrates
what comes back, same grounding discipline as every other tool:

  - `session_view(sessions, current_id)`   : compact model-facing session list
  - `order_pair(meta_a, meta_b)`           : earlier/later by date (never trust
                                             the model to know which came first)
  - `session_rollup(scorecards)`           : per-indicator MEDIAN synthetic
                                             scorecard for delta_truth
  - `compare_scorecards(early, late)`      : delta_truth + counts, chat-shaped

Aggregation notes: medians of body-scale-normalized indicator values across a
session's swings are legitimate (same golfer, same normalization). Ball-flight
carries are NOT aggregated — quality tiers differ per swing, so distance
questions stay per-swing (load_swing + get_ball_flight).

Pure stdlib; imported by the chat Lambda (Dockerfile COPYs it).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coaching_progress import delta_truth  # evaluated comparator (pg_readable)

# how many swings per session a rollup samples (newest first). Bounds the
# worst-case artifact fetches inside one API-Gateway-limited chat turn.
MAX_ROLLUP_SWINGS = 4


def session_view(sessions: list[dict], current_id: str | None = None,
                 max_sessions: int = 20) -> list[dict]:
    """session_meta.cluster_sessions output -> compact model-facing list."""
    out = []
    for s in sessions[:max_sessions]:
        out.append({
            "session_id": s["session_id"],
            "label": s.get("label"),
            "date": s.get("date"),
            "n_swings": s["n_swings"],
            "swings": [{
                "swing_id": sw.get("job_id"),
                "name": sw.get("name"),
                "date": sw.get("date"),
                **{k: sw[k] for k in ("club", "setting", "notes") if sw.get(k)},
                **({"is_current": True} if sw.get("job_id") == current_id else {}),
            } for sw in s["swings"]],
        })
    return out


def order_pair(a: dict, b: dict) -> tuple[dict, dict, bool]:
    """(a, b) with .date -> (earlier, later, dated). Falls back to given order
    (dated=False) when either date is missing/equal — the tool result then says
    the order was assumed, so the model can disclose it."""
    da, db = a.get("date") or "", b.get("date") or ""
    if da and db and da != db:
        return (a, b, True) if da < db else (b, a, True)
    return a, b, False


def session_rollup(scorecards: list[dict]) -> dict:
    """Synthetic scorecard whose indicator values are per-indicator MEDIANS
    across the session's swings. pro_band/pro_median come from the first swing
    carrying the indicator (they're reference constants); confidence_tier is
    'low' if ANY sampled swing measured it low (conservative: one bad capture
    poisons the aggregate rather than hiding inside it)."""
    inds: dict[str, dict] = {}
    keys: list[str] = []
    for sc in scorecards:
        for k in (sc.get("indicators") or {}):
            if k not in keys:
                keys.append(k)
    for k in keys:
        rows = [sc["indicators"][k] for sc in scorecards if k in (sc.get("indicators") or {})]
        vals = [r["value"] for r in rows if isinstance(r.get("value"), (int, float))]
        if not vals:
            continue
        tier = "low" if any(r.get("confidence_tier") == "low" for r in rows) \
            else (rows[0].get("confidence_tier") or "med")
        pcts = [r["percentile"] for r in rows if isinstance(r.get("percentile"), (int, float))]
        inds[k] = {
            "value": round(statistics.median(vals), 2),
            "percentile": round(statistics.median(pcts), 1) if pcts else None,
            "pro_median": rows[0].get("pro_median"),
            "pro_band": rows[0].get("pro_band"),
            "confidence_tier": tier,
            "n_swings": len(vals),
        }
    return {"indicators": inds, "meta": {"rollup_of": len(scorecards)}}


def compare_scorecards(early: dict, late: dict) -> dict:
    """delta_truth (A=earlier, B=later) reshaped for chat: per-indicator rows
    the model can quote, plus improved/regressed/unchanged counts. Only
    narratable (reliable-in-both) indicators carry values."""
    truth = delta_truth(early, late)
    rows, improved, regressed, unchanged = [], [], [], []
    for k, t in truth.items():
        if not t.get("narratable"):
            rows.append({"key": k, "comparable": False, "reason": t.get("reason")})
            continue
        d = t["direction"]
        (improved if d == "improved" else regressed if d == "regressed" else unchanged).append(k)
        rows.append({
            "key": k, "comparable": True,
            "earlier_value": t["valA"], "later_value": t["valB"],
            "delta": round(t["valB"] - t["valA"], 2),
            "change_vs_tour": d,               # improved = moved TOWARD the pro band
            "meaningful_move": t["meaningful"],
            "pro_band": t["band"],
        })
    return {
        "compared": True,
        "indicators": rows,
        "improved": improved, "regressed": regressed, "unchanged": unchanged,
        "n_improved": len(improved), "n_regressed": len(regressed),
        "how_to_phrase": (
            "improved/regressed are RELATIVE TO THE TOUR BAND (moved toward or away "
            "from it), already computed - narrate them, do not re-derive from raw "
            "deltas. Never call a change an improvement unless change_vs_tour says so. "
            "Metrics with comparable=false are not reliable enough to compare."),
    }
