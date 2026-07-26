"""Motion Caddie — coaching Q&A CHATBOT with tool calling.

Successor to the one-shot `coaching_ask.py`. Instead of dumping the whole
scorecard into a single structured-output call, the model is given **tools** and
must FETCH the data it reasons over:

    list_indicators / get_indicator / get_flagged_observations /
    get_swing_summary / explain_term / compare_indicator

Grounding becomes structural: the model can only state a number it actually
pulled from a tool result. A deterministic transcript verifier
(`verify_chat_grounding`) then checks the answer against the tool log.

Two interchangeable backends (so the whole loop is testable offline):
  - AnthropicBackend  : claude-opus-4-8 native tool use (prod / AWS path).
                        Needs ANTHROPIC_API_KEY.
  - ScriptedBackend   : canned turns for deterministic tests (no key, no network).

CLI:
    python coaching_chat.py --scorecard <stem>_scorecard.json --question "How was my tempo?"
    python coaching_chat.py --scorecard <stem>_scorecard.json            # interactive REPL
    python coaching_chat.py --scorecard <a> --compare <b> --question "Did my hips improve?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import coaching_llm_summary_v2 as v2  # KB loader + KB block builder (no clip numbers)
from coaching_persona import DEFAULT_PERSONA, load_persona #Loads the narration persona
import ball_flight as BF  # physics ball-flight sim (McNally CVPR'23W baseline)

KB_PATH = PROJECT_ROOT / "Data" / "coaching" / "indicator_kb.json"
CONF_PATH = PROJECT_ROOT / "Data" / "coaching" / "indicator_confidence.json"
DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOOL_ITERS = 6  # hard cap on tool round-trips per user turn (cost + loop guard)

# --------------------------------------------------------------------------- #
# SwingContext — the ONLY thing tools can read (grounding boundary)
# --------------------------------------------------------------------------- #


def _load_confidence() -> dict:
    """Per-indicator tiers. The file nests them under an "indicators" key."""
    try:
        data = json.loads(CONF_PATH.read_text(encoding="utf-8"))
        return data.get("indicators", data)
    except OSError:
        return {}


@dataclass
class SwingContext:
    """Wraps one scorecard (and optionally a second, for progress questions)
    plus the KB and per-indicator confidence. Pure data accessor — no model."""

    a: dict
    kb: dict
    b: dict | None = None
    ball: dict | None = None            # ball_3d.json (measured track + flight fit)
    conf: dict = field(default_factory=_load_confidence)
    a_label: str = "this swing"
    b_label: str = "the earlier swing"

    @classmethod
    def from_files(cls, a_path: str | Path, b_path: str | Path | None = None,
                   ball_path: str | Path | None = None) -> "SwingContext":
        a = json.loads(Path(a_path).read_text(encoding="utf-8"))
        b = json.loads(Path(b_path).read_text(encoding="utf-8")) if b_path else None
        ball = None
        if ball_path and Path(ball_path).exists():
            try:
                ball = json.loads(Path(ball_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                ball = None
        return cls(a=a, kb=v2.load_kb(), b=b, ball=ball)

    # -- per-indicator helpers ------------------------------------------------
    def has(self, key: str, which: str = "a") -> bool:
        sc = self.b if which == "b" else self.a
        return bool(sc) and key in sc.get("indicators", {})

    def confidence_tier(self, key: str, which: str = "a") -> str:
        sc = self.b if which == "b" else self.a
        ind = (sc or {}).get("indicators", {}).get(key, {})
        if ind.get("confidence_tier"):
            return ind["confidence_tier"]
        return self.conf.get(key, {}).get("tier", "med")

    def in_range(self, key: str, which: str = "a") -> bool:
        sc = self.b if which == "b" else self.a
        v = sc["indicators"][key]
        lo, hi = v["pro_band"]
        return lo <= v["value"] <= hi

    def indicator_view(self, key: str, which: str = "a") -> dict:
        """Structured, grounded read of one indicator + its KB card."""
        sc = self.b if which == "b" else self.a
        v = sc["indicators"][key]
        card = self.kb.get("indicators", {}).get(key, {})
        tier = self.confidence_tier(key, which)
        return {
            "key": key,
            "label": card.get("label", key),
            "value": v["value"],
            "unit": card.get("unit", ""),
            "tour_median": v.get("pro_median"),
            "pro_band": v.get("pro_band"),
            "percentile": v.get("percentile"),
            "in_tour_range": self.in_range(key, which),
            "confidence_tier": tier,
            "reliable": tier != "low",
            "measures": card.get("measures", ""),
            "why": card.get("why", ""),
        }


# --------------------------------------------------------------------------- #
# Tools — deterministic functions over a SwingContext. Each returns a JSON-able
# dict. "Not measured" is a normal result (so the model refuses, not crashes).
# --------------------------------------------------------------------------- #

ToolFn = Callable[[SwingContext, dict], dict]


def _t_list_indicators(ctx: SwingContext, _inp: dict) -> dict:
    out = []
    for key in ctx.a.get("indicators", {}):
        card = ctx.kb.get("indicators", {}).get(key, {})
        tier = ctx.confidence_tier(key)
        out.append({
            "key": key,
            "label": card.get("label", key),
            "plain_name": card.get("plain_name", ""),
            "confidence_tier": tier,
            "reliable": tier != "low",
        })
    return {"indicators": out, "count": len(out)}


def _t_get_indicator(ctx: SwingContext, inp: dict) -> dict:
    key = (inp or {}).get("key", "")
    if not ctx.has(key):
        return {"measured": False, "key": key,
                "note": "This metric is not measured for this swing. Do not guess a value."}
    view = ctx.indicator_view(key)
    view["measured"] = True
    if not view["reliable"]:
        view["note"] = ("This metric is LOW CONFIDENCE from a single camera. Do not state its "
                        "value or judge it; tell the golfer it isn't reliable enough to assess.")
    return view


def _t_get_flagged_observations(ctx: SwingContext, _inp: dict) -> dict:
    flags = [f for f in ctx.a.get("feedback", []) if f.get("severity") == "review"]
    return {"flagged": [{"key": f["indicator"], "label": f.get("label"),
                         "event": f.get("event"), "message": f.get("message"),
                         "percentile": f.get("percentile")} for f in flags],
            "count": len(flags)}


def _t_get_swing_summary(ctx: SwingContext, _inp: dict) -> dict:
    inds = ctx.a.get("indicators", {})
    reliable = [k for k in inds if ctx.confidence_tier(k) != "low"]
    in_range = [k for k in reliable if ctx.in_range(k)]
    flagged = [f["indicator"] for f in ctx.a.get("feedback", []) if f.get("severity") == "review"]
    return {
        "n_indicators": len(inds),
        "n_reliable": len(reliable),
        "n_in_tour_range": len(in_range),
        "n_flagged": len(flagged),
        "flagged_keys": flagged,
        "events_detected": list(ctx.a.get("events", {}).keys()),
        "player": ctx.a.get("meta", {}).get("player"),
    }


def _t_explain_term(ctx: SwingContext, inp: dict) -> dict:
    term = (inp or {}).get("term", "").strip().lower()
    gloss = {k.lower(): val for k, val in ctx.kb.get("glossary", {}).items()}
    if term in gloss:
        return {"term": term, "in_glossary": True, "gloss": gloss[term]}
    # also look in per-indicator glosses
    for card in ctx.kb.get("indicators", {}).values():
        for t, g in (card.get("glosses") or {}).items():
            if t.lower() == term:
                return {"term": term, "in_glossary": True, "gloss": g}
    return {"term": term, "in_glossary": False,
            "note": "Not in the approved glossary. Explain only in plain words you are sure of, or say you can't define it."}


def _t_compare_indicator(ctx: SwingContext, inp: dict) -> dict:
    key = (inp or {}).get("key", "")
    if ctx.b is None:
        return {"comparable": False, "key": key,
                "note": "No earlier swing is loaded, so progress can't be compared."}
    if not ctx.has(key, "a") or not ctx.has(key, "b"):
        return {"comparable": False, "key": key,
                "note": "This metric isn't measured in both swings."}
    if ctx.confidence_tier(key, "a") == "low" or ctx.confidence_tier(key, "b") == "low":
        return {"comparable": False, "key": key, "confidence_tier": "low",
                "note": "Low-confidence metric; not reliable enough to compare."}
    va = ctx.a["indicators"][key]["value"]
    vb = ctx.b["indicators"][key]["value"]
    delta = round(va - vb, 2)
    return {
        "comparable": True, "key": key,
        "earlier_value": vb, "current_value": va, "delta": delta,
        "direction": "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged",
        "current_in_tour_range": ctx.in_range(key, "a"),
        "earlier_in_tour_range": ctx.in_range(key, "b"),
    }


def _t_estimate_ball_flight(ctx: SwingContext, inp: dict) -> dict:
    inp = inp or {}
    meta = ctx.a.get("meta", {})
    club = inp.get("club") or meta.get("club")
    tier = inp.get("skill_tier")
    if tier not in BF.TIERS:
        # GolfDB demo clips carry a tour player's NAME (letters + spaces, e.g.
        # "TIGER WOODS") -> tour/lpga launch numbers. Uploads carry a filename
        # stem ("IMG_8107"), a job id, or nothing -> amateur.
        player = str(meta.get("player") or "").strip().lower()
        is_name = (player and player != "unknown"
                   and any(c.isalpha() for c in player)
                   and all(c.isalpha() or c.isspace() for c in player))
        tier = ("lpga" if str(meta.get("sex", "")).lower().startswith("f") else "tour") \
            if is_name else "amateur"
    overrides = {k: inp[k] for k in ("ball_speed_mph", "launch_angle_deg",
                                     "backspin_rpm", "sidespin_rpm") if k in inp}

    # swing-aware nudge: THIS swing's measured hand speed vs the GolfDB pro
    # median scales the assumed ball speed (capped ±12% inside estimate_for_club).
    # Low-confidence single-camera signal — disclosed in the result, never a
    # substitute for a golfer-stated ball speed.
    speed_scale = 1.0
    hs = ctx.a.get("indicators", {}).get("hand_speed_impact_bs")
    if hs and isinstance(hs.get("value"), (int, float)) and hs.get("pro_median"):
        speed_scale = float(hs["value"]) / float(hs["pro_median"])

    res = BF.estimate_for_club(club, tier, overrides, speed_scale=speed_scale)
    if res.get("estimated") and speed_scale != 1.0 and \
            "ball_speed_mph" not in res["assumed_launch"]["overridden_by_golfer"]:
        pct = round((min(max(speed_scale, 0.88), 1.12) - 1.0) * 100)
        if abs(pct) >= 3:      # don't narrate a within-noise nudge
            res["swing_speed_adjustment"] = (
                f"assumed ball speed nudged {pct:+d}% because this swing's measured hand "
                f"speed is {'above' if pct > 0 else 'below'} tour-typical (rough single-"
                f"camera estimate)")
    return res


def _t_get_ball_flight(ctx: SwingContext, inp: dict) -> dict:
    """THIS swing's ball flight: measured from the video track when available,
    else the physics-simulation fallback. Display numbers are pre-rounded -
    the grounding verifier matches answer numbers against these exactly."""
    ball = ctx.ball or {}
    q = ball.get("quality")
    if q in ("measured", "partial"):
        fit = ball.get("fit", {})
        fl = ball.get("flight", {})
        az = fit.get("azimuth_deg") or 0.0
        direction = ("right of the camera line" if az > 3
                     else "left of the camera line" if az < -3
                     else "straight down the camera line")
        res: dict = {
            "quality": q,
            "launch_angle_deg": fit.get("launch_deg"),
            "start_direction_deg": az,
            "start_direction": direction,
            "ball_speed_mph": fit.get("ball_speed_mph"),
            "carry_yd": fl.get("carry_yd"),
            "apex_yd": fl.get("apex_yd"),
            "flight_time_s": fl.get("flight_time_s"),
            "n_track_points": ball.get("n_track_points"),
            "direction_note": ball.get("azimuth_note"),
        }
        if q == "measured":
            ci = ball.get("ci_10_90") or {}
            if ci.get("carry_yd"):
                res["carry_range_yd"] = [round(ci["carry_yd"][0]), round(ci["carry_yd"][1])]
            if ci.get("speed_mph"):
                res["speed_range_mph"] = [round(ci["speed_mph"][0]), round(ci["speed_mph"][1])]
            res["how_to_phrase"] = (
                "MEASURED: the ball was tracked in the video and these numbers come from a "
                "physics fit to that track. Answer distance questions confidently - e.g. "
                "'your carry was about " + str(res["carry_yd"]) + " yards, measured from "
                "your video' - and give the range only if asked about precision.")
        else:
            res["speed_source"] = "club-typical (not measured)"
            res["how_to_phrase"] = (
                "PARTIAL: launch direction and angle were measured from the video ball track, "
                "but ball speed was assumed from club norms. State direction and launch "
                "confidently; give carry as an estimate informed by the measured launch.")
        traj = ball.get("trajectory_world")
        if traj:
            res["_ui_trajectory"] = traj
        return res
    res = _t_estimate_ball_flight(ctx, inp)
    res["quality"] = "simulated"
    res["how_to_phrase"] = ("SIMULATED: the ball was not trackable in this video; this is a "
                            "physics simulation from typical launch conditions. Always say so.")
    return res


# (name -> (json-schema, fn)). Schemas are the model-facing tool contract.
TOOLS: dict[str, tuple[dict, ToolFn]] = {
    "list_indicators": ({
        "description": "List every swing metric that is measured for this golfer, with its "
                       "plain-language name and whether it is reliable. Call this first to see "
                       "what you can answer.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }, _t_list_indicators),
    "get_indicator": ({
        "description": "Get the measured value of ONE swing metric vs the tour-pro range "
                       "(value, tour median, pro band, percentile, in-range flag, confidence). "
                       "This is your only source of a metric's number — never state a value you "
                       "did not get from here.",
        "input_schema": {"type": "object",
                         "properties": {"key": {"type": "string", "description": "indicator key, e.g. tempo_ratio"}},
                         "required": ["key"], "additionalProperties": False},
    }, _t_get_indicator),
    "get_flagged_observations": ({
        "description": "List the metrics that fell outside the typical tour range for this swing "
                       "(the things that 'stood out').",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }, _t_get_flagged_observations),
    "get_swing_summary": ({
        "description": "High-level summary: how many metrics were measured, how many are in the "
                       "tour range, how many were flagged, and which swing events were detected.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }, _t_get_swing_summary),
    "explain_term": ({
        "description": "Look up the approved plain-language definition of a golf term from the "
                       "controlled glossary. Use this instead of inventing a definition.",
        "input_schema": {"type": "object",
                         "properties": {"term": {"type": "string"}},
                         "required": ["term"], "additionalProperties": False},
    }, _t_explain_term),
    "compare_indicator": ({
        "description": "Compare ONE metric between the current swing and an earlier swing "
                       "(only works if an earlier swing was loaded). Returns both values, the "
                       "change, and the direction.",
        "input_schema": {"type": "object",
                         "properties": {"key": {"type": "string"}},
                         "required": ["key"], "additionalProperties": False},
    }, _t_compare_indicator),
    "get_ball_flight": ({
        "description": "Get THIS swing's ball flight. Uses the MEASURED ball track from the "
                       "video when available (quality 'measured': speed/launch/carry from the "
                       "actual tracked ball, with confidence ranges; 'partial': launch "
                       "direction/angle measured, speed assumed from club norms; 'simulated': "
                       "no track - physics simulation). ALWAYS call this first for questions "
                       "about how far or where the ball went, carry, height, or trajectory, "
                       "and follow the result's how_to_phrase guidance.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }, _t_get_ball_flight),
    "estimate_ball_flight": ({
        "description": "WHAT-IF simulator (different club or golfer-stated launch numbers). "
                       "For this swing's actual flight call get_ball_flight instead. "
                       "SIMULATE the likely ball flight for this swing's club with a physics "
                       "model (McNally et al. 2023). Returns estimated carry, curve, apex and "
                       "flight time plus the launch conditions assumed. This is an ESTIMATE "
                       "from typical launch conditions for the club — the ball itself is not "
                       "tracked in the video. Use it when the golfer asks how far the ball "
                       "went / would go, or about carry or trajectory. Pass any launch numbers "
                       "the golfer states as inputs.",
        "input_schema": {"type": "object", "properties": {
            "club": {"type": "string",
                     "description": "override the recorded club (driver, 3 wood, hybrid, "
                                    "5/7/9 iron, wedge) — e.g. for what-if questions"},
            "skill_tier": {"type": "string", "enum": ["tour", "lpga", "amateur"],
                           "description": "who to assume typical launch numbers for; "
                                          "default: tour for named pros, else amateur"},
            "ball_speed_mph": {"type": "number", "description": "golfer-stated override"},
            "launch_angle_deg": {"type": "number", "description": "golfer-stated override"},
            "backspin_rpm": {"type": "number", "description": "golfer-stated override"},
            "sidespin_rpm": {"type": "number",
                             "description": "golfer-stated override; >0 fade, <0 draw"},
        }, "additionalProperties": False},
    }, _t_estimate_ball_flight),
}


def tool_specs(with_compare: bool) -> list[dict]:
    """Anthropic-format tool list. `compare_indicator` only offered when a 2nd clip is loaded."""
    names = list(TOOLS)
    if not with_compare:
        names.remove("compare_indicator")
    return [{"name": n, **TOOLS[n][0]} for n in names]


def dispatch_tool(ctx: SwingContext, name: str, inp: dict) -> dict:
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"unknown tool {name!r}"}
    return spec[1](ctx, inp or {})


def model_visible(result: dict) -> dict:
    """Tool-result view sent to the model. Keys prefixed "_ui" carry UI-only
    payloads (e.g. trajectory points) — they stay in tool_log for the frontend
    but are stripped here, and excluded from the grounding-number pool."""
    return {k: v for k, v in result.items() if not k.startswith("_ui")}


# --------------------------------------------------------------------------- #
# System prompt (refusal taxonomy + tool discipline). No clip numbers here —
# the KB block carries definitions only; all values come from tools.
# --------------------------------------------------------------------------- #

SYSTEM_RULES = """You are Motion Caddie, a friendly golf-swing assistant. You answer a golfer's
questions about THEIR swing using ONLY data you fetch with the tools provided. The swing was
already measured; you do not re-analyze video.

How to work:
  - To answer anything about a metric, CALL `get_indicator` (or `compare_indicator`) first. Never
    state a number, percentile, or in/out-of-range judgement you did not get from a tool result.
  - Use `list_indicators` to see what is measurable, `get_flagged_observations` / `get_swing_summary`
    for overviews, and `explain_term` for golf-term definitions. Gloss any golf term briefly.

Handle broad and multi-part questions (don't punt to "which one?"):
  - BROAD asks ("biggest takeaways", "how did I do", "summarize my swing", "what's the gist"):
    call `get_swing_summary` and `get_flagged_observations`, then `get_indicator` on a few notable
    metrics, and give a short rundown — a couple of strengths, then anything outside the tour range
    (and, if an earlier swing is loaded, what changed). A few bullet-like lines is ideal.
  - MULTIPLE metrics in one question, or "all of them" / "everything": fetch EACH with
    `get_indicator` (or `compare_indicator` when they ask about change) and give one short line per
    metric. Do not answer only the first one.
  - FOLLOW-UPS and references ("all of the above", "and my hips?", "those"): resolve them from the
    conversation so far, then fetch and answer.

BALL FLIGHT (how far / where did it go, carry, height, trajectory):
  - Call `get_ball_flight` first. Its `quality` field tells you how to answer:
      "measured"  -> the ball WAS tracked in the video; speed/launch/carry come from a physics
                     fit to the real track. Answer distance questions confidently ("your carry
                     was about 260 yards, measured from your video"); cite the confidence range
                     only if asked about precision.
      "partial"   -> launch direction + angle are measured; ball speed is assumed from club
                     norms. State direction/launch confidently; frame carry as an estimate
                     informed by the measured launch.
      "simulated" -> no usable ball track; a physics simulation from typical launch conditions.
                     ALWAYS disclose that it is a simulated estimate, not a measurement.
  - `estimate_ball_flight` is only for what-if questions (a different club, golfer-stated
    launch numbers). Quote numbers exactly as returned; follow each result's how_to_phrase.

When to REFUSE (do not guess):
  - UNMEASURED: the question is about something not in `list_indicators` and not simulable
    (grip, which direction the ball started, club face/path, swing plane, wrist hinge,
    clubhead speed, club choice, ...). Say plainly you can't tell from what was measured.
  - LOW CONFIDENCE: `get_indicator` returns reliable=false (e.g. arm bend from one camera). Do not
    reveal or judge that value; say it isn't reliable enough to assess.
  - FIX / ADVICE: the golfer asks what to change, fix, drill, or practice. Say you can describe the
    swing but not prescribe fixes.

Style: warm, plain, short. 1-3 sentences for most answers. Invent nothing. Use only glossary wording
for golf terms."""


def build_system(ctx: SwingContext) -> list[dict]:
    """Build the chat system prompt from rules, persona, and KB."""
    persona_block = load_persona(DEFAULT_PERSONA)

    return [
        {"type": "text", "text": SYSTEM_RULES},
        {"type": "text", "text": persona_block},
        {
            "type": "text",
            "text": v2.build_kb_block(ctx.kb),
            "cache_control": {"type": "ephemeral"},
        },
    ]


# --------------------------------------------------------------------------- #
# Backends — normalized to Anthropic wire format (content-block dicts)
# --------------------------------------------------------------------------- #


@dataclass
class ModelTurn:
    """One assistant turn, normalized. `content` is appendable to `messages`."""
    content: list[dict]
    stop_reason: str

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class Backend:
    def create(self, system: list[dict], messages: list[dict], tools: list[dict]) -> ModelTurn:
        raise NotImplementedError


class AnthropicBackend(Backend):
    """claude-opus-4-8 native tool use.

    Thinking is OFF by default: this is short grounded Q&A that does not need it,
    and disabling it sidesteps two footguns — (a) with thinking ON you MUST resend
    the assistant's thinking blocks (with signatures) alongside tool_results in the
    SAME loop or the API 400s, and (b) thinking tokens count against max_tokens,
    raising truncation risk. If thinking is enabled we DO preserve the blocks."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
                 timeout: float = 40, max_retries: int = 2, thinking: bool = False):
        import anthropic
        self.max_tokens = max_tokens
        self.thinking = thinking
        provider = os.environ.get("CHAT_BACKEND_PROVIDER", "anthropic").lower()
        if provider == "bedrock":
            # Claude on Amazon Bedrock (Mantle client — same Messages API surface,
            # SigV4 auth, no ANTHROPIC_API_KEY). Bedrock model IDs carry an
            # `anthropic.` prefix; override with BEDROCK_MODEL if the account
            # requires an inference-profile id (us.anthropic....).
            self.model = os.environ.get("BEDROCK_MODEL", f"anthropic.{model}")
            self._client = anthropic.AnthropicBedrockMantle(
                aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                timeout=timeout, max_retries=max_retries)
        elif provider == "bedrock-runtime":
            # Legacy bedrock-runtime (InvokeModel) path — separate entitlement from
            # Mantle. Model ids here are inference profiles (us.anthropic....v1:0).
            self.model = os.environ.get(
                "BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
            self._client = anthropic.AnthropicBedrock(
                aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                timeout=timeout, max_retries=max_retries)
        else:
            self.model = model
            self._client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)

    def create(self, system, messages, tools) -> ModelTurn:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        msg = self._client.messages.create(**kwargs)
        content: list[dict] = []
        for b in msg.content:
            if b.type == "text":
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            elif b.type == "thinking":  # preserve verbatim (signature required for resend)
                content.append({"type": "thinking", "thinking": b.thinking, "signature": b.signature})
            elif b.type == "redacted_thinking":
                content.append({"type": "redacted_thinking", "data": b.data})
        return ModelTurn(content=content, stop_reason=msg.stop_reason)


class ScriptedBackend(Backend):
    """Deterministic offline backend. `turns` is a list of (content_blocks, stop_reason).
    Each create() pops the next scripted turn, ignoring the input — for tests."""

    def __init__(self, turns: list[tuple[list[dict], str]]):
        self._turns = list(turns)
        self.calls: list[dict] = []  # records what the loop sent (for assertions)

    def create(self, system, messages, tools) -> ModelTurn:
        import copy
        self.calls.append({"messages": copy.deepcopy(messages), "n_tools": len(tools)})
        if not self._turns:
            return ModelTurn(content=[{"type": "text", "text": "(no more scripted turns)"}], stop_reason="end_turn")
        content, stop = self._turns.pop(0)
        return ModelTurn(content=content, stop_reason=stop)


# The Anthropic Messages API has native tool use; the Codex CLI does not, so we
# emulate ONE step of the loop with a structured-output decision: the model
# either requests tool calls or gives a final answer. The Conversation loop
# handles dispatch + re-prompting the same way for both backends.
CODEX_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["call_tools", "final"]},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "key": {"type": "string", "description": "indicator key, or '' if N/A"},
                    "term": {"type": "string", "description": "glossary term, or '' if N/A"},
                },
                "required": ["name", "key", "term"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string", "description": "final answer text, or '' when calling tools"},
    },
    "required": ["action", "tool_calls", "answer"],
    "additionalProperties": False,
}


def _render_history(messages: list[dict]) -> str:
    """Serialize the Anthropic-format transcript into text Codex can read."""
    lines: list[str] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            lines.append(f"{role.upper()}: {content}")
            continue
        for b in content:
            t = b.get("type")
            if t == "text":
                lines.append(f"{role.upper()}: {b['text']}")
            elif t == "tool_use":
                lines.append(f"ASSISTANT called {b['name']}({json.dumps(b.get('input', {}))})")
            elif t == "tool_result":
                lines.append(f"TOOL RESULT: {b['content']}")
    return "\n".join(lines)


class CodexBackend(Backend):
    """Runs the tool-use loop through the authed Codex CLI (no Anthropic key needed)."""

    def __init__(self, model: str | None = None, timeout: int = 120):
        self.model = model
        self.timeout = timeout
        self._seq = 0

    def create(self, system, messages, tools) -> ModelTurn:
        sys_text = "\n\n".join(b["text"] for b in system if b.get("type") == "text")
        if tools:
            catalog = "\n".join(
                f"- {t['name']}(" + ", ".join((t.get("input_schema", {}).get("properties") or {}).keys())
                + f"): {t['description']}" for t in tools)
            tool_block = ("\nTOOLS YOU CAN CALL (fetch data before stating any number):\n" + catalog +
                          "\n\nDecide your next step for the LATEST user message. If you still need "
                          "data, set action=\"call_tools\" and list tool_calls (name + key or term; "
                          "leave the other \"\"). When you have enough, set action=\"final\" and write "
                          "the answer using ONLY numbers from TOOL RESULTS. Follow the refusal rules.")
        else:
            tool_block = ("\nNo tools available now — set action=\"final\" and answer from the TOOL "
                          "RESULTS already gathered.")
        prompt = (sys_text + "\n\n=== CONVERSATION ===\n" + _render_history(messages) + "\n" + tool_block)
        out = v2.call_codex(prompt, model=self.model, timeout=self.timeout,
                            schema=CODEX_DECISION_SCHEMA)
        calls = out.get("tool_calls") or []
        if out.get("action") == "call_tools" and calls:
            content = []
            for tc in calls:
                self._seq += 1
                inp = {}
                if tc.get("key"):
                    inp["key"] = tc["key"]
                if tc.get("term"):
                    inp["term"] = tc["term"]
                content.append({"type": "tool_use", "id": f"c{self._seq}",
                                "name": tc.get("name", ""), "input": inp})
            return ModelTurn(content=content, stop_reason="tool_use")
        return ModelTurn(content=[{"type": "text", "text": out.get("answer", "")}], stop_reason="end_turn")


# --------------------------------------------------------------------------- #
# The tool-use loop (multi-turn aware)
# --------------------------------------------------------------------------- #


@dataclass
class TurnResult:
    answer: str
    tool_log: list[dict]          # [{name, input, result}]
    iterations: int
    stopped_reason: str           # "end_turn" | "max_iters"


class Conversation:
    """Holds message history so follow-up questions keep context. One per session."""

    def __init__(self, ctx: SwingContext, backend: Backend, max_tool_iters: int = MAX_TOOL_ITERS):
        self.ctx = ctx
        self.backend = backend
        self.max_tool_iters = max_tool_iters
        self.system = build_system(ctx)
        self.tools = tool_specs(with_compare=ctx.b is not None)
        self.messages: list[dict] = []

    def ask(self, question: str) -> TurnResult:
        self.messages.append({"role": "user", "content": question})
        tool_log: list[dict] = []
        for i in range(1, self.max_tool_iters + 1):
            turn = self.backend.create(self.system, self.messages, self.tools)
            self.messages.append({"role": "assistant", "content": turn.content})
            sr = turn.stop_reason
            if sr == "pause_turn":
                continue  # server-side pause: resend the partial assistant turn to resume
            if sr == "tool_use" and turn.tool_uses:
                results = []
                for tu in turn.tool_uses:
                    res = dispatch_tool(self.ctx, tu["name"], tu.get("input", {}))
                    tool_log.append({"name": tu["name"], "input": tu.get("input", {}), "result": res})
                    results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                    "content": json.dumps(model_visible(res))})
                self.messages.append({"role": "user", "content": results})
                continue
            # end_turn / stop_sequence / max_tokens / refusal → this turn is done
            return TurnResult(turn.text, tool_log, i, sr or "end_turn")
        # hit the tool-round cap. Fold the "answer now" instruction into the LAST user
        # message (the tool_result block) so we never send two user messages in a row,
        # then make one final tool-free call.
        last = self.messages[-1]
        instr = ("You have gathered enough. Answer the question now from the tool results "
                 "above; do not request more tools.")
        if last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append({"type": "text", "text": instr})
        else:
            self.messages.append({"role": "user", "content": instr})
        turn = self.backend.create(self.system, self.messages, [])
        self.messages.append({"role": "assistant", "content": turn.content})
        return TurnResult(turn.text, tool_log, self.max_tool_iters, "max_iters")


# --------------------------------------------------------------------------- #
# Deterministic transcript grounding verifier
# --------------------------------------------------------------------------- #

import re

PRESCRIPTIVE = [r"\byou should\b", r"\btry to\b", r"\bwork on\b", r"\bfocus on\b",
                r"\bmake sure\b", r"\bto fix\b", r"\bpractice\b", r"\bdrill\b",
                r"\byou need to\b", r"\baim to\b"]

# a turn that DECLINES ("I can't tell you what to work on", "I don't give fixes")
# echoes fix-words without giving advice — don't count those as prescriptive.
_REFUSAL_CUE = re.compile(
    r"can'?t (?:tell|say|assess|judge|compare|give|prescribe)|prescrib|"
    r"isn'?t (?:reliable|something)|not reliable enough|don'?t give|"
    r"not something (?:this|the)|out of scope|outside what|i can describe|"
    r"rather not|no earlier session|not measured|don'?t (?:coach|tell you what)", re.I)

# integers this small are almost always phrasing ("3-to-1 tempo", "1-2 sentences",
# "six weeks" as digits) rather than a measured value — skip them to avoid false
# positives. Any number with a decimal point, or an integer >= this, must be grounded.
_SMALL_INT_CUTOFF = 13


def _numbers_in(obj) -> list[float]:
    """Recursively collect every numeric leaf in a tool-result object."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


_RANGE_WORDS = ("in range", "tour range", "within", "outside", "above", "below",
                "more than", "less than", "tour-like", "in the range")


def _norm(text: str) -> str:
    """Normalize smart quotes/dashes so regexes match the model's real output."""
    return (text or "").translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))


# In "38-68" (an en/em-dash range after _norm, or a plain hyphen range) the "-"
# is a separator, not a minus sign — only read it as a sign when it does NOT
# directly follow a digit or decimal point.
_NUM_TOKEN = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")


def _is_measurementish(tok: str) -> bool:
    """Number tokens the verifier holds to grounding (see _SMALL_INT_CUTOFF)."""
    return ("." in tok) or abs(float(tok)) >= _SMALL_INT_CUTOFF


def _display_match(tok: str, target: float) -> bool:
    """True if `tok` restates `target` at the token's own displayed precision —
    "21" for 21.25, "38" for 37.5, "68" for 68.1. Floor of 0.05 keeps the old
    flat tolerance for decimal tokens."""
    decimals = len(tok.split(".", 1)[1]) if "." in tok else 0
    return abs(float(tok) - target) <= max(0.05, 0.5 * 10.0 ** -decimals) + 1e-9


def verify_chat_grounding(ctx: SwingContext, result: TurnResult) -> dict:
    """Check the final answer against the tool log:
      - low_confidence_leak : answer states a low-confidence metric's value or judges its range
      - ungrounded_range     : answer asserts in/out-of-range for a metric never fetched
      - ungrounded_number    : answer states a numeric value not present in any RELIABLE tool result
      - prescriptive         : answer gives a fix/advice (unless it's a refusal echoing fix-words)
    Naming a low-confidence metric *to decline it* is allowed (not a leak). Returns {grounded,...}."""
    answer = _norm(result.answer or "")
    answer_l = answer.lower()
    is_refusal = bool(_REFUSAL_CUE.search(answer))
    violations: list[dict] = []

    fetched_ok: set[str] = set()
    fetched_lowconf: set[str] = set()
    lowconf_values: dict[str, float] = {}
    grounded_nums: set[float] = set()
    sim_ok = False   # a successful flight simulation legitimizes range phrasing
                     # ("above tour-typical") that narrates the tool's own notes
    for entry in result.tool_log:
        r = model_visible(entry["result"])  # UI-only payloads can't ground an answer
        if entry["name"] == "estimate_ball_flight" and r.get("estimated"):
            sim_ok = True
        if entry["name"] == "get_ball_flight" and r.get("quality"):
            sim_ok = True
        if entry["name"] == "get_flagged_observations":
            # each flag IS a tool-asserted out-of-range judgment for its metric,
            # so narrating it with range words is grounded
            for f in r.get("flagged", []):
                if f.get("key"):
                    fetched_ok.add(f["key"])
        if entry["name"] in ("get_indicator", "compare_indicator"):
            key = r.get("key", "")
            if r.get("reliable") is False or r.get("confidence_tier") == "low":
                fetched_lowconf.add(key)
                for vf in ("value", "current_value", "earlier_value"):
                    if isinstance(r.get(vf), (int, float)):
                        lowconf_values[key] = round(float(r[vf]), 2)
                # the tour band/median are KB constants, citable even in a refusal
                for n in _numbers_in([r.get("pro_band"), r.get("tour_median")]):
                    grounded_nums.add(round(n, 2))
                continue  # the low-confidence MEASUREMENT is never allowed → keep it out of grounded_nums
            if r.get("measured") or r.get("comparable"):
                fetched_ok.add(key)
        for n in _numbers_in(r):  # reliable indicators + summary/flagged/list tool numbers
            grounded_nums.add(round(n, 2))

    answer_tokens = _NUM_TOKEN.findall(answer)

    # leak = the low-conf VALUE is stated (exactly, or display-rounded for a
    # measurement-looking token), or a range judgment is made while NOT refusing
    for key in fetched_lowconf:
        val = lowconf_values.get(key)
        leaked_value = val is not None and any(
            abs(val - float(t)) <= 0.05 or (_is_measurementish(t) and _display_match(t, val))
            for t in answer_tokens)
        label = ctx.kb.get("indicators", {}).get(key, {}).get("label", "").lower()
        judged = (not is_refusal) and bool(label) and label in answer_l and \
            any(w in answer_l for w in _RANGE_WORDS)
        if leaked_value or judged:
            violations.append({"type": "low_confidence_leak", "key": key})

    # every "measurement-looking" number in the answer must trace to a reliable
    # tool result — allowing display rounding ("21 degrees" for a fetched 21.25)
    for tok in answer_tokens:
        if not _is_measurementish(tok):
            continue
        if not any(_display_match(tok, g) for g in grounded_nums):
            violations.append({"type": "ungrounded_number", "value": float(tok)})

    # range words present but no reliable get_indicator/compare call (and no
    # successful flight simulation, whose notes use range phrasing) → ungrounded
    if any(p in answer_l for p in _RANGE_WORDS) and not fetched_ok and not sim_ok:
        violations.append({"type": "ungrounded_range_claim"})

    if not is_refusal and any(re.search(p, answer_l) for p in PRESCRIPTIVE):
        violations.append({"type": "prescriptive"})

    return {"grounded": len(violations) == 0, "violations": violations,
            "fetched_reliable": sorted(fetched_ok), "fetched_low_conf": sorted(fetched_lowconf)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def make_backend(name: str = "auto", model: str | None = None) -> Backend:
    """Backend selector. 'auto' → anthropic if a key is set, else codex (authed CLI)."""
    if name == "auto":
        name = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "codex"
    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[chat] ANTHROPIC_API_KEY not set — use --backend codex.", file=sys.stderr)
            sys.exit(2)
        return AnthropicBackend(model=model or DEFAULT_MODEL)
    if name == "codex":
        return CodexBackend(model=model)
    raise ValueError(f"unknown backend {name!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorecard", required=True, help="current swing *_scorecard.json")
    p.add_argument("--compare", default=None, help="optional earlier *_scorecard.json (progress)")
    p.add_argument("--question", default=None, help="one-shot question (omit for interactive REPL)")
    p.add_argument("--backend", choices=["auto", "codex", "anthropic"], default="auto")
    p.add_argument("--model", default=None, help="model override (backend-specific)")
    p.add_argument("--show-tools", action="store_true", help="print the tool call log per turn")
    args = p.parse_args()

    ctx = SwingContext.from_files(args.scorecard, args.compare)
    backend = make_backend(args.backend, args.model)
    convo = Conversation(ctx, backend)

    def run(q: str):
        res = convo.ask(q)
        g = verify_chat_grounding(ctx, res)
        print(f"\nA: {res.answer}")
        if args.show_tools:
            for e in res.tool_log:
                print(f"   · {e['name']}({e['input']}) -> {json.dumps(e['result'])[:90]}")
        flag = "" if g["grounded"] else f"  [grounding: {[v['type'] for v in g['violations']]}]"
        print(f"   ({res.iterations} step(s), {len(res.tool_log)} tool call(s){flag})")

    if args.question:
        run(args.question)
        return
    print("Motion Caddie chat — ask about your swing. Ctrl-C / 'quit' to exit.")
    while True:
        try:
            q = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("quit", "exit", ""):
            break
        run(q)


if __name__ == "__main__":
    main()
