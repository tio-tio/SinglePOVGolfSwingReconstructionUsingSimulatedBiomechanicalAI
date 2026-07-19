"""LLM Interpretation Layer v2 — knowledge-base grounded, provider-agnostic.

Successor to coaching_llm_summary.py. Same job (turn a structured scorecard into
a beginner-friendly, plain-English swing explanation) with three upgrades:

  1. KNOWLEDGE BASE GROUNDING. The model narrates only from vetted per-indicator
     "explainer cards" + a controlled glossary (Data/coaching/indicator_kb.json),
     using those glosses verbatim and inventing no golf terminology.

  2. STRUCTURED-OUTPUT GROUNDING. The model returns JSON: the prose PLUS a list
     of claims, each tagged with the indicator key(s) it's about and a claim_type
     (in_range / out_of_range / encouragement / gloss). verify_grounding() then
     DETERMINISTICALLY checks every factual claim against the scorecard — so
     grounding is mechanically verified, not just judged by another LLM.

  3. MULTIMODAL (optional). Pass --image <scorecard.png or swing frame> for
     visual context (still forbidden from contradicting the measured metrics).

Two backends, same KB + same deterministic verifier:
  --backend codex      (default) uses the already-authed Codex CLI with
                       --output-schema (structured output) and -i (image).
  --backend anthropic  uses the Anthropic SDK (claude-opus-4-8). Adds prompt
                       caching; needs ANTHROPIC_API_KEY / `ant auth login`.

No backend needed for --dry-run or the grounding verifier (both pure-Python).
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
KB_PATH = PROJECT_ROOT / "Data" / "coaching" / "indicator_kb.json"
CODEX_BIN = shutil.which("codex.cmd") or shutil.which("codex") or "codex"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"

# --------------------------------------------------------------------------- #
# Prompt assembly (shared across backends)
# --------------------------------------------------------------------------- #

RULES = """You write a short, beginner-friendly explanation of one golfer's swing
from MEASURED motion data (already computed — do not re-analyze anything). This
is a one-shot writing task: produce the explanation only, no questions, no preamble.

Strict rules:
  - 2-3 short paragraphs, warm and plain. Gloss any golf term in a few words.
  - Lead with what they did WELL (metrics inside the tour range).
  - Then describe EACH flagged observation as a neutral description of what
    their body did — never as faults, never as instructions/tips to fix. Cover
    every flagged observation; group related ones into the same paragraph if it
    reads better, but do not silently drop any.
  - NEVER mention any metric whose confidence is "low"; we don't trust that number.
  - PRECISION: only call a metric "in range"/"tour-like" if THAT exact metric is
    inside its pro_band. Never lump several metrics under one label and call the
    group in range. When unsure, say nothing about that metric.
  - Don't add phase/timing details unless the metric name itself says so
    (e.g. *_top vs *_impact vs *_address).
  - Use ONLY the wording and glosses from the KNOWLEDGE BASE below. Do not
    introduce golf concepts, drills, or terminology that aren't in the KB.
  - Invent nothing not in the data. End with one encouraging sentence.

You must also return a `claims` list that decomposes your explanation into the
factual statements you made, each tagged with the indicator key(s) it refers to
and a claim_type, so the claims can be checked against the data:
  - "in_range"      : you said this metric is tour-like / within the typical range
  - "out_of_range"  : you described this metric as differing from the tour range
  - "encouragement" : general encouragement (indicator_keys may be empty)
  - "gloss"         : a plain-language explanation of a term (cite the metric it relates to)
Every in_range / out_of_range claim MUST list the exact indicator key(s) it covers."""

# JSON schema for structured output (no numeric/length constraints — unsupported
# by structured-output engines). additionalProperties:false everywhere for strict.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "The user-facing explanation, 2-3 short paragraphs.",
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "indicator_keys": {"type": "array", "items": {"type": "string"}},
                    "claim_type": {
                        "type": "string",
                        "enum": ["in_range", "out_of_range", "encouragement", "gloss"],
                    },
                },
                "required": ["text", "indicator_keys", "claim_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["explanation", "claims"],
    "additionalProperties": False,
}


def load_kb() -> dict:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def build_kb_block(kb: dict) -> str:
    """Build the stable knowledge-base prompt sent to the LLM.

    Supports both the original 0.1 KB fields and the expanded 0.2 fields
    generated from the Markdown knowledge cards.
    """
    lines = [
        "KNOWLEDGE BASE — use only these definitions and interpretation rules.",
        "Describe measured movement neutrally. Do not invent causes, diagnoses, "
        "shot outcomes, corrections, or ideal targets.",
        "",
        "GLOSSARY:",
    ]

    for term, gloss in kb.get("glossary", {}).items():
        lines.append(f"  - {term}: {gloss}")

    lines.append("\nINDICATOR CARDS:")

    for key, card in kb.get("indicators", {}).items():
        lines.append(
            f"\n[{key}] {card.get('label', key)} "
            f"({card.get('plain_name', '')}) — "
            f"event: {card.get('event', '')}, "
            f"unit: {card.get('unit', '')}"
        )

        if card.get("measures"):
            lines.append(f"  measures: {card['measures']}")

        if card.get("beginner_explanation"):
            lines.append(
                f"  beginner_explanation: {card['beginner_explanation']}"
            )

        if card.get("why"):
            lines.append(f"  why: {card['why']}")

        if card.get("handedness_note"):
            lines.append(
                f"  handedness_note: {card['handedness_note']}"
            )

        evidence = card.get("evidence_classification", {})
        if evidence:
            if evidence.get("directly_measured"):
                lines.append(
                    "  directly_measured: "
                    f"{evidence['directly_measured']}"
                )

            if evidence.get("derived_comparison"):
                lines.append(
                    "  derived_comparison: "
                    f"{evidence['derived_comparison']}"
                )

            if evidence.get("not_supported"):
                lines.append(
                    "  not_supported: "
                    f"{evidence['not_supported']}"
                )

        if card.get("in_range_phrasing"):
            lines.append(
                f"  in_range: {card['in_range_phrasing']}"
            )

        if card.get("out_low_phrasing"):
            lines.append(
                f"  if_below_range: {card['out_low_phrasing']}"
            )

        if card.get("out_high_phrasing"):
            lines.append(
                f"  if_above_range: {card['out_high_phrasing']}"
            )

        comparison_language = card.get("comparison_language", [])
        if comparison_language:
            lines.append("  approved_comparison_language:")

            for statement in comparison_language:
                lines.append(f"    - {statement}")

        limitations = card.get("limitations", [])
        if limitations:
            lines.append("  limitations:")

            for limitation in limitations:
                lines.append(f"    - {limitation}")

        allowed_inferences = card.get("allowed_inferences", [])
        if allowed_inferences:
            lines.append("  allowed_inferences:")

            for inference in allowed_inferences:
                lines.append(f"    - {inference}")

        prohibited_inferences = card.get("prohibited_inferences", [])
        if prohibited_inferences:
            lines.append("  prohibited_inferences:")

            for inference in prohibited_inferences:
                lines.append(f"    - {inference}")

        if card.get("glosses"):
            for term, gloss in card["glosses"].items():
                lines.append(f"  gloss[{term}]: {gloss}")

    return "\n".join(lines)


def build_scorecard_text(sc: dict) -> str:
    """Per-clip content."""
    inds = sc["indicators"]
    fb = sc["feedback"]
    lines = ["MEASURED METRICS (you vs tour-pro, with confidence):"]
    for k, v in inds.items():
        in_band = v["pro_band"][0] <= v["value"] <= v["pro_band"][1]
        lines.append(
            f"  - {k}: you={v['value']}  tour_median={v['pro_median']}  "
            f"pro_band={v['pro_band']}  percentile={v['percentile']}  "
            f"in_tour_range={in_band}  confidence={v.get('confidence_tier', 'med')}"
        )
    lines.append("\nFLAGGED OBSERVATIONS (severity=review only):")
    real = [f for f in fb if f.get("severity") == "review"]
    if not real:
        lines.append("  (none — everything confidently measured is within the tour range)")
    for f in real:
        lines.append(f"  - {f['label']} at {f['event']}: {f['message']} (you p{f['percentile']})")
    lines.append("\nNow write the explanation and the claims list.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Deterministic grounding verification (no LLM, no API key)
# --------------------------------------------------------------------------- #

def verify_grounding(sc: dict, claims: list[dict]) -> dict:
    """Mechanically check the model's claims against the scorecard.

    Returns {grounded, violations, coverage, n_flagged, n_claims}. A violation is
    a hard factual error the LLM judge previously had to catch by reading prose."""
    inds = sc["indicators"]
    low_conf = {k for k, v in inds.items() if v.get("confidence_tier") == "low"}
    violations: list[dict] = []

    def in_band(k: str) -> bool:
        b = inds[k]["pro_band"]
        return b[0] <= inds[k]["value"] <= b[1]

    for cl in claims:
        ctype = cl.get("claim_type")
        for k in cl.get("indicator_keys", []):
            if k not in inds:
                violations.append({"type": "unknown_metric", "key": k, "claim": cl["text"]})
                continue
            if k in low_conf:
                violations.append({"type": "low_confidence_leakage", "key": k, "claim": cl["text"]})
                continue
            if ctype == "in_range" and not in_band(k):
                violations.append({"type": "claimed_in_range_but_out", "key": k, "claim": cl["text"]})
            elif ctype == "out_of_range" and in_band(k):
                violations.append({"type": "claimed_out_but_in_range", "key": k, "claim": cl["text"]})

    flagged = [f["indicator"] for f in sc["feedback"] if f.get("severity") == "review"]
    mentioned = {k for cl in claims for k in cl.get("indicator_keys", [])}
    covered = [k for k in flagged if k in mentioned]
    coverage = (len(covered) / len(flagged)) if flagged else None

    return {
        "grounded": len(violations) == 0,
        "violations": violations,
        "coverage": coverage,
        "n_flagged": len(flagged),
        "n_claims": len(claims),
    }


def _coerce_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# Backend: Codex CLI (default — already authed on this machine)
# --------------------------------------------------------------------------- #

_SEQ = 0
_SEQ_LOCK = threading.Lock()


def call_codex(prompt: str, image: str | None = None, model: str | None = None,
               timeout: int = 240, schema: dict | None = None) -> dict:
    """Run codex exec with a JSON output schema; return the parsed object.
    `schema` defaults to the single-clip OUTPUT_SCHEMA; callers (e.g. the progress
    layer) pass their own."""
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        seq = _SEQ
    tmp = PROJECT_ROOT / "Data" / "coaching"
    tmp.mkdir(parents=True, exist_ok=True)
    schema_file = tmp / f"_schema_{os.getpid()}_{seq}.json"
    out_file = tmp / f"_v2out_{os.getpid()}_{seq}.json"
    schema_file.write_text(json.dumps(schema or OUTPUT_SCHEMA), encoding="utf-8")
    cmd = [CODEX_BIN, "exec", "--sandbox", "read-only",
           "--output-schema", str(schema_file), "-o", str(out_file)]
    if model:
        cmd += ["-c", f'model="{model}"']
    if image:
        cmd += ["-i", str(image)]
    cmd += ["-"]  # prompt via stdin (avoids Windows newline mangling)
    try:
        cp = subprocess.run(cmd, input=prompt, timeout=timeout, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", shell=False)
        if cp.returncode != 0:
            raise RuntimeError(f"codex exit {cp.returncode}: {(cp.stderr or '')[:300]}")
        raw = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else cp.stdout
        return _coerce_json(raw)
    finally:
        for f in (schema_file, out_file):
            try:
                f.unlink()
            except OSError:
                pass


def _run_codex(sc: dict, kb: dict, image: str | None, model: str | None) -> tuple[dict, dict]:
    prompt = RULES + "\n\n" + build_kb_block(kb) + "\n\n" + build_scorecard_text(sc)
    if image:
        prompt += ("\n\n(An image of the SAME swing is attached for visual context "
                   "only. Never let it override or contradict the measured metrics.)")
    out = call_codex(prompt, image=image, model=model)
    meta = {"backend": "codex", "model": model or "codex-default", "multimodal": bool(image)}
    return out, meta


# --------------------------------------------------------------------------- #
# Backend: Anthropic SDK (prompt caching; needs a key)
# --------------------------------------------------------------------------- #

def _run_anthropic(sc: dict, kb: dict, image: str | None, model: str,
                   thinking: bool) -> tuple[dict, dict]:
    import anthropic

    system = [
        {"type": "text", "text": RULES},
        {"type": "text", "text": build_kb_block(kb), "cache_control": {"type": "ephemeral"}},
    ]
    content: list[dict] = []
    if image:
        data = Path(image).read_bytes()
        media = mimetypes.guess_type(image)[0] or "image/png"
        content.append({"type": "image", "source": {"type": "base64", "media_type": media,
                        "data": base64.standard_b64encode(data).decode("ascii")}})
        content.append({"type": "text", "text":
            "The image is the SAME swing, for visual context only. Never let it "
            "override or contradict the measured metrics below."})
    content.append({"type": "text", "text": build_scorecard_text(sc)})

    kwargs = dict(model=model, max_tokens=4000, system=system,
                  messages=[{"role": "user", "content": content}],
                  output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    msg = anthropic.Anthropic().messages.create(**kwargs)
    out = _coerce_json(next(b.text for b in msg.content if b.type == "text"))
    meta = {"backend": "anthropic", "model": msg.model, "multimodal": bool(image),
            "usage": {"input_tokens": msg.usage.input_tokens,
                      "output_tokens": msg.usage.output_tokens,
                      "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0),
                      "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0)}}
    return out, meta


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def summarize(scorecard_json: Path, backend: str = "codex", model: str | None = None,
              image: str | None = None, thinking: bool = True) -> dict:
    sc = json.loads(Path(scorecard_json).read_text(encoding="utf-8"))
    kb = load_kb()
    if backend == "anthropic":
        out, meta = _run_anthropic(sc, kb, image, model or DEFAULT_ANTHROPIC_MODEL, thinking)
    else:
        out, meta = _run_codex(sc, kb, image, model)

    grounding = verify_grounding(sc, out.get("claims", []))
    sc["llm_summary_v2"] = out["explanation"]
    sc["llm_claims"] = out.get("claims", [])
    sc["llm_grounding"] = grounding
    sc["llm_meta"] = meta
    Path(scorecard_json).write_text(json.dumps(sc, indent=2), encoding="utf-8")
    return sc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KB-grounded swing explanation (v2)")
    p.add_argument("--scorecard", required=True, help="Path to a *_scorecard.json")
    p.add_argument("--backend", choices=["codex", "anthropic"], default="codex")
    p.add_argument("--model", default=None, help="Override model (codex: -c model=...; anthropic: model id)")
    p.add_argument("--image", default=None, help="Optional scorecard PNG / swing frame for multimodal context")
    p.add_argument("--no-thinking", action="store_true", help="Anthropic backend: disable adaptive thinking")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble + print the prompt and schema; no model call")
    args = p.parse_args()

    if args.dry_run:
        sc = json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
        kb = load_kb()
        print("=" * 70, "\nRULES:\n", RULES[:500], "...")
        print("=" * 70, "\nKB BLOCK (head):\n", build_kb_block(kb)[:800], "...")
        print("=" * 70, "\nSCORECARD TEXT:\n", build_scorecard_text(sc))
        print("=" * 70, "\nOUTPUT SCHEMA:\n", json.dumps(OUTPUT_SCHEMA, indent=2))
        raise SystemExit(0)

    sc = summarize(Path(args.scorecard), backend=args.backend, model=args.model,
                   image=args.image, thinking=not args.no_thinking)
    print("\n" + "=" * 60 + f"\nLLM SWING EXPLANATION (KB-grounded, {sc['llm_meta']['backend']})\n" + "=" * 60)
    print(sc["llm_summary_v2"])
    g = sc["llm_grounding"]
    print("\n" + "=" * 60 + "\nGROUNDING CHECK\n" + "=" * 60)
    print(f"grounded={g['grounded']}  violations={len(g['violations'])}  "
          f"coverage={g['coverage']}  claims={g['n_claims']}")
    for v in g["violations"]:
        print(f"  ✗ {v['type']}: {v.get('key','')} — {v['claim'][:80]}")
    if "usage" in sc["llm_meta"]:
        u = sc["llm_meta"]["usage"]
        print(f"\ntokens: in={u['input_tokens']} out={u['output_tokens']} "
              f"cache_read={u['cache_read_input_tokens']} cache_write={u['cache_creation_input_tokens']}")
