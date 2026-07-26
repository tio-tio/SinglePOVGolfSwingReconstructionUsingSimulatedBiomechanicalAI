#!/usr/bin/env python3
"""
Build Motion Caddie's structured indicator knowledge base from Markdown cards.

By default, this script writes:

    Data/coaching/indicator_kb.generated.json

It does not overwrite the active indicator_kb.json unless --write is supplied.

Examples
--------
Preview-build the generated JSON:

    python Scripts/build_indicator_kb_from_markdown.py

Validate without writing a file:

    python Scripts/build_indicator_kb_from_markdown.py --check

Replace the active knowledge base after reviewing the generated file:

    python Scripts/build_indicator_kb_from_markdown.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "Data" / "coaching" / "knowledge_cards"
ACTIVE_KB_PATH = ROOT / "Data" / "coaching" / "indicator_kb.json"
GENERATED_KB_PATH = ROOT / "Data" / "coaching" / "indicator_kb.generated.json"
BACKUP_KB_PATH = ROOT / "Data" / "coaching" / "indicator_kb.backup.json"


REQUIRED_CARD_SECTIONS = {
    "Metadata",
    "Measurement definition",
    "Beginner explanation",
    "Why it matters",
    "Evidence classification",
    "Approved interpretation language",
    "Approved glossary",
    "Comparison language",
    "Confidence and camera limitations",
    "Allowed inferences",
    "Prohibited inferences",
    "Approved response examples",
    "Prohibited response examples",
}


EXPECTED_INDICATOR_KEYS = {
    "shoulder_turn_top_deg",
    "hip_turn_top_deg",
    "x_factor_top_deg",
    "hip_turn_impact_deg",
    "spine_tilt_address_deg",
    "spine_tilt_impact_deg",
    "posture_loss_deg",
    "head_sway_max_pct",
    "head_lift_max_pct",
    "left_arm_bend_top_deg",
    "right_arm_bend_top_deg",
    "lead_knee_flex_address_deg",
    "lead_knee_flex_impact_deg",
    "hip_lateral_shift_pct",
    "tempo_ratio",
    "hand_speed_impact_bs",
}


class CardParseError(ValueError):
    """Raised when a Markdown knowledge card cannot be parsed safely."""


def clean_inline_markdown(value: str) -> str:
    """Remove simple Markdown formatting from an inline value."""
    value = value.strip()
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"\*([^*]+)\*", r"\1", value)
    return value.strip()


def normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def split_sections(text: str) -> tuple[str, dict[str, str]]:
    """
    Return the document title and all level-two sections.

    Level-three headings remain inside their parent section so they can be
    parsed later as interpretation or example subsections.
    """
    lines = text.splitlines()

    title = ""
    for line in lines:
        if line.startswith("# "):
            title = normalize_heading(line[2:])
            break

    if not title:
        raise CardParseError("Missing level-one title")

    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for line in lines:
        if line.startswith("## "):
            current_section = normalize_heading(line[3:])
            sections[current_section] = []
            continue

        if current_section is not None:
            sections[current_section].append(line)

    return title, {
        heading: "\n".join(content).strip()
        for heading, content in sections.items()
    }


def parse_metadata(section: str) -> dict[str, str]:
    metadata: dict[str, str] = {}

    pattern = re.compile(
        r"^\s*-\s+\*\*(?P<key>[^*]+):\*\*\s*(?P<value>.+?)\s*$"
    )

    for line in section.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        key = normalize_heading(match.group("key")).lower()
        value = clean_inline_markdown(match.group("value"))
        metadata[key] = value

    return metadata


def parse_paragraph(section: str) -> str:
    lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return " ".join(lines).strip()


def parse_bullets(section: str) -> list[str]:
    bullets: list[str] = []
    current: list[str] = []

    for raw_line in section.splitlines():
        line = raw_line.strip()

        if line.startswith("- "):
            if current:
                bullets.append(" ".join(current).strip())
            current = [clean_inline_markdown(line[2:])]
        elif line and current and not line.startswith("#"):
            current.append(clean_inline_markdown(line))

    if current:
        bullets.append(" ".join(current).strip())

    return bullets


def parse_labelled_bullets(section: str) -> dict[str, str]:
    """
    Parse bullets shaped like:

        - **Directly measured:** Shoulder rotation...
    """
    result: dict[str, str] = {}

    pattern = re.compile(
        r"^\s*-\s+\*\*(?P<label>[^*]+):\*\*\s*(?P<value>.+?)\s*$"
    )

    for line in section.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        label = normalize_heading(match.group("label")).lower()
        label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
        value = clean_inline_markdown(match.group("value"))
        result[label] = value

    return result


def parse_glossary(section: str) -> dict[str, str]:
    glossary: dict[str, str] = {}

    pattern = re.compile(
        r"^\s*-\s+\*\*(?P<term>[^*]+):\*\*\s*(?P<definition>.+?)\s*$"
    )

    for line in section.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        term = clean_inline_markdown(match.group("term"))
        definition = clean_inline_markdown(match.group("definition"))
        glossary[term.lower()] = definition

    return glossary


def parse_subsections(section: str) -> dict[str, str]:
    """
    Parse level-three headings and their paragraph content.

    Example:
        ### Inside the reference range
        The measured shoulder turn...
    """
    result: dict[str, list[str]] = {}
    current: str | None = None

    for raw_line in section.splitlines():
        line = raw_line.strip()

        if line.startswith("### "):
            current = normalize_heading(line[4:])
            result[current] = []
            continue

        if current is not None and line:
            if line.startswith("- "):
                result[current].append(clean_inline_markdown(line[2:]))
            elif not line.startswith("#"):
                result[current].append(clean_inline_markdown(line))

    return {
        heading: " ".join(content).strip()
        for heading, content in result.items()
        if content
    }


def find_subsection(
    subsections: dict[str, str],
    phrase: str,
    *,
    required: bool = True,
) -> str | None:
    phrase_lower = phrase.lower()

    for heading, content in subsections.items():
        if phrase_lower in heading.lower():
            return content

    if required:
        raise CardParseError(
            f"Missing subsection containing heading: {phrase!r}"
        )

    return None


def validate_sections(path: Path, sections: dict[str, str]) -> None:
    missing = REQUIRED_CARD_SECTIONS - set(sections)

    if missing:
        raise CardParseError(
            f"{path.name}: missing required section(s): "
            + ", ".join(sorted(missing))
        )


def parse_card(path: Path, existing_card: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    if text.lstrip().startswith("```"):
        raise CardParseError(
            f"{path.name}: file still contains an outer Markdown code fence"
        )

    title, sections = split_sections(text)
    validate_sections(path, sections)

    metadata = parse_metadata(sections["Metadata"])

    indicator_key = metadata.get("indicator key")
    if not indicator_key:
        raise CardParseError(f"{path.name}: missing Indicator key metadata")

    label = metadata.get("label")
    plain_name = metadata.get("plain-language name")
    swing_phase = metadata.get("swing phase")
    unit = metadata.get("unit")

    required_metadata = {
        "Label": label,
        "Plain-language name": plain_name,
        "Swing phase": swing_phase,
        "Unit": unit,
    }

    missing_metadata = [
        name for name, value in required_metadata.items() if not value
    ]

    if missing_metadata:
        raise CardParseError(
            f"{path.name}: missing metadata: {', '.join(missing_metadata)}"
        )

    interpretations = parse_subsections(
        sections["Approved interpretation language"]
    )
    approved_examples = parse_subsections(
        sections["Approved response examples"]
    )

    in_range = find_subsection(interpretations, "Inside")
    out_low = find_subsection(
        interpretations,
        "Below",
        required=False,
    )
    out_high = find_subsection(
        interpretations,
        "Above",
        required=False,
    )

    # Preserve the established machine-facing event value when one already
    # exists. The Markdown phase remains available as swing_phase_display.
    event = existing_card.get("event", swing_phase)

    card: dict[str, Any] = {
        "label": label,
        "plain_name": plain_name,
        "event": event,
        "swing_phase_display": swing_phase,
        "unit": existing_card.get("unit", unit.lower()),
        "measures": parse_paragraph(sections["Measurement definition"]),
        "beginner_explanation": parse_paragraph(
            sections["Beginner explanation"]
        ),
        "why": parse_paragraph(sections["Why it matters"]),
        "in_range_phrasing": in_range,
        "out_low_phrasing": out_low,
        "out_high_phrasing": out_high,
        "glosses": parse_glossary(sections["Approved glossary"]),
        "evidence_classification": parse_labelled_bullets(
            sections["Evidence classification"]
        ),
        "comparison_language": parse_bullets(
            sections["Comparison language"]
        ),
        "limitations": parse_bullets(
            sections["Confidence and camera limitations"]
        ),
        "allowed_inferences": parse_bullets(
            sections["Allowed inferences"]
        ),
        "prohibited_inferences": parse_bullets(
            sections["Prohibited inferences"]
        ),
        "approved_examples": approved_examples,
        "prohibited_examples": parse_bullets(
            sections["Prohibited response examples"]
        ),
        "source_file": path.name,
        "source_title": title,
    }

    handedness_note = metadata.get("handedness note")
    if handedness_note:
        card["handedness_note"] = handedness_note

    return indicator_key, card


def load_existing_kb() -> dict[str, Any]:
    if not ACTIVE_KB_PATH.exists():
        raise FileNotFoundError(
            f"Active knowledge base not found: {ACTIVE_KB_PATH}"
        )

    return json.loads(ACTIVE_KB_PATH.read_text(encoding="utf-8"))


def update_schema_documentation(kb: dict[str, Any]) -> None:
    schema = kb.setdefault("_schema", {})
    indicator_schema = schema.setdefault("indicator card", {})

    indicator_schema.update(
        {
            "beginner_explanation": (
                "plain-language explanation for a beginner"
            ),
            "swing_phase_display": (
                "human-readable swing phase from the Markdown source"
            ),
            "evidence_classification": (
                "directly measured, derived comparison, and unsupported claims"
            ),
            "comparison_language": (
                "approved neutral phrases for comparing two swings"
            ),
            "limitations": (
                "confidence, camera, and interpretation limitations"
            ),
            "allowed_inferences": (
                "claims the LLM may make from this metric"
            ),
            "prohibited_inferences": (
                "claims the LLM must not make from this metric"
            ),
            "approved_examples": (
                "approved response examples grouped by situation"
            ),
            "prohibited_examples": (
                "response examples the LLM must not produce"
            ),
            "handedness_note": (
                "optional handedness limitation for side-specific metrics"
            ),
            "source_file": "Markdown source filename",
            "source_title": "Markdown source document title",
        }
    )


def build_kb() -> dict[str, Any]:
    existing_kb = load_existing_kb()
    existing_indicators = existing_kb.get("indicators", {})

    card_paths = sorted(CARDS_DIR.glob("*.md"))

    if not card_paths:
        raise FileNotFoundError(
            f"No Markdown cards found in {CARDS_DIR}"
        )

    parsed_indicators: dict[str, dict[str, Any]] = {}
    source_for_key: dict[str, str] = {}

    for path in card_paths:
        # Read the indicator key first so the existing card can be preserved
        # for machine-facing fields such as event identifiers.
        _, preliminary_sections = split_sections(
            path.read_text(encoding="utf-8")
        )
        preliminary_metadata = parse_metadata(
            preliminary_sections.get("Metadata", "")
        )
        preliminary_key = preliminary_metadata.get("indicator key", "")
        existing_card = existing_indicators.get(preliminary_key, {})

        indicator_key, card = parse_card(path, existing_card)

        if indicator_key in parsed_indicators:
            raise CardParseError(
                f"Duplicate indicator key {indicator_key!r} in "
                f"{source_for_key[indicator_key]} and {path.name}"
            )

        parsed_indicators[indicator_key] = card
        source_for_key[indicator_key] = path.name

    found_keys = set(parsed_indicators)
    missing = EXPECTED_INDICATOR_KEYS - found_keys
    unexpected = found_keys - EXPECTED_INDICATOR_KEYS

    if missing or unexpected:
        messages: list[str] = []

        if missing:
            messages.append(
                "missing expected keys: " + ", ".join(sorted(missing))
            )

        if unexpected:
            messages.append(
                "unexpected keys: " + ", ".join(sorted(unexpected))
            )

        raise CardParseError("; ".join(messages))

    # Preserve the existing JSON indicator order. This keeps generated prompt
    # ordering stable and avoids unnecessary diffs.
    ordered_indicators: dict[str, dict[str, Any]] = {}

    for key in existing_indicators:
        if key in parsed_indicators:
            ordered_indicators[key] = parsed_indicators[key]

    for key in sorted(parsed_indicators):
        if key not in ordered_indicators:
            ordered_indicators[key] = parsed_indicators[key]

    kb = dict(existing_kb)
    kb["_version"] = "0.2.0"
    kb["_about"] = (
        "Grounded coaching knowledge base generated from human-readable "
        "Markdown indicator cards. The LLM must describe measured metrics "
        "without inventing causes, diagnoses, shot outcomes, or corrections."
    )
    kb["indicators"] = ordered_indicators

    update_schema_documentation(kb)

    # Merge approved card glosses into the shared glossary without deleting
    # established global entries.
    global_glossary = dict(kb.get("glossary", {}))

    for card in ordered_indicators.values():
        for term, definition in card.get("glosses", {}).items():
            global_glossary.setdefault(term, definition)

    kb["glossary"] = global_glossary

    return kb


def validate_built_kb(kb: dict[str, Any]) -> None:
    indicators = kb.get("indicators")

    if not isinstance(indicators, dict):
        raise CardParseError("Generated KB has no indicators dictionary")

    if set(indicators) != EXPECTED_INDICATOR_KEYS:
        raise CardParseError(
            "Generated KB indicator keys do not match the expected set"
        )

    required_fields = {
        "label",
        "plain_name",
        "event",
        "unit",
        "measures",
        "why",
        "in_range_phrasing",
        "glosses",
        "evidence_classification",
        "limitations",
        "allowed_inferences",
        "prohibited_inferences",
        "approved_examples",
        "prohibited_examples",
    }

    problems: list[str] = []

    for key, card in indicators.items():
        missing = [
            field
            for field in sorted(required_fields)
            if field not in card or card[field] in ("", None, [], {})
        ]

        if missing:
            problems.append(
                f"{key}: missing or empty fields: {', '.join(missing)}"
            )

    if problems:
        raise CardParseError("\n".join(problems))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(kb: dict[str, Any], destination: Path | None) -> None:
    indicators = kb["indicators"]

    print(f"Validated cards: {len(indicators)}")
    print(f"Knowledge-base version: {kb.get('_version')}")

    for key, card in indicators.items():
        print(f"  PASS  {key} <- {card['source_file']}")

    if destination is not None:
        print(f"\nWrote: {destination}")
    else:
        print("\nValidation completed without writing a file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build indicator_kb.json from Markdown knowledge cards."
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate all cards without writing output.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help=(
            "Back up and replace Data/coaching/indicator_kb.json. "
            "Use only after reviewing the generated file."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        kb = build_kb()
        validate_built_kb(kb)

        if args.check:
            print_summary(kb, destination=None)
            return 0

        if args.write:
            shutil.copy2(ACTIVE_KB_PATH, BACKUP_KB_PATH)
            write_json(ACTIVE_KB_PATH, kb)
            print_summary(kb, destination=ACTIVE_KB_PATH)
            print(f"Backup: {BACKUP_KB_PATH}")
            return 0

        write_json(GENERATED_KB_PATH, kb)
        print_summary(kb, destination=GENERATED_KB_PATH)
        print(
            "\nReview the generated file before running this script "
            "with --write."
        )
        return 0

    except (CardParseError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())