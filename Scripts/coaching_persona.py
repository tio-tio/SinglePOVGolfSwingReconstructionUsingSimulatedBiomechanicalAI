#!/usr/bin/env python3
"""
Shared coaching-persona loader for Motion Caddie.

Personas are stored as Markdown files under:

    Data/coaching/personas/

The loader returns the persona content as plain prompt text so the same persona
can be used by both the one-shot explanation layer and the conversational chat.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = PROJECT_ROOT / "Data" / "coaching" / "personas"

DEFAULT_PERSONA = "traditional"


class PersonaLoadError(ValueError):
    """Raised when a requested persona cannot be loaded safely."""


def persona_path(persona: str = DEFAULT_PERSONA) -> Path:
    """
    Resolve a persona name to its Markdown file.

    Example:
        traditional -> Data/coaching/personas/traditional.md
    """
    safe_name = persona.strip().lower()

    if not safe_name:
        raise PersonaLoadError("Persona name cannot be empty")

    if not safe_name.replace("-", "").replace("_", "").isalnum():
        raise PersonaLoadError(
            f"Invalid persona name: {persona!r}"
        )

    path = PERSONAS_DIR / f"{safe_name}.md"

    if not path.exists():
        available = available_personas()
        available_text = ", ".join(available) if available else "none"

        raise PersonaLoadError(
            f"Persona {safe_name!r} was not found. "
            f"Available personas: {available_text}"
        )

    return path


def available_personas() -> list[str]:
    """Return all available persona names."""
    if not PERSONAS_DIR.exists():
        return []

    return sorted(path.stem for path in PERSONAS_DIR.glob("*.md"))


def load_persona(persona: str = DEFAULT_PERSONA) -> str:
    """
    Load a persona Markdown file and return a stable prompt block.

    Markdown headings and bullets are preserved because they provide useful
    structure to the LLM prompt.
    """
    path = persona_path(persona)
    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise PersonaLoadError(
            f"Persona file is empty: {path}"
        )

    if content.startswith("```") or content.endswith("```"):
        raise PersonaLoadError(
            f"Persona file contains an outer Markdown code fence: {path.name}"
        )

    return (
        "COACHING PERSONA\n"
        "Follow this persona for tone, response structure, and communication "
        "style. It must never override measured data, knowledge-base rules, "
        "tool results, or deterministic grounding requirements.\n\n"
        f"{content}"
    )


if __name__ == "__main__":
    print("Available personas:")

    for name in available_personas():
        print(f"  - {name}")

    print("\nLoaded default persona:\n")
    print(load_persona())