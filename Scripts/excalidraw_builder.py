"""Minimal builder for Excalidraw JSON files.

Excalidraw's file format (v2) is documented at
https://github.com/excalidraw/excalidraw/blob/master/dev-docs/data-formats.md.
This module exposes builders for the element types we need — rectangle,
text, arrow — plus a helper to assemble a complete .excalidraw payload.

Generated files open directly at excalidraw.com (File → Open) or in any
IDE plugin that reads Excalidraw.
"""
from __future__ import annotations

import json
import random
from typing import Optional


def _id() -> str:
    return f"e{random.randint(10**10, 10**11 - 1)}"


def _nonce() -> int:
    return random.randint(1, 2**31 - 1)


def rect(x: float, y: float, w: float, h: float, *,
          text: Optional[str] = None,
          stroke_color: str = "#1e1e1e",
          background_color: str = "#ffffff",
          fill_style: str = "solid",
          stroke_width: float = 2,
          stroke_style: str = "solid",
          roughness: int = 1,
          roundness: bool = True,
          font_size: int = 20,
          text_color: Optional[str] = None,
          text_align: str = "center",
          opacity: int = 100,
          group_id: Optional[str] = None) -> list[dict]:
    """A rounded rectangle, optionally with bound centered text. Returns
    a list of 1 or 2 elements (rectangle + text)."""
    rid = _id()
    elements = []

    elements.append({
        "id": rid,
        "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": background_color,
        "fillStyle": fill_style,
        "strokeWidth": stroke_width,
        "strokeStyle": stroke_style,
        "roughness": roughness,
        "opacity": opacity,
        "groupIds": [group_id] if group_id else [],
        "frameId": None,
        "roundness": {"type": 3} if roundness else None,
        "seed": _nonce(),
        "version": 1,
        "versionNonce": _nonce(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1700000000000,
        "link": None,
        "locked": False,
    })

    if text:
        tid = _id()
        text_elem = {
            "id": tid,
            "type": "text",
            "x": x + 4, "y": y + h / 2 - font_size / 2,
            "width": w - 8, "height": font_size * 1.25,
            "angle": 0,
            "strokeColor": text_color or stroke_color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [group_id] if group_id else [],
            "frameId": None,
            "roundness": None,
            "seed": _nonce(),
            "version": 1,
            "versionNonce": _nonce(),
            "isDeleted": False,
            "boundElements": [],
            "updated": 1700000000000,
            "link": None,
            "locked": False,
            "text": text,
            "fontSize": font_size,
            "fontFamily": 1,  # 1=Virgil (hand-drawn), 2=Helvetica, 3=Cascadia
            "textAlign": text_align,
            "verticalAlign": "middle",
            "containerId": rid,
            "originalText": text,
            "lineHeight": 1.25,
            "baseline": int(font_size * 0.9),
        }
        elements[0]["boundElements"] = [{"id": tid, "type": "text"}]
        elements.append(text_elem)

    return elements


def text(x: float, y: float, txt: str, *,
          font_size: int = 16,
          stroke_color: str = "#1e1e1e",
          text_align: str = "left",
          font_family: int = 1,
          width: Optional[float] = None,
          group_id: Optional[str] = None) -> dict:
    """Standalone text element. `width` controls wrapping; if None we
    estimate from the text length."""
    if width is None:
        # Rough estimate — about 0.6 of font_size per character
        width = max(60, int(len(txt.split("\n")[0]) * font_size * 0.6))
    lines = txt.count("\n") + 1
    height = int(font_size * 1.25 * lines)
    return {
        "id": _id(),
        "type": "text",
        "x": x, "y": y, "width": width, "height": height,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [group_id] if group_id else [],
        "frameId": None,
        "roundness": None,
        "seed": _nonce(),
        "version": 1,
        "versionNonce": _nonce(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1700000000000,
        "link": None,
        "locked": False,
        "text": txt,
        "fontSize": font_size,
        "fontFamily": font_family,
        "textAlign": text_align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": txt,
        "lineHeight": 1.25,
        "baseline": int(font_size * 0.9),
    }


def arrow(x1: float, y1: float, x2: float, y2: float, *,
           stroke_color: str = "#1e1e1e",
           stroke_width: float = 2,
           start_arrowhead: Optional[str] = None,
           end_arrowhead: Optional[str] = "arrow",
           start_id: Optional[str] = None,
           end_id: Optional[str] = None,
           group_id: Optional[str] = None,
           dashed: bool = False) -> dict:
    """A straight arrow from (x1,y1) to (x2,y2). If start_id/end_id are
    provided, bind to those element IDs so the arrow follows them when
    moved."""
    el = {
        "id": _id(),
        "type": "arrow",
        "x": x1, "y": y1,
        "width": x2 - x1, "height": y2 - y1,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": stroke_width,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [group_id] if group_id else [],
        "frameId": None,
        "roundness": {"type": 2},
        "seed": _nonce(),
        "version": 1,
        "versionNonce": _nonce(),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1700000000000,
        "link": None,
        "locked": False,
        "startBinding": ({"elementId": start_id, "focus": 0, "gap": 4}
                          if start_id else None),
        "endBinding": ({"elementId": end_id, "focus": 0, "gap": 4}
                        if end_id else None),
        "lastCommittedPoint": None,
        "startArrowhead": start_arrowhead,
        "endArrowhead": end_arrowhead,
        "points": [[0, 0], [x2 - x1, y2 - y1]],
        "elbowed": False,
    }
    return el


def assemble(elements: list[dict],
              background_color: str = "#ffffff") -> dict:
    """Wrap a flat list of elements into a complete .excalidraw payload."""
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "motion-caddie/golf-capstone",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": background_color,
        },
        "files": {},
    }


def save(elements: list[dict], path: str,
          background_color: str = "#ffffff") -> None:
    """Convenience: assemble + write JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assemble(elements, background_color), f, indent=1)
