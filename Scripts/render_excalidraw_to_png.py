"""Render .excalidraw JSON files to PNG.

Walks the elements list and draws each one with matplotlib:
  - rectangle → FancyBboxPatch (with rounded corners optional)
  - text      → ax.text with bound-container layout matching Excalidraw
  - arrow     → FancyArrowPatch + arrowhead

The output is "clean schematic" style (sharp lines, geometric) rather than
trying to mimic Excalidraw's hand-drawn aesthetic. Trades sketchiness for
crisp slide rendering.

Usage:
    python render_excalidraw_to_png.py             # render all .excalidraw files
    python render_excalidraw_to_png.py FILE...     # render specific files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DIR  = PROJECT_ROOT / "Data" / "visualizations" / "excalidraw"


def _fill_style_to_mpl(style: str, color: str, opacity: float = 1.0):
    """Map Excalidraw fillStyle to matplotlib hatch + alpha args."""
    if style == "solid":
        return {"facecolor": color, "alpha": opacity}
    if style == "hachure":
        return {"facecolor": color, "alpha": opacity * 0.35, "hatch": "//"}
    if style == "cross-hatch":
        return {"facecolor": color, "alpha": opacity * 0.4, "hatch": "xx"}
    if style == "zigzag":
        return {"facecolor": color, "alpha": opacity * 0.35, "hatch": "\\\\"}
    return {"facecolor": "white", "alpha": 0.0}


def render(in_path: Path, out_path: Path):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    elements = data["elements"]

    # Determine canvas extent from element bounds
    xs, ys = [], []
    for e in elements:
        if e.get("isDeleted"):
            continue
        if e["type"] in ("rectangle", "text"):
            x, y, w, h = e["x"], e["y"], e.get("width", 0), e.get("height", 0)
            xs.extend([x, x + w]); ys.extend([y, y + h])
        elif e["type"] == "arrow":
            x, y = e["x"], e["y"]
            for px, py in e.get("points", [[0, 0]]):
                xs.append(x + px); ys.append(y + py)
    if not xs:
        return
    xmin, xmax = min(xs) - 20, max(xs) + 20
    ymin, ymax = min(ys) - 20, max(ys) + 20

    # Lay out a figure that matches the canvas aspect ratio
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    target_w = 19.2  # inches
    target_h = target_w * canvas_h / canvas_w
    fig, ax = plt.subplots(figsize=(target_w, target_h), dpi=100,
                            facecolor=data.get("appState", {}).get("viewBackgroundColor", "#ffffff"))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)   # invert Y because Excalidraw is screen-coords
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Build an id → element map so arrow bindings can be resolved
    id_to_elem = {e["id"]: e for e in elements if not e.get("isDeleted")}

    # Pass 1: rectangles + arrows (background)
    for e in elements:
        if e.get("isDeleted"): continue

        if e["type"] == "rectangle":
            x, y, w, h = e["x"], e["y"], e["width"], e["height"]
            stroke = e.get("strokeColor", "#000000")
            bg     = e.get("backgroundColor", "transparent")
            sw     = e.get("strokeWidth", 1) * 0.75
            roundness = e.get("roundness")
            opacity = e.get("opacity", 100) / 100
            stroke_style = e.get("strokeStyle", "solid")
            line_dash = "--" if stroke_style == "dashed" else (":" if stroke_style == "dotted" else "-")
            fill_style = e.get("fillStyle", "solid")

            if roundness:
                radius = 0.04
                fill_args = _fill_style_to_mpl(fill_style, bg, opacity) if bg != "transparent" else {"facecolor": "none"}
                box = FancyBboxPatch((x, y), w, h,
                                      boxstyle=f"round,pad=0,rounding_size={min(w, h) * radius}",
                                      linewidth=sw, edgecolor=stroke,
                                      linestyle=line_dash, **fill_args)
            else:
                fill_args = _fill_style_to_mpl(fill_style, bg, opacity) if bg != "transparent" else {"facecolor": "none"}
                box = patches.Rectangle((x, y), w, h, linewidth=sw,
                                          edgecolor=stroke, linestyle=line_dash,
                                          **fill_args)
            ax.add_patch(box)

        elif e["type"] == "arrow":
            x0, y0 = e["x"], e["y"]
            points = e.get("points", [[0, 0]])
            stroke = e.get("strokeColor", "#000000")
            sw = e.get("strokeWidth", 1) * 0.75
            stroke_style = e.get("strokeStyle", "solid")
            line_dash = "--" if stroke_style == "dashed" else (":" if stroke_style == "dotted" else "-")
            # Excalidraw arrows are typically 2-point straight lines
            if len(points) >= 2:
                p1 = (x0 + points[0][0], y0 + points[0][1])
                p2 = (x0 + points[-1][0], y0 + points[-1][1])
                end_head = e.get("endArrowhead")
                arrow = FancyArrowPatch(p1, p2,
                                          arrowstyle="-|>" if end_head == "arrow" else "->",
                                          mutation_scale=14,
                                          color=stroke, lw=sw,
                                          linestyle=line_dash)
                ax.add_patch(arrow)

    # Pass 2: text on top
    for e in elements:
        if e.get("isDeleted") or e["type"] != "text": continue
        x, y, w, h = e["x"], e["y"], e["width"], e["height"]
        txt = e.get("text", "")
        if not txt: continue
        font_size = e.get("fontSize", 16) * 0.85
        color = e.get("strokeColor", "#000000")
        text_align = e.get("textAlign", "left")
        v_align = e.get("verticalAlign", "top")

        # Map alignment + position to matplotlib semantics
        if text_align == "center":
            tx = x + w / 2; ha = "center"
        elif text_align == "right":
            tx = x + w; ha = "right"
        else:
            tx = x; ha = "left"
        if v_align == "middle":
            ty = y + h / 2; va = "center"
        else:
            ty = y; va = "top"

        ax.text(tx, ty, txt, fontsize=font_size, color=color,
                 ha=ha, va=va, family="sans-serif",
                 wrap=True)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor(),
                 bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        files = sorted(DEFAULT_DIR.glob("*.excalidraw"))
    print(f"Rendering {len(files)} Excalidraw files to PNG...")
    for f in files:
        out_path = f.with_suffix(".rendered.png")
        try:
            render(f, out_path)
            sz = out_path.stat().st_size // 1024
            print(f"  ok  {f.name}  -> {out_path.name}  ({sz} KB)")
        except Exception as e:
            print(f"  ERR {f.name}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
