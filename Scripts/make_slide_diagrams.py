"""Generate architecture diagrams + composite tiles for slides 11, 15-new, and 16.

Outputs (all 1920x1080 PNG, slide-ready):
  diagram_self_distillation.png         — the training/inference architecture
  diagram_youtube_pipeline.png          — 8-stage YouTube → labeled clips funnel
  panel_three_corroborations.png        — MP-Heavy / GolfPose / Sapiens parallel finding
  panel_llm_experiments.png             — 2-up tile: event detector + data labeler

Run: python make_slide_diagrams.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = Path(__file__).parent.parent / "Data" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared palette
GOLD     = "#FFB300"   # winner / MotionBERT-Full
PALE_GOLD = "#FFE082"
TEAL     = "#26A69A"   # Sapiens / foundation
RED      = "#E53935"   # GolfPose / surprise
PURPLE   = "#AB47BC"   # LLM
INDIGO   = "#5C6BC0"   # 2D backbone
GRAY     = "#546E7A"
DARK     = "#1F1F1F"
SOFT_BG  = "#FAFAFA"
TRAIN_BG = "#FFF8E1"
INFER_BG = "#E3F2FD"


def add_box(ax, x, y, w, h, label, sub=None, facecolor="#fafafa",
             edgecolor=DARK, linewidth=1.5, label_size=12, sub_size=10,
             title_color=DARK, sub_color="#555555"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.012",
                           linewidth=linewidth, edgecolor=edgecolor,
                           facecolor=facecolor)
    ax.add_patch(rect)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, label, fontsize=label_size,
                fontweight="bold", color=title_color, ha="center", va="center")
        ax.text(x + w / 2, y + h * 0.30, sub, fontsize=sub_size,
                color=sub_color, ha="center", va="center")
    else:
        ax.text(x + w / 2, y + h / 2, label, fontsize=label_size,
                fontweight="bold", color=title_color, ha="center", va="center")


def add_arrow(ax, x0, y0, x1, y1, color=DARK, lw=2.0,
              connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                   arrowstyle="-|>", mutation_scale=22,
                                   color=color, lw=lw,
                                   connectionstyle=connectionstyle))


def setup_fig():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    return fig, ax


# =========================================================================
# Diagram 1 — Self-distillation architecture
# =========================================================================

def diagram_self_distillation():
    fig, ax = setup_fig()

    # Title
    ax.text(0.5, 0.965, "Self-Distillation Architecture",
             fontsize=26, fontweight="bold", color=DARK, ha="center")
    ax.text(0.5, 0.93, "Use a frontier LLM once at training-time so a small fast model serves users at inference-time",
             fontsize=14, color="#555555", ha="center", style="italic")

    # ============== PHASE 1: TRAINING (top half) ==============
    # Background panel
    train_bg = FancyBboxPatch((0.02, 0.50), 0.96, 0.39,
                                boxstyle="round,pad=0.005,rounding_size=0.01",
                                linewidth=0, facecolor=TRAIN_BG)
    ax.add_patch(train_bg)
    ax.text(0.04, 0.86, "①  TRAINING PHASE",
             fontsize=18, fontweight="bold", color=GOLD)
    ax.text(0.04, 0.835, "one-time   ·   ~$300   ·   ~24 hours on AWS",
             fontsize=11, color="#666666", style="italic")

    # Three input boxes
    add_box(ax, 0.06, 0.66, 0.18, 0.10,
             "GolfDB",
             "1,400 human-labeled clips",
             facecolor="white", edgecolor=GRAY)
    add_box(ax, 0.30, 0.66, 0.20, 0.10,
             "YouTube Expansion Pipeline",
             "8-stage automation, ~$0.75/clip\n(Experiment #2)",
             facecolor="white", edgecolor=PURPLE,
             linewidth=2.2, title_color=PURPLE, sub_size=9)
    add_box(ax, 0.56, 0.66, 0.20, 0.10,
             "LLM Event Labels",
             "Codex GPT-5 on 24-frame grid\n(Experiment #1)",
             facecolor="white", edgecolor=PURPLE,
             linewidth=2.2, title_color=PURPLE, sub_size=9)

    # Combined dataset box
    add_box(ax, 0.30, 0.535, 0.46, 0.10,
             "Combined Dataset",
             "2,400+ clips with mixed gold + LLM-pseudo labels",
             facecolor="#FFF3E0", edgecolor=GOLD,
             linewidth=2.5, title_color=GOLD)
    # Arrows into combined
    add_arrow(ax, 0.15, 0.66, 0.42, 0.635, color=GRAY)
    add_arrow(ax, 0.40, 0.66, 0.50, 0.635, color=PURPLE)
    add_arrow(ax, 0.66, 0.66, 0.58, 0.635, color=PURPLE)

    # Trained model
    add_box(ax, 0.80, 0.535, 0.16, 0.10,
             "1D-CNN Event Detector",
             "~5 MB · <1 sec inference",
             facecolor="#E8F5E9", edgecolor="#2E7D32",
             linewidth=2.5, title_color="#2E7D32", sub_size=10)
    add_arrow(ax, 0.76, 0.585, 0.80, 0.585, color="#2E7D32", lw=3.0)

    # ============== PHASE 2: INFERENCE (bottom half) ==============
    infer_bg = FancyBboxPatch((0.02, 0.04), 0.96, 0.40,
                                boxstyle="round,pad=0.005,rounding_size=0.01",
                                linewidth=0, facecolor=INFER_BG)
    ax.add_patch(infer_bg)
    ax.text(0.04, 0.41, "②  INFERENCE PHASE",
             fontsize=18, fontweight="bold", color=INDIGO)
    ax.text(0.04, 0.385, "per-user   ·   ~$0.001 marginal   ·   ~1 second total wall-clock",
             fontsize=11, color="#666666", style="italic")

    pipeline_stages = [
        ("phone.mp4", "user upload\n(any single-POV)", "white", GRAY),
        ("MediaPipe Lite", "2D landmarks\n107 FPS · CPU", "white", INDIGO),
        ("MotionBERT-Full", "3D lift\n6,900 FPS · GPU", "white", GOLD),
        ("1D-CNN Detector", "8 swing events\nFROM TRAINING ABOVE", "#E8F5E9", "#2E7D32"),
        ("UE5 / Scorecard", "BVH + CSV + 3D viz", "white", GRAY),
    ]
    n = len(pipeline_stages)
    box_w = 0.155
    gap = 0.034
    total_w = n * box_w + (n - 1) * gap
    start_x = (1 - total_w) / 2
    y = 0.18
    centers = []
    for i, (lbl, sub, fc, ec) in enumerate(pipeline_stages):
        x = start_x + i * (box_w + gap)
        is_trained = (lbl == "1D-CNN Detector")
        add_box(ax, x, y, box_w, 0.12, lbl, sub, facecolor=fc, edgecolor=ec,
                 linewidth=2.5 if is_trained else 1.8,
                 title_color=ec, sub_size=9)
        centers.append((x + box_w / 2, y + 0.06))

    for i in range(n - 1):
        x0 = centers[i][0] + box_w / 2 + 0.003
        x1 = centers[i + 1][0] - box_w / 2 - 0.003
        add_arrow(ax, x0, y + 0.06, x1, y + 0.06, color=DARK, lw=2)

    # Big down arrow showing "trained model goes into inference"
    add_arrow(ax, 0.88, 0.535 - 0.005, 0.88, 0.30,
              color="#2E7D32", lw=3.5,
              connectionstyle="arc3,rad=0")
    ax.text(0.90, 0.42, "trained model\ndeployed", fontsize=10,
             color="#2E7D32", fontweight="bold", style="italic")

    # Footer
    ax.text(0.5, 0.025,
             "The two LLM experiments enable this architecture: #2 generates training data, #1 proves a small model can hit competitive PCE.",
             fontsize=10, color="#555555", ha="center", style="italic")

    out = OUT_DIR / "diagram_self_distillation.png"
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# =========================================================================
# Diagram 2 — YouTube data expansion pipeline (funnel)
# =========================================================================

def diagram_youtube_pipeline():
    fig, ax = setup_fig()

    ax.text(0.5, 0.96, "YouTube Data Expansion Pipeline",
             fontsize=26, fontweight="bold", color=DARK, ha="center")
    ax.text(0.5, 0.92, "8 fully-automated stages: queries in → GolfDB-schema labeled clips out",
             fontsize=14, color="#555555", ha="center", style="italic")

    stages = [
        ("1\nSEARCH",        "yt-dlp",          "20 queries",  "55 candidates",   GRAY),
        ("2\nDOWNLOAD",      "yt-dlp 480p",     "55 URLs",     "55 mp4s",         GRAY),
        ("3\nDETECT SWINGS", "MediaPipe Lite",  "55 videos",   "12 swings",       INDIGO),
        ("4\nEXTRACT",       "160x160 crop",    "12 swings",   "12 clips",        INDIGO),
        ("5\nLLM LABEL",     "Codex / GPT-5",   "12 clips",    "12 JSON labels",  PURPLE),
        ("6\nFILTER",        "monotonic + spacing", "12 labels", "4 clips",        RED),
        ("7\nMERGE",         "GolfDB schema",   "4 clips",     "+4 rows",         GOLD),
    ]
    n = len(stages)

    # Layout
    margin_x = 0.04
    avail = 1 - 2 * margin_x
    box_w = avail / n - 0.005
    box_h = 0.22
    y = 0.42

    for i, (title, sub, in_n, out_n, color) in enumerate(stages):
        x = margin_x + i * (box_w + 0.005)
        # Title block
        rect = FancyBboxPatch((x, y), box_w, box_h,
                               boxstyle="round,pad=0.005,rounding_size=0.012",
                               linewidth=2.5, edgecolor=color, facecolor="white")
        ax.add_patch(rect)
        # Stage number + name
        ax.text(x + box_w / 2, y + box_h * 0.75, title,
                 fontsize=13, fontweight="bold", color=color,
                 ha="center", va="center", linespacing=1.2)
        # Tool / method
        ax.text(x + box_w / 2, y + box_h * 0.42, sub,
                 fontsize=10, color=DARK, ha="center", va="center", style="italic")
        # I/O counts
        ax.text(x + box_w / 2, y + box_h * 0.20,
                 f"{in_n}  →  {out_n}",
                 fontsize=9, color="#666666", ha="center", va="center")
        # Arrow to next stage
        if i < n - 1:
            arrow_x0 = x + box_w + 0.001
            arrow_x1 = x + box_w + 0.0045
            add_arrow(ax, arrow_x0, y + box_h / 2, arrow_x1, y + box_h / 2,
                       color=DARK, lw=1.5)

    # Final result callout
    final_y = 0.18
    final_w = 0.50
    final_x = (1 - final_w) / 2
    add_box(ax, final_x, final_y, final_w, 0.12,
             "Combined Dataset",
             "GolfDB (1,400 gold) + MotionCaddie-Extra (4 LLM-pseudo) = 1,404 clips\nSchema-identical → drop-in compatible with all downstream code",
             facecolor="#FFF3E0", edgecolor=GOLD, linewidth=2.5,
             title_color=GOLD, sub_size=10)
    # Arrow from stage 7 down to combined
    last_x = margin_x + (n - 1) * (box_w + 0.005) + box_w / 2
    add_arrow(ax, last_x, y - 0.005, last_x, final_y + 0.13, color=GOLD, lw=2.5)
    add_arrow(ax, last_x, final_y + 0.135, final_x + final_w / 2, final_y + 0.13,
              color=GOLD, lw=2.5)

    # POC stats footer
    ax.text(0.5, 0.06,
             "POC RESULT (1 hour wall-clock, ~$3 spend):  55 candidates → 4 final clips  ·  7.3% end-to-end yield",
             fontsize=12, color=DARK, ha="center", fontweight="bold")
    ax.text(0.5, 0.025,
             "Scaling math:  15,000 candidates → ~1,000 final clips at ~$300 + 24 hours on AWS",
             fontsize=11, color="#555555", ha="center", style="italic")

    out = OUT_DIR / "diagram_youtube_pipeline.png"
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# =========================================================================
# Diagram 3 — Three corroborations panel
# =========================================================================

def panel_three_corroborations():
    fig, ax = setup_fig()

    ax.text(0.5, 0.95, "Three Independent Corroborations of the Same Effect",
             fontsize=25, fontweight="bold", color=DARK, ha="center")
    ax.text(0.5, 0.91,
             "Smoother 2D landmarks → displaced wrist-Y peaks → our heuristic event detector misses them",
             fontsize=14, color="#555555", ha="center", style="italic")

    # 3 cards
    cards = [
        ("MediaPipe Heavy", "Bigger pose model",
         "Smoothing within MediaPipe\ndisplaces peak frames",
         "PCE@5 = 0.052", "(MP Lite: 0.090 — bigger LOSES)",
         INDIGO),
        ("GolfPose 17+0", "Golf-fine-tuned model",
         "Domain training adds\ntemporal smoothing",
         "PCE@5 = 0.042–0.076", "(MotionBERT: 2-2.4× higher)",
         RED),
        ("Sapiens-Pose 1B", "Foundation model (Meta)",
         "Foundation-model architecture\nover-smooths predictions",
         "PCE@5 = 0.043", "(though 1.2% implausible — 12× cleaner anatomy)",
         TEAL),
    ]

    card_w = 0.27
    card_h = 0.50
    gap = 0.035
    total = 3 * card_w + 2 * gap
    start_x = (1 - total) / 2

    for i, (model, type_, mechanism, headline, sub, color) in enumerate(cards):
        x = start_x + i * (card_w + gap)
        y = 0.27
        rect = FancyBboxPatch((x, y), card_w, card_h,
                               boxstyle="round,pad=0.008,rounding_size=0.015",
                               linewidth=3, edgecolor=color, facecolor="#fafafa")
        ax.add_patch(rect)
        # Lightbulb / model label
        ax.text(x + card_w / 2, y + card_h * 0.88, f"#{i+1}",
                 fontsize=14, color=color, fontweight="bold", ha="center")
        ax.text(x + card_w / 2, y + card_h * 0.78, model,
                 fontsize=18, fontweight="bold", color=DARK, ha="center")
        ax.text(x + card_w / 2, y + card_h * 0.71, type_,
                 fontsize=11, color="#666666", style="italic", ha="center")
        # Mechanism
        ax.text(x + card_w / 2, y + card_h * 0.50, mechanism,
                 fontsize=12, color=DARK, ha="center", va="center",
                 linespacing=1.5)
        # Headline number
        ax.text(x + card_w / 2, y + card_h * 0.25, headline,
                 fontsize=20, fontweight="bold", color=color, ha="center")
        ax.text(x + card_w / 2, y + card_h * 0.15, sub,
                 fontsize=9, color="#666666", ha="center", style="italic")

    # Convergent conclusion
    concl_w = 0.78
    concl_x = (1 - concl_w) / 2
    add_box(ax, concl_x, 0.08, concl_w, 0.14,
             "Convergent conclusion",
             "The bottleneck is NOT the 2D pose model. It's the heuristic event detector.\n"
             "Replace it with a learned 1D-CNN over MotionBERT 3D trajectories → expected 3-5× PCE.",
             facecolor="#FFF8E1", edgecolor=GOLD, linewidth=2.5,
             title_color=GOLD, sub_size=12)

    # Arrows from each card down to conclusion
    for i in range(3):
        x = start_x + i * (card_w + gap) + card_w / 2
        add_arrow(ax, x, 0.27 - 0.005, x, 0.22 + 0.005, color=GOLD, lw=1.8)

    out = OUT_DIR / "panel_three_corroborations.png"
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


# =========================================================================
# Diagram 4 — LLM experiments tile (event detector + data labeler, side-by-side)
# =========================================================================

def panel_llm_experiments():
    fig, ax = setup_fig()

    ax.text(0.5, 0.965, "Two LLM Experiments — Complementary, Not Competing",
             fontsize=24, fontweight="bold", color=DARK, ha="center")
    ax.text(0.5, 0.93,
             "Same model (Codex / GPT-5 with vision) tested in two roles: inference-time event detector vs training-time data labeler",
             fontsize=13, color="#555555", ha="center", style="italic")

    # Two big panels
    panel_h = 0.78
    panel_w = 0.46
    gap = 0.03
    y = 0.07
    x1 = (1 - 2 * panel_w - gap) / 2
    x2 = x1 + panel_w + gap

    # === LEFT: Event detector ===
    rect = FancyBboxPatch((x1, y), panel_w, panel_h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=3, edgecolor=PURPLE, facecolor="#F3E5F5")
    ax.add_patch(rect)
    ax.text(x1 + panel_w / 2, y + panel_h - 0.04,
             "EXPERIMENT #1", fontsize=12, fontweight="bold", color=PURPLE, ha="center")
    ax.text(x1 + panel_w / 2, y + panel_h - 0.085,
             "LLM as Event Detector", fontsize=20, fontweight="bold",
             color=DARK, ha="center")
    ax.text(x1 + panel_w / 2, y + panel_h - 0.12,
             "(inference-time)", fontsize=11, color="#666666",
             ha="center", style="italic")

    # Mini bar chart for headline
    bars = [
        ("Codex (GPT-5)",  0.258, PURPLE),
        ("MotionBERT-Full",  0.125, GOLD),
        ("ViTPose+MotionBERT", 0.098, PALE_GOLD),
        ("MediaPipe Lite", 0.074, INDIGO),
    ]
    chart_x = x1 + 0.03
    chart_w = panel_w - 0.06
    chart_y = y + 0.18
    chart_h = 0.35
    bar_h = chart_h / (len(bars) + 1)
    for i, (lbl, val, color) in enumerate(bars):
        by = chart_y + chart_h - (i + 1) * bar_h - bar_h * 0.15
        bar_max = 0.30
        bar_pixel_w = val / bar_max * (chart_w - 0.13)
        # Bar
        rect = patches.Rectangle((chart_x + 0.12, by), bar_pixel_w,
                                   bar_h * 0.65, color=color,
                                   edgecolor=DARK, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(chart_x + 0.119, by + bar_h * 0.325, lbl,
                 fontsize=9, color=DARK, ha="right", va="center")
        ax.text(chart_x + 0.12 + bar_pixel_w + 0.005,
                 by + bar_h * 0.325, f"{val:.3f}",
                 fontsize=10, fontweight="bold", color=DARK, ha="left", va="center")
    ax.text(chart_x + chart_w / 2, chart_y - 0.005,
             "PCE@5 on same 32 clips  (higher = better)",
             fontsize=10, color="#666666", ha="center", style="italic")

    # Headline + caveat
    ax.text(x1 + panel_w / 2, y + 0.12,
             "+106% over best specialized pipeline",
             fontsize=15, fontweight="bold", color=PURPLE, ha="center")
    ax.text(x1 + panel_w / 2, y + 0.085,
             "with zero domain training, just a labeled grid of 24 sampled frames",
             fontsize=10, color=DARK, ha="center", style="italic")
    ax.text(x1 + panel_w / 2, y + 0.05,
             "Caveat:  ~50 sec/clip   ·   ~$0.10/clip   ·   non-deterministic   ·   no landmarks",
             fontsize=9, color="#999999", ha="center", style="italic")

    # === RIGHT: Data labeler ===
    rect = FancyBboxPatch((x2, y), panel_w, panel_h,
                           boxstyle="round,pad=0.008,rounding_size=0.012",
                           linewidth=3, edgecolor=PURPLE, facecolor="#F3E5F5")
    ax.add_patch(rect)
    ax.text(x2 + panel_w / 2, y + panel_h - 0.04,
             "EXPERIMENT #2", fontsize=12, fontweight="bold", color=PURPLE, ha="center")
    ax.text(x2 + panel_w / 2, y + panel_h - 0.085,
             "LLM as Data Labeler", fontsize=20, fontweight="bold",
             color=DARK, ha="center")
    ax.text(x2 + panel_w / 2, y + panel_h - 0.12,
             "(training-time)", fontsize=11, color="#666666",
             ha="center", style="italic")

    # Funnel of yields
    stages = [
        ("20 queries",     1.00),
        ("55 candidates",  0.85),
        ("55 downloads",   0.70),
        ("12 swings",      0.55),
        ("12 LLM labels",  0.40),
        ("4 final clips",  0.25),
    ]
    fx = x2 + 0.04
    fw = panel_w - 0.08
    fy = y + 0.20
    fh = 0.36
    for i, (lbl, frac) in enumerate(stages):
        by = fy + fh - (i + 1) * (fh / len(stages))
        bar_w = frac * fw
        bx = fx + (fw - bar_w) / 2
        rect = patches.Rectangle((bx, by), bar_w, fh / len(stages) * 0.75,
                                   color=PURPLE, alpha=0.65, edgecolor=DARK, linewidth=0.6)
        ax.add_patch(rect)
        ax.text(fx + fw / 2, by + fh / len(stages) * 0.4, lbl,
                 fontsize=11, fontweight="bold", color=DARK,
                 ha="center", va="center")
    ax.text(fx + fw / 2, fy + fh + 0.015,
             "POC funnel  (yields at each gate)",
             fontsize=10, color="#666666", ha="center", style="italic")

    # Headline + caveat
    ax.text(x2 + panel_w / 2, y + 0.12,
             "~$0.75 per final clip end-to-end",
             fontsize=15, fontweight="bold", color=PURPLE, ha="center")
    ax.text(x2 + panel_w / 2, y + 0.085,
             "competitive with Mechanical Turk; schema-identical to GolfDB",
             fontsize=10, color=DARK, ha="center", style="italic")
    ax.text(x2 + panel_w / 2, y + 0.05,
             "Caveat:  ~7% conversion   ·   labels noisy   ·   YouTube selection bias   ·   POC scale only",
             fontsize=9, color="#999999", ha="center", style="italic")

    out = OUT_DIR / "panel_llm_experiments.png"
    fig.savefig(out, dpi=100, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("Generating slide diagrams + tiles...")
    for fn in [diagram_self_distillation, diagram_youtube_pipeline,
                panel_three_corroborations, panel_llm_experiments]:
        out = fn()
        sz = out.stat().st_size // 1024
        print(f"  wrote {out.name}  ({sz} KB)")


if __name__ == "__main__":
    main()
