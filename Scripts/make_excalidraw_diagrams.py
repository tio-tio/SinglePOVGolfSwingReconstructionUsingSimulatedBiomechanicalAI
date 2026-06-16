"""Generate four slide-ready Excalidraw diagrams.

Outputs to Data/visualizations/excalidraw/*.excalidraw — open at
excalidraw.com (File → Open) or in any Excalidraw plugin.

Diagrams:
  diagram_self_distillation       — training-time vs inference-time architecture
  diagram_youtube_pipeline        — 8-stage YouTube → labeled clips funnel
  panel_three_corroborations      — MP Heavy / GolfPose / Sapiens pattern
  panel_llm_experiments           — 2-up LLM experiments tile
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import excalidraw_builder as ex

OUT_DIR = Path(__file__).parent.parent / "Data" / "visualizations" / "excalidraw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette — uses Excalidraw's recommended colors so they look "native"
GOLD       = "#f0a847"   # winner / MotionBERT-Full
PALE_GOLD  = "#fde2b3"
PURPLE     = "#9b59b6"
LIGHT_PURP = "#e8d5f0"
TEAL       = "#1ba39c"
LIGHT_TEAL = "#c8e6e3"
RED        = "#d9485f"
LIGHT_RED  = "#fcd6dc"
INDIGO     = "#5b6cdb"
LIGHT_INDI = "#dde0fa"
GREEN      = "#3a7c50"
LIGHT_GREE = "#cce8d4"
GRAY       = "#5c6370"
DARK       = "#1e1e1e"
SOFT_BG_T  = "#fff8e1"   # warm wash, training phase
SOFT_BG_I  = "#e3f2fd"   # cool wash, inference phase


# =============================================================================
# Diagram 1 — Self-distillation
# =============================================================================

def diagram_self_distillation():
    els = []

    # Title
    els.append(ex.text(80, 30, "Self-Distillation Architecture",
                        font_size=36, stroke_color=DARK, font_family=1, width=900))
    els.append(ex.text(80, 80,
                        "Use a frontier LLM at training time → small fast model serves users at inference time",
                        font_size=18, stroke_color=GRAY, font_family=1, width=1200))

    # ===== TRAINING PHASE (top) =====
    els.extend(ex.rect(60, 140, 1800, 320,
                        background_color=SOFT_BG_T,
                        stroke_color=GOLD, stroke_width=1,
                        fill_style="solid", roughness=0))
    els.append(ex.text(80, 160, "①  TRAINING PHASE",
                        font_size=24, stroke_color=GOLD, font_family=1, width=400))
    els.append(ex.text(80, 195, "one-time   ·   ~$300   ·   ~24 hours on AWS",
                        font_size=14, stroke_color=GRAY, font_family=1, width=400))

    # Three input boxes
    box_golfdb = ex.rect(120, 250, 280, 110, text="GolfDB\n1,400 human-labeled clips",
                          stroke_color=GRAY, background_color="#ffffff",
                          font_size=16, stroke_width=2)
    els.extend(box_golfdb)
    golfdb_id = box_golfdb[0]["id"]

    box_yt = ex.rect(440, 250, 300, 110,
                      text="YouTube Pipeline\n~1,000 LLM-labeled clips\n(Experiment #2)",
                      stroke_color=PURPLE, background_color="#ffffff",
                      font_size=16, stroke_width=3, text_color=PURPLE)
    els.extend(box_yt)
    yt_id = box_yt[0]["id"]

    box_llm = ex.rect(780, 250, 300, 110,
                       text="LLM Event Labels\nCodex GPT-5 vision\n(Experiment #1)",
                       stroke_color=PURPLE, background_color="#ffffff",
                       font_size=16, stroke_width=3, text_color=PURPLE)
    els.extend(box_llm)
    llm_id = box_llm[0]["id"]

    # Combined dataset
    box_combined = ex.rect(440, 400, 640, 50,
                            text="Combined Dataset — 2,400+ clips, mixed gold + LLM-pseudo labels",
                            stroke_color=GOLD, background_color="#fff3e0",
                            font_size=16, stroke_width=3, text_color=GOLD)
    els.extend(box_combined)
    combined_id = box_combined[0]["id"]

    # Trained CNN
    box_cnn_train = ex.rect(1180, 320, 290, 100,
                             text="1D-CNN Event Detector\n~5 MB  ·  <1 sec inference",
                             stroke_color=GREEN, background_color="#cce8d4",
                             font_size=16, stroke_width=3, text_color=GREEN)
    els.extend(box_cnn_train)
    cnn_train_id = box_cnn_train[0]["id"]

    # Arrows into Combined
    els.append(ex.arrow(260, 360, 540, 410, stroke_color=GRAY,
                         start_id=golfdb_id, end_id=combined_id))
    els.append(ex.arrow(590, 360, 670, 410, stroke_color=PURPLE,
                         start_id=yt_id, end_id=combined_id))
    els.append(ex.arrow(930, 360, 850, 410, stroke_color=PURPLE,
                         start_id=llm_id, end_id=combined_id))
    # Arrow Combined → CNN
    els.append(ex.arrow(1080, 425, 1180, 370, stroke_color=GREEN, stroke_width=3,
                         start_id=combined_id, end_id=cnn_train_id))

    # ===== INFERENCE PHASE (bottom) =====
    els.extend(ex.rect(60, 540, 1800, 380,
                        background_color=SOFT_BG_I,
                        stroke_color=INDIGO, stroke_width=1,
                        fill_style="solid", roughness=0))
    els.append(ex.text(80, 560, "②  INFERENCE PHASE",
                        font_size=24, stroke_color=INDIGO, font_family=1, width=400))
    els.append(ex.text(80, 595,
                        "per-user   ·   ~$0.001 marginal   ·   ~1 second total wall-clock",
                        font_size=14, stroke_color=GRAY, font_family=1, width=600))

    # Pipeline stages
    pipeline = [
        ("phone.mp4",       "user upload",        "#ffffff", GRAY),
        ("MediaPipe Lite",  "2D landmarks · 107 FPS · CPU", "#ffffff", INDIGO),
        ("MotionBERT-Full", "3D lift · 6,900 FPS · GPU",     "#ffffff", GOLD),
        ("1D-CNN Detector", "8 swing events\nFROM TRAINING ABOVE", LIGHT_GREE, GREEN),
        ("UE5 / Scorecard", "BVH + CSV + 3D viz", "#ffffff", GRAY),
    ]
    n = len(pipeline)
    box_w = 320
    gap = 25
    total_w = n * box_w + (n - 1) * gap
    x0 = (1920 - total_w) / 2

    stage_ids = []
    for i, (title, sub, bg, color) in enumerate(pipeline):
        x = x0 + i * (box_w + gap)
        box = ex.rect(x, 700, box_w, 150,
                       text=f"{title}\n\n{sub}",
                       stroke_color=color, background_color=bg,
                       font_size=18, stroke_width=3 if i == 3 else 2,
                       text_color=color)
        els.extend(box)
        stage_ids.append(box[0]["id"])

    for i in range(n - 1):
        x1 = x0 + (i + 1) * (box_w + gap) - gap
        x2 = x0 + (i + 1) * (box_w + gap)
        els.append(ex.arrow(x1, 775, x2, 775, stroke_color=DARK, stroke_width=2,
                             start_id=stage_ids[i], end_id=stage_ids[i + 1]))

    # Dashed green arrow training → inference
    cnn_inf_idx = 3
    cnn_inf_x = x0 + cnn_inf_idx * (box_w + gap) + box_w / 2
    els.append(ex.arrow(cnn_inf_x, 420, cnn_inf_x, 700,
                         stroke_color=GREEN, stroke_width=3, dashed=True,
                         start_id=cnn_train_id, end_id=stage_ids[3]))
    els.append(ex.text(cnn_inf_x + 20, 545, "trained model deployed",
                        font_size=14, stroke_color=GREEN, font_family=1, width=300))

    # Footer
    els.append(ex.text(80, 950,
                        "Two LLM experiments enable this:  Experiment #2 generates training data  ·  Experiment #1 proves a small model can hit competitive PCE.",
                        font_size=14, stroke_color=GRAY, font_family=1, width=1700))

    out_path = OUT_DIR / "diagram_self_distillation.excalidraw"
    ex.save(els, str(out_path))
    return out_path


# =============================================================================
# Diagram 2 — YouTube data expansion pipeline
# =============================================================================

def diagram_youtube_pipeline():
    els = []

    els.append(ex.text(80, 30, "YouTube Data Expansion Pipeline",
                        font_size=36, stroke_color=DARK, font_family=1, width=900))
    els.append(ex.text(80, 80,
                        "8 fully-automated stages: queries in → GolfDB-schema labeled clips out",
                        font_size=18, stroke_color=GRAY, font_family=1, width=1200))

    stages = [
        ("1", "SEARCH",        "yt-dlp",              "20 queries → 55 cand",      GRAY),
        ("2", "DOWNLOAD",      "yt-dlp 480p",         "55 URLs → 55 mp4s",         GRAY),
        ("3", "DETECT SWINGS", "MediaPipe Lite",      "55 vids → 12 swings",       INDIGO),
        ("4", "EXTRACT",       "160×160 crop",        "12 swings → 12 clips",      INDIGO),
        ("5", "LLM LABEL",     "Codex / GPT-5",       "12 clips → 12 labels",      PURPLE),
        ("6", "FILTER",        "monotonic + spacing", "12 labels → 4 clips",       RED),
        ("7", "MERGE",         "GolfDB schema",       "4 clips → +4 rows",         GOLD),
    ]

    n = len(stages)
    margin_x = 60
    avail = 1920 - 2 * margin_x
    box_w = (avail - 30 * (n - 1)) / n
    box_h = 260
    y = 250

    stage_ids = []
    for i, (num, title, tool, io, color) in enumerate(stages):
        x = margin_x + i * (box_w + 30)
        bg_map = {GRAY: "#ffffff", INDIGO: LIGHT_INDI, PURPLE: LIGHT_PURP,
                   RED: LIGHT_RED, GOLD: PALE_GOLD}
        bg = bg_map.get(color, "#ffffff")

        box = ex.rect(x, y, box_w, box_h,
                       stroke_color=color, background_color=bg,
                       stroke_width=3)
        els.extend(box)
        stage_ids.append(box[0]["id"])

        # Number large
        els.append(ex.text(x + box_w / 2 - 20, y + 20, num,
                            font_size=48, stroke_color=color, font_family=1, width=80))
        # Title
        els.append(ex.text(x + 10, y + 100, title,
                            font_size=20, stroke_color=DARK, font_family=1,
                            width=box_w - 20, text_align="center"))
        # Tool
        els.append(ex.text(x + 10, y + 145, tool,
                            font_size=14, stroke_color=DARK, font_family=1,
                            width=box_w - 20, text_align="center"))
        # I/O
        els.append(ex.text(x + 10, y + 200, io,
                            font_size=12, stroke_color=GRAY, font_family=1,
                            width=box_w - 20, text_align="center"))

        if i < n - 1:
            x_next = margin_x + (i + 1) * (box_w + 30)
            els.append(ex.arrow(x + box_w + 4, y + box_h / 2,
                                 x_next - 4, y + box_h / 2,
                                 stroke_color=DARK,
                                 start_id=stage_ids[i]))

    # Combined dataset callout
    combined_box = ex.rect(560, 600, 800, 130,
                            text="Combined Dataset\n\nGolfDB (1,400 gold) + MotionCaddie-Extra (4 LLM-pseudo) = 1,404 clips\nSchema-identical → drop-in compatible with all downstream code",
                            stroke_color=GOLD, background_color=PALE_GOLD,
                            stroke_width=3, font_size=14, text_color=GOLD)
    els.extend(combined_box)
    combined_id = combined_box[0]["id"]

    # Arrow from stage 7 down to combined
    last_x = margin_x + (n - 1) * (box_w + 30) + box_w / 2
    els.append(ex.arrow(last_x, y + box_h + 5, last_x, 605,
                         stroke_color=GOLD, stroke_width=3,
                         start_id=stage_ids[-1], end_id=combined_id))
    els.append(ex.arrow(last_x - 4, 665, 1360 - 4, 665, stroke_color=GOLD,
                         stroke_width=3, start_id=combined_id))

    # POC + scaling footers
    els.append(ex.text(80, 800,
                        "POC RESULT  (1 hour wall-clock, ~$3 spend):  55 candidates → 4 final clips  ·  7.3% end-to-end yield",
                        font_size=20, stroke_color=DARK, font_family=1, width=1800))
    els.append(ex.text(80, 850,
                        "Scaling math:  15,000 candidates → ~1,000 final clips at ~$300 + 24 hours on AWS",
                        font_size=16, stroke_color=GRAY, font_family=1, width=1700))

    out_path = OUT_DIR / "diagram_youtube_pipeline.excalidraw"
    ex.save(els, str(out_path))
    return out_path


# =============================================================================
# Diagram 3 — Three corroborations
# =============================================================================

def panel_three_corroborations():
    els = []

    els.append(ex.text(80, 30, "Three Independent Corroborations",
                        font_size=36, stroke_color=DARK, font_family=1, width=1100))
    els.append(ex.text(80, 80,
                        "Smoother 2D landmarks → displaced wrist-Y peaks → our heuristic event detector misses them",
                        font_size=18, stroke_color=GRAY, font_family=1, width=1500))

    cards = [
        ("#1", "MediaPipe Heavy", "Bigger pose model",
         "Smoothing within MediaPipe\ndisplaces peak frames",
         "PCE@5 = 0.052",
         "(MP Lite: 0.090 — bigger LOSES)",
         INDIGO, LIGHT_INDI),
        ("#2", "GolfPose 17+0", "Golf-fine-tuned model",
         "Domain training adds\ntemporal smoothing",
         "PCE@5 = 0.042–0.076",
         "(MotionBERT: 2-2.4× higher)",
         RED, LIGHT_RED),
        ("#3", "Sapiens-Pose 1B", "Foundation model (Meta)",
         "Foundation architecture\nover-smooths predictions",
         "PCE@5 = 0.043",
         "(though 1.2% implausible —\n12× cleaner anatomy)",
         TEAL, LIGHT_TEAL),
    ]

    card_w = 540
    card_h = 470
    gap = 60
    total_w = 3 * card_w + 2 * gap
    x0 = (1920 - total_w) / 2
    y0 = 180

    card_ids = []
    for i, (n, model, type_, mechanism, headline, sub, color, bg) in enumerate(cards):
        x = x0 + i * (card_w + gap)
        card = ex.rect(x, y0, card_w, card_h,
                        stroke_color=color, background_color=bg,
                        stroke_width=4)
        els.extend(card)
        card_ids.append(card[0]["id"])

        els.append(ex.text(x + card_w / 2 - 20, y0 + 30, n,
                            font_size=28, stroke_color=color, font_family=1, width=80))
        els.append(ex.text(x + 10, y0 + 90, model,
                            font_size=28, stroke_color=DARK, font_family=1,
                            width=card_w - 20, text_align="center"))
        els.append(ex.text(x + 10, y0 + 140, type_,
                            font_size=16, stroke_color=GRAY, font_family=1,
                            width=card_w - 20, text_align="center"))
        els.append(ex.text(x + 30, y0 + 200, mechanism,
                            font_size=18, stroke_color=DARK, font_family=1,
                            width=card_w - 60, text_align="center"))
        els.append(ex.text(x + 10, y0 + 320, headline,
                            font_size=28, stroke_color=color, font_family=1,
                            width=card_w - 20, text_align="center"))
        els.append(ex.text(x + 30, y0 + 380, sub,
                            font_size=14, stroke_color=GRAY, font_family=1,
                            width=card_w - 60, text_align="center"))

    # Conclusion
    concl_x = (1920 - 1400) / 2
    concl = ex.rect(concl_x, 720, 1400, 160,
                     stroke_color=GOLD, background_color=PALE_GOLD,
                     stroke_width=4)
    els.extend(concl)
    concl_id = concl[0]["id"]
    els.append(ex.text(concl_x + 10, 745, "Convergent Conclusion",
                        font_size=24, stroke_color=GOLD, font_family=1,
                        width=1380, text_align="center"))
    els.append(ex.text(concl_x + 30, 790,
                        "The bottleneck is NOT the 2D pose model — it's the heuristic event detector.\n"
                        "Replace it with a learned 1D-CNN over MotionBERT 3D trajectories → expected 3-5× PCE.",
                        font_size=16, stroke_color=DARK, font_family=1,
                        width=1340, text_align="center"))

    # Arrows from each card → conclusion
    for i in range(3):
        x = x0 + i * (card_w + gap) + card_w / 2
        els.append(ex.arrow(x, y0 + card_h + 5, x, 715,
                             stroke_color=GOLD, stroke_width=3,
                             start_id=card_ids[i], end_id=concl_id))

    out_path = OUT_DIR / "panel_three_corroborations.excalidraw"
    ex.save(els, str(out_path))
    return out_path


# =============================================================================
# Diagram 4 — LLM experiments tile
# =============================================================================

def panel_llm_experiments():
    els = []

    els.append(ex.text(80, 30, "Two LLM Experiments — Complementary, Not Competing",
                        font_size=32, stroke_color=DARK, font_family=1, width=1700))
    els.append(ex.text(80, 80,
                        "Same model (Codex / GPT-5 with vision) tested in two roles",
                        font_size=18, stroke_color=GRAY, font_family=1, width=1500))

    panel_w = 870
    panel_h = 800
    gap = 50
    x_left = (1920 - 2 * panel_w - gap) / 2
    x_right = x_left + panel_w + gap
    y = 150

    # ===== Left: Event detector =====
    left = ex.rect(x_left, y, panel_w, panel_h,
                    stroke_color=PURPLE, background_color=LIGHT_PURP,
                    stroke_width=4)
    els.extend(left)

    els.append(ex.text(x_left + 30, y + 30, "EXPERIMENT #1",
                        font_size=18, stroke_color=PURPLE, font_family=1, width=300))
    els.append(ex.text(x_left + 30, y + 70, "LLM as Event Detector",
                        font_size=32, stroke_color=DARK, font_family=1, width=800))
    els.append(ex.text(x_left + 30, y + 120, "(inference-time)",
                        font_size=16, stroke_color=GRAY, font_family=1, width=400))

    # Bar chart
    bars = [
        ("Codex (GPT-5)",  0.258, PURPLE),
        ("MotionBERT-Full", 0.125, GOLD),
        ("ViTPose + MotionBERT", 0.098, PALE_GOLD),
        ("MediaPipe Lite",  0.074, INDIGO),
    ]
    chart_x = x_left + 50
    chart_w = panel_w - 100
    chart_y = y + 200
    bar_h = 50
    bar_gap = 30
    bar_max = 0.30
    for i, (lbl, val, color) in enumerate(bars):
        by = chart_y + i * (bar_h + bar_gap)
        bar_pixel_w = val / bar_max * (chart_w - 220)
        # Label
        els.append(ex.text(chart_x, by + 10, lbl,
                            font_size=14, stroke_color=DARK, font_family=1, width=210, text_align="right"))
        # Bar
        bar = ex.rect(chart_x + 220, by, bar_pixel_w, bar_h,
                       stroke_color=color, background_color=color,
                       stroke_width=2, fill_style="solid")
        els.extend(bar)
        # Value
        els.append(ex.text(chart_x + 230 + bar_pixel_w, by + 14, f"{val:.3f}",
                            font_size=16, stroke_color=DARK, font_family=1, width=80))

    els.append(ex.text(chart_x, chart_y + 4 * (bar_h + bar_gap) + 10,
                        "PCE@5 on same 32 clips  (higher = better)",
                        font_size=12, stroke_color=GRAY, font_family=1, width=600))

    els.append(ex.text(x_left + 30, y + 600, "+106% over best specialized pipeline",
                        font_size=24, stroke_color=PURPLE, font_family=1, width=820, text_align="center"))
    els.append(ex.text(x_left + 30, y + 650,
                        "with zero domain training, just a labeled grid of 24 sampled frames",
                        font_size=14, stroke_color=DARK, font_family=1, width=820, text_align="center"))
    els.append(ex.text(x_left + 30, y + 720,
                        "Caveat:  ~50 sec/clip   ·   ~$0.10/clip   ·   non-deterministic   ·   no landmarks",
                        font_size=12, stroke_color=GRAY, font_family=1, width=820, text_align="center"))

    # ===== Right: Data labeler =====
    right = ex.rect(x_right, y, panel_w, panel_h,
                     stroke_color=PURPLE, background_color=LIGHT_PURP,
                     stroke_width=4)
    els.extend(right)

    els.append(ex.text(x_right + 30, y + 30, "EXPERIMENT #2",
                        font_size=18, stroke_color=PURPLE, font_family=1, width=300))
    els.append(ex.text(x_right + 30, y + 70, "LLM as Data Labeler",
                        font_size=32, stroke_color=DARK, font_family=1, width=800))
    els.append(ex.text(x_right + 30, y + 120, "(training-time)",
                        font_size=16, stroke_color=GRAY, font_family=1, width=400))

    # Funnel
    funnel_stages = [
        ("20 queries",     720),
        ("55 candidates",  620),
        ("55 downloads",   520),
        ("12 swings",      420),
        ("12 LLM labels",  300),
        ("4 final clips",  180),
    ]
    fy_start = y + 200
    fy_step = 65
    for i, (lbl, w) in enumerate(funnel_stages):
        by = fy_start + i * fy_step
        bx = x_right + (panel_w - w) / 2
        funnel = ex.rect(bx, by, w, 50,
                          stroke_color=PURPLE, background_color=PURPLE,
                          stroke_width=2, opacity=70, fill_style="solid",
                          text=lbl, font_size=16, text_color="#ffffff")
        els.extend(funnel)

    els.append(ex.text(x_right + 30, fy_start + 6 * fy_step + 30,
                        "POC funnel  (yields at each gate)",
                        font_size=12, stroke_color=GRAY, font_family=1,
                        width=820, text_align="center"))

    els.append(ex.text(x_right + 30, y + 600, "~$0.75 per final clip end-to-end",
                        font_size=24, stroke_color=PURPLE, font_family=1, width=820, text_align="center"))
    els.append(ex.text(x_right + 30, y + 650,
                        "competitive with Mechanical Turk; schema-identical to GolfDB",
                        font_size=14, stroke_color=DARK, font_family=1, width=820, text_align="center"))
    els.append(ex.text(x_right + 30, y + 720,
                        "Caveat:  ~7% conversion   ·   labels noisy   ·   YouTube selection bias   ·   POC scale",
                        font_size=12, stroke_color=GRAY, font_family=1, width=820, text_align="center"))

    out_path = OUT_DIR / "panel_llm_experiments.excalidraw"
    ex.save(els, str(out_path))
    return out_path


# =============================================================================

def main():
    print("Generating Excalidraw diagrams...")
    import random
    random.seed(42)
    for fn in [diagram_self_distillation, diagram_youtube_pipeline,
                panel_three_corroborations, panel_llm_experiments]:
        out = fn()
        sz = out.stat().st_size // 1024
        print(f"  wrote {out.name}  ({sz} KB)")
    print(f"\nOpen any of these at https://excalidraw.com (File → Open)")


if __name__ == "__main__":
    main()
