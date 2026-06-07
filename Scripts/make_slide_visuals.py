"""Generate slide-ready visualizations for the Model Evaluation slide.

Produces under Data/visualizations/:
  slide_leaderboard.png         — clean top-7 horizontal bar chart
  slide_eval_panel.png          — combined 16:9 slide panel (leaderboard + findings)

Both are sized for direct Google-Slides drop-in (1920x1080 panel, 1400x700 chart).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
METRICS_PARQUET = PROJECT_ROOT / "Data" / "all_metrics.parquet"
OUT_DIR = PROJECT_ROOT / "Data" / "visualizations"

# Family classification + colors
FAMILY_COLORS = {
    "MotionBERT-Full":   "#FFB300",  # warm gold = winner
    "MotionBERT-Lite":   "#FFE082",  # pale gold
    "GolfPose":          "#E53935",  # crimson red = golf-fine-tuned surprise
    "Sapiens":           "#26A69A",  # teal = foundation pose model
    "LLM":               "#AB47BC",  # purple = vision-language model
    "2D-only":           "#5C6BC0",  # indigo blue
}


def display_name(model: str) -> str:
    if "motionbert_full_from_" in model:
        base = model.replace("motionbert_full_from_", "")
        return f"MotionBERT-Full ← {pretty_2d(base)}"
    if "motionbert_lite_from_" in model:
        base = model.replace("motionbert_lite_from_", "")
        return f"MotionBERT-Lite ← {pretty_2d(base)}"
    if "golfpose3d_from_" in model:
        base = model.replace("golfpose3d_from_", "")
        return f"GolfPose 17+0 ← {pretty_2d(base)}"
    if model == "llm_codex_gpt5":
        return "GPT-5 (Codex, vision-only)"
    if model.startswith("llm_"):
        return model.replace("llm_", "LLM: ")
    return pretty_2d(model)


def pretty_2d(name: str) -> str:
    mapping = {
        "mediapipe_lite":     "MediaPipe Lite",
        "mediapipe_heavy":    "MediaPipe Heavy",
        "movenet_thunder":    "MoveNet Thunder",
        "movenet_lightning":  "MoveNet Lightning",
        "yolov8n_pose":       "YOLOv8n-Pose",
        "yolov8m_pose":       "YOLOv8m-Pose",
        "vitpose_base":       "ViTPose Base",
        "sapiens_pose_1b":    "Sapiens-Pose 1B",
    }
    return mapping.get(name, name)


def family_of(model: str) -> str:
    if "llm_" in model: return "LLM"
    if "from_sapiens" in model and "motionbert_full" in model: return "MotionBERT-Full"
    if "from_sapiens" in model and "motionbert_lite" in model: return "MotionBERT-Lite"
    if "sapiens" in model: return "Sapiens"
    if "motionbert_full" in model: return "MotionBERT-Full"
    if "motionbert_lite" in model: return "MotionBERT-Lite"
    if "golfpose"        in model: return "GolfPose"
    return "2D-only"


def build_leaderboard():
    df = pd.read_parquet(METRICS_PARQUET)
    lb = df.groupby("model").agg(
        pce5=("pce_at_5", "mean"),
        pce3=("pce_at_3", "mean"),
        pce1=("pce_at_1", "mean"),
        fps=("fps_inference", "median"),
        bone_cv=("bone_cv_mean", "median"),
        jitter=("jitter_mean_px", "median"),
        n_clips=("clip_id", "count"),
    ).reset_index()
    lb["display"] = lb["model"].apply(display_name)
    lb["family"]  = lb["model"].apply(family_of)
    lb = lb.sort_values("pce5", ascending=False).reset_index(drop=True)
    lb["rank"] = lb.index + 1
    return lb


# =============================================================================
# Plot 1: standalone horizontal bar chart (top-7)
# =============================================================================

def render_leaderboard_chart(lb: pd.DataFrame, out_path: Path):
    """Full leaderboard — every pipeline tested, no analysis annotations."""
    rows = lb.iloc[::-1]   # reverse so rank 1 is at top of bar chart
    n = len(rows)

    fig, ax = plt.subplots(figsize=(15, max(8, n * 0.42)), dpi=140)
    fig.patch.set_facecolor("white")

    colors = [FAMILY_COLORS[f] for f in rows["family"]]
    bars = ax.barh(np.arange(n), rows["pce5"], color=colors,
                    edgecolor="#1f1f1f", linewidth=0.6, height=0.72)

    # Annotate PCE value at end of each bar — and flag any row whose
    # sample size is much smaller than the corpus (e.g. LLM subset).
    full_corpus_n = int(rows["n_clips"].max())
    for bar, pce, row_n in zip(bars, rows["pce5"], rows["n_clips"]):
        label = f"{pce:.3f}"
        if row_n < full_corpus_n * 0.9:
            label += f"  (n={int(row_n)})"
        ax.text(bar.get_width() + 0.0015, bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="left", fontsize=12,
                 fontweight="bold", color="#1f1f1f")

    # Y-axis labels: rank + display name
    ytick_labels = [f"#{int(row['rank']):>2}   {row['display']}"
                    for _, row in rows.iterrows()]
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(ytick_labels, fontsize=11)
    ax.set_xlabel(
        f"PCE@5   (mean across {full_corpus_n:,} GolfDB clips unless noted, higher = better)",
        fontsize=13,
    )
    ax.set_title(f"Model Leaderboard — all {n} pipelines tested",
                  fontsize=17, fontweight="bold", pad=16)

    # Style cleanup
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, axis="y", length=0)
    ax.set_xlim(0, max(rows["pce5"]) * 1.18)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)

    # Family legend below the chart
    legend_handles = [patches.Patch(facecolor=c, edgecolor="#1f1f1f",
                                      linewidth=0.6, label=f)
                       for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=11,
              frameon=False, title="Model family", title_fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Plot 2: combined 16:9 slide panel
# =============================================================================

def render_slide_panel(lb: pd.DataFrame, out_path: Path):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=110)  # 1920x1080 @ 110 dpi
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0.55, wspace=0.25,
                          left=0.04, right=0.98, top=0.92, bottom=0.06)

    # Title
    fig.suptitle("Model Evaluation — 20 Pipelines Tested on 1,400 GolfDB Clips",
                  fontsize=24, fontweight="bold", y=0.97)

    # ---- Top 2/3: leaderboard chart ----
    ax_lb = fig.add_subplot(gs[0:2, :])
    top = lb.head(7).iloc[::-1]
    colors = [FAMILY_COLORS[f] for f in top["family"]]
    bars = ax_lb.barh(np.arange(len(top)), top["pce5"], color=colors,
                       edgecolor="#1f1f1f", linewidth=0.8, height=0.7)
    for bar, pce in zip(bars, top["pce5"]):
        ax_lb.text(bar.get_width() + 0.0015, bar.get_y() + bar.get_height() / 2,
                    f"{pce:.3f}", va="center", ha="left", fontsize=14, fontweight="bold")
    ytick_labels = [f"#{int(row['rank'])}   {row['display']}"
                     for _, row in top.iterrows()]
    ax_lb.set_yticks(np.arange(len(top)))
    ax_lb.set_yticklabels(ytick_labels, fontsize=13)
    ax_lb.set_xlabel("PCE@5  (higher = better)", fontsize=14)
    for s in ("top", "right", "left"): ax_lb.spines[s].set_visible(False)
    ax_lb.tick_params(left=False, axis="y", length=0)
    ax_lb.set_xlim(0, max(top["pce5"]) * 1.22)
    ax_lb.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax_lb.set_axisbelow(True)
    ax_lb.set_title("Top 7 of 20 pipelines, sorted by PCE@5",
                     fontsize=15, pad=8, loc="left")

    # GolfPose surprise annotation
    golfpose_rows = top[top["family"] == "GolfPose"]
    if len(golfpose_rows):
        gp_idx_in_top = top.index.get_loc(golfpose_rows.index[0])
        gp_pce = golfpose_rows.iloc[0]["pce5"]
        ax_lb.annotate(
            "THE SURPRISE\ngolf-specific model\nunderperforms generic",
            xy=(gp_pce + 0.003, gp_idx_in_top),
            xytext=(gp_pce + 0.04, gp_idx_in_top + 0.6),
            fontsize=12, ha="left", color="#B71C1C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.6,
                             connectionstyle="arc3,rad=0.2"),
        )

    legend_handles = [patches.Patch(facecolor=c, edgecolor="#1f1f1f",
                                      linewidth=0.8, label=f)
                       for f, c in FAMILY_COLORS.items()]
    ax_lb.legend(handles=legend_handles, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=11,
                  frameon=False, title="Model family", title_fontsize=11)

    # ---- Bottom: 3 finding cards ----
    findings = [
        ("CHEAP BEATS EXPENSIVE\n(on the 2D side)",
         "MediaPipe Lite beats Heavy on\n"
         "golf-event accuracy by 73%,\n"
         "runs 4× faster on CPU.\n\n"
         "Coworker's MediaPipe Heavy\n"
         "default was the worst variant.",
         FAMILY_COLORS["2D-only"]),
        ("3D LIFT IS ESSENTIALLY FREE",
         "MotionBERT-Full adds +89%\n"
         "PCE on MediaPipe Lite, runs\n"
         "at 7,000 FPS post-2D.\n\n"
         "Adding 3D is the largest\n"
         "single accuracy win we found.",
         FAMILY_COLORS["MotionBERT-Full"]),
        ("GOLF-SPECIFIC ≠ BETTER\n(the surprise)",
         "GolfPose, fine-tuned on actual\n"
         "golf swings, lost to generic\n"
         "MotionBERT-Full by 2–2.4×.\n\n"
         "Tiny training set (4 subjects)\n"
         "didn't generalize to GolfDB.",
         FAMILY_COLORS["GolfPose"]),
    ]
    for col, (title, body, accent_color) in enumerate(findings):
        ax = fig.add_subplot(gs[2, col])
        ax.axis("off")
        # Card background
        rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                               boxstyle="round,pad=0.02",
                               linewidth=2.5, edgecolor=accent_color,
                               facecolor="#fafafa", transform=ax.transAxes)
        ax.add_patch(rect)
        # Title bar
        ax.text(0.5, 0.86, title, transform=ax.transAxes,
                 fontsize=14, fontweight="bold", color=accent_color,
                 va="top", ha="center", linespacing=1.3)
        # Body
        ax.text(0.07, 0.55, body, transform=ax.transAxes,
                 fontsize=12, va="top", ha="left", color="#1f1f1f",
                 linespacing=1.45)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lb = build_leaderboard()
    print(f"[viz] loaded leaderboard with {len(lb)} models")

    out1 = OUT_DIR / "slide_leaderboard.png"
    render_leaderboard_chart(lb, out1)
    print(f"[viz] wrote {out1.relative_to(PROJECT_ROOT)}  ({out1.stat().st_size // 1024} KB)")

    # Remove stale eval-panel image if it exists from a previous run
    stale = OUT_DIR / "slide_eval_panel.png"
    if stale.exists():
        stale.unlink()
        print(f"[viz] removed stale {stale.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
