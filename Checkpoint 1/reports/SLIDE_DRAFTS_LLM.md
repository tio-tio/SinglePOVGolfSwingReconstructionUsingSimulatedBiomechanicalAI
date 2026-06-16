# Slide Drafts — LLM Experiments Additions

**For:** Motion Caddie GD checkpoint 1 deck
**Author:** Banjo
**Date:** 2026-06-04

Four new/updated slides covering today's work: Sapiens foundation-model test, LLM-as-event-detector experiment, LLM-as-data-labeler POC, and the architectural payoff (self-distillation).

Each section below is **paste-ready** — title, body, speaker notes, and the image file to drop on the slide.

---

## ✏️ UPDATED — Slide 11 (Model Selection)

**Change:** the leaderboard image already includes the LLM bar from today's session. Update the title/subtitle and speaker notes to land the new findings.

### Title
> **Model Selection — 24 Pipelines Tested on 1,400 GolfDB Clips**

### Subtitle
> *7 × 2D pose · 10 × MotionBERT 3D · 3 × GolfPose · 1 × Sapiens foundation model · 1 × LLM (Codex GPT-5)*

### Body — the image
**Use:** `Checkpoint 1/images/10_leaderboard_full.png` (full leaderboard, already in deck)

### Body — three callout boxes below the chart
- 🥇 **Specialized winner:** `MediaPipe Lite + MotionBERT-Full` — PCE@5 = 0.170, 6,900 FPS, $0 marginal
- 🚨 **LLM null hypothesis:** `Codex GPT-5 (vision)` — PCE@5 = **0.258** (n=32) — **2.06× better** than specialized winner
- ⚠️ **Three independent corroborations:** MediaPipe Heavy, GolfPose, **and** Sapiens-1B foundation model all underperform on PCE despite smoother outputs

### Speaker notes
> We tested 24 pipelines across the full 1,400-clip GolfDB corpus on a 9-metric framework. Two findings that reframe the deck:
>
> **First** — we tested Meta's Sapiens-Pose 1-billion-parameter foundation model. It produces the cleanest output we've seen: 1.2% implausible joints, 12× anatomically better than MediaPipe Lite, 100% detection rate. But on PCE@5 it lands at rank 21. That's the same pattern as MediaPipe Heavy and GolfPose — three independent corroborations of "smoother 2D doesn't help our heuristic event detector."
>
> **Second** — we tested whether a frontier vision-language model could skip the entire specialized pipeline. Codex with GPT-5 vision, fed a 6×4 grid of 24 sampled frames, hit PCE@5 = 0.258 on 32 clips. That's more than double our best specialized pipeline on the same clips. Zero domain training, just a prompt and a labeled grid.
>
> This isn't an indictment of the specialized pipeline — we still need landmarks for UE5 and biomechanics. It's evidence that the event detector specifically is the bottleneck, and a learned replacement is the next investment.

---

## 🆕 NEW SLIDE — between current Slide 11 and Slide 12

### Title
> **Three Corroborations: The Event Detector Is The Bottleneck**

### Body — the image (full-slide)
**Use:** `Checkpoint 1/images/14_panel_three_corroborations.png`

### Body text (optional, below the image if there's room)
Three different architectural choices — bigger model, golf-specific fine-tune, foundation model — all produce smoother 2D landmarks. All three score lower on PCE@5 than their lighter / generic counterparts. Same mechanism each time: smoothing displaces the wrist-Y peak frames our heuristic detector relies on.

### Speaker notes
> Across the 24 pipelines we hit a recurring pattern. MediaPipe Heavy beats Lite on traditional pose-estimation benchmarks but loses on PCE@5 — it smooths too much. GolfPose, the only model fine-tuned on actual golf swings, also underperformed generic MotionBERT by 2-2.4×. And Sapiens-1B, the largest model we tested, achieved 12× better anatomical consistency but still landed at rank 21 on PCE.
>
> Three architectural categories, same failure mode. The signal is no longer "we picked the wrong model" — it's "the bottleneck is downstream of the 2D pose."
>
> What that bottleneck is: our event detector is a simple wrist-Y argmin / argmax heuristic. It was deliberately simple so we could fairly compare pose models. Replace it with a learned 1D-CNN over MotionBERT 3D trajectories and we should see 3-5× the PCE — that's our next-sprint priority.

---

## 🆕 NEW SLIDE — LLM Experiments

### Title
> **Two LLM Experiments — Complementary, Not Competing**

### Subtitle
> *Same model (Codex GPT-5) tested in two roles*

### Body — the image (full-slide)
**Use:** `Checkpoint 1/images/15_panel_llm_experiments.png`

### Speaker notes
> We ran two LLM experiments this sprint. Both used Codex with GPT-5 vision, but for completely different jobs.
>
> **Experiment one — LLM as event detector.** Inference-time use. Sample 24 frames, lay out in a grid, ask the LLM "at what frame does each of these 8 events occur?" Result on 32 clips: PCE@5 = 0.258, more than double our best specialized pipeline. Cost ~$0.10 per clip, runtime ~50 seconds. Strong upper bound on what's achievable.
>
> **Experiment two — LLM as data labeler.** Training-time use. Eight-stage pipeline scrapes YouTube, finds swings via MediaPipe, extracts clips, asks Codex to label the 8 events on each, filters for structural sanity, merges with GolfDB. POC produced 4 schema-compatible labeled clips from 55 candidates at $0.75 per final clip — competitive with Mechanical Turk for non-expert work.
>
> Neither result alone is the whole story. Together they unlock the architecture on the next slide.

---

## 🆕 NEW SLIDE — Self-Distillation Architecture

### Title
> **Self-Distillation — The Architecture The Two Experiments Enable**

### Subtitle
> *LLM at training-time → small fast model at inference-time*

### Body — the image (full-slide)
**Use:** `Checkpoint 1/images/12_diagram_self_distillation.png`

### Speaker notes
> This is the production architecture the two LLM experiments unlock. It splits the LLM into two roles:
>
> **At training time, one-time, ~$300 + 24 hours on AWS:** combine GolfDB's 1,400 human-labeled clips with ~1,000 LLM-labeled clips from our YouTube expansion pipeline. Use the combined dataset to train a tiny 1D-CNN event detector over MotionBERT 3D landmark trajectories. The trained model is small — about 5 megabytes — and fast — under a second per swing.
>
> **At inference time, per user, ~$0.001:** existing pipeline — MediaPipe Lite, MotionBERT-Full, smoothing — feeds the trained CNN, which outputs the 8 swing events. No internet call, no LLM cost per user, no rate limits.
>
> The result: a model that approximates the LLM's PCE@5 of 0.26 but runs locally for free. We expect 0.40-0.50 in practice given the larger training set. That's product-quality.
>
> This is the path Slide 16's "train a smarter event detector" was pointing at all along. The two LLM experiments make it concrete with cost numbers and a clear cutoff between training-time spend and inference-time deployment.

---

## 🆕 OPTIONAL NEW SLIDE — YouTube Expansion Pipeline

If you want to show the data-labeler pipeline in more detail (maybe for the AWS migration conversation):

### Title
> **YouTube Data Expansion — How LLM Labeling Scales**

### Body — the image (full-slide)
**Use:** `Checkpoint 1/images/13_diagram_youtube_pipeline.png`

### Speaker notes
> Eight automated stages turn YouTube search queries into GolfDB-schema labeled training data, no human in the loop. Stage 1 pulls candidate URLs via yt-dlp. Stages 2-4 download and clip swings using MediaPipe Lite as a cheap swing detector. Stage 5 hands each clip to Codex for the 8-event labels. Stage 6 filters out obviously broken labels — non-monotonic order, impossible spacing, swing windows that don't cover the clip. Stage 7 merges into the combined dataset.
>
> Our POC ran 55 candidates and produced 4 high-quality labeled clips — 7.3% end-to-end yield. The bottleneck stages are swing detection (60% drop) and the structural filter (33% drop after labeling). At AWS scale, we estimate $300 plus 24 hours for 1,000 labeled clips — that's ~30% of the size of GolfDB, generated automatically at ~$0.30 per clip in compute.
>
> Two pre-AWS fixes needed: single-person check after bbox detection (one of our clips was a split-screen video) and tune the swing detector's threshold to lift the 22% yield. Both are well-scoped engineering tasks for next sprint.

---

## 🔁 UPDATED — Slide 16 (Plan, Risks & Next Steps)

The current "Next Steps" bullets are all done. Replace them with concrete forward-looking work:

### Replace "Next Steps" column with:

**Done this sprint (already on disk):**
- ✅ 24-pipeline benchmark on full GolfDB
- ✅ 3D pipeline with 5 UE5-ready export formats
- ✅ Smoothing layer (−51% jitter)
- ✅ LLM event-detector experiment (PCE@5 = 0.258)
- ✅ LLM data-labeler pipeline (POC, 4 clips, $0.75/clip)

**Next 2 weeks (concrete, scoped):**
- ➔ Train 1D-CNN event detector on combined GolfDB + YouTube clips → expected 3-5× PCE
- ➔ Wire BVH/CSV into UE5 Control Rig (handoff to Theo)
- ➔ Collect 20 real phone clips to validate generalization (Lawrence)
- ➔ Scale YouTube labeler to 1,000 clips on AWS (~$300, ~24h)

### Replace "Key Risks" with this updated set:

- ⚠ **Domain-specific fine-tuning isn't a free win** — GolfPose lost to generic MotionBERT by 2-2.4×. Any future fine-tune needs much larger / more diverse training data.
- ⚠ **Sapiens result shows: smoother 2D ≠ better events** — three architectures confirmed this. Must invest in event detector, not pose model.
- ⚠ **LLM labels are noisy** — 7% conversion rate in our pipeline. Mitigation: structural filters + dual-labeler consensus at scale.
- ⚠ **Real phone video untested** — all 1,400 clips are GolfDB pre-cropped. 20-clip collection in next 2 weeks.

---

## 📂 File checklist for slide editing

Drop these on the corresponding slides:

| Slide | Image | Where in Checkpoint 1 |
|---|---|---|
| 11 (updated) | `10_leaderboard_full.png` | Already in deck |
| Three corroborations (new) | `14_panel_three_corroborations.png` | `Checkpoint 1/images/` |
| LLM experiments (new) | `15_panel_llm_experiments.png` | `Checkpoint 1/images/` |
| Self-distillation (new) | `12_diagram_self_distillation.png` | `Checkpoint 1/images/` |
| YouTube pipeline (optional) | `13_diagram_youtube_pipeline.png` | `Checkpoint 1/images/` |

All four new images are **1940×1080 PNG**, sized to fill a 16:9 slide.

---

## 🎬 Suggested ordering in the deck

After the additions:

```
 1. Title
 2. Problem
 3. Why It Matters
 4. User Journey
 5. Competitive Landscape
 6. MVP Scope
 7-9. Data + EDA
 10. 2D → 3D Mapping
 11. Model Selection (24 pipelines, updated)
 12. NEW: Three Corroborations
 13. Technical Pipeline (existing)
 14. Technical Pipeline 4-panel demo
 15. NEW: Two LLM Experiments
 16. NEW: Self-Distillation Architecture
 17. Evaluation Metrics
 18. Preview
 19. Plan, Risks & Next Steps (updated)
 20. Thank you / Q&A
```

That gives the deck a clean narrative arc: data → model search → "the bottleneck isn't what we thought" → LLM experiments → architectural solution → preview → plan.
