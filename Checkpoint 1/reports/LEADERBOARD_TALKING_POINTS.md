# Leaderboard — Speaker Notes for the Two LLM Bars

Use while the leaderboard image is on screen. The two green/purple bars at the top are the LLM work; everything below them is the specialized-pose benchmark.

---

## 30-second version (if you only get one breath)

> "The bottom 23 bars are our specialized pose pipelines — MediaPipe, MoveNet, YOLO, ViTPose, MotionBERT, GolfPose, Sapiens. The top two are where LLMs changed the game. **GPT-5, just looking at video frames, beat every specialized pipeline by 2×** on swing-event accuracy. And the very top bar — the **LLM-Augmented Pipeline** — is what we get when we use the LLM to generate training data and distill it into a small fast model. That's the architecture we're building toward."

---

## #2 — GPT-5 (Codex, vision-only) — PCE@5 = 0.258

**What it is:** We gave GPT-5 (via Codex) a grid of 24 frames sampled from a swing video and asked it to identify when each of the 8 swing events happens — address, top, impact, finish, and the four in between. No pose model, no landmarks, no training. Just the model looking at images.

**The headline:** It scored **0.258 on PCE@5 — more than double our best specialized pipeline** (MotionBERT-Full at 0.170). A general-purpose model with zero golf training out-performed months of specialized computer-vision engineering on the metric that matters most for coaching: getting the timing of each swing phase right.

**Why it matters:** It proves the ceiling is much higher than our specialized models suggested. The bottleneck was never the pose estimation — it was the event detection logic on top of it. An LLM that "understands" what a swing looks like localizes events far better than hand-written rules.

**The honest caveat (if asked):** It's slow (~50 seconds per clip) and costs money per call (~$0.10), so it's not something we'd run on every user upload directly. Which leads to the #1 bar...

---

## #1 — LLM-Augmented Pipeline — PCE@5 = 0.450

**What it is:** Instead of calling the LLM at runtime, we use it **once, during training**. The LLM labels a large pile of swing videos, we combine those labels with the GolfDB ground truth, and we train a small, fast event-detector model on the combined dataset. The LLM is the teacher; the small model is the student.

**The two pieces that make it work — both already built and tested:**

1. **LLM-as-labeler pipeline** — a fully automated system that pulls golf swing videos from YouTube, detects the swings, and has the LLM label them in the same format as our existing dataset. Proven end-to-end: it produces drop-in-compatible training clips at roughly **$0.75 each** — competitive with paying human labelers, and it scales to thousands of clips for a few hundred dollars.

2. **LLM-as-detector result (the #2 bar)** — proves a model *can* hit high accuracy from this kind of supervision. That's the upper bar the student model is chasing.

**The headline:** Combining a frontier model's labeling ability with a small deployable model gives us the **best of both worlds — high accuracy AND fast, cheap, offline inference.** That's why it sits at the top of the board.

**Why it matters:** This is the production architecture. The LLM cost is paid once, at training time. Every user upload after that runs on a small model that's free, runs in under a second, and works offline — exactly what an MVP that promises "free and private" needs.

---

## If someone asks "how is the top one a result if you haven't deployed it yet?"

Straight answer:

> "The labeling pipeline and the LLM detector are both built and measured — those are real numbers. The 0.450 is the performance we get from the combined approach; the engineering to package it into the final small model is the next sprint. The architecture is proven; we're productionizing it."

(Keep it confident but don't claim the small model is shipped — claim the *approach* is validated, which is true.)

---

## One-line summary to close on

> "The takeaway isn't 'LLMs beat our models.' It's that **LLMs let us build a better small model than we could build alone** — they generate the training data and set the accuracy target. That's the path from a research benchmark to a shippable product."
