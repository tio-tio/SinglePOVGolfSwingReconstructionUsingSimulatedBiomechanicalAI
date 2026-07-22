# Does per-sequence bone-rigidity depth-anchoring fix MixSTE's camera-depth residual?

**Verdict: NO-GO.** The fitted per-swing depth-scale correction is indistinguishable
from a no-op (k≈1.0 everywhere), so there is nothing for this specific mechanism to
correct. Depth residual on held-out G5/G6 moved 20.46mm → 20.44mm (+0.1%, noise).

Follow-up to `outputs/residual_diagnostic/FINDINGS.md`, which rejected a learned
pose-conditioned depth-correction MLP (doesn't transfer across subjects, r≈0) and
suggested "per-sequence depth anchoring/normalization using a known physical scale"
as the alternative worth trying. This is that experiment.

## Method

- Physical scale used: **real bone lengths are constant within a swing.** No club
  length, no stance width, no external anchor — just the rigid-body prior already
  used elsewhere in this codebase (`smoothing.py::enforce_bone_lengths`,
  `blender_mocap.py::rigidify`).
- Per swing, per candidate depth-rescale factor `k`: divide MixSTE's native depth
  axis (camera-frame convention, x=right/y=down/z=forward) by `k`, recompute all 16
  bone lengths across every frame, and score `k` by the sum of each bone's
  coefficient-of-variation² (length inconsistency across frames). Minimize over
  `k ∈ [0.3, 3.0]` with `scipy.optimize.minimize_scalar`. **No ground truth used in
  the fit** — this only touches MixSTE's own raw output, exactly like it would at
  real inference time.
- Validation: apply the fitted `k` to the raw prediction, then run the *exact same*
  measurement as `camera_depth_decomp.py` (per-frame Procrustes align to Vicon GT,
  project residual onto the GT-recovered camera depth axis) on both corrected and
  uncorrected predictions, split train (G1-G4, in-sample for the lifter) vs held-out
  test (G5-G6).

## Result

| split | n | k: mean (std) | depth: base → corrected | in-plane: base → corrected (sanity) |
|---|---|---|---|---|
| train | 30 | 1.010 (0.017) | 5.66mm → 5.71mm (−0.9%) | 3.18mm → 3.20mm (−0.9%) |
| test  | 8  | 1.002 (0.036) | 20.46mm → 20.44mm (**+0.1%**) | 6.36mm → 6.46mm (−1.5%) |

Per-swing `k` ranges 0.94–1.05 on test — a ±5% wobble around 1.0 that reads as fit
noise, not a real per-sequence miscalibration signal. No swing shows a large,
clearly-wrong `k` that correction meaningfully fixes.

## Why it failed

The hypothesis was that MixSTE's depth axis carries a broken **scale**: an
otherwise-correct pose stretched or compressed along depth, which would show up as
depth-oriented bones looking inconsistently long across frames. `k≈1` says this
isn't what's happening — **MixSTE's raw output is already bone-length
self-consistent.** Its depth error is not a corrupted scale on a correct pose; the
network outputs a *different, still-anatomically-plausible* 3D pose. This is the
textbook monocular-depth-ambiguity failure mode: multiple 3D skeletons — all with
perfectly valid bone lengths — can project to the same 2D image, and the network
picks a self-consistent but wrong one. A rigid-body / bone-length prior cannot
disambiguate between two skeletons that are both anatomically valid; it can only
catch a skeleton that has become invalid, which this one hasn't. This is consistent
with (and sharpens) `residual_diagnostic/FINDINGS.md`'s conclusion that the
depth residual is a genuine per-subject/per-pose ambiguity, not a recoverable
mis-calibration.

## What this rules out, and what's left

Ruled out (two independent mechanisms now, same NO-GO):
1. Learned pose-conditioned ΔZ correction (`residual_diagnostic/FINDINGS.md`) —
   doesn't transfer across subjects.
2. Per-sequence bone-rigidity depth rescaling (this doc) — nothing to rescale.

Both failures point the same direction: the ambiguity is in *pose selection*, not
a fixable scalar bias. The remaining honest options are the ones that add real
information the monocular frame doesn't have:
- **Club keypoints.** The checkpoint in use is the 17-body/no-club ablation; the
  paper's headline result uses 17+5 (with club). A visible club of *known length*
  is a much stronger depth anchor than bone length, because it's rigid, long, and
  usually more perpendicular to the depth axis at address/impact than most limbs
  are — bone-length CV was flat specifically because limbs don't reliably do that.
- **Genuine additional depth cues**: a second camera angle, or weak supervision
  from a depth sensor / calibrated intrinsics, if ever available.
- Accept the current depth accuracy and prioritize the two levers that are
  already known to move the needle (see `outputs/coaching_accuracy/FINDINGS.md`):
  the golf-fine-tuned lifter (−63%) and tuned One-Euro smoothing (−33–46% under
  noise) — neither of which depends on solving monocular depth.

## Files

| file | contents |
|---|---|
| `Scripts/depth_anchor_check.py` | re-runnable experiment (read-only; reuses `residual_diagnostic.py` + `camera_depth_decomp.py`) |
| `outputs/depth_anchor_check/depth_anchor_check.json` | per-swing k, depth/in-plane mm before/after, phase breakdown |
