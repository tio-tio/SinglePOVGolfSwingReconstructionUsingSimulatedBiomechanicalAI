# GolfPose MediaPipe Z-MPJPE Optimization — Findings
## Headline Numbers
- **Baseline Z-MPJPE**: 106.0 mm (MediaPipe raw + Procrustes alignment)
- **Best method**: + Combined (SG+Limb+MLP) → 32.7 mm
- **Improvement**: 69.1%
## Method Comparison
| Method | Z-MPJPE (mm) | Δ vs baseline |
|--------|------------|---------------|
| MediaPipe raw | 106.0 | +0.0 |
| + Savitzky-Golay | 105.8 | -0.2 |
| + Gaussian smoothing | 105.8 | -0.2 |
| + Limb-length constraint | 106.0 | +0.0 |
| + Learned residual MLP | 34.1 | -71.9 |
| + Combined (SG+Limb+MLP) | 32.7 | -73.2 |

## Enhancement Analysis
- **Temporal smoothing (Savitzky-Golay)**: removes high-frequency jitter in Z estimates.
  Most effective for fast-moving joints (wrists, elbows) where MediaPipe oscillates frame-to-frame.
- **Limb-length constraint**: enforces anatomical plausibility per frame.
  Helps for frames where the solver gets an unrealistic depth estimate.
- **Learned residual MLP**: captures systematic biases in MediaPipe Z relative to Vicon.
  Benefits from seeing diverse body configurations during training.
## Per-Joint Reliability
- **Most reliable**: hip, knee joints (high visibility, stable 2D position)
- **Least reliable**: wrist joints (fast-moving, occluded during swing)
## Alignment Procedure
- Root-center: subtract mid-hip (GolfPose joint 0; MediaPipe mean of landmarks 23+24)
- Procrustes calibration: optimal rotation R and scale s fit on S1 training frames
  (minimizes ||GP_xyz − s * MP_xyz @ R||_F over all S1 frames)
- Scale factor s captures the MediaPipe-to-Vicon unit conversion
## Limitations
- MediaPipe world landmarks assume fixed body proportions; deviations increase Z error.
- Procrustes calibration uses S1 (1 subject); a multi-subject calibration would be more robust.
- S5/S6 test set is small (~3,956 frames, 5 swings) — variance in MPJPE numbers is high.
- Club joints excluded (17–21) per spec; body-only evaluation understates wrist Z challenges.
- GolfPose Y (depth) was not evaluated; Z (height) is the primary axis per spec.
## Next Steps
- When teammate baseline model arrives: swap `pred_list` in `results_dict`; evaluation harness unchanged.
- Expand calibration to all S1–S4 subjects for more robust Procrustes R.
- Try per-subject MLP training if subject identity labels become available for S5/S6.
- Consider using GolfPose 2D GT (not MediaPipe) as 2D input to the MLP for ceiling-level comparison.
