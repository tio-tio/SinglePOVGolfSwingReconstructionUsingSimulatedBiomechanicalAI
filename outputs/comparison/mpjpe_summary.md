# MPJPE Summary

| Method | Overall Z-MPJPE (mm) | Best joint | Worst joint | Δ vs baseline |
|--------|---------------------|-----------|------------|---------------|
| MediaPipe raw | 106.0 | l_hip (5.3mm) | r_wrist (165.3mm) | +0.0mm |
| + Savitzky-Golay | 105.8 | l_hip (5.2mm) | r_wrist (163.6mm) | -0.2mm |
| + Gaussian smoothing | 105.8 | l_hip (5.2mm) | r_wrist (163.6mm) | -0.2mm |
| + Limb-length constraint | 106.0 | l_hip (5.3mm) | r_wrist (165.3mm) | +0.0mm |
| + Learned residual MLP | 34.1 | l_hip (13.3mm) | r_elbow (57.8mm) | -71.9mm |
| + Combined (SG+Limb+MLP) | 32.7 | l_hip (12.5mm) | r_elbow (57.3mm) | -73.2mm |
