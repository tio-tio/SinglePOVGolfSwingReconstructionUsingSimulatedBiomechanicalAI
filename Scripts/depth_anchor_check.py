"""Does per-sequence bone-rigidity depth-anchoring reduce MixSTE's camera-depth
residual? The follow-up experiment residual_diagnostic.py's FINDINGS.md pointed
at: not a learned pose-conditioned MLP (NO-GO, doesn't transfer across subjects),
but "per-sequence depth anchoring/normalization" using a known physical scale.

The physical scale used here: real bone lengths are constant within a swing.
MixSTE's depth axis carries ~92% of its residual energy (camera_depth_decomp.py),
while in-plane is comparatively accurate (6.4mm vs 20.5mm on held-out subjects).
So: for each swing, find a single global depth-axis rescale factor k that makes
every bone's length (recomputed with the corrected depth) as SELF-CONSISTENT
as possible across frames -- no ground truth involved, this only uses MixSTE's
own raw camera-frame output. Apply k, then re-run the exact same
procrustes_frame + recover_camera evaluation as camera_depth_decomp.py to see
whether it actually reduces error against Vicon truth on held-out G5/G6.

Read-only. Does not touch the default lifter path. Requires
scratchpad/mixste_preds_G1-G6.pkl (built by residual_diagnostic.py).

Run: python3 Scripts/depth_anchor_check.py
"""
from __future__ import annotations
import sys, pickle, json, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "Scripts"))
from residual_diagnostic import (load_vicon, procrustes_frame, JOINT_NAMES,
                                 TRAIN_SUBJ, TEST_SUBJ, detect_phases, PHASES)
from camera_depth_decomp import recover_camera

OUT = PROJ / "outputs" / "depth_anchor_check"
OUT.mkdir(parents=True, exist_ok=True)

# Same topology as export_ue5.H36M17_PARENTS; JOINT_NAMES here just renames
# joint 8/9 (thorax->neck, neck->nose) but the parent-index skeleton is identical.
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
BONES = [(j, PARENTS[j]) for j in range(17) if PARENTS[j] >= 0]
DEPTH_AXIS = 2  # MixSTE's native camera-frame convention: x=right,y=down,z=forward


def bone_length_cv_loss(k: float, pc: np.ndarray) -> float:
    """Sum of per-bone coefficient-of-variation^2 of bone length, after
    rescaling the depth axis by 1/k. pc: (T,17,3) root-relative, native frame."""
    corrected = pc.copy()
    corrected[..., DEPTH_AXIS] = corrected[..., DEPTH_AXIS] / k
    loss = 0.0
    for child, parent in BONES:
        L = np.linalg.norm(corrected[:, child] - corrected[:, parent], axis=-1)
        m = L.mean()
        if m < 1e-9:
            continue
        loss += (L.std() / m) ** 2
    return loss


def fit_depth_scale(pred: np.ndarray) -> float:
    """Per-swing: find k minimizing bone-length self-inconsistency. k>1 means
    the raw prediction's depth axis is OVER-scaled (bones with depth content
    look too long/variable) relative to what's physically self-consistent."""
    pc = pred - pred[:, :1]
    res = minimize_scalar(bone_length_cv_loss, args=(pc,), bounds=(0.3, 3.0),
                          method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def apply_depth_anchor(pred: np.ndarray, k: float) -> np.ndarray:
    pc = pred - pred[:, :1]
    pc[..., DEPTH_AXIS] = pc[..., DEPTH_AXIS] / k
    return pc + pred[:, :1]


def eval_camera_depth(pred: np.ndarray, gt: np.ndarray, xy: np.ndarray):
    """Same measurement as camera_depth_decomp.py: procrustes_frame align to
    GT, project residual onto GT-recovered camera depth axis."""
    aligned, gc, _ = procrustes_frame(pred, gt)
    R = gc - aligned
    B = recover_camera(xy, gt[:, :17, :])
    Rcam = R @ B.T
    inplane = np.linalg.norm(Rcam[..., :2], axis=-1)
    depth = np.abs(Rcam[..., 2])
    return inplane, depth, R, gc


def main():
    data = load_vicon()
    preds = pickle.load(open(PROJ / "scratchpad" / "mixste_preds_G1-G6.pkl", "rb"))
    print(f"[data] {len(preds)} swings loaded from cache")

    rows = []
    agg = {"train": {"base_ip": [], "base_dp": [], "corr_ip": [], "corr_dp": []},
          "test":  {"base_ip": [], "base_dp": [], "corr_ip": [], "corr_dp": []}}
    depth_by_phase = {"train": {p: {"base": [], "corr": []} for p in PHASES},
                      "test":  {p: {"base": [], "corr": []} for p in PHASES}}

    for (s, sw), pred in sorted(preds.items()):
        gt, xy = data[s][sw]
        split = "train" if s in TRAIN_SUBJ else "test"

        k = fit_depth_scale(pred)
        pred_corr = apply_depth_anchor(pred, k)

        base_ip, base_dp, R_base, gc = eval_camera_depth(pred, gt, xy)
        corr_ip, corr_dp, R_corr, _ = eval_camera_depth(pred_corr, gt, xy)

        rows.append({
            "subj": s, "swing": sw, "split": split, "k": round(k, 3),
            "base_depth_mm": round(float(base_dp.mean() * 1000), 1),
            "corr_depth_mm": round(float(corr_dp.mean() * 1000), 1),
            "base_inplane_mm": round(float(base_ip.mean() * 1000), 1),
            "corr_inplane_mm": round(float(corr_ip.mean() * 1000), 1),
        })

        agg[split]["base_ip"].append(base_ip.reshape(-1))
        agg[split]["base_dp"].append(base_dp.reshape(-1))
        agg[split]["corr_ip"].append(corr_ip.reshape(-1))
        agg[split]["corr_dp"].append(corr_dp.reshape(-1))

        labels, _, _ = detect_phases(gc)
        for p in PHASES:
            sel = labels == p
            if sel.any():
                depth_by_phase[split][p]["base"].append(base_dp[sel].reshape(-1))
                depth_by_phase[split][p]["corr"].append(corr_dp[sel].reshape(-1))

    summary = {}
    for split in ("train", "test"):
        base_ip = np.concatenate(agg[split]["base_ip"]) * 1000
        base_dp = np.concatenate(agg[split]["base_dp"]) * 1000
        corr_ip = np.concatenate(agg[split]["corr_ip"]) * 1000
        corr_dp = np.concatenate(agg[split]["corr_dp"]) * 1000
        base_ip2, base_dp2 = (base_ip**2).sum(), (base_dp**2).sum()
        corr_ip2, corr_dp2 = (corr_ip**2).sum(), (corr_dp**2).sum()
        summary[split] = {
            "baseline_depth_mm": round(float(base_dp.mean()), 2),
            "corrected_depth_mm": round(float(corr_dp.mean()), 2),
            "depth_reduction_pct": round(100 * (1 - corr_dp.mean() / base_dp.mean()), 1),
            "baseline_inplane_mm": round(float(base_ip.mean()), 2),
            "corrected_inplane_mm": round(float(corr_ip.mean()), 2),
            "inplane_change_pct": round(100 * (1 - corr_ip.mean() / base_ip.mean()), 1),
            "baseline_depth_energy_share": round(float(base_dp2 / (base_ip2 + base_dp2)), 3),
            "corrected_depth_energy_share": round(float(corr_dp2 / (corr_ip2 + corr_dp2)), 3),
        }
        summary[split]["depth_by_phase_mm"] = {}
        for p in PHASES:
            b = depth_by_phase[split][p]["base"]
            c = depth_by_phase[split][p]["corr"]
            if b:
                summary[split]["depth_by_phase_mm"][p] = {
                    "base": round(float(np.concatenate(b).mean() * 1000), 1),
                    "corr": round(float(np.concatenate(c).mean() * 1000), 1),
                }

    k_vals = {"train": [r["k"] for r in rows if r["split"] == "train"],
             "test": [r["k"] for r in rows if r["split"] == "test"]}
    summary["k_stats"] = {
        split: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
               "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}
        for split, v in k_vals.items()
    }

    out = {"per_swing": rows, "summary": summary}
    json.dump(out, open(OUT / "depth_anchor_check.json", "w"), indent=2)

    print("\n" + "=" * 70)
    print("DEPTH-ANCHORING CHECK (per-swing bone-rigidity depth rescale)")
    print("=" * 70)
    for split in ("train", "test"):
        s = summary[split]
        print(f"\n[{split}]  n_swings={len(k_vals[split])}  "
              f"k: mean={summary['k_stats'][split]['mean']} "
              f"std={summary['k_stats'][split]['std']} "
              f"range=[{summary['k_stats'][split]['min']}, {summary['k_stats'][split]['max']}]")
        print(f"  depth   : {s['baseline_depth_mm']}mm -> {s['corrected_depth_mm']}mm "
              f"({s['depth_reduction_pct']:+.1f}%)")
        print(f"  in-plane: {s['baseline_inplane_mm']}mm -> {s['corrected_inplane_mm']}mm "
              f"({s['inplane_change_pct']:+.1f}%)  (sanity: should stay ~flat)")
        print(f"  depth_energy_share: {s['baseline_depth_energy_share']} -> "
              f"{s['corrected_depth_energy_share']}")
        print("  depth by phase (base -> corr, mm):")
        for p in PHASES:
            d = s["depth_by_phase_mm"].get(p)
            if d:
                print(f"    {p:<16} {d['base']:6.1f} -> {d['corr']:6.1f}")

    print("\nper-swing detail:")
    for r in rows:
        print(f"  {r['subj']} {r['swing']:<8} [{r['split']}]  k={r['k']:.3f}  "
              f"depth {r['base_depth_mm']:6.1f}->{r['corr_depth_mm']:6.1f}mm  "
              f"inplane {r['base_inplane_mm']:6.1f}->{r['corr_inplane_mm']:6.1f}mm")

    verdict = "GO" if summary["test"]["depth_reduction_pct"] > 5 and \
        abs(summary["test"]["inplane_change_pct"]) < 10 else "NO-GO"
    print(f"\n[+] wrote {OUT / 'depth_anchor_check.json'}")
    print(f"\nVERDICT (held-out test, >5% depth reduction + <10% in-plane damage = GO): {verdict}")
    return out


if __name__ == "__main__":
    main()
