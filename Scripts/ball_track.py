"""Ball tracking + physics-fit flight estimation from a single swing video.

Runs alongside the 2D pose stage (same video, separate pass) and produces the
`<stem>_ball_3d.json` artifact: the measured 2D ball track, launch direction,
and — when the track is long enough — a measured ball speed / carry with a
confidence band from a physics fit (ball_flight.simulate_flight projected
through a pinhole camera model).

Quality tiers (deploy/BALL_TRACKING_PLAN.md):
  measured  — >=15 subpixel track points: free fit; speed/launch/azimuth/carry
              are measured, reported with a bootstrap confidence band.
  partial   — 8..14 points: launch angle + azimuth are measured; ball speed
              comes from the club envelope (hand-speed nudged upstream), carry
              is simulated-with-measured-shape.
  simulated — no confident track: caller should fall back to the pure
              simulation path (estimate_for_club).

Monocular scale caveat: depth is anchored on the golfer's standing pixel
height (assumed 1.75 m unless --golfer-height-m). Azimuth is relative to the
camera's optical axis, not the target line (single camera, no aim reference).

Dependencies: numpy + cv2 only (both in the processing image); the optimizer
is a small built-in Nelder-Mead so the cloud image needs no scipy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from ball_flight import CLUB_DEFAULTS, simulate_flight, normalize_club

YD_M = 0.9144
DEFAULT_F_PX = 910.0        # ~26 mm-equiv phone lens on a 720x1280 portrait video
F_PX_JITTER = 0.08          # focal prior uncertainty folded into the bootstrap
MEASURED_MIN_PTS = 15
PARTIAL_MIN_PTS = 8


# --------------------------------------------------------------------------
# pose helpers
# --------------------------------------------------------------------------

def load_pose_2d(csv_path: str | Path) -> dict[int, list[tuple[float, float]]]:
    per_frame: dict[int, list[tuple[float, float]]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            per_frame.setdefault(int(row["frame"]), []).append(
                (float(row["x"]), float(row["y"])))
    return per_frame


def standing_height_px(pose: dict[int, list[tuple[float, float]]]) -> float:
    """95th-pct per-frame pixel span ~= standing height (impact frame is
    crouched and under-measures, which inflates the depth anchor)."""
    spans = []
    for pts in pose.values():
        ys = [p[1] for p in pts]
        spans.append(max(ys) - min(ys))
    return float(np.percentile(spans, 95))


def impact_from_pose(pose: dict[int, list[tuple[float, float]]]) -> int | None:
    """Fallback impact estimate: peak frame-to-frame mean landmark speed
    (the downswing-to-impact burst dominates every swing clip)."""
    frames = sorted(pose)
    if len(frames) < 10:
        return None
    speeds = []
    for a, b in zip(frames, frames[1:]):
        pa, pb = np.array(pose[a]), np.array(pose[b])
        n = min(len(pa), len(pb))
        speeds.append((b, float(np.mean(np.linalg.norm(pb[:n] - pa[:n], axis=1)))))
    return max(speeds, key=lambda s: s[1])[0]


# --------------------------------------------------------------------------
# 2D tracking (R5 discriminator stack + subpixel + predictive extension)
# --------------------------------------------------------------------------

def _aligned_grays(video: str | Path, lo: int, hi: int, ref_frame: int):
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    grays = {}
    for i in range(max(0, lo), min(total, hi)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            grays[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cap.release()
    ref = np.float32(grays[ref_frame])
    aligned = {}
    for i, g in grays.items():
        shift, _ = cv2.phaseCorrelate(ref, np.float32(g))
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        aligned[i] = cv2.warpAffine(g, M, (W, H))
    return aligned, W, H, fps


def track_ball(video: str | Path, impact: int,
               pose: dict[int, list[tuple[float, float]]],
               n_after: int = 90) -> dict:
    """Detect + link the ball track after impact. Returns dict with the
    subpixel track, launch origin, and diagnostics."""
    aligned, W, H, fps = _aligned_grays(video, impact - 8, impact + n_after, impact)
    frames_sorted = sorted(aligned)
    med = np.median(np.stack([aligned[i] for i in frames_sorted]), axis=0)
    med16 = med.astype(np.int16)

    def golfer_bbox(f: int):
        pts = pose.get(f) or pose[min(pose, key=lambda k: abs(k - f))]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs) - 70, min(ys) - 70, max(xs) + 70, max(ys) + 70)

    # per-frame small moving blobs: median-diff AND consecutive-diff
    cands: dict[int, list[tuple[float, float, int]]] = {}
    prev = None
    for i in frames_sorted:
        d = np.abs(aligned[i].astype(np.int16) - med16).astype(np.uint8)
        _, th = cv2.threshold(d, 16, 255, cv2.THRESH_BINARY)
        if prev is not None:
            dc = np.abs(aligned[i].astype(np.int16) - prev).astype(np.uint8)
            _, thc = cv2.threshold(dc, 12, 255, cv2.THRESH_BINARY)
            thc = cv2.dilate(thc, np.ones((5, 5), np.uint8))
            th = cv2.bitwise_and(th, thc)
        prev = aligned[i].astype(np.int16)
        x0, y0, x1, y1 = golfer_bbox(i)
        th[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = 0
        th[int(0.62 * H):, :] = 0
        th[:int(0.06 * H), :] = 0
        th[:, :12] = 0; th[:, -12:] = 0
        n, _, stats, cents = cv2.connectedComponentsWithStats(th)
        out = []
        for j in range(1, n):
            _, _, bw, bh, area = stats[j]
            if 2 <= area <= 80 and bw <= 16 and bh <= 16:
                out.append((float(cents[j][0]), float(cents[j][1]), int(area)))
        cands[i] = out

    kp = pose.get(impact, [])
    origin = ((float(np.mean([p[0] for p in kp])) if kp else W / 2),
              (float(max(p[1] for p in kp)) if kp else 0.75 * H))

    def chase(f0, x0, y0):
        track = [(f0, x0, y0)]
        vel = None
        fi, xi, yi = f0, x0, y0
        gaps = 0
        for fn in range(f0 + 1, frames_sorted[-1] + 1):
            if fn not in cands:
                continue
            px, py = (xi + vel[0] * (fn - fi), yi + vel[1] * (fn - fi)) if vel else (xi, yi)
            gate = 45 if vel is None else max(8, 0.7 * math.hypot(*vel) * (fn - fi) + 6)
            pick, bd = None, gate
            for (x, y, a) in cands[fn]:
                dd = math.hypot(x - px, y - py)
                if dd < bd:
                    nv = ((x - xi) / (fn - fi), (y - yi) / (fn - fi))
                    if vel is None and nv[1] > -2:
                        continue
                    pick, bd = (x, y, nv), dd
            if pick:
                if math.hypot(*pick[2]) < 1.5:          # ball never hovers early
                    gaps += 1
                    if gaps > 8:
                        break
                    continue
                xi, yi, fi = pick[0], pick[1], fn
                vel = pick[2]
                track.append((fn, xi, yi))
                gaps = 0
            else:
                gaps += 1
                if gaps > 8:
                    break
        return track

    def truncate_ballistic(track):
        if len(track) < 4:
            return track
        steps = [((b[1] - a[1]) / (b[0] - a[0]), (b[2] - a[2]) / (b[0] - a[0]))
                 for a, b in zip(track, track[1:])]
        vx0 = np.mean([s[0] for s in steps[:3]]); vy0 = np.mean([s[1] for s in steps[:3]])
        ax = 0 if abs(vx0) >= abs(vy0) else 1
        sgn = 1 if (vx0 if ax == 0 else vy0) >= 0 else -1
        stall = rev = 0
        for k, s in enumerate(steps):
            stall = stall + 1 if math.hypot(*s) < 1.2 else 0
            rev = rev + 1 if s[ax] * sgn < 0 else 0
            if stall >= 3 or rev >= 3:
                return track[:k + 2 - max(stall, rev)]
        return track

    best, best_score = None, 1e18
    for f0 in range(impact + 1, impact + 10):
        for (x0, y0, a0) in cands.get(f0, []):
            track = truncate_ballistic(chase(f0, x0, y0))
            if len(track) < PARTIAL_MIN_PTS:
                continue
            head = track[:8]
            sp = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0])
                  for a, b in zip(head, head[1:])]
            if sp[0] < 4 or sp[0] * 1.15 < sp[-1]:
                continue
            xs = np.array([p[1] for p in head]); ys = np.array([p[2] for p in head])
            if xs.ptp() >= ys.ptp():
                A = np.vstack([xs, np.ones_like(xs)]).T
                m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
                rms = float(np.sqrt(np.mean((ys - (m * xs + c)) ** 2)))
            else:
                A = np.vstack([ys, np.ones_like(ys)]).T
                m, c = np.linalg.lstsq(A, xs, rcond=None)[0]
                rms = float(np.sqrt(np.mean((xs - (m * ys + c)) ** 2)))
            if rms > 5.0:
                continue
            ovx, ovy = x0 - origin[0], y0 - origin[1]
            tvx = np.mean([b[1] - a[1] for a, b in zip(head[:4], head[1:4])])
            tvy = np.mean([b[2] - a[2] for a, b in zip(head[:4], head[1:4])])
            dot = (ovx * tvx + ovy * tvy) / (math.hypot(ovx, ovy) * math.hypot(tvx, tvy) + 1e-9)
            if ovy > 0 or dot < math.cos(math.radians(40)):
                continue
            # anti-causality: our ball cannot exist before impact
            vx0_, vy0_ = ((head[1][1] - head[0][1]) / (head[1][0] - head[0][0]),
                          (head[1][2] - head[0][2]) / (head[1][0] - head[0][0]))
            pre_hits = 0
            for fb in range(impact - 6, impact):
                if fb not in cands:
                    continue
                k = track[0][0] - fb
                bxp, byp = x0 - vx0_ * k, y0 - vy0_ * k
                if math.hypot(bxp - origin[0], byp - origin[1]) < 220:
                    continue                     # pre-impact club, not a neighbor ball
                if any(math.hypot(x - bxp, y - byp) < 25 for x, y, a in cands[fb]):
                    pre_hits += 1
            if pre_hits >= 2:
                continue
            score = 10 * rms - 5 * min(len(track), 18) - 2 * sp[0] - 40 * dot
            if score < best_score:
                best, best_score = track, score

    if best is None:
        return {"track": [], "origin": origin, "fps": fps, "W": W, "H": H}

    # predictive extension: chase past the end with a decaying threshold,
    # searching the median-diff image around the predicted position. Guarded by
    # ballistic consistency — the receding ball only ever slows and stays on
    # course, so a speed-up or direction jump means we've latched onto clutter.
    ext = list(best)
    med32 = med.astype(np.float32)
    while len(ext) >= 3 and len(ext) < len(best) + 25:
        (fa, xa, ya), (fb, xb, yb) = ext[-3], ext[-1]
        vel = ((xb - xa) / (fb - fa), (yb - ya) / (fb - fa))
        fn = ext[-1][0] + 1
        if fn not in aligned or fn > frames_sorted[-1]:
            break
        px, py = xb + vel[0], yb + vel[1]
        if not (14 < px < W - 14 and 0.06 * H < py < 0.62 * H):
            break
        d = np.abs(aligned[fn].astype(np.float32) - med32)
        r = 6
        x0i, y0i = int(px) - r, int(py) - r
        patch = d[max(0, y0i):y0i + 2 * r + 1, max(0, x0i):x0i + 2 * r + 1]
        if patch.size == 0 or patch.max() < 8:      # threshold decay floor
            break
        yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
        nx, ny = float(max(0, x0i) + xx), float(max(0, y0i) + yy)
        step = math.hypot(nx - xb, ny - yb)
        pv = math.hypot(*vel)
        if step > max(3.0, 1.5 * pv):               # ball never speeds back up
            break
        if pv > 1.0 and step > 1.0:
            cosang = ((nx - xb) * vel[0] + (ny - yb) * vel[1]) / (step * pv + 1e-9)
            if cosang < math.cos(math.radians(35)):
                break
        ext.append((fn, nx, ny))

    # subpixel refinement: intensity-weighted centroid on the median-diff patch
    refined = []
    for f, x, y in ext:
        d = np.abs(aligned[f].astype(np.float32) - med32)
        x0i, y0i = int(round(x)) - 7, int(round(y)) - 7
        p = d[max(0, y0i):y0i + 15, max(0, x0i):x0i + 15].copy()
        if p.size == 0 or p.max() < 1e-6:
            refined.append((f, x, y))
            continue
        p[p < p.max() * 0.3] = 0
        ys_, xs_ = np.mgrid[0:p.shape[0], 0:p.shape[1]]
        s = p.sum()
        refined.append((f, float((p * xs_).sum() / s + max(0, x0i)),
                        float((p * ys_).sum() / s + max(0, y0i))))

    return {"track": refined, "origin": origin, "fps": fps, "W": W, "H": H}


# --------------------------------------------------------------------------
# physics fit (R6/R7)
# --------------------------------------------------------------------------

def _nelder_mead(loss, x0, steps, max_iter=300, ftol=0.02):
    """Minimal Nelder-Mead (no scipy in the processing image)."""
    n = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += steps[i]
        simplex.append(p)
    fv = [loss(p) for p in simplex]
    for _ in range(max_iter):
        order = np.argsort(fv)
        simplex = [simplex[i] for i in order]
        fv = [fv[i] for i in order]
        if abs(fv[-1] - fv[0]) < ftol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = loss(xr)
        if fr < fv[0]:
            xe = centroid + 2 * (centroid - simplex[-1])
            fe = loss(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = loss(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fv[i] = loss(simplex[i])
    k = int(np.argmin(fv))
    return list(simplex[k]), float(fv[k])


class FlightFitter:
    def __init__(self, track, impact, fps, origin_px, pose, W, H,
                 f_px=DEFAULT_F_PX, golfer_m=1.75, horizon_y=None,
                 backspin_rpm=3275.0):
        self.cx, self.cy = W / 2, H / 2
        self.f_px = f_px
        self.spin = backspin_rpm
        self.horizon_y = horizon_y if horizon_y is not None else self.cy
        h_px = standing_height_px(pose)
        self.Zg = f_px * golfer_m / h_px
        self.T = np.array([(origin_px[0] - self.cx) * self.Zg / f_px,
                           (origin_px[1] - self.cy) * self.Zg / f_px, self.Zg])
        self.times = np.array([(f - impact) / fps for f, _, _ in track])
        self.obs = np.array([(x, y) for _, x, y in track])

    def _cam_pts(self, speed, launch, azim, f_px=None, n_traj=240):
        """Simulated flight as camera-frame points (meters, absolute - tee at
        self.T). Returns (pts, tt, sim_result)."""
        f_px = f_px or self.f_px
        pitch = math.atan((self.cy - self.horizon_y) / f_px)
        r = simulate_flight(speed, launch, self.spin, 0.0, 0.0, trajectory_points=n_traj)
        traj = np.array(r["trajectory"])
        tt = np.linspace(0, r["flight_time_s"], len(traj))
        P = traj * YD_M                              # [downrange, height, side] m
        a = math.radians(azim)
        Xc = P[:, 0] * math.sin(a) + P[:, 2] * math.cos(a)
        Zc = P[:, 0] * math.cos(a) - P[:, 2] * math.sin(a)
        Yc = -P[:, 1]
        cp, sp = math.cos(pitch), math.sin(pitch)
        Zc2 = Zc * cp - Yc * sp
        Yc2 = Zc * sp + Yc * cp
        return np.stack([Xc, Yc2, Zc2], axis=1) + self.T, tt, r

    def _project(self, speed, launch, azim, dt=0.0, f_px=None):
        f_px = f_px or self.f_px
        pts, tt, r = self._cam_pts(speed, launch, azim, f_px=f_px)
        out = []
        for t in self.times - dt:
            t = max(t, 0.0)
            k = int(np.searchsorted(tt, t))
            k = min(max(k, 1), len(tt) - 1)
            w = (t - tt[k - 1]) / (tt[k] - tt[k - 1] + 1e-9)
            p = pts[k - 1] * (1 - w) + pts[k] * w
            out.append((self.cx + f_px * p[0] / p[2], self.cy + f_px * p[1] / p[2])
                       if p[2] > 0.3 else (1e6, 1e6))
        return np.array(out), r

    def _loss(self, v, obs=None, f_px=None):
        dt = v[3] if len(v) > 3 else 0.0
        if not (40 <= v[0] <= 210 and 2 <= v[1] <= 45 and -60 <= v[2] <= 60
                and -0.25 <= dt <= 0.25):
            return 1e9
        pr, _ = self._project(v[0], v[1], v[2], dt, f_px=f_px)
        o = self.obs if obs is None else obs
        errs = np.sort(np.linalg.norm(pr - o, axis=1))
        keep = max(6, int(len(errs) * 0.85))        # trimmed mean: tolerate tail junk
        return float(np.mean(errs[:keep]))

    def fit_free(self):
        best, bl = None, 1e18
        for s in (90, 110, 130, 150, 170):
            for la in (8, 14, 20, 26):
                for az in (-15, 0, 10, 22):
                    l = self._loss((s, la, az))
                    if l < bl:
                        best, bl = [s, la, az], l
        best, bl = _nelder_mead(lambda v: self._loss(v), best + [0.0], [8, 2, 3, 0.05])
        return best, bl

    def fit_constrained(self, speed):
        best, bl = None, 1e18
        for la in (8, 12, 16, 20, 25):
            for az in (-20, -5, 5, 15, 25):
                l = self._loss((speed, la, az))
                if l < bl:
                    best, bl = [la, az], l
        v, bl = _nelder_mead(lambda v: self._loss((speed, v[0], v[1], v[2])),
                             best + [0.0], [2, 3, 0.05])
        return [speed] + v, bl

    def bootstrap(self, base, n=10, seed=7):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            idx = sorted(rng.choice(len(self.obs), size=len(self.obs), replace=True))
            fj = self.f_px * (1 + rng.uniform(-F_PX_JITTER, F_PX_JITTER))
            obs = self.obs[idx]
            times = self.times[idx]
            saved_t, saved_o = self.times, self.obs
            self.times, self.obs = times, obs
            v, _ = _nelder_mead(lambda v: self._loss(v, f_px=fj), base,
                                [6, 1.5, 2, 0.04][:len(base)], max_iter=200)
            _, r = self._project(v[0], v[1], v[2], v[3] if len(v) > 3 else 0.0,
                                 f_px=fj)
            self.times, self.obs = saved_t, saved_o
            rows.append((v[0], v[1], v[2], r["carry_yd"], r["apex_yd"],
                         r["flight_time_s"]))
        B = np.array(rows)
        pct = lambda c: [round(float(np.percentile(B[:, c], 10)), 1),
                         round(float(np.percentile(B[:, c], 90)), 1)]
        return {"speed_mph": pct(0), "launch_deg": pct(1), "azimuth_deg": pct(2),
                "carry_yd": pct(3), "apex_yd": pct(4), "flight_time_s": pct(5)}


# --------------------------------------------------------------------------
# horizon estimate (for camera pitch)
# --------------------------------------------------------------------------

def estimate_horizon_y(video: str | Path, frame: int) -> float | None:
    """Sky/ground boundary via per-row brightness gradient in the top 2/3 of a
    frame — coarse but only feeds a small pitch correction."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H = g.shape[0]
    rows = g[: int(H * 0.66)].mean(axis=1)
    rows = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 9), 0).ravel()
    grad = np.diff(rows)
    k = int(np.argmin(grad))                        # sharpest bright->dark drop
    return float(k) if 0.2 * H < k < 0.62 * H else None


# --------------------------------------------------------------------------
# top-level analysis
# --------------------------------------------------------------------------

def analyze(video: str | Path, landmarks_csv: str | Path,
            scorecard_json: str | Path | None = None,
            club: str | None = None, speed_scale: float = 1.0,
            golfer_m: float = 1.75) -> dict:
    pose = load_pose_2d(landmarks_csv)
    impact = None
    if scorecard_json and Path(scorecard_json).exists():
        sc = json.loads(Path(scorecard_json).read_text())
        impact = (sc.get("events") or {}).get("impact")
        club = club or (sc.get("meta") or {}).get("club")
    if impact is None:
        impact = impact_from_pose(pose)
    if impact is None:
        return {"quality": "simulated", "reason": "no impact frame"}

    t = track_ball(video, int(impact), pose)
    track = t["track"]
    n = len(track)
    club_key = normalize_club(club) or "driver"
    env_speed = CLUB_DEFAULTS[club_key]["amateur"][0] * max(0.88, min(1.12, speed_scale))
    spin = CLUB_DEFAULTS[club_key]["amateur"][2]

    result: dict = {
        "engine": "balltrack-v1",
        "impact_frame": int(impact),
        "fps": t["fps"],
        "track_2d": [[f, round(x, 2), round(y, 2)] for f, x, y in track],
        "n_track_points": n,
        "azimuth_note": "azimuth is relative to the camera axis (single camera, no aim line)",
    }
    if n < PARTIAL_MIN_PTS:
        result["quality"] = "simulated"
        result["reason"] = f"track too short ({n} pts)"
        return result

    horizon = estimate_horizon_y(video, int(impact))
    fitter = FlightFitter(track, int(impact), t["fps"], t["origin"], pose,
                          t["W"], t["H"], horizon_y=horizon, golfer_m=golfer_m,
                          backspin_rpm=spin)

    if n >= MEASURED_MIN_PTS:
        v, resid = fitter.fit_free()
        _, r = fitter._project(v[0], v[1], v[2], v[3])
        ci = fitter.bootstrap(v)
        result.update({
            "quality": "measured",
            "fit": {"ball_speed_mph": round(v[0], 1), "launch_deg": round(v[1], 1),
                    "azimuth_deg": round(v[2], 1), "launch_dt_s": round(v[3], 3),
                    "residual_px": round(resid, 1)},
            "flight": {"carry_yd": round(r["carry_yd"]), "apex_yd": round(r["apex_yd"]),
                       "side_yd": round(r["side_yd"]),
                       "flight_time_s": round(r["flight_time_s"], 1)},
            "ci_10_90": ci,
        })
    else:
        v, resid = fitter.fit_constrained(env_speed)
        _, r = fitter._project(v[0], v[1], v[2], v[3])
        result.update({
            "quality": "partial",
            "fit": {"ball_speed_mph": round(env_speed, 1), "speed_source": "club_envelope",
                    "launch_deg": round(v[1], 1), "azimuth_deg": round(v[2], 1),
                    "launch_dt_s": round(v[3], 3), "residual_px": round(resid, 1)},
            "flight": {"carry_yd": round(r["carry_yd"]), "apex_yd": round(r["apex_yd"]),
                       "side_yd": round(r["side_yd"]),
                       "flight_time_s": round(r["flight_time_s"], 1)},
        })
    # UI arc + replay-frame trajectory come from the fitted sim
    rr = simulate_flight(result["fit"]["ball_speed_mph"], result["fit"]["launch_deg"],
                         spin, 0.0, 0.0, trajectory_points=48)
    result["trajectory_world"] = [[round(a, 1), round(b, 1), round(c, 1)]
                                  for a, b, c in rr["trajectory"]]
    cam, tt, _ = fitter._cam_pts(result["fit"]["ball_speed_mph"],
                                 result["fit"]["launch_deg"],
                                 result["fit"]["azimuth_deg"], n_traj=48)
    rel = cam - fitter.T                             # meters relative to the tee
    result["trajectory_cam_m"] = [[round(float(t), 3)] + [round(float(v), 3) for v in p]
                                  for t, p in zip(tt, rel)]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--landmarks", required=True, help="2D landmarks CSV")
    ap.add_argument("--scorecard", help="scorecard JSON (events.impact, meta.club)")
    ap.add_argument("--club")
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--golfer-height-m", type=float, default=1.75)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyze(a.video, a.landmarks, a.scorecard, a.club, a.speed_scale,
                  a.golfer_height_m)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[ball_track] quality={res.get('quality')} n={res.get('n_track_points', 0)} "
          f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
