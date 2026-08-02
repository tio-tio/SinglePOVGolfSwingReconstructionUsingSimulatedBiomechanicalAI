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
import copy
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from ball_flight import CLUB_DEFAULTS, simulate_flight, normalize_club
from video_orientation import open_capture

YD_M = 0.9144
REF_SHORT_SIDE = 720.0      # DEFAULT_F_PX and every _PX gate below were tuned here
DEFAULT_F_PX = 910.0        # ~26 mm-equiv phone lens on a 720x1280 portrait video
F_PX_JITTER = 0.08          # focal prior uncertainty folded into the bootstrap
MEASURED_MIN_PTS = 15
PARTIAL_MIN_PTS = 8
PITCH_CLAMP_RAD = math.radians(25.0)
SPEED_GRID = tuple(range(60, 211, 10))      # mph, the profiled scale direction
SPEED_PRIOR_SIGMA = 0.15                    # 1 sigma on log(speed / club envelope)
SPEED_PRIOR_WEIGHT_PX = 1.5                 # pixel-equivalent cost of a 1-sigma miss
BAND_DELTA_PX = 1.0                         # profile width that defines the band
BAND_MEASURED_FRAC = 0.28                   # carry band this wide => speed is measured
ANCHOR_JITTER = 0.10                        # depth-anchor (golfer height / focal) prior
# Real ball tracks on the validation clips reproject at 2.5-3.2 px; the best
# chain the negative clip can offer sits at 5.6. 4.5 separates them with room.
MAX_FIT_RESIDUAL_PX = 4.5
MAX_CANDIDATES = 10                         # tracks re-ranked by ballistic fit
PREFIX_HEAD_PTS = 10                        # vetted head the flight is anchored on
PREFIX_GROW_TOL_PX = 7.0                    # how far a later point may stray
MAX_LAUNCH_DT_S = 0.12                      # flight must start at impact
LENGTH_RESID_TRADE = 1.5                    # track points worth one px of residual


def default_f_px(W: int, H: int) -> float:
    """Focal length in pixels for a frame this size.

    Focal length in PIXELS is a property of the lens AND the sampling grid, so
    the 26 mm-equiv phone lens that images at 910 px on 720x1280 images at
    1365 px on 1080x1920. Anchoring on the SHORT side keeps that true for
    landscape clips as well as portrait.
    """
    return DEFAULT_F_PX * min(W, H) / REF_SHORT_SIDE


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

def _aligned_grays(video: str | Path, lo: int, hi: int, ref_frame: int,
                   clahe: bool = True):
    """Decode [lo, hi) as grayscale, stabilise onto ref_frame.

    Preprocessing that matters for a 3-6 px backlit ball (L3):
      * sequential decode — per-frame POS_FRAMES seeking is slow and, on some
        phone H.264/HEVC files, lands on the wrong frame;
      * CLAHE — the ball is a faint dot against a dark range at these venues,
        and local contrast equalisation lifts it well clear of the diff floor;
      * Hanning-windowed phase correlation — without a window the FFT sees the
        frame edges as a step discontinuity, and the resulting shift error
        smears every high-contrast structure (netting, rails) into the
        median-diff as ball-sized clutter.
    """
    cap = open_capture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lo, hi = max(0, lo), min(total, hi)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
    grays = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for i in range(lo, hi):
        ok, f = cap.read()                      # sequential: no per-frame seek
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays[i] = eq.apply(g) if eq is not None else g
    cap.release()
    if ref_frame not in grays:
        ref_frame = min(grays, key=lambda k: abs(k - ref_frame))
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    ref = np.float32(grays[ref_frame])
    aligned = {}
    for i, g in grays.items():
        shift, _ = cv2.phaseCorrelate(ref, np.float32(g), win)
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        aligned[i] = cv2.warpAffine(g, M, (W, H))
    return aligned, W, H, fps


def _cells(tr, q=8):
    return {(f, int(x / q), int(y / q)) for f, x, y in tr}


def _same_chain(a: set, b: set) -> bool:
    """Two candidate tracks are the same physical chain if their points largely
    coincide (keying on the start pixel alone lets one chain fill the whole
    shortlist and pushes the real ball off it)."""
    return len(a & b) >= 0.5 * min(len(a), len(b))


def track_ball(video: str | Path, impact: int,
               pose: dict[int, list[tuple[float, float]]],
               n_after: int = 90, clahe: bool = True) -> dict:
    """Detect + link the ball track after impact. Returns dict with the
    subpixel track, launch origin, and diagnostics."""
    aligned, W, H, fps = _aligned_grays(video, impact - 8, impact + n_after, impact,
                                        clahe=clahe)
    frames_sorted = sorted(aligned)
    med = np.median(np.stack([aligned[i] for i in frames_sorted]), axis=0)
    med16 = med.astype(np.int16)

    def golfer_bbox(f: int):
        pts = pose.get(f) or pose[min(pose, key=lambda k: abs(k - f))]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs) - 70, min(ys) - 70, max(xs) + 70, max(ys) + 70)

    # per-frame small moving blobs: median-diff AND consecutive-diff
    masks: dict[int, np.ndarray] = {}
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
        masks[i] = th

    # Blob gate stays tight (R5). Relaxing it to admit motion-streaks and adding
    # a static-clutter occupancy mask were both measured on the validation clips
    # (L3 matrix) and neither helped: the extra candidates were clubhead and
    # netting chains, and on the negative clip they manufactured fits. The win
    # came from the preprocessing above plus ballistic re-ranking below.
    cands: dict[int, list[tuple[float, float, int]]] = {}
    for i, th in masks.items():
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

    scored: list[tuple[float, list]] = []
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
            scored.append((score, track))

    if not scored:
        return {"track": [], "candidates": [], "origin": origin, "fps": fps,
                "W": W, "H": H}

    # Shortlist distinct candidates (same start frame + start pixel = same
    # chain). The 2D score cannot reliably separate the ball from the clubhead
    # leaving the same origin at the same moment, so the caller re-ranks these
    # by how well each actually fits a ball flight.
    scored.sort(key=lambda s: s[0])
    shortlist: list[list] = []
    picked: list[set] = []
    for _, track in scored:
        c = _cells(track)
        if any(_same_chain(c, p) for p in picked):
            continue
        picked.append(c)
        shortlist.append(track)
        if len(shortlist) >= MAX_CANDIDATES:
            break

    med32 = med.astype(np.float32)

    def extend(best):
        """predictive extension: chase past the end with a decaying threshold,
        searching the median-diff image around the predicted position. Guarded by
        ballistic consistency — the receding ball only ever slows and stays on
        course, so a speed-up or direction jump means we've latched onto clutter."""
        ext = list(best)
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
        return ext

    def refine(ext):
        """subpixel: intensity-weighted centroid on the median-diff patch"""
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
        return refined

    candidates = [refine(extend(t)) for t in shortlist]
    return {"track": candidates[0], "candidates": candidates, "origin": origin,
            "fps": fps, "W": W, "H": H}


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
                 f_px=None, golfer_m=1.75, horizon_y=None,
                 backspin_rpm=3275.0, speed_prior_mph=None,
                 speed_prior_sigma=SPEED_PRIOR_SIGMA):
        self.cx, self.cy = W / 2, H / 2
        f_px = self.f_px = default_f_px(W, H) if f_px is None else f_px
        # Reprojection error is measured in 720p-EQUIVALENT pixels. The same
        # angular misfit costs proportionally more raw pixels on a bigger frame,
        # and every _PX constant above was tuned on 720x1280 clips, so scaling
        # the error once here keeps all of them meaningful at any resolution.
        self.px_scale = min(W, H) / REF_SHORT_SIDE
        self.spin = backspin_rpm
        # Soft prior keeping ball speed near the club envelope. A ~0.6 s track
        # constrains the trajectory's SHAPE (launch, azimuth) tightly but its
        # SCALE barely at all — unpriored, the fit walks to the top of the speed
        # grid and reports tour-pro carries for every swing (L2).
        self.speed_prior = speed_prior_mph
        self.speed_prior_sigma = speed_prior_sigma
        self.horizon_y = horizon_y if horizon_y is not None else self.cy
        # Camera pitch from the horizon row. With the camera pitched UP by t, a
        # world point at camera height and infinite range projects to
        # cy + f*tan(t), so t = atan((horizon_y - cy)/f). (Getting this sign
        # backwards biases the fitted launch angle by +2t — see L1 in
        # deploy/BALL_TRACKING_PLAN.md.) Clamped: the horizon estimate is a
        # coarse brightness-gradient guess and a wild value must not swing the fit.
        self.pitch = max(-PITCH_CLAMP_RAD, min(PITCH_CLAMP_RAD,
                                               math.atan((self.horizon_y - self.cy) / f_px)))
        h_px = standing_height_px(pose)
        self.Zg = f_px * golfer_m / h_px
        self.T = np.array([(origin_px[0] - self.cx) * self.Zg / f_px,
                           (origin_px[1] - self.cy) * self.Zg / f_px, self.Zg])
        self.times = np.array([(f - impact) / fps for f, _, _ in track])
        self.obs = np.array([(x, y) for _, x, y in track])
        self.t_max = float(self.times.max()) if len(self.times) else 1.0

    def _cam_pts(self, speed, launch, azim, f_px=None, n_traj=240,
                 max_time_s=15.0):
        """Simulated flight as camera-frame points (meters, absolute - tee at
        self.T). Returns (pts, tt, sim_result)."""
        f_px = f_px or self.f_px
        pitch = self.pitch
        r = simulate_flight(speed, launch, self.spin, 0.0, 0.0,
                            max_time_s=max_time_s, trajectory_points=n_traj)
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
        self._last_level = np.stack([Xc, Yc, Zc], axis=1)   # tee-relative, level
        return np.stack([Xc, Yc2, Zc2], axis=1) + self.T, tt, r

    def level_pts(self, speed, launch, azim, n_traj=48):
        """Flight relative to the tee in a LEVEL frame (x right, y DOWN, z
        downrange), i.e. before the camera-pitch rotation.

        The 3D replay is a leveled frame, so handing it camera-frame points made
        the arc inherit the camera's tilt — on a 6.8 deg downward tilt the ball
        "landed" 21 m below the tee."""
        _, tt, _ = self._cam_pts(speed, launch, azim, n_traj=n_traj)
        return self._last_level, tt

    def _project(self, speed, launch, azim, dt=0.0, f_px=None, fast=False):
        """fast=True simulates only as far as the observed track reaches — the
        loss never looks past it, and the full 6-9 s flight is ~10x the work."""
        f_px = f_px or self.f_px
        if fast:
            pts, tt, r = self._cam_pts(speed, launch, azim, f_px=f_px, n_traj=64,
                                       max_time_s=self.t_max + 0.4)
        else:
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

    def pixel_loss(self, v, obs=None, f_px=None):
        """Trimmed mean reprojection error in 720p-equivalent pixels (no prior)."""
        dt = v[3] if len(v) > 3 else 0.0
        if not (40 <= v[0] <= 210 and 2 <= v[1] <= 45 and -60 <= v[2] <= 60
                and -0.25 <= dt <= 0.25):
            return 1e9
        pr, _ = self._project(v[0], v[1], v[2], dt, f_px=f_px, fast=True)
        o = self.obs if obs is None else obs
        errs = np.sort(np.linalg.norm(pr - o, axis=1))
        keep = max(6, int(len(errs) * 0.85))        # trimmed mean: tolerate tail junk
        return float(np.mean(errs[:keep])) / self.px_scale

    def _speed_penalty(self, speed):
        """Prior cost in pixel-equivalent units (0 when no prior is set)."""
        if not self.speed_prior:
            return 0.0
        z = math.log(max(speed, 1e-3) / self.speed_prior) / self.speed_prior_sigma
        return SPEED_PRIOR_WEIGHT_PX * z * z

    def _loss(self, v, obs=None, f_px=None):
        pl = self.pixel_loss(v, obs=obs, f_px=f_px)
        return pl if pl >= 1e9 else pl + self._speed_penalty(v[0])

    def fit_at_speed(self, speed, obs=None, f_px=None, with_prior=True):
        """Best (launch, azimuth, dt) with ball speed held fixed.

        Multi-start: the (launch, azimuth) surface has local minima, and a
        single start leaves spikes in the speed profile that corrupt both the
        MAP estimate and the confidence band."""
        loss = self._loss if with_prior else self.pixel_loss
        grid = []
        for la in (5, 9, 13, 17, 21, 26, 32):
            for az in (-25, -12, 0, 12, 25):
                grid.append((loss((speed, la, az), obs=obs, f_px=f_px), la, az))
        grid.sort(key=lambda g: g[0])
        best_v, best_l = None, 1e18
        for _, la, az in grid[:3]:
            v, l = _nelder_mead(
                lambda p: loss((speed, p[0], p[1], p[2]), obs=obs, f_px=f_px),
                [la, az, 0.0], [2, 3, 0.05])
            if l < best_l:
                best_v, best_l = v, l
        return best_v, best_l

    def profile_speed(self, speeds=SPEED_GRID, obs=None, f_px=None,
                      with_prior=True):
        """Profile likelihood along ball speed: [(speed, loss, [launch, az, dt])].

        Speed is the badly-conditioned direction of this fit (a ~0.6 s track
        pins the trajectory's shape far better than its scale), so it gets swept
        explicitly rather than left to the optimizer — and the width of this
        profile is the honest uncertainty (see calibrate_band)."""
        out = []
        for s in speeds:
            v, l = self.fit_at_speed(s, obs=obs, f_px=f_px, with_prior=with_prior)
            out.append((float(s), float(l), list(v)))
        return out

    def fit_free(self):
        """Free 3-param fit via the speed profile + a joint polish.

        A single joint Nelder-Mead from one coarse start lands in local minima
        here (L2: it returned 119 mph on noise-free 150 mph synthetic data)."""
        s_best, l_best, v_best = min(self.profile_speed(), key=lambda r: r[1])
        for s in (s_best - 6, s_best - 3, s_best + 3, s_best + 6):
            if not (SPEED_GRID[0] <= s <= SPEED_GRID[-1]):
                continue
            v, l = self.fit_at_speed(s)
            if l < l_best:
                s_best, l_best, v_best = s, l, v
        v, l = _nelder_mead(lambda p: self._loss(p), [s_best] + list(v_best),
                            [5, 1.5, 2, 0.04])
        if l <= l_best:
            return list(v), float(l)
        return [float(s_best)] + list(v_best), float(l_best)

    def fit_constrained(self, speed):
        v, bl = self.fit_at_speed(speed)
        return [speed] + list(v), bl

    def _with_anchor(self, k):
        """Shallow clone with the depth anchor scaled by k (golfer-height and
        focal-prior uncertainty both act on the anchor)."""
        o = copy.copy(self)
        o.Zg = self.Zg * k
        o.T = self.T * k
        return o

    def confidence_band(self, delta_px=BAND_DELTA_PX, with_prior=True):
        """Every solution within delta_px of the best, over the speed profile
        AND the depth-anchor prior. This is the flat direction the old bootstrap
        never sampled: it resampled track points and jittered focal length while
        re-optimising from the same speed, so it reported +-5 yd on a carry that
        the pixels only pin to about +-60."""
        rows = []
        for k in (1 - ANCHOR_JITTER, 1.0, 1 + ANCHOR_JITTER):
            fit = self if k == 1.0 else self._with_anchor(k)
            for s, l, v in fit.profile_speed(with_prior=with_prior):
                rows.append((l, s, v, fit))
        lmin = min(r[0] for r in rows)
        keep = [r for r in rows if r[0] <= lmin + delta_px]
        cols: dict[str, list[float]] = {k: [] for k in
                                        ("speed_mph", "launch_deg", "azimuth_deg",
                                         "carry_yd", "apex_yd", "flight_time_s")}
        for _, s, v, fit in keep:
            _, r = fit._project(s, v[0], v[1], v[2])
            cols["speed_mph"].append(s)
            cols["launch_deg"].append(v[0])
            cols["azimuth_deg"].append(v[1])
            cols["carry_yd"].append(r["carry_yd"])
            cols["apex_yd"].append(r["apex_yd"])
            cols["flight_time_s"].append(r["flight_time_s"])
        return {k: [round(min(vals), 1), round(max(vals), 1)]
                for k, vals in cols.items()}, len(keep)


# --------------------------------------------------------------------------
# horizon estimate (for camera pitch)
# --------------------------------------------------------------------------

def estimate_horizon_y(video: str | Path, frame: int) -> float | None:
    """Sky/ground boundary via per-row brightness gradient in the top 2/3 of a
    frame — coarse but only feeds a small pitch correction.

    open_capture matters here specifically: on a sideways phone frame the
    horizon runs vertically and a row scan finds nothing meaningful."""
    cap = open_capture(video)
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
            golfer_m: float = 1.75, debug: bool = False) -> dict:
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

    # Two preprocessing passes. CLAHE rescues a ball lost in bright sky/pole
    # clutter (IMG_8107: 8 -> 14 points) but floods a dark night bay with
    # competing blobs and loses a ball that was already high-contrast
    # (IMG_3434). Neither setting wins everywhere, so run both and let the
    # ballistic ranking below choose — it is a far better judge than any global
    # preprocessing flag.
    t = track_ball(video, int(impact), pose, clahe=False)
    t_eq = track_ball(video, int(impact), pose, clahe=True)
    pool: list[list] = []
    picked: list[set] = []
    for cand in (t.get("candidates") or []) + (t_eq.get("candidates") or []):
        c = _cells(cand)
        if any(_same_chain(c, p) for p in picked):
            continue
        picked.append(c)
        pool.append(cand)

    club_key = normalize_club(club) or "driver"
    env_speed = CLUB_DEFAULTS[club_key]["amateur"][0] * max(0.88, min(1.12, speed_scale))
    spin = CLUB_DEFAULTS[club_key]["amateur"][2]
    horizon = estimate_horizon_y(video, int(impact))

    def make_fitter(tr):
        return FlightFitter(tr, int(impact), t["fps"], t["origin"], pose,
                            t["W"], t["H"], horizon_y=horizon, golfer_m=golfer_m,
                            backspin_rpm=spin, speed_prior_mph=env_speed)

    def trim_tail(cand):
        """Keep the longest PREFIX of the chain that is still one ball flight.

        The head of a chain is the trustworthy part — it has to clear the launch
        cone, speed and anti-causality filters at initiation. The tail is
        speculative: the predictive extension keeps chasing after the ball has
        faded and picks up clutter, which then drags the whole fit (IMG_3434's
        real f290-305 ball ran on to f330 and its residual went 2.5 -> 12 px,
        past the rejection gate). So: fit the head, then walk forward for as long
        as the observations keep agreeing with that flight."""
        if len(cand) <= PREFIX_HEAD_PTS:
            return cand
        cur = list(cand)
        for _ in range(2):
            head = cur[:PREFIX_HEAD_PTS]
            fh = make_fitter(head)
            vh, _ = fh.fit_constrained(env_speed)
            if fh.pixel_loss(vh) > MAX_FIT_RESIDUAL_PX:
                return cur                   # head is not a flight; let it be rejected
            fa = make_fitter(cur)
            pr, _ = fa._project(vh[0], vh[1], vh[2], vh[3])
            errs = np.linalg.norm(pr - fa.obs, axis=1) / fa.px_scale
            k = PREFIX_HEAD_PTS
            while k < len(cur) and errs[k] <= PREFIX_GROW_TOL_PX:
                k += 1
            if k == len(cur):
                break
            cur = cur[:k]
        return cur

    # Re-rank the 2D shortlist by ballistic fit: the clubhead leaves the same
    # origin at the same instant as the ball and looks just as track-like in 2D,
    # but it cannot be reprojected as a flight (L3/L4).
    # Score each chain at full length AND trimmed. Trimming rescues a real track
    # that the predictive extension ran past (IMG_3434), but a head-anchored fit
    # is short-baseline and mispredicts the far end, so it would also throw away
    # good late points on a clean track (IMG_8107 fits all 15 of its points at
    # 2.0 px). Offering both and letting the gate + length ranking choose avoids
    # having to get that call right up front.
    ranked = []
    for cand in pool:
        if len(cand) < PARTIAL_MIN_PTS:
            continue
        variants = [cand]
        trimmed = trim_tail(cand)
        if PARTIAL_MIN_PTS <= len(trimmed) < len(cand):
            variants.append(trimmed)
        for var in variants:
            cf = make_fitter(var)
            cv_, _ = cf.fit_constrained(env_speed)
            resid = cf.pixel_loss(cv_)
            # The physical gates decide which chains are ball flights AT ALL, so
            # they belong here rather than only on the winner: otherwise a long
            # clutter chain that happens to squeak under the residual bar wins
            # the ranking and takes the whole clip down with it.
            valid = resid <= MAX_FIT_RESIDUAL_PX and abs(cv_[3]) <= MAX_LAUNCH_DT_S
            ranked.append((valid, resid, var, cand))
    if debug:
        for valid, resid, cand, _ in sorted(ranked, key=lambda rr: rr[1]):
            print(f"  [rank] {'ok ' if valid else '   '} n={len(cand):2d} "
                  f"f{cand[0][0]}..{cand[-1][0]} resid={resid:7.2f}")
    # Among valid flights, trade track length against fit quality: more frames
    # condition the fit better, but not at any price in reprojection error.
    good = [r for r in ranked if r[0]]
    parent = None
    if good:
        _, _, track, parent = max(good, key=lambda r: len(r[2]) - LENGTH_RESID_TRADE * r[1])
    elif ranked:
        track = min(ranked, key=lambda r: r[1])[2]
    else:
        track = t["track"] or t_eq["track"]

    # Reclaim tail points the (deliberately conservative, short-baseline) trim
    # dropped: the winning fit is well conditioned, so re-test the parent chain's
    # remaining points against IT. More points tighten the confidence band.
    if parent is not None and len(parent) > len(track):
        fw = make_fitter(track)
        vw, _ = fw.fit_constrained(env_speed)
        fp = make_fitter(parent)
        pr, _ = fp._project(vw[0], vw[1], vw[2], vw[3])
        errs = np.linalg.norm(pr - fp.obs, axis=1) / fp.px_scale
        k = len(track)
        while k < len(parent) and errs[k] <= PREFIX_GROW_TOL_PX:
            k += 1
        if k > len(track):
            grown = parent[:k]
            fg = make_fitter(grown)
            vg, _ = fg.fit_constrained(env_speed)
            if (fg.pixel_loss(vg) <= MAX_FIT_RESIDUAL_PX
                    and abs(vg[3]) <= MAX_LAUNCH_DT_S):
                track = grown
    n = len(track)

    result: dict = {
        "engine": "balltrack-v2",
        "impact_frame": int(impact),
        "fps": t["fps"],
        "track_2d": [[f, round(x, 2), round(y, 2)] for f, x, y in track],
        "n_track_points": n,
        "n_candidates": len(pool),
        "azimuth_note": "azimuth is relative to the camera axis (single camera, no aim line)",
    }
    if n < PARTIAL_MIN_PTS:
        result["quality"] = "simulated"
        result["reason"] = f"track too short ({n} pts)"
        return result

    fitter = make_fitter(track)
    v, _ = fitter.fit_free() if n >= MEASURED_MIN_PTS else fitter.fit_constrained(env_speed)
    resid = fitter.pixel_loss(v)                 # report PIXEL error, not loss+prior
    _, r = fitter._project(v[0], v[1], v[2], v[3])

    # A real ball flight reprojects to a couple of pixels. Anything worse is
    # clutter that survived the 2D filters (a rising club, a net edge, another
    # bay's ball) — report no measurement rather than a confident wrong one.
    reject = None
    if resid > MAX_FIT_RESIDUAL_PX:
        reject = f"{resid:.1f} px reprojection residual over {n} points"
    elif abs(v[3]) > MAX_LAUNCH_DT_S:
        # launch_dt_s slides the flight's start in time. A real ball leaves at
        # impact, so a fit that needs to start it several frames early (or late)
        # is bending the model around clutter, not tracking a golf ball.
        reject = f"fitted launch is {abs(v[3]):.2f}s from impact"
    if reject:
        result["quality"] = "simulated"
        result["reason"] = f"track does not fit a ball flight ({reject})"
        result["rejected_fit"] = {"launch_deg": round(v[1], 1),
                                  "azimuth_deg": round(v[2], 1),
                                  "launch_dt_s": round(v[3], 3),
                                  "residual_px": round(resid, 1)}
        return result

    # Reported band = posterior (pixels + priors). The measured/partial decision
    # uses the PIXEL-ONLY band, so a tight prior can never talk us into calling
    # a speed "measured" that the video never actually pinned down.
    band, n_band = fitter.confidence_band()
    raw_band, _ = fitter.confidence_band(with_prior=False)
    rlo, rhi = raw_band["carry_yd"]
    speed_measured = (n >= MEASURED_MIN_PTS
                      and (rhi - rlo) / max((rlo + rhi) / 2.0, 1.0) <= BAND_MEASURED_FRAC)

    fit = {"launch_deg": round(v[1], 1), "azimuth_deg": round(v[2], 1),
           "launch_dt_s": round(v[3], 3), "residual_px": round(resid, 1)}
    if speed_measured:
        fit["ball_speed_mph"] = round(v[0], 1)
    else:
        # The pixels do not pin the scale: keep the measured SHAPE and take the
        # scale from the club envelope, re-fitting launch/azimuth at that speed.
        v, _ = fitter.fit_constrained(env_speed)
        resid = fitter.pixel_loss(v)
        _, r = fitter._project(v[0], v[1], v[2], v[3])
        fit.update({"ball_speed_mph": round(env_speed, 1),
                    "speed_source": "club_envelope",
                    "launch_deg": round(v[1], 1), "azimuth_deg": round(v[2], 1),
                    "launch_dt_s": round(v[3], 3), "residual_px": round(resid, 1)})
    result.update({
        "quality": "measured" if speed_measured else "partial",
        "fit": fit,
        "flight": {"carry_yd": round(r["carry_yd"]), "apex_yd": round(r["apex_yd"]),
                   "side_yd": round(r["side_yd"]),
                   "flight_time_s": round(r["flight_time_s"], 1)},
        "ci_10_90": band,
        "band_note": (f"range over every flight within {BAND_DELTA_PX:.0f} px of the best "
                      f"fit ({n_band} solutions), including depth-anchor uncertainty"),
    })
    # UI arc + replay-frame trajectory come from the fitted sim
    rr = simulate_flight(result["fit"]["ball_speed_mph"], result["fit"]["launch_deg"],
                         spin, 0.0, 0.0, trajectory_points=48)
    result["trajectory_world"] = [[round(a, 1), round(b, 1), round(c, 1)]
                                  for a, b, c in rr["trajectory"]]
    lvl, tt = fitter.level_pts(result["fit"]["ball_speed_mph"],
                               result["fit"]["launch_deg"],
                               result["fit"]["azimuth_deg"], n_traj=48)
    result["trajectory_tee_m"] = [[round(float(t), 3)] + [round(float(v), 3) for v in p]
                                  for t, p in zip(tt, lvl)]
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
