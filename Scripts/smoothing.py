"""Temporal smoothing for 3D pose sequences before Unreal Engine handoff.

Operates on (T, 17, 3) H36M-17 arrays (the lifter output loaded by
export_ue5.load_3d_parquet_as_h36m). Insert one call to `smooth_sequence`
between lifting and export to de-jitter the CSV/JSON/BVH and the HTML preview
at once.

Golf constraint: impact is 2-3 frames of very fast wrist/club motion, so the
default filter is the One Euro filter, which backs off its smoothing as joint
speed rises and therefore preserves the impact snap. Savitzky-Golay is offered
as an offline alternative that preserves derivatives well on slower clips.

Pipeline order inside `smooth_sequence`:
  1. interpolate_gaps  - linear-fill short low-confidence runs (needs conf)
  1b. interpolate_gaps_angular (optional) - re-fill arm-joint gaps by angular
      momentum around the swing plane, including gaps step 1 can't touch
  2. temporal filter   - one_euro_filter | savgol_smooth
  3. enforce_bone_lengths (optional) - rigidify the skeleton frame-to-frame
  4. enforce_joint_angle_limits (optional) - undo anatomically impossible bends
  5. enforce_grip_constraint (optional) - pull the wrists back together
  6. enforce_swing_plane (optional) - pull the grip point back onto its own arc
  7. enforce_torso_clearance (optional) - push a wrist/elbow out of the torso
  8. enforce_follow_through_deceleration (optional) - suppress isolated
     grip-speed spikes without flattening real multi-frame events
  9. freeze_arm_pose_during_holds (optional) - freeze the arms to a single
     pose wherever the torso proves they should be still, without touching
     forearm length

Joint angle limits: elbows and knees are hinges that cannot fold past a certain
point (~15-20 deg between the two bones at the joint is near the true anatomical
limit for able-bodied adults - a golf swing never gets close). The lifter has no
anatomical model, so a depth-ambiguous frame can occasionally predict a wrist or
ankle almost coincident with the shoulder/hip in 3D, an angle no human elbow or
knee can reach. Step 4 rotates the distal joint back out to the angle floor,
about the joint, preserving the segment's bone length exactly (so it composes
losslessly with step 3, unlike step 5).

Grip constraint: both hands hold the same club, so the lifter (which predicts
every joint independently, per frame, with no rig or club) can drift the
wrists apart in a way that is never physically possible - worst during the
fastest part of the swing, when hand tracking is noisiest. Step 5 clamps
left/right wrist distance to a plausible grip width by keeping each wrist ON
the sphere of its own forearm length around its elbow throughout (alternating
nudge-together / re-project-onto-sphere), so it never re-trades step 3's
bone-length precision to fix the hands-floating-apart error - an earlier
version that moved wrists freely did, and on fast real footage that meant a
forearm collapsing to under half length for several consecutive frames.

Swing plane: real swings trace the hands/club through one arc, not a cloud -
step 5 fixes hand-to-hand distance but says nothing about whether the PAIR, as a
unit, is drifting off that arc frame to frame. Step 6 fits a plane to the grip
point's trajectory (robust to address/finish, which genuinely aren't on the
backswing-to-impact plane) and pulls outlier frames back toward it, shifting
both wrists by the same vector so it never fights step 5's hand-to-hand distance.

Torso clearance: the lifter has no body-volume model, so at a self-occlusion
frame (arm crossing in front of/behind the torso from camera) it can place a
wrist or elbow geometrically inside where the torso actually is - impossible
for a real arm, and the single most visible failure mode of all of these
(the limb visibly clips through the body in the rendered mesh). Step 7 pushes
any wrist/elbow closer than `min_torso_clearance` to the spine axis
(hip_center -> thorax) radially outward until it clears, along whichever
radial direction it was already leaning (so the arm doesn't jump to the wrong
side of the body). Runs last since this is the most visually severe artifact.

Angular gap filling: a golf swing is rotational, not linear, so step 1's
straight-line fill visibly cuts the corner of the true arc across anything
more than a couple of frames, and cannot fill a gap that runs to either end
of the clip at all (no "after"/"before" anchor to draw a line to). Step 1b
fits the swing plane for its orientation, then per arm joint fits a circle
to that joint's position RELATIVE TO ITS OWN SHOULDER (not a single fixed
pivot averaged over the clip - the shoulder itself moves through the swing,
so a static pivot falls behind it and the arm visibly droops instead of
reading as an extension of the shoulder), and interpolates/extrapolates the
joint's ANGLE around that circle - matching angular velocity at whatever
anchor frame(s) exist - before reconstructing off the shoulder's own actual
position at each frame. Validated against a real held-out trajectory: cuts
reconstruction error roughly 3x on an interior gap and roughly 9x on a
boundary gap versus step 1 alone (see the module tests). Off by default
pending more real-world mileage - it depends on the swing-plane/circle fits
being sane, which needs enough confident frames to fit in the first place.
Ends with a final bone-length re-lock: a joint can get reconstructed here
because ITS OWN confidence was low while its child stays raw (that child's
own confidence was fine, so this step's gap-fill never touches it) - orphaning
the child from its newly-moved parent. Measured on real footage: a forearm
left over double its true rigid length for several consecutive frames,
reading as the hand trailing (or leading from the wrong place relative to)
the elbow rather than as its extension.

Follow-through deceleration: the lifter has no notion of which speed changes
are real swing dynamics (the downswing's acceleration into impact, the
wrist-release whip) and which are per-frame tracking noise - both look like
"this frame's speed differs from the previous frame's". Step 8 tells them
apart by comparing each frame's grip-point speed only to its own local
neighbourhood (a Hampel-style outlier test: median + scaled MAD over a small
window of nearby frames, excluding itself). An isolated noise spike stands
out against calm neighbours and gets clamped down to the local threshold; a
real multi-frame event (release, downswing) raises its neighbourhood right
along with it, so it's never flagged. Shifts both wrists by the same vector
(so it never fights step 5's hand-to-hand distance).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from export_ue5 import H36M17_IDX, H36M17_PARENTS


# ---------------------------------------------------------------------------
# 1. Gap handling
# ---------------------------------------------------------------------------

def interpolate_gaps(xyz: np.ndarray,
                     conf: np.ndarray | None,
                     min_conf: float = 0.3,
                     max_gap: int = 5) -> np.ndarray:
    """Linearly interpolate joint positions across short low-confidence runs.

    A "gap" is a maximal run of consecutive frames where conf < min_conf for a
    given joint. Gaps no longer than `max_gap` and bounded by valid frames on
    both sides are filled by linear interpolation; longer gaps and gaps at the
    sequence ends are left untouched (no reliable anchor to interpolate from).

    If `conf` is None this is a no-op (the caller has no per-joint confidence).
    """
    if conf is None:
        return xyz.astype(np.float32, copy=True)

    out = xyz.astype(np.float32, copy=True)
    T, J, _ = out.shape
    for j in range(J):
        valid = conf[:, j] >= min_conf
        if valid.all() or valid.sum() < 2:
            continue
        t = 0
        while t < T:
            if valid[t]:
                t += 1
                continue
            start = t
            while t < T and not valid[t]:
                t += 1
            end = t  # first valid frame after the gap (or T)
            gap_len = end - start
            if start == 0 or end == T or gap_len > max_gap:
                continue  # unbounded or too long -> leave as-is
            lo, hi = start - 1, end
            for k in range(start, end):
                w = (k - lo) / (hi - lo)
                out[k, j] = (1.0 - w) * out[lo, j] + w * out[hi, j]
    return out


# ---------------------------------------------------------------------------
# 1b. Angular-momentum gap filling (swing-plane-relative, arm joints)
# ---------------------------------------------------------------------------

def _fit_circle_2d(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """Kasa least-squares circle fit to 2D points. Returns (center_u, center_v,
    radius). Linear (not iterative): u^2+v^2 = 2*cu*u + 2*cv*v + (r^2-cu^2-cv^2),
    solved for the three unknowns on the right by ordinary least squares."""
    A = np.stack([2.0 * u, 2.0 * v, np.ones_like(u)], axis=1)
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv, c = sol
    r = math.sqrt(max(float(c) + cu * cu + cv * cv, 0.0))
    return float(cu), float(cv), r


def _in_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (e1, e2) spanning the plane perpendicular to `normal`."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, normal) * normal
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def interpolate_gaps_angular(xyz: np.ndarray,
                             conf: np.ndarray | None,
                             joints: tuple = (
                                 # elbows MUST be processed before wrists: the wrist anchors to
                                 # the elbow, so it needs the elbow's already-reconstructed
                                 # position, not its raw one (see docstring).
                                 ("left_elbow", H36M17_IDX["left_elbow"], H36M17_IDX["left_shoulder"]),
                                 ("right_elbow", H36M17_IDX["right_elbow"], H36M17_IDX["right_shoulder"]),
                                 ("left_wrist", H36M17_IDX["left_wrist"], H36M17_IDX["left_elbow"]),
                                 ("right_wrist", H36M17_IDX["right_wrist"], H36M17_IDX["right_elbow"]),
                             ),
                             min_conf: float = 0.3,
                             max_extrapolate: int = 90,
                             lead_factor: float = 1.0) -> np.ndarray:
    """Fill low-confidence gaps for arm joints using constant ANGULAR
    velocity around a two-link shoulder->elbow->wrist chain, instead of
    linear position interpolation (`interpolate_gaps`).

    A golf swing is fundamentally rotational (the arm whips as an extension
    of the shoulder), so linear interpolation of raw XYZ visibly cuts the
    corner of the true arc for anything more than a couple of frames - and
    cannot extrapolate at all past the last confident frame, since it has no
    "after" anchor to draw a line to. This instead:
      1. Fits the swing plane (reusing `_fit_plane_robust`) to the grip
         point's (wrist-pair midpoint) confident frames, for its ORIENTATION
         only (the normal/in-plane basis - a stable property of the whole
         swing motion).
      2. Per joint, re-expresses its confident samples RELATIVE TO ITS OWN
         ANCHOR - the elbow orbits the shoulder, the wrist orbits the
         (already-reconstructed) elbow, matching the arm's real two-link
         kinematic chain - at each of those frames, then fits a 2D circle
         (center, radius) to that relative trajectory within the plane.
         Wrist-relative-to-SHOULDER is NOT a good circle: the elbow flexes
         independently, so that distance swings by nearly 2x through a
         swing and a single-circle fit puts the reconstructed wrist
         implausibly far out. Wrist-relative-to-ELBOW (forearm length) and
         elbow-relative-to-shoulder (upper-arm length) are each close to a
         true rigid segment, so the circle-fit assumption actually holds.
         Anchoring to the anchor joint's own per-frame tracked/reconstructed
         position - not a single fixed pivot averaged over the whole clip -
         also matters on its own: the anchor itself translates and rotates
         through the swing, so a static pivot falls behind it and the
         reconstructed limb visibly droops instead of reading as an
         extension of its parent joint.
      3. Per gap, interpolates (or, at a sequence boundary, extrapolates)
         the joint's ANGLE around that circle - matching both angle and
         angular velocity at whatever anchor frame(s) exist, with the
         boundary angular velocities tangent-clamped to a small multiple of
         the secant slope so a genuine reversal (e.g. the angular velocity
         crossing zero at the top of the backswing) doesn't make the cubic
         Hermite overshoot in the middle - and reads the radius and
         out-of-plane offset off the nearest anchor(s). Also floors the
         wrist's boundary angular velocity at `lead_factor` times its
         elbow's own (already-reconstructed) rate, in whichever direction
         they agree on: a real swing's hand rotates at least as fast as the
         arm it's attached to (leads it), but each joint's angle is fit
         independently from its OWN sparse boundary samples with no such
         coupling by default, so a thin/noisy wrist-side estimate could
         otherwise read as the hand trailing the elbow instead.
      4. Reconstructs 3D position as the anchor's ACTUAL (already-updated,
         for the wrist's elbow anchor) position at that frame plus the
         interpolated angle/radius/offset - so the limb always hangs off
         wherever its anchor really is, not a fixed point.

    A joint whose OWN confidence is fine but whose ANCHOR just got
    reconstructed is treated as a gap too, not left at its raw position: a
    confident wrist next to a low-confidence elbow means the elbow is about
    to move to a physically-plausible circular position, and the wrist's raw
    sample was tracked relative to wherever the OLD (unreliable) elbow was,
    not the new one - on this project's own real footage, leaving it alone
    orphaned a wrist up to 0.51m from an elbow with a true rigid length of
    0.225m, reading as the hand trailing the elbow instead of extending from
    it. Forcing the wrist through the same circular reconstruction there also
    means it goes through the leading-hand check in step 3 above, which
    otherwise never runs for a frame the wrist itself didn't need filling.

    Call this AFTER `interpolate_gaps`: it only touches the specified arm
    joints, and it will happily fill (extrapolate through) a gap running to
    either end of the clip, which `interpolate_gaps` explicitly refuses to
    touch - so it should run second, overwriting the untouched frames
    `interpolate_gaps` left behind for those joints. `max_extrapolate` caps
    how many frames of pure extrapolation are trusted before giving up (an
    open-ended gap far from any real data is guesswork past some point).

    Assumes the shoulder is reasonably well tracked throughout (shoulders
    are large, rarely self-occluded landmarks, unlike wrists/elbows) - any
    brief shoulder gaps should already be closed by `interpolate_gaps`
    upstream of this call. The elbow anchor for the wrist may itself be
    reconstructed rather than raw-tracked (that's the point - elbows and
    wrists tend to go untracked together during self-occlusion), which is
    exactly why elbows are processed first in the default `joints` order.
    """
    if conf is None:
        return xyz.astype(np.float32, copy=True)
    out = xyz.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < 8:
        return out.astype(np.float32)

    # This function's whole model assumes each anchor->joint segment has a
    # stable length (that's what makes the per-joint circle fit meaningful).
    # The raw lifter output does NOT reliably have that - shoulder-to-elbow
    # distance alone was observed swinging ~4x (0.11m to 0.44m) across a
    # single real clip - so bone-length-lock internally first rather than
    # assume the caller already has. Harmless if the caller bone-locks again
    # later (idempotent up to the same reference lengths).
    out = enforce_bone_lengths(out)

    li, ri = H36M17_IDX["left_wrist"], H36M17_IDX["right_wrist"]
    grip = 0.5 * (out[:, li] + out[:, ri])
    conf_grip = np.minimum(conf[:, li], conf[:, ri])
    good_grip = conf_grip >= min_conf
    if good_grip.sum() < 8:
        return out.astype(np.float32)
    _plane_centroid, normal, _ = _fit_plane_robust(grip[good_grip], exclude_threshold=float("inf"))
    e1, e2 = _in_plane_basis(normal)

    # Per-joint angle trajectories (in the shared swing-plane u,v basis, so
    # angular velocities are directly comparable across joints even though
    # each is measured around a different circle center) - lets a later
    # joint in `joints` (the wrist) read back an earlier one's (the elbow's)
    # rotation rate, for the leading-hand enforcement below.
    angle_by_joint: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    # Which frames each joint actually reconstructed - so a child whose OWN
    # confidence is fine but whose anchor was just reconstructed still gets
    # treated as a gap (see docstring).
    touched_by_joint: dict[int, np.ndarray] = {}

    for _name, j_idx, anchor_idx in joints:
        jconf = conf[:, j_idx]
        anchor_touched = touched_by_joint.get(anchor_idx)
        own_good = jconf >= min_conf
        good = own_good if anchor_touched is None else (own_good & ~anchor_touched)
        idx_good = np.where(good)[0]
        if len(idx_good) < 4:
            continue

        anchor = out[:, anchor_idx]        # (T,3) shoulder's own tracked position, per frame
        pos = out[:, j_idx]
        rel = pos - anchor                 # joint position relative to ITS shoulder, per frame
        u = rel @ e1
        v = rel @ e2
        w = rel @ normal  # out-of-plane component

        cu, cv, _r_global = _fit_circle_2d(u[good], v[good])
        angle_raw = np.arctan2(v - cv, u - cu)
        angle = np.full(T, np.nan)
        angle[idx_good] = np.unwrap(angle_raw[idx_good])
        radius = np.hypot(u - cu, v - cv)
        anchor_angle, anchor_idx_good = angle_by_joint.get(anchor_idx, (None, None))
        angle_by_joint[j_idx] = (angle, idx_good)

        def _anchor_velocity(frame: int, before: bool) -> float:
            """Local angular velocity of the anchor's OWN already-
            reconstructed trajectory near `frame` (3-frame window on the
            `before`/after side), for the leading-hand check below. 0.0 if
            the anchor wasn't itself angular-gap-filled (e.g. the elbow's
            own anchor, the shoulder, is never processed by this loop)."""
            if anchor_angle is None:
                return 0.0
            win_a = (anchor_idx_good[anchor_idx_good <= frame][-3:] if before
                     else anchor_idx_good[anchor_idx_good >= frame][:3])
            if len(win_a) < 2:
                return 0.0
            return float(np.polyfit(win_a, anchor_angle[win_a], 1)[0])

        def _lead(v_self: float, v_anchor: float) -> float:
            """A real golf swing's distal segment (wrist relative to elbow)
            rotates at least as fast as its proximal one (elbow relative to
            shoulder) - the hand leads, almost always, rather than trailing.
            Independent per-joint circle-fit interpolation has no such
            coupling: each joint's angle is reconstructed purely from its
            OWN sparse boundary samples, so a noisy or thin anchor for the
            wrist's own boundary velocity can end up UNDER its elbow's,
            reading as the hand lagging behind the arm instead of leading
            the whip. When both velocities agree in rotational direction (or
            this joint's own estimate is degenerate/zero), floor this
            joint's magnitude at `lead_factor` times its anchor's - never
            below it, so it can lead but is never forced to trail. Left
            alone when they disagree in sign: that's the existing overshoot
            guard's territory (a genuine independent reversal), not this
            check's."""
            if v_anchor == 0.0:
                return v_self
            if v_self == 0.0 or v_self * v_anchor > 0:
                sign = 1.0 if v_anchor > 0 else -1.0
                return sign * max(abs(v_self), lead_factor * abs(v_anchor))
            return v_self

        touched = np.zeros(T, dtype=bool)
        t = 0
        while t < T:
            if good[t]:
                t += 1
                continue
            start = t
            while t < T and not good[t]:
                t += 1
            end = t  # first good frame at/after end of gap, or T
            gap_len = end - start
            if gap_len > max_extrapolate:
                continue
            lo = start - 1
            has_lo, has_hi = lo >= 0, end < T
            if not has_lo and not has_hi:
                continue
            touched[start:end] = True

            if has_lo:
                win = idx_good[idx_good <= lo][-3:]
                ang_v_lo = float(np.polyfit(win, angle[win], 1)[0]) if len(win) >= 2 else 0.0
                ang_lo, r_lo, w_lo = angle[lo], radius[lo], w[lo]
            if has_hi:
                win = idx_good[idx_good >= end][:3]
                ang_v_hi = float(np.polyfit(win, angle[win], 1)[0]) if len(win) >= 2 else 0.0
                ang_hi, r_hi, w_hi = angle[end], radius[end], w[end]
                if has_lo:
                    # re-anchor ang_hi's 2pi branch to the one nearest a
                    # constant-velocity prediction from the lo side, so a
                    # gap spanning more than half a turn doesn't get
                    # unwrapped the "short way" by mistake.
                    predicted = ang_lo + ang_v_lo * (end - lo)
                    ang_hi = predicted + ((ang_hi - predicted + math.pi) % (2 * math.pi) - math.pi)

                    # Monotone-Hermite-style tangent clamping: a cubic
                    # Hermite matching value+slope at both ends can badly
                    # OVERSHOOT in the middle when a boundary slope disagrees
                    # in SIGN with the overall secant trend - exactly what a
                    # swing reversal (top of backswing, angular velocity
                    # crossing zero) produces, and exactly the situation
                    # classic monotone-cubic-interpolation methods (Fritsch-
                    # Carlson) zero the tangent for. We only have noisy 2-3
                    # frame boundary velocity estimates, not enough signal to
                    # reliably place *where* in the gap a reversal happens or
                    # how sharp it is - so when a boundary tangent disagrees
                    # in sign with the secant, trust the secant and drop the
                    # tangent to 0 rather than let it pull the curve into a
                    # large excursion. When they agree, still cap magnitude
                    # to a small multiple of the secant as a second guard.
                    span = end - lo
                    avg_v = (ang_hi - ang_lo) / span if span > 0 else 0.0
                    cap = 3.0 * abs(avg_v) + math.radians(2.0)
                    ang_v_lo = ang_v_lo if ang_v_lo * avg_v >= 0 else 0.0
                    ang_v_hi = ang_v_hi if ang_v_hi * avg_v >= 0 else 0.0
                    ang_v_lo = float(np.clip(ang_v_lo, -cap, cap))
                    ang_v_hi = float(np.clip(ang_v_hi, -cap, cap))

            if has_lo:
                ang_v_lo = _lead(ang_v_lo, _anchor_velocity(lo, before=True))
            if has_hi:
                ang_v_hi = _lead(ang_v_hi, _anchor_velocity(end, before=False))

            for k in range(start, end):
                if has_lo and has_hi:
                    span = end - lo
                    tt = (k - lo) / span
                    h00 = 2 * tt**3 - 3 * tt**2 + 1
                    h10 = tt**3 - 2 * tt**2 + tt
                    h01 = -2 * tt**3 + 3 * tt**2
                    h11 = tt**3 - tt**2
                    ang_k = h00 * ang_lo + h10 * span * ang_v_lo + h01 * ang_hi + h11 * span * ang_v_hi
                    frac = tt
                    r_k = (1 - frac) * r_lo + frac * r_hi
                    w_k = (1 - frac) * w_lo + frac * w_hi
                elif has_lo:
                    ang_k = ang_lo + ang_v_lo * (k - lo)
                    r_k, w_k = r_lo, w_lo
                else:
                    ang_k = ang_hi + ang_v_hi * (k - end)
                    r_k, w_k = r_hi, w_hi

                pu = cu + r_k * math.cos(ang_k)
                pv = cv + r_k * math.sin(ang_k)
                out[k, j_idx] = anchor[k] + pu * e1 + pv * e2 + w_k * normal

        touched_by_joint[j_idx] = touched

    # Backstop, not the primary fix: `touched_by_joint` above makes a child
    # go through proper circular reconstruction whenever its anchor moved,
    # which is what actually fixes the orphaned-wrist case. This just
    # re-snaps every child back onto its parent's rigid length along
    # whatever direction it currently has, for anything that still slipped
    # through (e.g. a joint with too few confident samples to reconstruct at
    # all, which `continue`s before ever recording `touched_by_joint`) - a
    # no-op for frames properly reconstructed above (their radius already IS
    # close to the rigid length).
    out = enforce_bone_lengths(out)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 2a. One Euro filter (adaptive low-pass, default)
# ---------------------------------------------------------------------------

def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_filter(xyz: np.ndarray,
                    fps: float = 30.0,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    d_cutoff: float = 1.0) -> np.ndarray:
    """One Euro filter (Casiez et al. 2012) applied per joint and axis.

    Adaptive low-pass: at low speeds the cutoff approaches `min_cutoff`
    (heavy smoothing, kills tremor); as speed rises the cutoff grows by
    `beta * |velocity|` (light smoothing, low lag). Tuning:
      - lower min_cutoff -> smoother slow phases (more lag)
      - higher beta      -> snappier fast phases (less lag at impact)
    """
    T = xyz.shape[0]
    if T < 2:
        return xyz.astype(np.float32, copy=True)

    dt = 1.0 / float(fps) if fps and fps > 0 else 1.0 / 30.0
    out = np.empty_like(xyz, dtype=np.float32)
    x_prev = xyz[0].astype(np.float32)
    dx_prev = np.zeros_like(x_prev)
    out[0] = x_prev
    a_d = _alpha(d_cutoff, dt)
    for t in range(1, T):
        x = xyz[t].astype(np.float32)
        dx = (x - x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        # vectorized alpha per joint/axis
        tau = 1.0 / (2.0 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * x_prev
        out[t] = x_hat
        x_prev = x_hat
        dx_prev = dx_hat
    return out


# ---------------------------------------------------------------------------
# 2b. Savitzky-Golay (offline alternative)
# ---------------------------------------------------------------------------

def savgol_smooth(xyz: np.ndarray,
                  window: int = 7,
                  polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing along the time axis.

    Window is forced odd and clamped to the sequence length; polyorder is
    clamped below the window. Offline (uses future frames) and preserves
    peaks/derivatives better than a moving average.
    """
    from scipy.signal import savgol_filter

    T = xyz.shape[0]
    win = int(window)
    if win % 2 == 0:
        win += 1
    if win > T:
        win = T if T % 2 == 1 else T - 1
    if win < 3:
        return xyz.astype(np.float32, copy=True)
    poly = min(int(polyorder), win - 1)
    return savgol_filter(xyz, window_length=win, polyorder=poly,
                         axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Bone-length stabilization
# ---------------------------------------------------------------------------

def enforce_bone_lengths(xyz: np.ndarray,
                         parents: tuple[int, ...] = H36M17_PARENTS,
                         ref: str = "median") -> np.ndarray:
    """Rigidify the skeleton: re-project each joint so every bone keeps a
    constant length across all frames.

    For each bone (child -> parent) a reference length is taken as the
    median (or mean) of its per-frame lengths. Then, walking the skeleton
    root->leaf, each child is placed along its current bone direction at the
    reference length from the already-corrected parent. The root joint is
    untouched, so global translation is preserved.

    Relies on H36M17_PARENTS having parent index < child index (true for the
    H36M-17 ordering), so a single forward pass processes parents first.
    """
    T, J, _ = xyz.shape
    out = xyz.astype(np.float32, copy=True)

    # reference length per joint (0 for root)
    ref_len = np.zeros(J, dtype=np.float32)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        lengths = np.linalg.norm(xyz[:, j] - xyz[:, p], axis=1)
        ref_len[j] = np.median(lengths) if ref == "median" else lengths.mean()

    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                # degenerate frame: keep parent position
                out[t, j] = out[t, p]
            else:
                out[t, j] = out[t, p] + vec / norm * ref_len[j]
    return out


# ---------------------------------------------------------------------------
# 4. Joint angle limits (elbow / knee hinge floor)
# ---------------------------------------------------------------------------

# (proximal, joint, distal) index triples for each hinge, named by the middle joint.
HINGES: tuple[tuple[str, int, int, int], ...] = (
    ("left_elbow",  H36M17_IDX["left_shoulder"], H36M17_IDX["left_elbow"],  H36M17_IDX["left_wrist"]),
    ("right_elbow", H36M17_IDX["right_shoulder"], H36M17_IDX["right_elbow"], H36M17_IDX["right_wrist"]),
    ("left_knee",   H36M17_IDX["left_hip"],   H36M17_IDX["left_knee"],   H36M17_IDX["left_ankle"]),
    ("right_knee",  H36M17_IDX["right_hip"],  H36M17_IDX["right_knee"],  H36M17_IDX["right_ankle"]),
)


def enforce_joint_angle_limits(xyz: np.ndarray,
                               hinges: tuple = HINGES,
                               min_angle_deg: float = 15.0) -> np.ndarray:
    """Undo anatomically impossible hinge-joint flexion, per frame.

    For each (proximal, joint, distal) triple - e.g. (shoulder, elbow, wrist) -
    the interior angle at `joint` is the angle between the proximal-ward and
    distal-ward bone vectors: 180 deg is a fully straight limb, and it can only
    decrease toward `min_angle_deg` as the limb bends. Anything predicted below
    that floor is not a tight bend, it's a lifter error (typically the distal
    joint landing at nearly the same depth as the proximal one).

    The correction rotates the distal joint within the (proximal, distal) plane
    about `joint` until the angle equals `min_angle_deg`, holding the joint's
    own position and the segment's bone length (|joint - distal|) fixed - so it
    never fights `enforce_bone_lengths` and composes losslessly with it.
    """
    out = xyz.astype(np.float32, copy=True)
    T = out.shape[0]
    min_rad = math.radians(min_angle_deg)
    for _name, p_idx, j_idx, d_idx in hinges:
        for t in range(T):
            joint = out[t, j_idx]
            v1 = out[t, p_idx] - joint
            v2 = out[t, d_idx] - joint
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            u1 = v1 / n1
            along = float(np.dot(v2, u1))
            perp = v2 - along * u1
            n_perp = np.linalg.norm(perp)
            if n_perp < 1e-8:
                continue  # degenerate: v2 parallel/antiparallel to v1
            angle = math.atan2(n_perp, along)
            if angle >= min_rad:
                continue
            u2 = perp / n_perp
            v2_new = n2 * (math.cos(min_rad) * u1 + math.sin(min_rad) * u2)
            out[t, d_idx] = joint + v2_new
    return out


# ---------------------------------------------------------------------------
# 5. Grip stabilization (two-hand coupling)
# ---------------------------------------------------------------------------

def enforce_grip_constraint(xyz: np.ndarray,
                            conf: np.ndarray | None = None,
                            left_idx: int = H36M17_IDX["left_wrist"],
                            right_idx: int = H36M17_IDX["right_wrist"],
                            left_elbow_idx: int = H36M17_IDX["left_elbow"],
                            right_elbow_idx: int = H36M17_IDX["right_elbow"],
                            max_distance: float = 0.12) -> np.ndarray:
    """Clamp left/right wrist distance so the hands never exceed a plausible
    golf grip width, per frame - without breaking the elbow-to-wrist bone
    length `enforce_bone_lengths`/`enforce_joint_angle_limits` already made
    rigid.

    Both hands share one club, so wrist-to-wrist distance is small (roughly a
    hand's width for an overlap/interlock grip, a bit more for ten-finger) and
    should not vary much through the swing. The lifter has no notion of a
    club or a second hand, so nothing stops it drifting the wrists apart -
    and during the fastest part of the swing, when hand tracking is noisiest,
    that drift can be large.

    An earlier version moved each wrist freely along the line between them,
    with no regard for the elbow. On this project's own real swing footage
    that collapsed a forearm to under half its true length for several
    consecutive frames right through the fastest part of the downswing - a
    forearm visibly concertina-ing rather than staying a rigid arm, which
    reads as the hand dragging behind the elbow and looking floppy, not as a
    hand's-width grip correction. Fixed by keeping each wrist ON the sphere
    of its own (already bone-locked) forearm length around its elbow the
    whole time: alternate nudging the pair together along the line between
    them and re-projecting each back onto its own sphere, a few iterations of
    alternating projection, so the correction gets as close to `max_distance`
    as two fixed-radius spheres allow without ever breaking either forearm's
    length. If `conf` is given, the less confident wrist absorbs more of each
    nudge (it's the one more likely to be wrong); otherwise the correction is
    split evenly.
    """
    out = xyz.astype(np.float64, copy=True)
    T = out.shape[0]
    for t in range(T):
        l = out[t, left_idx]
        r = out[t, right_idx]
        le = out[t, left_elbow_idx]
        re_ = out[t, right_elbow_idx]
        len_l = float(np.linalg.norm(l - le))
        len_r = float(np.linalg.norm(r - re_))

        if conf is not None:
            cl, cr = float(conf[t, left_idx]), float(conf[t, right_idx])
            total = cl + cr
            w_left = cr / total if total > 1e-8 else 0.5
        else:
            w_left = 0.5
        w_right = 1.0 - w_left

        for _ in range(4):
            vec = r - l
            dist = float(np.linalg.norm(vec))
            if dist <= max_distance or dist < 1e-8:
                break
            excess = dist - max_distance
            direction = vec / dist
            l = l + direction * excess * w_left
            r = r - direction * excess * w_right
            if len_l > 1e-6:
                l = le + (l - le) / np.linalg.norm(l - le) * len_l
            if len_r > 1e-6:
                r = re_ + (r - re_) / np.linalg.norm(r - re_) * len_r

        out[t, left_idx] = l
        out[t, right_idx] = r
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Swing-plane consistency
# ---------------------------------------------------------------------------

def _fit_plane_robust(points: np.ndarray,
                      exclude_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit plane (centroid, unit normal, keep-mask) via PCA, trimmed
    against address/finish. Address and finish are real, legitimate positions
    that are NOT on the backswing-to-impact swing plane (arms hang near the
    body, a different orientation) - fitting to every frame would tilt the
    plane toward them and mis-flag the actual swing arc as the outlier.

    Fit once on everything, then drop any frame whose residual from that
    first pass exceeds `exclude_threshold` and refit on what's left. This is
    a magnitude threshold, not a fixed top-K/percent: a clip with only two
    extreme frames (address, finish) drops only those two, however many
    frames it has - unlike a fixed trim fraction, which would either grab
    unrelated frames to fill a quota or fail to fully exclude the extremes
    depending on clip length. The returned keep-mask says which frames were
    trusted to belong to the swing-plane arc at all - the caller should only
    correct those, not the frames excluded from the fit."""
    T = len(points)
    c0 = points.mean(axis=0)
    _, _, vt0 = np.linalg.svd(points - c0, full_matrices=False)
    n0 = vt0[-1]
    resid0 = np.abs((points - c0) @ n0)
    keep = resid0 <= exclude_threshold
    if keep.sum() < 4:
        return c0, n0, np.ones(T, dtype=bool)  # too few trustworthy frames to trim
    kept = points[keep]
    c = kept.mean(axis=0)
    _, _, vt = np.linalg.svd(kept - c, full_matrices=False)
    return c, vt[-1], keep


def enforce_swing_plane(xyz: np.ndarray,
                        left_idx: int = H36M17_IDX["left_wrist"],
                        right_idx: int = H36M17_IDX["right_wrist"],
                        max_offplane: float = 0.08,
                        exclude_multiplier: float = 3.0,
                        max_correction_frac: float = 0.35,
                        verbose: bool = True) -> np.ndarray:
    """Pull the grip point (wrist-pair midpoint) back toward its own best-fit
    swing plane wherever a single frame has drifted off it.

    `enforce_grip_constraint` only sees hand-to-hand distance, so a pair of
    hands that stayed correctly together but drifted, as a unit, off the arc
    the real swing traces is invisible to it. This catches that: fit a plane
    to the grip-point trajectory (see `_fit_plane_robust` for why address/
    finish - anything beyond `exclude_multiplier * max_offplane` from the
    initial fit - are excluded from the fit), then for any TRUSTED frame
    whose grip point is more than `max_offplane` from that plane, shift it
    back along the plane normal until it's exactly `max_offplane` away. Both
    wrists are shifted by the same vector, so their distance from each other
    - step 5's job - is untouched. Frames excluded from the fit (presumed
    genuine address/finish) are left alone, not forced onto the swing arc.

    Guard: the whole premise is ONE swing tracing ONE plane per clip. On a
    clip that isn't that (multiple swings, walk-up/practice footage, a long
    capture), the "best-fit plane" is meaningless and correcting toward it
    actively distorts the pose. There's no ground truth at inference time to
    detect this directly, so this uses a cheap proxy: if more than
    `max_correction_frac` of TRUSTED frames need correction, the fit itself
    is presumed unreliable and the whole step is skipped, unmodified.
    Calibrated on two real clips: a clean 138-frame single swing needed
    correction on 25% of frames (a real, localized lifter-error cluster,
    visually confirmed to help) vs. a 399-frame clip whose off-plane residual
    drifted systematically over the full clip (not a localized cluster) and
    needed correction on 44% of frames - not a single swing plane, a
    genuinely different failure mode. The 0.35 default sits between those
    two data points; it is a heuristic, not a derived bound, and may need
    revisiting with more clips.
    """
    out = xyz.astype(np.float32, copy=True)
    T = out.shape[0]
    if T < 5:
        return out  # not enough frames to define a meaningful plane
    grip = 0.5 * (out[:, left_idx] + out[:, right_idx])
    centroid, normal, keep = _fit_plane_robust(
        grip.astype(np.float64), exclude_threshold=exclude_multiplier * max_offplane)
    resid = (grip.astype(np.float64) - centroid) @ normal
    needs_fix = keep & (np.abs(resid) > max_offplane)
    n_trusted = int(keep.sum())
    frac = float(needs_fix.sum()) / n_trusted if n_trusted else 0.0
    if frac > max_correction_frac:
        if verbose:
            print(f"[swing-plane] {needs_fix.sum()}/{n_trusted} trusted frames "
                  f"({frac*100:.0f}%) off-plane, above {max_correction_frac*100:.0f}% guard - "
                  f"this doesn't look like a single-swing-plane clip, skipping correction")
        return out
    for t in range(T):
        if not needs_fix[t]:
            continue
        d = resid[t]
        shift = -(abs(d) - max_offplane) * np.sign(d) * normal
        out[t, left_idx] = out[t, left_idx] + shift
        out[t, right_idx] = out[t, right_idx] + shift
    return out


# ---------------------------------------------------------------------------
# 7. Torso clearance (limb/torso collision)
# ---------------------------------------------------------------------------

TORSO_LIMB_JOINTS: tuple[tuple[str, int], ...] = (
    ("left_wrist",  H36M17_IDX["left_wrist"]),
    ("right_wrist", H36M17_IDX["right_wrist"]),
    ("left_elbow",  H36M17_IDX["left_elbow"]),
    ("right_elbow", H36M17_IDX["right_elbow"]),
)


def enforce_torso_clearance(xyz: np.ndarray,
                            joints: tuple = TORSO_LIMB_JOINTS,
                            hip_idx: int = H36M17_IDX["hip_center"],
                            thorax_idx: int = H36M17_IDX["thorax"],
                            min_clearance: float = 0.12) -> np.ndarray:
    """Push a wrist/elbow radially off the spine axis (hip_center -> thorax)
    whenever it comes closer than `min_clearance`.

    The lifter has no body-volume model, so a self-occlusion frame (arm
    swinging in front of or behind the torso from the camera's view) can
    place a wrist geometrically past where the torso actually is - visible
    as the limb clipping straight through the body mesh, the single most
    obvious failure mode of an otherwise-plausible-looking skeleton.

    For each frame, the joint is projected onto the (clamped) spine segment;
    if its distance from that projection is below `min_clearance`, it's
    pushed straight out along the same radial direction it already had
    (never flipped to the opposite side of the body - that would be a worse,
    more visible jump than the clipping it's fixing). `min_clearance`
    approximates the torso's cross-section radius (roughly 0.10-0.16m,
    tighter front-to-back than side-to-side) plus margin for the limb's own
    thickness; 0.12m is a conservative round number between those, matching
    the existing `max_hand_distance` convention.
    """
    out = xyz.astype(np.float32, copy=True)
    T = out.shape[0]
    for t in range(T):
        a = out[t, hip_idx].astype(np.float64)
        b = out[t, thorax_idx].astype(np.float64)
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-8:
            continue
        for _name, j_idx in joints:
            p = out[t, j_idx].astype(np.float64)
            frac = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
            proj = a + frac * ab
            radial = p - proj
            dist = float(np.linalg.norm(radial))
            if dist >= min_clearance or dist < 1e-6:
                continue
            out[t, j_idx] = (proj + radial / dist * min_clearance).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 8. Follow-through deceleration
# ---------------------------------------------------------------------------

def enforce_follow_through_deceleration(xyz: np.ndarray,
                                        left_idx: int = H36M17_IDX["left_wrist"],
                                        right_idx: int = H36M17_IDX["right_wrist"],
                                        top_window: tuple[float, float] = (0.1, 0.9),
                                        spike_window: int = 5,
                                        spike_k: float = 3.5,
                                        min_speed_mad: float = 0.01) -> np.ndarray:
    """Suppress isolated grip-point speed spikes after the top-of-backswing,
    without touching genuine multi-frame swing events like the wrist-release
    whip or the downswing's own acceleration into impact.

    An earlier version of this clamped speed to a decaying envelope off the
    post-impact peak - but a real release is BY DEFINITION a sudden speed
    increase, which looked identical to a tracking-noise spike to that
    envelope, so it flattened the whip along with the noise (hands read as
    "limp", never catching up to the club). A single frame of bad tracking
    and a real kinematic event both differ from what came right before them;
    the two only become distinguishable by also looking at what comes right
    AFTER. Noise is a one-frame outlier: the neighbours on both sides stay
    normal. A real event moves its neighbours too: the frames around a whip
    are also elevated, rising into it and decaying out of it.

    So for each frame's grip-point speed, compare it only to its own local
    window of `spike_window` neighbouring frames (excluding itself) via a
    Hampel-style outlier test: median + `spike_k` scaled MADs, with
    `min_speed_mad` as a floor so a near-still local window (e.g. right at
    the top of the backswing) doesn't make an ordinary small speed look like
    an infinite-sigma outlier. Only frames that exceed that LOCAL threshold
    get their speed clamped down to it (direction preserved); a sustained
    rise never exceeds its own neighbourhood's threshold, since the
    neighbourhood rises with it. Both wrists are shifted by the same vector
    so this never fights `enforce_grip_constraint`'s hand-to-hand distance.
    """
    raw = xyz.astype(np.float64, copy=True)
    out = xyz.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < 10:
        return out.astype(np.float32)

    grip_raw = 0.5 * (raw[:, left_idx] + raw[:, right_idx])
    height = -grip_raw[:, 1]  # camera convention: y=down, so -y = up/height
    lo, hi = int(top_window[0] * T), int(top_window[1] * T)
    if hi <= lo:
        return out.astype(np.float32)
    top = lo + int(np.argmax(height[lo:hi]))
    if top >= T - 2:
        return out.astype(np.float32)

    vecs = grip_raw[top + 1:] - grip_raw[top:-1]  # vecs[i] = grip[top+i+1] - grip[top+i]
    speeds = np.linalg.norm(vecs, axis=1)
    n = len(speeds)
    half = spike_window // 2
    corrected = vecs.copy()

    for i in range(n):
        neighbor_idx = [j for j in range(max(0, i - half), min(n, i + half + 1)) if j != i]
        if len(neighbor_idx) < 2:
            continue
        neighbor_speeds = speeds[neighbor_idx]
        med = float(np.median(neighbor_speeds))
        mad = float(np.median(np.abs(neighbor_speeds - med))) * 1.4826
        thresh = med + spike_k * max(mad, min_speed_mad)
        if speeds[i] > thresh and speeds[i] > 1e-9:
            corrected[i] = vecs[i] * (thresh / speeds[i])

    grip_out = grip_raw.copy()
    for i in range(n):
        grip_out[top + 1 + i] = grip_out[top + i] + corrected[i]

    delta = grip_out[top:] - grip_raw[top:]
    out[top:, left_idx] = raw[top:, left_idx] + delta
    out[top:, right_idx] = raw[top:, right_idx] + delta
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 9. Arm-pose freeze during proven-still holds
# ---------------------------------------------------------------------------

def freeze_arm_pose_during_holds(xyz: np.ndarray,
                                 fps: float = 30.0,
                                 top_window: tuple[float, float] = (0.1, 0.9),
                                 still_speed: float = 0.008,
                                 min_hold_frames: int = 5,
                                 blend_frames: int = 5,
                                 left_shoulder_idx: int = H36M17_IDX["left_shoulder"],
                                 right_shoulder_idx: int = H36M17_IDX["right_shoulder"],
                                 left_wrist_idx: int = H36M17_IDX["left_wrist"],
                                 right_wrist_idx: int = H36M17_IDX["right_wrist"],
                                 segments: tuple = (
                                     (H36M17_IDX["left_shoulder"], H36M17_IDX["left_elbow"]),
                                     (H36M17_IDX["left_elbow"], H36M17_IDX["left_wrist"]),
                                     (H36M17_IDX["right_shoulder"], H36M17_IDX["right_elbow"]),
                                     (H36M17_IDX["right_elbow"], H36M17_IDX["right_wrist"]),
                                 )) -> np.ndarray:
    """Freeze the arms to a single held pose wherever the torso proves they
    should be still (a finish hold, address, any pause) - hands "winging
    around" when the body itself isn't moving - without touching bone length.

    A first attempt at this (re-filtering just the wrist's direction around
    the elbow, adaptively, everywhere) made no visible difference. Tracing
    the actual failure on this project's own real footage showed why: the
    shoulder midpoint barely moves at all for most of the back half of both
    clips (frame-to-frame displacement ~0.001-0.02m - the golfer is
    genuinely holding still) while in that SAME window the elbow AND wrist
    both keep jumping 0.05-0.27m per frame, at comparable magnitude to each
    other. That rules out "wrist wobbling around a stable elbow" - the whole
    arm is being mis-reconstructed while the body it's attached to isn't
    moving, the same per-frame-independent depth-ambiguity failure this
    project already diagnosed at the top-of-backswing (see
    `interpolate_gaps_angular`'s docstring), recurring here because the
    finish-hold pose also folds the arms in close to the torso from this
    camera angle. Since the noise is large in amplitude (not small residual
    jitter) and comparable between elbow and wrist, no per-joint adaptive
    filter using that corrupted signal's OWN speed can tell it apart from
    real motion. This instead uses an INDEPENDENT, trustworthy signal - the
    shoulder's own speed - to decide when the arm should be still at all.

    A continuous filter gated by that signal (blend toward a heavily-damped
    reference, weighted by torso speed) was tried first and rejected: over-
    smoothing the reference direction meant it lagged behind real motion, so
    right at a hold's edge - e.g. the brief pause as the swing transitions
    from backswing to downswing, where the TORSO can be nearly still for a
    few frames while the arms are about to move fast - blending toward that
    stale reference produced a bigger, artificial jump than the noise it was
    meant to fix (measured: swing-window peak speed as high as 337% of
    truth). Explicit detect-and-hold avoids that: find contiguous runs of
    at least `min_hold_frames` where shoulder speed stays below
    `still_speed`, and only inside a qualifying run replace each arm
    segment's direction with that run's own mean direction - no reactive
    filtering, so there's no lag to overshoot at the edges (a short
    `blend_frames` crossfade in/out avoids a hard pop instead). To make
    doubly sure a mid-swing torso pause can never trigger a freeze, holds are
    only searched for AFTER the swing's own impact frame - found the same
    way `enforce_follow_through_deceleration` finds it (top-of-backswing =
    max grip height in `top_window`, impact = peak grip speed after that) -
    so this can only ever touch the follow-through/finish, never the swing
    itself. Each segment (shoulder->elbow, then elbow->wrist, per arm - order
    matters, elbow must be finalized before wrist reads it as its own
    proximal joint) is processed as a unit direction off its own already-
    rigid bone length, exactly like `interpolate_gaps_angular` and
    `stabilize_wrist_orientation` before it, so bone length is exactly
    preserved by construction, not just approximately.
    """
    out = xyz.astype(np.float64, copy=True)
    T = out.shape[0]
    if T < min_hold_frames + 2 * blend_frames:
        return out.astype(np.float32)

    grip = 0.5 * (out[:, left_wrist_idx] + out[:, right_wrist_idx])
    height = -grip[:, 1]  # camera convention: y=down, so -y = up/height
    lo, hi = int(top_window[0] * T), int(top_window[1] * T)
    if hi <= lo:
        return out.astype(np.float32)
    top = lo + int(np.argmax(height[lo:hi]))
    if top >= T - 2:
        return out.astype(np.float32)
    speeds = np.linalg.norm(np.diff(grip[top:], axis=0), axis=1)
    impact = top + int(np.argmax(speeds))

    shoulder_mid = 0.5 * (out[:, left_shoulder_idx] + out[:, right_shoulder_idx])
    torso_speed = np.zeros(T)
    torso_speed[1:] = np.linalg.norm(np.diff(shoulder_mid, axis=0), axis=1)
    torso_speed[0] = torso_speed[1] if T > 1 else 0.0
    still = torso_speed < still_speed
    still[:impact] = False  # never freeze before impact, even if the torso pauses mid-swing

    runs = []
    i = impact
    while i < T:
        if still[i]:
            j = i
            while j < T and still[j]:
                j += 1
            if j - i >= min_hold_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return out.astype(np.float32)

    for p_idx, d_idx in segments:
        proximal = out[:, p_idx]
        distal = out[:, d_idx]
        vec = distal - proximal
        length = np.linalg.norm(vec, axis=1, keepdims=True)
        direction = vec / np.clip(length, 1e-6, None)
        target = direction.copy()
        for (a, b) in runs:
            rep = direction[a:b].mean(axis=0)
            rep = rep / np.clip(np.linalg.norm(rep), 1e-6, None)
            target[a:b] = rep
            for k in range(1, blend_frames + 1):
                if a - k >= 0:
                    t = k / (blend_frames + 1)
                    blend = (1.0 - t) * direction[a - k] + t * rep
                    target[a - k] = blend / np.clip(np.linalg.norm(blend), 1e-6, None)
                if b - 1 + k < T:
                    t = k / (blend_frames + 1)
                    blend = (1.0 - t) * direction[b - 1 + k] + t * rep
                    target[b - 1 + k] = blend / np.clip(np.linalg.norm(blend), 1e-6, None)
        out[:, d_idx] = proximal + target * length
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def smooth_sequence(xyz: np.ndarray,
                    conf: np.ndarray | None = None,
                    method: str = "oneeuro",
                    fps: float = 30.0,
                    angular_gap_fill: bool = False,
                    bone_lock: bool = True,
                    angle_lock: bool = True,
                    grip_lock: bool = True,
                    plane_lock: bool = True,
                    torso_lock: bool = True,
                    min_conf: float = 0.3,
                    max_gap: int = 5,
                    max_extrapolate: int = 90,
                    lead_factor: float = 1.0,
                    min_cutoff: float = 1.0,
                    beta: float = 0.3,
                    window: int = 7,
                    polyorder: int = 2,
                    min_hinge_angle: float = 15.0,
                    max_hand_distance: float = 0.12,
                    max_offplane: float = 0.08,
                    plane_exclude_multiplier: float = 3.0,
                    plane_max_correction_frac: float = 0.35,
                    min_torso_clearance: float = 0.12,
                    deceleration_lock: bool = True,
                    spike_window: int = 5,
                    spike_k: float = 3.5,
                    min_speed_mad: float = 0.01,
                    arm_freeze: bool = True,
                    arm_freeze_still_speed: float = 0.008,
                    arm_freeze_min_hold_frames: int = 5,
                    arm_freeze_blend_frames: int = 5,
                    lead_arm_lock: bool = True,
                    lead_arm_min_angle: float = 155.0) -> np.ndarray:
    """Smooth a (T, 17, 3) H36M-17 sequence.

    method: "none" | "oneeuro" | "savgol". Gap interpolation runs first when
    `conf` is provided; angular gap-refill for arm joints runs next when
    `angular_gap_fill`; bone-length stabilization runs next when `bone_lock`;
    joint-angle-limit correction runs next when `angle_lock`; the lead-arm
    straightness lock runs next when `lead_arm_lock`; grip stabilization
    runs next when `grip_lock`; swing-plane consistency runs next when
    `plane_lock`; torso-clearance runs next when `torso_lock`; follow-
    through deceleration runs next when `deceleration_lock`; arm-pose
    freezing during proven-still holds runs last when `arm_freeze`.
    """
    if xyz.ndim != 3 or xyz.shape[1:] != (17, 3):
        raise ValueError(f"expected (T,17,3) H36M array, got {xyz.shape}")

    out = interpolate_gaps(xyz, conf, min_conf=min_conf, max_gap=max_gap)
    if angular_gap_fill:
        out = interpolate_gaps_angular(out, conf, min_conf=min_conf,
                                       max_extrapolate=max_extrapolate,
                                       lead_factor=lead_factor)

    if method == "none":
        pass
    elif method == "oneeuro":
        out = one_euro_filter(out, fps=fps, min_cutoff=min_cutoff, beta=beta)
    elif method == "savgol":
        out = savgol_smooth(out, window=window, polyorder=polyorder)
    else:
        raise ValueError(f"unknown smoothing method '{method}'")

    if bone_lock:
        out = enforce_bone_lengths(out)
    if angle_lock:
        out = enforce_joint_angle_limits(out, min_angle_deg=min_hinge_angle)
    if lead_arm_lock:
        # Confidence-filtered real footage: the left (lead, for a right-
        # handed golfer) elbow sits at a near-constant ~159-161 deg through
        # address, then dips to 40-150 deg through the exact windows this
        # session already identified as tracking-degraded, and as low as
        # 9-20 deg in the follow-through tail. A tighter floor than the
        # general anatomical one above forces the lead arm to stay close to
        # straight throughout - "the leading wrist locked to the same angle
        # as the arm" - since it's a one-sided floor (never straightens an
        # already-straighter frame further), not a hard single-value lock.
        out = enforce_joint_angle_limits(out, hinges=(HINGES[0],),
                                         min_angle_deg=lead_arm_min_angle)
    if grip_lock:
        out = enforce_grip_constraint(out, conf=conf, max_distance=max_hand_distance)
    if plane_lock:
        out = enforce_swing_plane(out, max_offplane=max_offplane,
                                  exclude_multiplier=plane_exclude_multiplier,
                                  max_correction_frac=plane_max_correction_frac)
    if torso_lock:
        out = enforce_torso_clearance(out, min_clearance=min_torso_clearance)
    if deceleration_lock:
        out = enforce_follow_through_deceleration(out, spike_window=spike_window,
                                                   spike_k=spike_k,
                                                   min_speed_mad=min_speed_mad)
    if arm_freeze:
        out = freeze_arm_pose_during_holds(out, fps=fps,
                                           still_speed=arm_freeze_still_speed,
                                           min_hold_frames=arm_freeze_min_hold_frames,
                                           blend_frames=arm_freeze_blend_frames)
    return out
