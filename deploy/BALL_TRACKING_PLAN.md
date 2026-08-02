# Ball Tracking — Research & Implementation Plan

> **2026-07-27 accuracy pass (L1–L6) — see "Accuracy pass" at the end of this file.**
> Reported symptoms: hallucinated flights, every ball drawn on a high arc, distance
> always over-estimated. All three reproduced and fixed; `engine` is now `balltrack-v2`.

Goal: a ball-tracking stage that runs in **parallel with the 2D skeleton stage**, followed by a
**3D uplift of the ball trajectory**, wired end-to-end into the deployed pipeline and validated on
`C:\Users\Banjo\Downloads\IMG_3434.MOV` and `C:\Users\Banjo\Downloads\IMG_8107.MOV`.

Loop state: **I6 COMPLETE, I7 (cloud E2E) IN PROGRESS** — image v12 building (v11 was missing
Scripts/adapters/ in the source zip). NOTE 2026-07-18: a git merge in the working tree
reverted uncommitted edits to 6 tracked files; all were restored from the built deploy
artifacts (proc-src.zip, chat-lambda package, S3 web copies). index.html/portal.html are in
merge conflict (teammate's ballflight.js work) — left untouched for the user to resolve;
their script tags will need the ?v=20260718a bump re-applied after resolution.

## Iteration log

### R1 (2026-07-17) — pipeline map + model survey
**Pipeline integration points (from codebase exploration):**
- Stage sequencer: `deploy/processing/processing_handler.py:_pipeline` (line 111) shells out to
  `Scripts/*.py` subprocesses; local mirror is `Scripts/demo.py` (`python Scripts/demo.py video.mov`,
  defaults `mediapipe_lite` + `golfpose3d`).
- 2D pose stage: `Scripts/adapters/mediapipe_adapter.py` via `BaseAdapter` ABC
  (`Scripts/eval_utils.py:216`) — `predict(video) -> DataFrame[frame,kp_idx,kp_name,x,y,conf]` in
  pixel coords, free parquet caching via `predict_and_cache()`. Registry:
  `Scripts/adapters/__init__.py:69`. **A ball detector fits this contract as a 1-keypoint adapter.**
- Frames are re-decoded per stage (`iter_frames`, `eval_utils.py:278`) — parallel ball stage can
  simply re-decode; no shared-decode refactor needed for v1.
- 3D frame: lifter output is hip-relative, unit-normalized, `h36m-camera` axes; **no camera
  intrinsics exist; absolute meters NOT recoverable**. Floor/leveling: `smoothing.level_and_ground`
  (`Scripts/smoothing.py:208`) — need to expose its rotation R to place ball in same frame.
  Body-scale (from `coaching_indicators.py`) is the only pixel→physical anchor.
- Sim ball flight to feed/replace: `Scripts/ball_flight.py` (`_ui_trajectory` at line 350),
  chat tool `estimate_ball_flight` (`Scripts/coaching_chat.py:236-294`), UI arc
  `deploy/web/chat.js:495 flightArc`.
- Artifacts: emit `<stem>_ball_3d.json`; cloud list `ARTIFACTS` at `processing_handler.py:50`.
- No existing ball/vision-detection code anywhere (ultralytics 8.4.53 already in local .venv;
  cloud image is CPU torch 2.4.1, numpy<2).

**Model candidates (initial survey):**
1. **WASB-SBDT** (BMVC2023, nttcom) — multi-sport ball detect+track strong baseline, pretrained
   weights in MODEL_ZOO (tennis/badminton/soccer etc., no golf) — https://github.com/nttcom/WASB-SBDT
2. **TrackNet family** (V1 tennis; V3 shuttlecock qaz812345/TrackNetV3; V4 motion-attention,
   arXiv 2409.14543) — heatmap nets on consecutive frames, built for tiny fast blurred balls.
3. **YOLO-based golf ball detection** — lit: Faster R-CNN vs YOLOv3/tiny (0.78 mAP, 357 fps,
   researchgate 347630649). Ultralytics already installed; Roboflow-style golf-ball datasets +
   nano model fine-tune is the low-lift trainable path.
4. **Classical motion baseline** — stabilize (phaseCorrelate) + frame-diff + physics-gated blob
   linking. Proven manually on IMG_3434.MOV (ball tracked frames 290–314, launch direction
   measurable). Zero deps, CPU-cheap — strong v1 / fallback candidate consistent with the
   pipeline's dependency-light ethos.

**3D uplift thought (to develop in R3+):** monocular 3D of a golf ball is ill-posed (ball too
small for size cue). Practical path: fit a physics trajectory (projectile + drag/Magnus from
`ball_flight.py`) whose camera projection matches the observed 2D track — i.e., 3D uplift =
constrained physics fit, anchored by body-scale + impact position, not a learned lifter.

### R2 (2026-07-17) — video ground truth + ball visibility ✅
- Ran `Scripts/demo.py` on both videos; bundles in `Data/demo/IMG_3434/` and `Data/demo/IMG_8107/`.
- **BUG found (fix in I-phase):** IMG_3434's 3D cache landed in
  `Data/eval_runs/golfpose3d_from_mediapipe_lite_lrfix/` (L/R repair fired) but `demo.py`'s
  scorecard-step cache lookup only checks the non-lrfix dir → scorecard skipped for IMG_3434.
  Cloud handler (`processing_handler.py:124-130`) prefers lrfix correctly; demo.py doesn't.
- **Ball visibility (both 720x1280@30fps, Topgolf, down-the-line):**
  - IMG_3434: impact f287; ball visible f290–314 (~0.8s, 24 frames), 3–6 px dot, drifts right,
    exits right frame edge near apex.
  - IMG_8107: impact f130 (event detector agrees — events: address 91 … impact 130); ball
    visible ~f131–147 (~0.5s), 3–5 px, climbs up-right, lost in sky/pole clutter after f147.
- **Detector implications:**
  - Ball is a 3–6 px backlit dot → generic YOLO at 640-px input sees a ~2 px target: hopeless
    without tiling. Need native-res inference (tiles/crops) or heatmap multi-frame nets
    (TrackNet/WASB) designed for tiny objects.
  - Stabilized frame-diff + blob linking works (verified manually on both clips) but needs
    physics-gated track initiation to reject clutter (flags, other bays' balls, birds).
  - **Synergy with existing stages:** event detector supplies the impact frame; pose supplies
    wrist/clubhead region at impact → constrain track initiation to that time+place. This is
    the key advantage over generic trackers.
  - Only ~15–25 frames of flight are observable → enough for launch direction + initial pixel
    speed, NOT for carry. Carry requires physics extrapolation (→ R3 physics-fit uplift).
### R3 (2026-07-17) — 3D uplift design (physics-fit) ✅
**Core idea: 3D uplift = constrained physics fit, not a learned lifter.** `simulate_flight()`
(`Scripts/ball_flight.py:157`) is already a full forward model — params
(ball_speed_mph, launch_angle_deg, azimuth_deg, backspin, sidespin) → 3D track
`[downrange_yd, height_yd, side_yd]` sampled over time. Fit: project the simulated 3D track
through a camera model and minimize pixel reprojection error against the observed 2D ball
track over the ~15–25 visible frames (t_i = (frame − impact_frame)/fps).

**Camera model (v1):** static pinhole (stabilization removes shake), principal point = image
center, focal length f_px from typical phone FOV prior (portrait 720x1280, main lens ≈ 26 mm
equiv) — treat f as fixed with a sensitivity band, or a 4th fit param with tight prior.
Camera yaw vs target line is unobservable → report azimuth RELATIVE TO CAMERA AXIS and label
it so (down-the-line assumption). No intrinsics exist anywhere in the repo (R1) — this is new.

**Identifiability:** early flight (~0.5 s) has little spin curvature → spin params stay at
club-default priors (envelope-gated, same philosophy as Phys-NN); fit only
(ball_speed, launch_angle, azimuth). ~20 obs / 3 params + physics priors = tractable.
Optimizer: tiny hand-rolled Nelder-Mead (repo is dependency-light; ball_flight is stdlib;
avoid scipy unless R6 shows it's already in the cloud image).

**Scale anchor:** ball diameter (42.7 mm, 3–6 px) is too small to use. Anchor = golfer:
pixel torso length at address ÷ assumed torso meters (~0.50 m; user height later) → depth of
launch point; launch origin = ball/tee position at impact (from pose feet/club region +
event detector impact frame). Carry scales linearly with this anchor → publish an
uncertainty band, not a point estimate.

**Output artifact `<stem>_ball_3d.json` (draft schema):**
- `track_2d`: [[frame, x_px, y_px, conf], …] (raw measured)
- `launch_measured`: {elevation_deg, azimuth_deg_rel_camera, n_frames, residual_px, quality}
- `fit`: {ball_speed_mph, launch_angle_deg, azimuth_deg, spin_assumed:{...}, carry_yd,
  side_yd, apex_yd, flight_time_s, engine:"physfit-v1", confidence, carry_range_yd:[lo,hi]}
- `trajectory_world`: simulate_flight trajectory points (feeds/replaces `_ui_trajectory` arc)
- `trajectory_replay`: ball points mapped into the replay frame — world yards → replay units
  via (pose torso units / torso meters), then apply the same leveling rotation + floor shift
  as the skeleton. **Refactor needed:** `level_and_ground` (`Scripts/smoothing.py:208`)
  returns only an info dict — expose R + shift in `info` so the ball step can reuse them.

**Renderable:** add optional `ball` key to `replay_3d.json` (viewers draw polyline+dot;
`grounded/floor_y` already in schema); chat `flightArc` (`deploy/web/chat.js:495`) renders the
measured arc with a "measured" badge vs today's "simulated".

**Fallback:** if no confident track (occlusion, face-on camera, indoor net), stage emits
`quality:"none"` and chat keeps current simulate-only behavior. Validation targets for the two
clips: IMG_3434 → azimuth clearly right-of-camera-axis (matches observed push/fade);
IMG_8107 → up-and-right, plausible Topgolf carry (150–230 restricted-flight).
### R4 (2026-07-17) — detector decision ✅
**Decision: v1 detector = classical stabilized frame-diff + physics-gated track linking.**
Rationale matrix:
| Candidate | Fit | Verdict |
|---|---|---|
| Classical diff+linking | Proven manually on BOTH clips; zero new deps (cv2 only); CPU-ms cheap; exploits impact-frame + launch-origin priors no generic model gets | **v1** |
| YOLO fine-tune | Roboflow golf datasets are small (100–185 imgs, mostly close-ups — not 3–6 px in-flight dots); ultralytics NOT in cloud image (new heavy dep); 640px input kills tiny targets without tiling | upgrade path |
| WASB / TrackNet | No golf weights; torch port + fine-tune data needed; strongest long-term for cluttered/low-contrast cases | upgrade path (v2 if v1 recall disappoints) |

Datasets noted for the upgrade path: Roboflow Universe "golf ball detection" (appleroot,
185 imgs), "golf ball tracker" (StandardData, 100 imgs), search `class:golfball`.

**v1 algorithm (BallDetector2D):**
1. Read impact_frame from scorecard events (or estimate from motion burst if absent);
   launch origin ≈ lowest wrist/club region at impact from 2D pose (already cached parquet).
2. Stabilize frames impact→+2.5 s vs impact frame via `cv2.phaseCorrelate` (proven here).
3. Consecutive-frame abs-diff, threshold, connected components ≤ ~30 px wide.
4. Track initiation: candidate chain must start within a cone from launch origin within
   ~5 frames of impact; growth by nearest-neighbor gated on smooth pixel velocity
   (monotone rise early, bounded accel) — rejects flags/birds/other bays' balls.
5. Output per-frame (x, y, conf) + quality (n_frames, gaps, residual to smooth poly).
**Eval metric (both clips):** manual GT tracks from today's analysis (IMG_3434 f290–314,
IMG_8107 f137–147 verified points) — require ≥80% of GT frames within 4 px, zero false
tracks; runtime < 2 s/clip CPU. Bake-off vs deep models deferred until v1 measured.
### R5 (2026-07-17) — prototype WORKS on both clips ✅
Prototype: scratchpad `proto_ball_track.py` (session 8a3e149f…; port to
`Scripts/adapters/ball_detector.py` in I1). Results vs manual GT:
- IMG_3434: **10/11 GT frames within 8 px**, 20-pt track f290–309 (real ball, club rejected)
- IMG_8107: **5/5 within 8 px**, 8-pt track f137–145; zero false tracks on either clip
- Runtime ~5 s/clip CPU (bar was 2 s — optimize by estimating phaseCorrelate shift on
  downscaled frames in I-phase; not blocking).

**The discriminator stack that made it work (each was added to kill a real observed failure):**
1. Foreground = median-background diff ∧ consecutive-frame diff (kills static clutter AND
   double-image artifacts; consecutive-diff alone produces ghost pairs).
2. Masks: golfer bbox from cached 2D pose (+70 px), below-horizon (0.62H), roof (0.06H),
   stabilization border bands (12 px — 30 px ate the ball's exit path).
3. Initiation: impact+1..impact+9 only (neighbor-bay tracks start late), first step upward,
   ≥4 px/frame, decelerating over first 8 pts.
4. Straightness: line-fit RMS ≤5 px over first 8 pts (kills club arcs, body parts).
5. Launch cone: origin→start direction within 40° of initial track velocity.
6. Anti-causality: back-extrapolate launch velocity into impact−6..impact−1; ≥2 moving-blob
   matches ⇒ neighbor bay's ball already in flight ⇒ reject. Exempt positions within 220 px
   of origin (that's the pre-impact club, not a neighbor ball).
7. Ballistic truncation: cut track at 3 consecutive near-zero steps (stall) or 3 consecutive
   dominant-axis sign reversals — the clubhead launches ball-like from the origin at impact+1
   but always stalls/reverses at the top of follow-through; a receding ball never does.
8. Score: 10·rms − 5·min(len,18) − 2·speed₀ − 40·cone_dot (min wins).
Launch origin heuristic: (mean pose x, max pose y) at impact — adequate; refine with
clubhead/tee detection later if needed.
### R6 (2026-07-17) — physics-fit uplift prototype WORKS (constrained form) ✅
Prototype: scratchpad `proto_phys_fit.py` (port in I-phase). Camera: pinhole, f_px≈910
(iPhone 26 mm-equiv prior, 720x1280), pitch from horizon line, depth anchored by golfer
standing height (95th-pct pixel span; impact-frame height is crouched and inflates depth).

**Key finding — speed is NOT identifiable monocularly:** the free 3-param fit
(speed, launch, azimuth) trades speed against depth/f: ±15% focal prior → 89–199 mph.
Fundamental scale ambiguity in a ~0.5 s 2D track; this is why launch monitors use radar.

**Production formulation — constrained fit:** fix ball speed from the existing envelope
(club default × hand-speed nudge, exactly what `estimate_for_club(speed_scale=…)` does),
fit ONLY (launch_angle, azimuth). Result is stable across ±15% focal prior:
- IMG_3434: launch 26°±3, azimuth +21..27° (right push — matches video), carry 206–212 yd,
  apex ~53 yd, ~7.6 s, resid 7–12 px
- IMG_8107: launch 10.4°±1 (flat), azimuth +17..21°, carry 196–202 yd, apex ~19 yd, ~5.3 s,
  resid 4–7 px
So: **measured = track + launch angle + azimuth (bands); simulated-but-informed = carry/apex
via envelope speed + measured shape; report carry as a band.** Azimuth is relative to camera
axis (labeled). scipy Nelder-Mead used locally — R7 must check the cloud image for scipy,
else hand-roll (~30 lines).
### R7 (2026-07-17) — speed IS extractable with subpixel + enough points ✅
(user direction: extract speed/launch confidently and estimate distance + other features)
Experiment: subpixel centroid refinement (intensity-weighted, median-diff patch) + bootstrap
CI (n=14, resampled points × ±8% focal jitter) on the FREE 3-param fit:
- IMG_3434, 20 pts: **speed 150–160 mph, launch 24.0–26.5°, azim 22–25°, carry 237–257 yd**
  — tight. Gravity curvature over 20 subpixel points pins the scale. (Chatbot's hand-speed
  sim guessed 241 — inside the band.)
- IMG_8107, 8 pts: speed 119–159 mph — too wide. **Track length is the gate.**
**Production rule:** ≥15 subpixel points → free fit, report speed/launch/azim/carry as
measured WITH CI band; <15 points → constrained fit (envelope speed, measured shape only),
label carry "estimated". Quality tiers: `measured` / `partial` / `simulated`.
**To widen the ≥15-pt regime (I-phase):** extend the chase past current cutoffs with
predictive search + threshold decay (8107's ball is visible past f147); more points also
narrow the CI. **To cut anchor bias:** user profile height (DB/product), iPhone focal-length
metadata from the .MOV container if present (probe in I-phase), mat-size scale reference.
**Other features to emit:** apex band, hang time, azimuth (push/pull direction), smash-factor
cross-check vs hand_speed indicator, sidespin/curvature fit once tracks are long enough
(needs the longer-chase work), descent angle at landing.
### R8 (pending) — consolidate: cloud-image constraints (scipy present? runtime budget,
numpy<2), artifact/DB wiring, chat tool + UI shape, implementation spec + I1–I7 task list.

### R8 ✅ — consolidation: NO scipy in cloud image (hand-rolled Nelder-Mead); web deploy =
s3 cp + CloudFront invalidation (E1OPRZIXL2IZX); chat Lambda zip via build_lambda_zip.sh;
processing image via CodeBuild loop (memory: aws-capstone-account).

## Implement→test→deploy log
### I1 ✅ — Scripts/ball_track.py: subpixel tracker + guarded predictive extension +
hand-rolled NM physics fit + launch_dt_s 4th param + 85% trimmed robust loss + quality tiers
(measured ≥15 pts / partial ≥8 / simulated) + bootstrap CI. demo.py _lrfix cache bug fixed.
Scripts/test_ball_track.py: both clips PASS (3434 measured n=16 resid 2.3 px: speed 164,
launch 26.5°, azim +23.5°, carry 259 [CI 261–271]; 8107 partial n=9: launch 6.9°,
azim +17.9°, envelope carry 184).
### I2 ✅ — Scripts/ball_step.py: ball_3d.json + trajectory_replay (mid-ankle anchor,
standing-height scale) + ball injected into replay_3d.json + overlay.mp4 re-encoded with
ball dot/trail (visually verified). Wired into demo.py + processing_handler (_pipeline,
non-fatal; ball_3d.json in ARTIFACTS, not REQUIRED).
### I3 ✅ — get_ball_flight chat tool (measured/partial/simulated + how_to_phrase per tier,
confident measured phrasing per user), SwingContext.ball, system-prompt BALL FLIGHT section,
verifier accepts tool's range phrasing; chat_handler job_ball_path (03_outputs fetch) +
ball_path_for (demo); build_lambda_zip.sh bundles ball_3d.json.
### I4 ✅ — replay3d.js: true-scale measured arc + animated tracer ball (far-plane fix:
was 100 m, clipped the arc; controls/ground scale with extent); chat.js arc label
quality-aware ("measured from your video").
### I5 ✅ — local E2E green: clean demo.py runs on both videos, all artifacts present;
scripted-backend chat turn grounded=True with confident measured answer; regression PASS.
### I6 ✅ — cloud deploys: image v11→v12 (v11 lacked Scripts/adapters/ — zip must include
Scripts RECURSIVELY), motion-caddie-processing → v12, motion-caddie-chat zip updated,
chat.js/replay3d.js to S3 (no-cache) + invalidation. Gotcha: git-bash mangles /paths in
CloudFront/logs CLI args — use PowerShell.
### I7 ✅ — cloud E2E VERIFIED on image v12 (2026-07-18)
- Jobs f0b3cebf7564… (IMG_3434) + 189323e45d67… (IMG_8107) re-triggered via S3
  copy-in-place after the v11 adapters-import failure; both produced the FULL artifact set
  incl. ball_3d.json in 03_outputs/.
- Cloud fits match local: 3434 measured (163.3 mph, launch 26.4°, azim +23.6°, carry 258);
  8107 partial (launch 7.1°, azim +18.1°, envelope carry 185).
- **Live chat (Bedrock Lambda) verified, grounded=True, zero violations:**
  - measured: "Your carry was about **258 yards**, measured from your video…"
  - partial: measured direction (+ camera-line caveat) with informed-estimate distance.

## Final phase status
- Deploys + e2e retest + LLM ball tool: ✅ DONE (I6/I7 — live and verified).
- **Git commit: BLOCKED on the user's in-progress merge** (index.html/portal.html in
  conflict with the incoming ballflight.js work). After the merge is resolved: commit
  Scripts/{ball_track,ball_step,test_ball_track}.py, Scripts/{demo,coaching_chat}.py,
  deploy/processing/processing_handler.py, deploy/chat_handler.py,
  deploy/chat/build_lambda_zip.sh, deploy/web/{chat.js,replay3d.js}, this plan; re-apply
  the ?v=20260718a script-tag bump to the resolved index.html/portal.html and re-upload
  them. Dev helpers to EXCLUDE from the commit: deploy/web/_ball_test.html,
  _balltest_replay.json, Scripts/{ball_track_proto,ball_fit_proto}.py (research artifacts —
  keep or drop at user's discretion).

## Final phase (user-added 2026-07-17, after I1–I7)
- Commit work to git.
- Deploy to BOTH the demo site and the dev portal (`/portal.html`) pages.
- Re-test everything end to end after deploy.
- **LLM tool access**: measured ball-tracking/estimation data must be exposed to the coaching
  LLM as a tool — extend/replace `estimate_ball_flight` in `Scripts/coaching_chat.py` (TOOLS
  registry line ~294) so chat can fetch the tracked trajectory + measured launch direction for
  a processed job, with grounding rules distinguishing measured vs simulated values.
- **Confident distance answers (user, 2026-07-17):** when ball_3d quality is `measured`, the
  chat tool returns measured speed/launch/azimuth + carry with its CI band, and the tool's
  display strings/grounding should let the model answer distance questions CONFIDENTLY
  ("your carry was about 245 yards, measured from the video") instead of today's heavy
  "simulated estimate" hedging — reserve the hedge for `partial` (direction measured, speed
  assumed) and `simulated` (no track) tiers. Tier + phrasing guidance must be explicit in the
  tool result so the grounding verifier accepts the confident phrasing.

## Accuracy pass (2026-07-27) — L1–L6

User report: "the model is hallucinating swings, always assumes a high arch when
sometimes the ball goes straight/flatter, and it over-estimates distance every time."
All three reproduced on real clips. The validation set gained a third clip,
`IMG_8110.MOV` (same venue, no usable ball flight), as a **negative** case.

### L1 — investigation (root causes, all measured rather than guessed)
1. **Camera-pitch sign inverted.** `_cam_pts` used `atan((cy - horizon_y)/f)`. With the
   camera pitched up by t the horizon projects to `cy + f*tan(t)`, so the sign was
   backwards and the fit absorbed the error into launch angle — **launch inflated by
   ~2x the camera pitch**. Since `estimate_horizon_y` is clamped to `0.2H..0.62H`
   (mostly above centre) the bias was almost always *upward*: hence "always a high
   arch". IMG_3434 came out at launch 26.5 deg, apex 80 yd, 9.2 s hang time —
   physically absurd for a driver.
2. **Optimizer landing in local minima.** A single joint Nelder-Mead from one coarse
   start returned 119 mph on *noise-free* synthetic data whose truth was 150 mph.
3. **Ball speed is barely identifiable** from a ~0.6 s track. Profiling the loss over
   speed, everything from 130-210 mph sat within 1 px of the best fit (carry 200-360
   yd). The old bootstrap resampled points and jittered focal length but never moved
   along that flat direction, so it reported +-5 yd on a quantity good to about +-60 —
   and the point estimate fell *outside* its own CI.
4. **No plausibility prior.** An unpriored free fit walks to the top of the speed grid,
   reporting tour-pro carries (259, and after the geometry fix 317 yd) for every swing.
   This is the distance over-estimate.
5. **No fit-quality gate.** Any chain passing the 2D filters became a confident
   "measured" flight. IMG_8110 produced a 30.2 deg launch / 62 yd apex / 204 yd carry
   from an object whose x never moved (444.9 -> 444.7 over 12 frames) while y decayed
   exponentially — a stationary rising object, not a ball.
6. **UI arc normalised x and y independently**, stretching every trajectory to fill the
   SVG box: a flat 190 yd drive and a 60 yd apex wedge drew the *identical* rainbow.
   The arc's shape carried no information at all.

### L2 — geometry, optimizer, honest uncertainty
- Pitch computed with the correct sign, clamped to +-25 deg.
- `profile_speed` sweeps the flat direction explicitly; `fit_at_speed` is multi-start;
  `_project(fast=True)` truncates the simulation to the observed window (~10x cheaper),
  which pays for the extra search.
- A soft log-normal prior ties ball speed to the club envelope (sigma 15%, 1.5 px per
  sigma), regularising the flat direction instead of letting it run away.
- `confidence_band` replaces `bootstrap`: every solution within 1 px of the best, over
  both the speed profile and a +-10% depth-anchor jitter.
- **The tier decision uses the PIXEL-ONLY band** while the reported band is the
  posterior — otherwise a tight prior talks you into calling a speed "measured" that
  the video never pinned down.
- The synthetic round-trip (an independent, first-principles projector) now recovers
  truth to +-0.1 mph / +-0.0 deg at every camera pitch, and is *invariant* to pitch as
  the geometry demands (spread 0.02 deg across -12..+12 deg).

### L3 — preprocessing (measured, not assumed)
Kept: sequential decode (per-frame `POS_FRAMES` seeking is slow and unreliable on phone
HEVC), Hanning-windowed phase correlation, and CLAHE — but CLAHE only as a *second
pass*, because it is not a global win: it rescues a ball lost in bright clutter
(IMG_8107: 8 -> 15 points) and loses one that was already high-contrast in a dark bay
(IMG_3434). Both passes feed a single candidate pool.

**Rejected after measurement** (12-config matrix across the 3 clips): motion-streak blob
gating (area<=220 with a fill-ratio test) and a static-clutter occupancy mask. Neither
improved any clip; the extra candidates were clubhead and netting chains, and on the
negative clip they manufactured fits. Reverted to the R5 blob gate.

### L4 — selection by ballistic fit (what actually kills the hallucinations)
The 2D score cannot separate the ball from the clubhead — both leave the same origin at
the same instant. The tracker now returns a *shortlist* (deduped by point overlap
rather than start pixel) and `analyze` re-ranks it by how well each chain reprojects as
a real flight:
- each chain is scored at full length **and** trimmed (the predictive extension runs
  past the ball and the clutter it picks up lands at the tail, where it does the most
  damage — IMG_3434's real track ran on to f330 and its residual went 2.5 -> 12 px);
- the physical gates are applied **during** ranking, not only to the winner;
- among valid flights, length is traded against residual (1.5 points per px), because
  ranking on mean residual alone rewards whittling a chain down to a fragment;
- the winner's well-conditioned fit is then used to reclaim tail points the
  conservative trim dropped.

Gates: residual <= 4.5 px (real tracks fit at 2.5-3.2; the best chain the negative clip
can offer is 5.6) and |launch_dt| <= 0.12 s (a real ball leaves at impact; the
hallucinated chains need to start 0.19-0.22 s away).

### L5 — the replay arc was tilted by the camera
`trajectory_cam_m` handed the 3D replay **camera-frame** points while the replay is a
*leveled* frame, so the arc inherited the camera tilt: on IMG_3434's 6.8 deg downward
camera the ball "landed" 21 m below the tee. `ball_track` now emits `trajectory_tee_m`
(level, tee at origin) and `ball_step` prefers it. Verified in the viewer: landing y
-18.46 -> -0.05 units, apex/carry ratio 0.140 against a true 0.139.

### L6 — UI honesty and regression tests
- `arcSVG` uses **one scale for both axes**, so the drawing carries the flight's real
  proportions. Verified in-browser: the measured arc draws at 39.3/280 = 0.140 against
  a true 29/209 = 0.139, and the measured vs what-if arcs now differ in height
  (39.3 vs 34.8) where before both were pinned to the full box height.
- The carry range is shown on both tiers in the card and returned to the chat model on
  both tiers (the grounding verifier only lets it quote numbers the tool returned).
- `Scripts/test_ball_track.py` rewritten: synthetic round-trip + pitch-invariance + a
  flat-launch case (a 7 deg drive must not come back lofted) + the three clips
  including the negative one. The synthetic half needs no video and always runs.

### Results on the validation clips
| clip | before | after |
|---|---|---|
| IMG_3434 | measured, launch 26.5 deg, apex 80 yd, carry 259 (CI 261-271), 9.2 s | partial, launch 15.1 deg, azim +21.9 deg, apex 29 yd, carry 209 (203-245), 6.2 s |
| IMG_8107 | partial, launch 6.9 deg, carry 184 | partial, launch 9.2 deg, azim +18.2 deg, apex 17 yd, carry 195 (187-232) |
| IMG_8110 | **partial, launch 30.2 deg, apex 62 yd, carry 204** (hallucinated) | **simulated** — rejected, "fitted launch is 0.22s from impact" |

Both real clips now land on `partial`: launch angle and start direction are measured,
distance is an envelope-anchored estimate **with a published range**. That is the
honest reading of a single 30 fps camera. 3434's flatter 15.1 deg / 29 yd apex and
8107's 9.2 deg / 17 yd apex are now clearly different shapes — exactly the "sometimes
it goes straight or flatter" case that used to be flattened into one high arc.
`measured` (confident speed and carry) is still reachable, but now requires the pixels
themselves to constrain the scale.

### Not done / follow-ups
- **Not deployed.** The cloud image and Lambda still run the old code; shipping needs
  the CodeBuild image loop, a chat-Lambda rezip, and `s3 cp` + CloudFront invalidation
  (see I6).
- The thresholds (4.5 px, 0.12 s, the 1.5 length/residual trade) are calibrated on three
  clips from one venue. Worth re-checking against a wider set before trusting them
  broadly.
- The `measured` tier is now rarely reached. Earning it back is a detection problem:
  longer tracks condition the scale better (n=30 visibly sharpens the speed profile).
  A learned tiny-object detector (TrackNet/WASB, R4's upgrade path) is the real answer.
