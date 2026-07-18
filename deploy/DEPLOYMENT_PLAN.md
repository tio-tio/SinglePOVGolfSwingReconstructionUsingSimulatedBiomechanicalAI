# MotionCaddie — Front-End Deployment Plan & Handoff

**Owner today:** Austin · **Proposed next owner:** Theo
**Status:** static demo + coach chat + **prototype** upload and account/consent flows working locally; S3 hosting + real backend not started
**Last updated:** 2026-07-05

---

## 2026-07-05 — Front-end reconciliation (Austin's demo ↔ Banjot's pushed sources)

Banjot pushed editable HTML sources from the same design lineage as this
`deploy/web/` demo (`coaching_chatbot_demo.html`, `serve_demo.py`,
`coaching_llm_features.html`, `Checkpoint 2/demo|interactive/`, and a chat backend
`deploy/chat_handler.py`). Rather than blindly merge, we reconciled: keep
`deploy/web/` as the shell and fold in only his **genuinely-new** capability.

**Done today (in `deploy/web/`):**
- **Added the coaching Q&A chat** as an "Ask the coach" tab on the Results screen
  (`chat.js` + markup in `index.html` + styles in `styles.css`). Engine ported from
  Banjot's `coaching_chatbot_demo.html`: grounded answers, refusal taxonomy
  (unmeasured / low-confidence / no-fixes), tool-call trace, grounding verifier,
  and cross-swing compare.
- **Unified the data contract:** the chat reads each swing's values from the SAME
  `assets/<id>/metrics.json` the "numbers" tab renders, so chat and table can't
  disagree. No new per-clip data files, no fabricated numbers.
- **Preserved all architecture constraints:** no framework, no build step. Backend
  swap stays a one-function change — `askChat()` in `chat.js` mirrors
  `loadClipBundle()`: offline in-page mock by default; set `window.API_BASE` to
  POST to the live `/chat` agent (`deploy/chat_handler.py`, Lambda container in
  `deploy/chat/`). Verified end-to-end headless (all metric/summary/refusal/compare
  paths, zero console errors).

**Deliberately LEFT OUT (with reasons):**
- **`Checkpoint 2/demo|interactive/1292_preview_3d.html`** — a Three.js 3D viewer
  loaded from the **unpkg CDN** via ES-module importmap. Violates no-framework /
  no-external-dependency. Our dependency-free canvas `Replay3D` already covers 3D
  replay. Kept as Checkpoint collateral only.
- **`Checkpoint 2/coaching_llm_features.html`** — a **dark-themed** static explainer
  (own palette, not this design system). It's a presentation artifact, not an app
  component. Left as slide collateral.
- **`serve_demo.py`** — kept as-is at repo root (Banjot's local live-chat bridge). It
  still serves *its own* `coaching_chatbot_demo.html`, not this `deploy/web/` app;
  wiring this app to its `/api/chat` is a small follow-up (serve `deploy/web/` + set
  `window.API_BASE`) — see below.

**Deferred (not blocking, do when convenient):**
1. **Widen the metric set 9 → full 15.** Banjot's sources carry the complete
   scorecard schema (e.g. `hip_turn_top_deg`, `spine_tilt_impact_deg`,
   `head_lift_max_pct`, knee-flex pair, trail-arm). This demo still shows Austin's
   hand-picked 9. The chat already degrades gracefully to whatever keys a clip has.
2. **Swap in Banjot's real copy/values.** Replace the hand-transcribed
   `metrics.json` labels/blurbs (and clip 0's values) with the real strings from his
   indicator dictionary / `1292_scorecard.json` to retire the reverse-engineering.
3. **Real live-chat wiring for this app** — point `serve_demo.py` (or a thin static
   server) at `deploy/web/` and set `window.API_BASE`, or stand up the chat Lambda
   Function URL. Needs per-clip scorecards where `chat_handler.py` expects them
   (`Data/demo/<id>/<id>_scorecard.json`).
4. **Fold `Checkpoint 2/demo/1292_*` in as a real-data clip** through the normal
   `assets/` pipeline (it's real pipeline output, same shape family).

---

## 2026-07-05 — Upload flow, prototype accounts + consent, visual polish

Reworked `deploy/web/` from a "pick a demo clip" demo into something that reads like
a product: a landing page with a real upload path, a prototype account + consent
flow, and a cleaner visual pass. **Everything added here is a front-end illusion —
no backend, no real auth, nothing leaves the browser.** New files: `auth.js`
(account + consent logic + the single `CONSENT_COPY` wording object) and
`privacy.html` (Data & Privacy page rendered from `CONSENT_COPY`).

**What shipped:**
- **Upload flow (prototype).** Landing leads with a primary **"Upload swing video"**
  CTA + a secondary **"Demo swings"** section (the three real pre-rendered clips).
  Picking a file shows its name, runs the *same* 7-step Analyze loader, then shows a
  results view that **plays back the user's actual file locally** (a `blob:` object
  URL — truthful, it's their raw clip) behind a persistent banner: *"Prototype. No
  real 3D analysis ran on your uploaded video — the metrics, 3D replay, and coach
  below are illustrative demo values."* The metrics/plain/coach/3D are a clearly
  labelled sample bundle. **No file is uploaded anywhere.**
- **Prototype account + consent.** Sign in / Create account as modals. Stores only
  safe profile metadata in `localStorage` (`displayName`, `username`,
  `consentGiven`, `modelUseConsent`, `consentTimestamp`, `consentVersion`); the
  password field is demo-only and its value is **never stored**. Consent is **two
  separate checkboxes**: a required, unchecked-by-default **analysis-use** consent
  that gates account creation, and a genuinely optional, off-by-default
  **model-improvement** opt-in. Both use honest "may"-wording. Signed-in state shows
  a **My swings** dashboard + working Sign out. All consent/privacy wording lives in
  one place: `CONSENT_COPY` in `auth.js`.
- **Polish.** Removed all UI emojis except the ⛳ logo (icons are inline SVG now),
  reframed samples as "Demo swings", tightened copy to a clean product tone.

**Prototype-only — what is NOT real yet:** uploaded-video analysis (no pipeline
runs), accounts (no server, no real auth, password never stored), and consent
enforcement (captured locally, honored by nobody yet because nothing is uploaded).
The honesty banner + the "prototype account only" microcopy + the privacy page's
"nothing is uploaded" wording are all true *today* and MUST stay until the backend
below actually exists.

---

## 2026-07-05 — 3D Swing View (headless-Blender render)

Added a **"3D Swing View"** tab on the results screen showing a precomputed Blender
render of the reconstructed 3D pose (three swing-phase stills: takeaway / mid /
follow-through). `Scripts/blender_mocap.py` turns the canonical mocap JSON into a
metric, unisex-scaled, constraint-rigged armature and renders headless.

**Body proxy — stylized capsule/tube mannequin (updated 2026-07-05):** the render
geometry is a neutral matte **capsule mannequin** — tapered rounded limb capsules, a
solid torso column, and rounded joint blobs (was ball-and-stick spheres+cylinders).
Only the *drawn geometry* in `build_in_blender`'s `if add_mesh:` block changed; the
numpy core, axis conversion, metric scaling, and FK solve are untouched. It's
deliberately stylized (single colour, no face/hands/clothing) — an artist's mannequin,
not anatomy. Because a solid body implies proportions that sticks didn't, the caption
now states the body is a **neutral template shape, not the golfer's actual build**.
Documented future upgrade if this ever goes production: a skinned mesh / **SMPL-X**
gender-neutral mannequin (see `BLENDER_HANDOFF_RESEARCH.md` upgrade paths) — not in
scope for the demo.

**How it works (offline asset step — the whole point):** Blender is an **offline /
headless asset-generation step**, never in the browser and never required by the
static site. Flow: `mocap JSON → headless Blender (Scripts/blender_mocap.py --mode
unisex --render) → compressed PNGs committed under deploy/web/assets/<id>/blender/ →
the static front end just displays them`. The tab is **presence-driven**: it appears
only for clips whose `clips.json` entry carries a `render.phases` list, so it lights
up automatically as future clips get renders and stays hidden otherwise (uploads
never show it). Generated with Blender 5.1; PNGs downscaled to 900px and pngquant'd
(~48 KB for all three). `.blend` files are **not** committed.

**Current-state note — interim source (be honest):** the render shipping now is
generated from the **MotionBERT-full lift** of clip 0, **NOT** the golfpose3d/MixSTE
production lifter (that's the only mocap JSON that exists today). The front-end
caption is deliberately **lifter-neutral** ("reconstructed 3D swing… reflects captured
motion and joint angles, not the golfer's actual body proportions") so it stays true
after the swap, and it never surfaces the FK `0.0000 mm` validation figure as an
accuracy claim (that number means the render *math* is lossless, not that the pose is
exact). **TODO(mixste-swap):** once the
model bundle lands, regenerate `Data/handoff/0/0_mocap.json` through the golfpose3d
(MixSTE) path and re-run the render command below to **overwrite the PNGs in
`deploy/web/assets/0/blender/` in place** — same filenames, no front-end change. This
TODO is mirrored in `app.js` at `loadRenderView()`.

**Asset status:** only **clip 0** has a mocap JSON (and therefore a render). Clips
**830 / 269 are placeholder swings with no mocap JSON** and are blocked on the model
bundle before they can be rendered — their `clips.json` entries have no `render`
block, so the tab correctly stays hidden for them.

**Regenerate / add a render:**
```bash
mkdir -p deploy/web/assets/0/blender
blender --background --python Scripts/blender_mocap.py -- \
  --input Data/handoff/0/0_mocap.json --height 1.78 --mode unisex \
  --render deploy/web/assets/0/blender/golfer_0_preview.png
# then downscale + compress the 3 _f<NNN> stills (sips -Z 900 + pngquant), drop the
# redundant base PNG, and — for a NEW clip — add a render.phases block to clips.json.
```

---

## 2026-07-05 — Interactive 3D replay: capsule mannequin (two versions)

The results-page **"3D replay — drag to spin"** now renders the reconstructed swing as a
lit, orbitable **capsule/tube mannequin** (matching the Blender stills' look), built
live in the browser. Two implementations ship:

- **Version B — in-app WebGL (primary).** `deploy/web/replay3d.js` (ES module) uses
  **three.js**, vendored locally under `deploy/web/vendor/` (`three.module.js` +
  `OrbitControls.js`, loaded via an importmap — **no CDN**, fully static-hostable). It
  builds tapered capsules + a torso box from the same `assets/<id>/replay_3d.json` joint
  positions (no GLB export), with real lighting, a ground shadow, orbit, and play/scrub.
  This is the one WebGL/framework dependency the project takes on — a deliberate,
  approved exception, kept self-contained (vendored, ~1.3 MB) rather than CDN.
- **Version A — standalone canvas (no dependencies).** `deploy/web/replay_capsule.html`
  is a self-contained page (no framework, no build, no external requests beyond one
  swing-data JSON) drawing the same capsule mannequin with the 2D canvas. Host it
  anywhere; it fetches `?data=<url>` (default `assets/0/replay_3d.json`). This is the
  portable, framework-free version.

`app.js` picks **B when WebGL is available**, else falls back to the original
dependency-free canvas skeleton (`Replay3D`). The WebGL/canvas capsule viewers frame on
the **body joints only**, which also fixes the earlier bug where the extrapolated
clubhead shrank/offset the figure. The geometry recipe (limb radii, joint blobs, torso)
mirrors `Scripts/blender_mocap.py`, so all three views read as the same mannequin.

---

## Going live on S3 — what the upload, accounts, and consent still need

This is the plan to turn the three prototype flows above into real ones once the
site is hosted (Phase 1) and a backend exists (Phases 3–4). The front-end seams are
deliberately localized so each is a contained change, not a rewrite.

### A. Upload → real analysis (extends Phase 3/4)
Today `startUpload()` in `app.js` sets `mode="upload"`, loads the sample bundle, and
plays the file back locally. To go live, that one function becomes:
1. **Gate first:** require signed-in + `consentGiven` before doing anything (see C).
2. **Get a presigned PUT URL** from a tiny `/presign` endpoint (Lambda Function URL
   or API Gateway) → returns a short-lived S3 PUT URL for
   `01_inputs/<userId>/<uploadId>.mp4` plus the `uploadId`.
3. **PUT the file straight to S3** from the browser (not through compute — big video
   shouldn't route through a Lambda). Show real upload progress.
4. **Show the existing 7-step loader as a genuine "processing" state** — real
   analysis isn't instant.
5. **Poll for results:** either poll `03_outputs/<uploadId>/` (presigned GETs) or a
   `/status?uploadId=` endpoint until the bundle exists, then call the *existing*
   `loadClipBundle()` pointed at those URLs. **Remove the prototype banner and the
   illustrative sample** only at this point.

Backend that has to exist for this (Phase 4):
- **`/presign` + `/status`** — tiny CPU Lambdas (or one small API).
- **Pipeline trigger:** S3 `01_inputs/` upload event → an async **Fargate/Batch GPU**
  job for the heavy path (raw video → MediaPipe 2D → 3D lift), then the **CPU Lambda**
  path (event detection → scorecard → Claude eval) → writes `overlay.mp4`,
  `replay_3d.json`, `metrics.json`, `explanation.json` to `03_outputs/<uploadId>/`.
  **Same asset shape the UI already reads**, so `loadClipBundle()` is the only
  front-end swap — exactly as designed.
- **CORS:** the S3 data bucket needs a CORS rule allowing presigned PUT from the
  CloudFront origin; `/presign` and `/status` must return CORS headers.
- **H.264/yuv420p** on any overlay the pipeline renders (browser playback gotcha).

### B. Accounts → real auth
`auth.js` is the single seam. `Auth.createAccount` / `signIn` / `signOut` currently
read/write `localStorage`. To go live, swap those three handlers for a hosted
identity provider (e.g. a Cognito user pool — confirm against Lawrence's account
setup, and note the demo constraint was explicitly *no* auth provider, so this is a
post-demo decision). Then:
- The front end stores the **auth token** the IdP issues, not a bare profile, and
  attaches it to `/presign`, `/status`, and `/chat` calls; the backend authorizes.
- **Password handling becomes the IdP's job** — our code still never sees or stores a
  raw password (keep it that way).
- Keep the UI identical; only the module internals change.

### C. Consent → captured, enforced, honored (the compliance piece)
Right now consent lives only in `localStorage` and binds nothing. Live, it has to be
real:
- **Persist it server-side**, keyed to the user, with the exact `consentVersion` from
  `CONSENT_COPY` so you know *which text* they agreed to.
- **Enforce the split in the pipeline:** the analysis-use consent is required to
  process at all; **`modelUseConsent` decides whether that user's uploaded video and
  derived pose data may be copied into any training/eval dataset.** If it's false, the
  pipeline must exclude their data from anything under `02_working/` /
  dataset-building — tag outputs `no-train` and never route them into a training set.
  This is what keeps the "may … help improve future models" wording honest: it's a
  real opt-out, not decoration.
- **Deletion + withdrawal path** (the privacy page promises these): a control to
  delete a user's uploads + derived data from S3 and any dataset, and to withdraw
  `modelUseConsent` later (which must also purge already-collected training copies).
- **Re-consent on version bump:** if `CONSENT_COPY.version` changes materially,
  prompt existing users to re-agree; the stored version lets you detect who's stale.
- ❗ **Honesty guardrail at cutover:** `privacy.html` currently says *"nothing is
  uploaded."* That sentence and the upload banner become false the moment real
  uploads start. **Update the privacy copy + remove the prototype banner in the same
  change that turns on real uploads — never before, never after.** `CONSENT_COPY` is
  the one place to edit the wording.

### D. Order of operations
Phase 1 (host the static site) can ship the prototype flows as-is — they're honest.
Real uploads (A) must not turn on before consent enforcement (C) and, ideally, real
auth (B) exist, because that's the moment user video actually leaves the browser.

---

## TL;DR for whoever picks this up
There's a working static front end at `deploy/web/` (vanilla HTML/CSS/JS, no
framework). It runs locally today off pre-rendered placeholder data for 3 fixed
clips. This doc explains what it is, how it connects to our AWS setup, and the
plan to take it from "local demo" → "hosted app with real uploads." Read the
"Current state" section first so you don't rebuild what exists.

---

## Current state (what already exists — don't redo)
- `deploy/web/index.html · styles.css · app.js · chat.js · auth.js · privacy.html`
  — editable vanilla front end, no build step. Flow: **Landing (upload CTA + demo
  swings) → Analyze (7-step loader) → Results** (Plain English + The Numbers + Ask
  the coach tabs, 9-metric table, 2D overlay, rotatable 3D replay). Plus a
  **prototype** upload flow, account/consent modals, and a "My swings" dashboard —
  see the two 2026-07-05 sections above for what's real vs prototype-only.
- `deploy/web/assets/<clip_id>/` — per-clip data: `metrics.json`,
  `explanation.json`, `replay_3d.json`, `overlay.mp4`, `raw.mp4`.
- Data is currently **placeholder** (clip 0 uses real MotionBERT 3D output;
  830/269 are labeled variants). All files carry `"placeholder": true`.
- The single backend swap point is `loadClipBundle(clipId)` in `app.js`, marked
  `TODO(live-backend)`. Everything routes through that one function.
- Run locally: `cd deploy/web && python -m http.server 8000` → localhost:8000.
- ⚠️ Video encoding gotcha: assets must be **H.264/yuv420p** or Chrome won't play
  them. Any new renders have to match.

## Our AWS reality (follow Lawrence's setup — this is decided)
- We use **Lawrence's shared account + IAM users** so the team can collaborate.
  Do NOT follow the Identity Center / `motion-caddie-demo-*` naming in the
  `deploy/infra` handoff docs — that's a different, superseded plan.
- Data bucket: `motioncaddie-capstone-data-lj-2026` (region **us-east-1**),
  folders: `01_inputs/ 02_working/ 03_outputs/ 04_docs/ 05_archive/`.
- LLM eval: **Claude API directly (claude-opus-4-8)** for the demo (best results);
  SageMaker + Hugging Face is the eventual production path.
- **Bedrock is blocked** on course credits — don't use it.

---

## The plan — from local demo to hosted app

### Phase 1 — Host the current static demo (no upload yet)
Goal: get the working static site live on the web so anyone can open it.
1. Create a **separate S3 bucket for the web app** (e.g. `motioncaddie-web-<...>`)
   — this is NOT the data bucket; it holds only the site files (html/css/js/assets).
   Confirm naming with Lawrence since it's his account.
2. `aws s3 sync deploy/web/ s3://<web-bucket>/` to upload the site.
3. Put **CloudFront** in front of the web bucket:
   - CloudFront is a CDN — it serves the site fast, over **HTTPS**, from edge
     locations, and lets us keep the bucket private (access via Origin Access
     Control). Raw S3 static hosting is HTTP-only, so CloudFront is what gives us
     a real `https://` URL.
4. Result: a public URL that serves the static demo. Still fixed clips, still
   pre-rendered — but now hosted, not just localhost.

### Phase 2 — Re-render the demo with REAL pipeline data
Goal: replace placeholder assets with real pipeline output.
- Blocked on the ~135 MB teammate model/data bundle (MixSTE checkpoint,
  `event_detector_tcn.pt`, cached 3D parquets, `Data/coaching/*.json`,
  `golfDB.pkl`). Once it lands (locally or in `02_working/`):
  1. Run the pipeline per clip (`Scripts/demo.py --golfdb-clip <id>`).
  2. Drop real `metrics.json / explanation.json / replay_3d.json / overlay.mp4`
     into `deploy/web/assets/<clip_id>/` (re-encode video to H.264).
  3. Upload the real renders to the data bucket `03_outputs/<clip_id>/`.
  4. No UI changes needed — same asset shape.
- The Claude eval path already exists: `coaching_explain.py --backend anthropic`
  (claude-opus-4-8, grounded). Set `ANTHROPIC_API_KEY` in env; never commit it.

### Phase 3 — Add the upload feature (the real product)
Goal: user uploads their own swing video instead of picking a fixed clip.
This is the big new build. Flow:

```
User uploads video (front end on CloudFront)
      ↓ upload straight to S3 01_inputs/  (presigned URL — see note)
Backend runs the pipeline on the new video
      ↓ writes 2D overlay + 3D reconstruction + metrics.json + eval
        to S3 03_outputs/<id>/
Front end polls / is notified, then displays the results
```

Front-end work for this:
- The upload UI **already exists** as a prototype (landing "Upload swing video" CTA +
  file picker + loader + results banner). Going live is the localized `startUpload()`
  change spelled out in **"Going live on S3 → A"** above — send the file to S3
  `01_inputs/` via presigned URL instead of playing it back locally.
- ⚠️ **Uploads should go directly to S3 via a presigned URL**, not through the
  app server — big video files shouldn't route through compute. The backend
  hands the browser a short-lived presigned PUT URL; the browser uploads straight
  to the bucket.
- A "processing…" state (reuse the existing 7-step Analyze screen) while the
  backend works, since real processing isn't instant.
- A results view that loads the real outputs from `03_outputs/<id>/`.

### Phase 4 — The backend / Lambda question (align before building)
The front end needs *something* to run the pipeline when a video is picked/uploaded.
Per the `deploy/infra` handoff docs, the intended design is:
- **Lambda (CPU) for the cached/light path** — takes a pre-computed 3D pose,
  runs event detection + scorecard + Claude eval, returns results. The front end
  calls a **Lambda Function URL**; Lambda IS the backend compute, not a
  "hand-off" layer.
- **The heavy GPU path (raw video → MediaPipe → 3D lifting) is NOT Lambda** —
  Lambda is CPU-only. That live-upload processing would need an async
  **Fargate/Batch** job triggered by the S3 upload. This is the harder, later
  piece.
- ❗ **Open item:** whether Lambda deploys cleanly under Lawrence's IAM setup is
  unconfirmed. Confirm with Lawrence before committing to the Lambda path. If
  Lambda is awkward, the fallback is running the pipeline in our SageMaker
  environment and having the front end call that.

---

## How the front end connects to Lawrence's S3 (summary)
- **App code** (`deploy/web/`) → its own **web bucket** + CloudFront (Phase 1).
  Code lives in git; the web bucket is just a hosting copy.
- **User-uploaded videos** → data bucket `01_inputs/` (Phase 3, via presigned URL).
- **Pipeline outputs** (overlay, 3D, metrics, eval) → data bucket `03_outputs/`.
- The front end reads results from `03_outputs/` (or from whatever the backend
  returns). Two buckets, two jobs: **web bucket = the site**, **data bucket =
  videos + results**.

## Things not to miss
- Keep `loadClipBundle()` as the single swap point — demo (local JSON) vs live
  (backend/S3) should be a one-function change, no UI rewrite.
- Never commit secrets (`ANTHROPIC_API_KEY` stays in env / Secrets Manager).
- Don't `git add -A` — repo has many unrelated untracked files; stage only
  `deploy/web/` and deploy outputs.
- All video assets must be H.264/yuv420p for browser playback.
- Everything region **us-east-1**.
- Confirm the web-bucket name + Lambda permissions with Lawrence before Phase 1/4.
- CORS: once the front end (CloudFront domain) calls a backend URL cross-origin,
  the backend must send CORS headers or the browser silently blocks the calls.

## Open decisions (need team/Lawrence sign-off)
1. Web-app S3 bucket name (Phase 1).
2. Lambda under Lawrence's account — works, or use SageMaker-hosted backend?
3. Who owns Phase 3 upload vs Phase 4 backend (front-end vs infra split)?
4. Demo scope for next presentation: hosted static (Phase 1) is the safe target;
   live upload (Phase 3) is stretch.
