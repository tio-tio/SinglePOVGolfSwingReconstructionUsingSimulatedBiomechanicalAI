# MotionCaddie — static demo front end

Pre-rendered, no-backend demo: pick a curated swing → animated "reading your
swing" checklist → results (plain-English read + metrics vs tour + 2D pose
overlay + rotatable 3D replay + a grounded **coaching Q&A chat**). Plain vanilla
HTML/CSS/JS — no build step.

## Run locally

```bash
cd deploy/web
python -m http.server 8000
# open http://localhost:8000
```

(Any static file server works; a server is needed because the app `fetch()`es
its JSON — `file://` won't do.)

## Layout

```
deploy/web/
  index.html          three screens (pick / analyze / results)
  styles.css          design tokens + layout (spec: warm off-white, forest green)
  app.js              screen flow, rendering, 3D canvas viewer
  assets/
    clips.json        manifest of curated clips
  chat.js             coaching Q&A engine (offline mock + live seam)
    <clip_id>/
      metrics.json      metric rows: you vs tour band, status, blurb
      explanation.json  plain-English eval (headline + tagged sections)
      replay_3d.json    per-frame 3D joints + rainbow bone map (drives the viewer)
      overlay.mp4       2D pose overlay render
      raw.mp4           original clip (gallery thumbnail)
```

## Coaching chat (the "Ask the coach" tab)

Folded in from Banjot's `coaching_chatbot_demo.html` (same design lineage). It's a
grounded Q&A over the selected swing: it only speaks to measured indicators, refuses
ball-flight/club questions and low-confidence metrics, never prescribes fixes, and
shows the tool calls behind each answer. Per-clip values are read from the SAME
`assets/<id>/metrics.json` the "numbers" tab renders, so chat and table can't disagree.

- **Offline (default):** an in-page mock answers, mirroring the real agent's
  behaviour + grounding verifier. No key, no network.
- **Live:** set `window.API_BASE` (e.g. the chat Lambda Function URL, or the origin
  of `serve_demo.py`). `askChat()` in `chat.js` is the single swap seam — it then
  POSTs `{clip_id, question, history[], compare_clip_id?}` to `${API_BASE}/chat` and
  renders the returned `{answer, grounded, violations, tool_log}`. The backend is
  `deploy/chat_handler.py` (packaged in `deploy/chat/` as a Lambda container).

### Running with the LIVE coach locally
The static server (`python -m http.server`) serves the demo with the offline mock.
For live answers, Banjot's `serve_demo.py` (repo root) serves the page AND exposes
`/api/chat` backed by the real agent — but note it currently serves its own
`coaching_chatbot_demo.html`, not this `deploy/web/` app. To point this app at it,
serve `deploy/web/` and set `window.API_BASE` to the `serve_demo.py` origin (see the
reconciliation note in `DEPLOYMENT_PLAN.md` for the remaining wiring).

## Data status — IMPORTANT

Every `assets/<id>/*.json` is currently tagged `"placeholder": true`:

- `replay_3d.json` is **real pipeline 3D** (the committed clip-0 output,
  `motionbert_full_from_mediapipe_lite`) reused for all three clips; the
  clubhead joint is extrapolated from forearm direction.
- `metrics.json` / `explanation.json` for clip 830 use the design-spec
  reference example; 269 and 0 are hand-made variants so the demo shows three
  distinct results.
- `overlay.mp4` for 830/269 are the old eval *comparison* renders (they have
  "Baseline / +SG Smooth" text baked in) — replace with clean
  `pipeline.py` overlay renders; clip 0's is a real clean overlay.
- All videos are re-encoded H.264/yuv420p (`imageio-ffmpeg`) — the originals
  were mp4v, which browsers won't play. Keep any future renders H.264.

To bake real results (needs the model/data bundle: `event_detector_tcn.pt`,
cached 3D parquets, `golfDB.pkl`, `Data/coaching/*.json`):

```bash
python Scripts/demo.py --golfdb-clip <id>          # scorecard + explanation
# then convert the bundle into assets/<id>/ (same JSON shapes as above)
```

The LLM read comes from `Scripts/coaching_explain.py --backend anthropic`
(claude-opus-4-8, grounded against the indicator KB; needs `ANTHROPIC_API_KEY`).

## Swapping to a live backend later

One function: `loadClipBundle(clipId)` at the top of `app.js`
(marked `TODO(live-backend)`). Replace its fetches with an API call returning
the same `{ metrics, explanation, replay, overlayUrl, rawUrl }` shape.

## Deploying to S3 + CloudFront (later — not done yet)

Everything is relative-path static, so:

```bash
# canonical command lives in deploy/infra/README.md ("Deploy & sync");
# bucket is motion-caddie-web-<acct> from the motion-caddie-web stack
aws s3 sync deploy/web/ s3://<WebBucketName>/ --delete --exclude ".DS_Store"
```

then front the bucket with CloudFront (Origin Access Control, default root
object `index.html`). Region: us-east-1. Rendered outputs also get copied to
`s3://motioncaddie-capstone-data-lj-2026/03_outputs/<clip_id>/` per team
convention (app code stays in git, not S3).
