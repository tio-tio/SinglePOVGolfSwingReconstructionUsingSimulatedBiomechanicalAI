/* MotionCaddie demo — static front end over pre-rendered pipeline results.
 * Three screens: pick → analyze (animated checklist) → results.
 * No backend: every clip's data is baked under assets/<clip_id>/.
 */
"use strict";

/* =========================================================================
 * Data access — the ONE place that knows where results come from.
 *
 * TODO(live-backend): to switch from pre-rendered files to a live backend,
 * replace the fetches below with a single call like
 *   const r = await fetch(`${API_BASE}/analyze/${clipId}`);
 * returning the same shape: { metrics, explanation, replay, overlayUrl, rawUrl }.
 * Nothing else in this file needs to change.
 * ========================================================================= */
async function loadClipBundle(clipId, jobBase = null) {
  // jobBase: absolute prefix for a processed upload (`${RESULTS_BASE}/<job_id>`);
  // the pipeline writes the same four files + videos as the baked demo assets.
  const base = jobBase || `assets/${clipId}`;
  const [metrics, explanation, replay, ball] = await Promise.all([
    fetch(`${base}/metrics.json`).then(r => r.json()),
    fetch(`${base}/explanation.json`).then(r => r.json()),
    fetch(`${base}/replay_3d.json`).then(r => r.json()),
    // measured ball track + flight fit — absent for demo clips / older jobs
    fetch(`${base}/ball_3d.json`).then(r => r.ok ? r.json() : null).catch(() => null),
  ]);
  return { metrics, explanation, replay, ball,
           overlayUrl: `${base}/overlay.mp4`, rawUrl: `${base}/raw.mp4` };
}

/* ================= swing library (persistent, this browser) =============== */
const LIB_KEY = "mc_library_v1";

function libLoad() {
  try { return JSON.parse(localStorage.getItem(LIB_KEY)) || []; }
  catch (e) { return []; }
}
function libSave(items) { localStorage.setItem(LIB_KEY, JSON.stringify(items)); }
function libAdd(entry) {
  const items = libLoad().filter(i => i.jobId !== entry.jobId);
  items.unshift(entry);                       // newest first
  libSave(items.slice(0, 50));
  renderLibrary();
}
function libSetStatus(jobId, status) {
  const items = libLoad();
  const it = items.find(i => i.jobId === jobId);
  if (it) { it.status = status; libSave(items); renderLibrary(); }
}
function libRemove(jobId) {
  libSave(libLoad().filter(i => i.jobId !== jobId));
  renderLibrary();
}

async function loadManifest() {
  return (await fetch("assets/clips.json").then(r => r.json())).clips;
}

/* ============================== app state =============================== */
const SAMPLE_ID = "830";   // demo swing used to illustrate a prototype upload result

const state = {
  clips: [],
  selectedId: null,
  mode: "demo",       // "demo" (real pre-rendered clip) | "upload" (prototype)
  upload: null,       // { name, url } for a prototype uploaded file
  pendingJob: null,   // job id of the upload currently processing in the cloud
  uploads: [],        // this session's prototype uploads (for the dashboard)
  bundle: null,       // loaded clip bundle for the results screen
  bundlePromise: null,
  viewer: null,       // Replay3D instance
};
window.__MC_STATE__ = state;  // debug/diagnostics handle (console + tooling)

const $ = (sel) => document.querySelector(sel);

/* HTML-escape anything interpolated into innerHTML — upload file names are
 * user-controlled, and job JSON becomes cloud-controlled once RESULTS_BASE
 * serves pipeline output. */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* monotonic navigation token: ignore stale async loads after quick switches */
let navSeq = 0;

/* Replace an element with a fresh clone (keeps id/attrs). Used for the replay
 * canvas so a new viewer never inherits a stale WebGL/2D drawing context. */
function resetCanvas(sel) {
  const old = $(sel);
  const fresh = old.cloneNode(false);
  old.replaceWith(fresh);
  return fresh;
}

/* ============================ screen router ============================= */
const SCREENS = { pick: "#screen-pick", analyze: "#screen-analyze", results: "#screen-results" };
const NAV_ORDER = ["pick", "analyze", "results"];

function goto(screen) {
  for (const [name, sel] of Object.entries(SCREENS)) {
    $(sel).hidden = name !== screen;
  }
  const idx = NAV_ORDER.indexOf(screen);
  document.querySelectorAll(".pstep").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
    el.classList.toggle("done", i < idx);
  });
  window.scrollTo({ top: 0 });
}

/* ============================ 1 · landing screen ======================= */
function renderGallery() {
  const gal = $("#clip-gallery");
  gal.innerHTML = "";
  for (const clip of state.clips) {
    const card = document.createElement("button");
    card.className = "clip-card";
    card.dataset.id = clip.id;
    card.innerHTML = `
      <video src="assets/${esc(clip.id)}/raw.mp4" poster="assets/${esc(clip.id)}/poster.jpg"
        preload="metadata" muted playsinline></video>
      <div class="clip-title">${esc(clip.title)}</div>
      <div class="clip-meta">${esc(clip.view)} · ${esc(clip.club)}</div>
      <div class="clip-cta">Open result</div>`;
    card.addEventListener("click", () => startDemo(clip.id));
    gal.appendChild(card);
  }
}

/* start a real pre-rendered demo swing */
function startDemo(id) {
  state.mode = "demo";
  state.selectedId = id;
  state.upload = null;
  setHash(id);
  runAnalyze();
}

/* quick-switch (recent rail / deep link): skip the analyze animation */
async function openSwing(id) {
  const seq = ++navSeq;
  state.mode = "demo";
  state.selectedId = id;
  state.upload = null;
  setHash(id);
  try {
    const bundle = await loadClipBundle(id);
    if (seq !== navSeq) return;          // user navigated on while we loaded
    state.bundle = bundle;
    renderResults();
    goto("results");
  } catch (e) {
    if (seq === navSeq) goto("pick");
  }
}

/* ---- deep links: #swing=<id> opens a result directly ---- */
let suppressHash = false;
function setHash(id) {
  suppressHash = true;
  location.hash = id != null ? `swing=${id}` : "";
  setTimeout(() => { suppressHash = false; }, 0);
}
function onHashChange() {
  if (suppressHash) return;
  const m = location.hash.match(/^#swing=([\w-]+)$/);
  if (m && state.clips.some(c => c.id === m[1])) openSwing(m[1]);
  else if (!location.hash) goto("pick");
}

/* ---- upload: real presigned S3 upload + queued cloud analysis ---- */
function onFileChosen(file) {
  if (!file) return;
  if (file.size > 200 * 1024 * 1024) {
    alert("That video is over the 200 MB upload cap — try a shorter clip.");
    return;
  }
  if (state.upload && state.upload.url) URL.revokeObjectURL(state.upload.url);
  state.upload = { name: file.name, url: URL.createObjectURL(file), file };
  const live = !!window.UPLOAD_URL;
  const box = $("#upload-status");
  box.hidden = false;
  box.innerHTML = `
    <div class="upload-file">
      <span class="upload-file-name" title="${esc(file.name)}">${esc(file.name)}</span>
      <span class="upload-badge">${live ? "Ready to upload" : "Prototype — no real analysis"}</span>
    </div>
    <button id="btn-analyze-upload" class="btn-primary" type="button">Analyze swing</button>`;
  $("#btn-analyze-upload").addEventListener("click", startUpload);
}

async function startUpload() {
  const upload = state.upload;     // snapshot: state.upload can change mid-await
  if (!upload) return;
  state.mode = "upload";
  if (!state.uploads.some(u => u.name === upload.name)) {
    state.uploads.push({ name: upload.name });
    renderDashboard();
  }

  // ---- REAL path: upload -> cloud pipeline -> auto-open the processed result.
  // The illustrative sample is only the fallback (offline preview, upload
  // failure, or a pipeline that outlasts our patience).
  if (window.UPLOAD_URL && window.RESULTS_BASE) {
    const seq = ++navSeq;
    startCloudAnalyzeUI(seq);
    try {
      // 1 · ask for a presigned slot
      const pr = await fetch(window.UPLOAD_URL, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ filename: upload.name,
                               content_type: upload.file.type || "video/mp4" }),
      });
      if (!pr.ok) throw new Error("presign http " + pr.status);
      const slot = await pr.json();
      // 2 · browser -> S3, direct
      const form = new FormData();
      for (const [k, v] of Object.entries(slot.fields)) form.append(k, v);
      form.append("file", upload.file);
      const up = await fetch(slot.url, { method: "POST", body: form });
      if (!up.ok) throw new Error("s3 http " + up.status);
      // 3 · library entry + wait for the pipeline (~2-3 min; poll up to 8)
      state.pendingJob = slot.job_id;
      libAdd({ jobId: slot.job_id, name: upload.name,
               date: new Date().toISOString(), status: "processing" });
      const ready = await pollJob(slot.job_id, 96, 5000);
      if (seq !== navSeq) return;              // user navigated away meanwhile
      if (ready) { openJob(slot.job_id); return; }
      showIllustrativeFallback();              // still grinding — sample + banner
      return;
    } catch (e) {
      console.warn("upload failed:", e);
      const failId = "local-" + Date.now();
      state.pendingJob = failId;               // renderResults -> "failed" banner
      libAdd({ jobId: failId, name: upload.name,
               date: new Date().toISOString(), status: "upload failed" });
      if (seq === navSeq) showIllustrativeFallback();
      return;
    }
  }

  // ---- offline preview: no backend wired, illustrative only
  showIllustrativeFallback();
}

/* the pre-cloud behavior: illustrative sample + honesty banner */
function showIllustrativeFallback() {
  state.mode = "upload";
  state.selectedId = SAMPLE_ID;
  setHash(null);
  runAnalyze();
}

/* analyze screen paced for the REAL pipeline: steps advance slowly, the last
 * one keeps pulsing until the result opens (or the wait falls back). */
function startCloudAnalyzeUI(seq) {
  goto("analyze");
  const note = $("#analyze-cloud-note"), elapsed = $("#analyze-elapsed");
  if (note) note.hidden = false;
  const items = [...document.querySelectorAll("#analyze-steps li")];
  items.forEach(li => li.classList.remove("doing", "done"));
  const t0 = Date.now();
  const timer = setInterval(() => {
    if (seq !== navSeq) { clearInterval(timer); if (note) note.hidden = true; return; }
    if (elapsed) {
      const s = Math.round((Date.now() - t0) / 1000);
      elapsed.textContent = `(${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} elapsed)`;
    }
  }, 1000);
  let i = 0;
  const tick = () => {
    if (seq !== navSeq) return;
    if (i > 0) items[i - 1].classList.replace("doing", "done");
    if (i < items.length) {
      items[i].classList.add("doing");
      i += 1;
      if (i < items.length) setTimeout(tick, 15000);   // ~90s across 7 real steps
    }
  };
  tick();
}

/* ---- the results-screen banner doubles as the upload progress line ---- */
function setUploadBanner(phase, jobId) {
  const text = $("#results-banner-text"), btn = $("#btn-view-real");
  if (!text || !btn) return;
  if (phase === "ready") {
    text.innerHTML = "<strong>Your real analysis is ready.</strong> The page below is still " +
      "the illustrative sample — open yours to see the measured result.";
    btn.hidden = false;
    btn.onclick = () => { btn.hidden = true; openJob(jobId); };
  } else if (phase === "failed") {
    text.innerHTML = "<strong>Upload failed.</strong> The result below is an illustrative " +
      "sample only — please check your connection and try the upload again.";
    btn.hidden = true;
  } else {  // processing (default copy lives in index.html)
    text.innerHTML = "<strong>Analyzing your swing in the cloud</strong> — this usually takes " +
      "2–3 minutes. Below is an illustrative sample result in the meantime; a " +
      "<strong>View your real analysis</strong> button will appear here when yours is ready " +
      "(it also lands under “Your swings”).";
    btn.hidden = true;
  }
}

/* poll a processed upload until its results land (only when RESULTS_BASE set).
 * "ready" requires ALL three result JSONs — metrics alone can land first.
 * NB: CloudFront rewrites S3 403s to 200/index.html (see web_hosting.yaml), so
 * r.ok alone would false-ready a job — require the real content-type too. */
async function jobIsReady(jobId) {
  const files = ["metrics.json", "explanation.json", "replay_3d.json"];
  // a real S3 object serves as its stored type (json/video/octet-stream); only
  // the CloudFront 403->index.html rewrite comes back as text/html
  const okReal = (r) =>
    r.ok && !(r.headers.get("content-type") || "").includes("html");
  const oks = await Promise.all([
    ...files.map(f =>
      fetch(`${window.RESULTS_BASE}/${jobId}/${f}`, { cache: "no-store" })
        .then(okReal).catch(() => false)),
    // HEAD the video so "Ready" never opens onto a broken player
    fetch(`${window.RESULTS_BASE}/${jobId}/overlay.mp4`,
          { method: "HEAD", cache: "no-store" })
      .then(okReal).catch(() => false),
  ]);
  return oks.every(Boolean);
}

async function pollJob(jobId, tries = 40, delayMs = 15000) {
  for (let i = 0; i < tries; i++) {
    try {
      if (await jobIsReady(jobId)) {
        libSetStatus(jobId, "ready");
        // if the uploader ended up on the illustrative fallback page, surface
        // the real result right where they are
        if (state.pendingJob === jobId) setUploadBanner("ready", jobId);
        return true;
      }
    } catch (e) { /* keep polling */ }
    await new Promise(res => setTimeout(res, delayMs));
  }
  libSetStatus(jobId, "processing (check back)");
  return false;
}

/* refresh-proofing: polling dies with the page, so on load prune the dead
 * "upload failed" entries and re-check every non-ready cloud job — flip the
 * finished ones to Ready, keep watching the young ones. */
async function reconcileLibrary() {
  const items = libLoad();
  const keep = items.filter(i => i.status !== "upload failed");
  if (keep.length !== items.length) libSave(keep);
  renderLibrary();
  if (!window.RESULTS_BASE) return;
  for (const it of keep) {
    if (it.status === "ready" || String(it.jobId).startsWith("local-")) continue;
    try {
      if (await jobIsReady(it.jobId)) { libSetStatus(it.jobId, "ready"); continue; }
    } catch (e) { /* fall through to polling */ }
    if (Date.now() - new Date(it.date).getTime() < 20 * 60 * 1000) pollJob(it.jobId);
    else libSetStatus(it.jobId, "processing (check back)");
  }
}

/* ========================== 2 · analyze screen ========================== */
const STEP_MS = 430;  // pre-rendered demo: tick quickly, then reveal results

function runAnalyze() {
  // kick off the data load in parallel with the animation
  const seq = ++navSeq;
  state.bundlePromise = loadClipBundle(state.selectedId);
  goto("analyze");

  const items = [...document.querySelectorAll("#analyze-steps li")];
  items.forEach(li => li.classList.remove("doing", "done"));

  let i = 0;
  const tick = () => {
    if (i > 0) items[i - 1].classList.replace("doing", "done");
    if (i < items.length) {
      items[i].classList.add("doing");
      i += 1;
      setTimeout(tick, STEP_MS);
    } else {
      state.bundlePromise.then(bundle => {
        if (seq !== navSeq) return;          // superseded by a newer navigation
        state.bundle = bundle;
        setTimeout(() => { if (seq === navSeq) { renderResults(); goto("results"); } }, 350);
      }).catch(err => {
        if (seq !== navSeq) return;
        alert(`Could not load results for clip ${state.selectedId}: ${err}`);
        goto("pick");
      });
    }
  };
  tick();
}

/* ========================== 3 · results screen ========================= */
function metricByKey(key) {
  return state.bundle.metrics.metrics.find(m => m.key === key);
}

function renderResults() {
  const { metrics, explanation, overlayUrl, replay } = state.bundle;
  const clip = state.clips.find(c => c.id === state.selectedId);
  const isUpload = state.mode === "upload";
  const isJob = !isUpload && !clip;            // a processed upload from the library

  // honesty banner + labelling for prototype uploads
  $("#results-banner").hidden = !isUpload;
  if (isUpload) {
    $("#results-clip-label").textContent = `Your upload · ${state.upload.name}`;
    $("#results-headline").textContent = "Illustrative result (prototype)";
    // banner reflects THIS upload's cloud job: ready (already?) / processing / failed
    const job = state.pendingJob && libLoad().find(i => i.jobId === state.pendingJob);
    setUploadBanner(job && job.status === "ready" ? "ready"
      : job && job.status === "upload failed" ? "failed" : "processing",
      state.pendingJob);
  } else if (isJob) {
    $("#results-clip-label").textContent = "Your swing · analyzed by the real pipeline";
    $("#results-headline").textContent = explanation.headline;
  } else {
    $("#results-clip-label").textContent = `${clip.title} · ${clip.view} · ${clip.club}`;
    $("#results-headline").textContent = explanation.headline;
  }

  renderPlain(explanation);
  initListen(explanation, isUpload ? "upload-preview" : String(state.selectedId));
  renderNumbers(metrics.metrics);
  populateNumbersCompare();
  // chat works on demo clips AND processed uploads (the backend fetches an
  // upload's scorecard from 03_outputs by its job id)
  let chatOk = true;
  if (isJob) chatOk = Chat.registerJob(state.selectedId, metrics);
  $("#tab-chat").hidden = !chatOk;
  if (chatOk) Chat.activate(state.selectedId);

  // ball-flight estimate card (simulated; same engine + nudge rule as the coach).
  // Demo clips: tour launch numbers for the clip's club. Uploads: amateur default,
  // club picker. Hidden on the illustrative prototype view (not this swing's data).
  if (window.BallFlight) {
    if (isUpload) BallFlight.unmount();
    else {
      const hs = isJob && (((state.bundle.scorecard || {}).indicators || {}).hand_speed_impact_bs);
      BallFlight.mount({ club: isJob ? null : clip.club,
                         tier: isJob ? "amateur" : "tour",
                         metricsRows: metrics.metrics,
                         handSpeed: hs ? { value: hs.value, median: hs.pro_median } : null,
                         ball: state.bundle.ball });
    }
  }

  // share link + scorecard download (demo clips only — uploads have no baked card,
  // and the hash router only resolves curated clip ids)
  $("#btn-share").hidden = isUpload || isJob;
  $("#btn-share").onclick = async () => {
    const url = `${location.origin}${location.pathname}#swing=${state.selectedId}`;
    try { await navigator.clipboard.writeText(url); $("#btn-share").textContent = "Link copied ✓"; }
    catch (e) { prompt("Copy this link:", url); }
    setTimeout(() => { $("#btn-share").textContent = "Share link"; }, 1800);
  };
  const dl = $("#btn-scorecard");
  dl.hidden = isUpload || isJob;
  if (!isUpload && !isJob) {
    dl.href = `assets/${state.selectedId}/scorecard.png`;
    dl.setAttribute("download", `motioncaddie_swing_${state.selectedId}_scorecard.png`);
  }

  const vid = $("#overlay-video");
  // upload mode plays back the user's ACTUAL file (truthful — their raw clip, no overlay claimed)
  vid.src = isUpload ? state.upload.url : overlayUrl;
  vid.load();
  renderEventMarkers(vid, metrics, replay, isUpload);

  if (state.viewer) state.viewer.destroy();
  // fresh canvas each time so a WebGL/2D context is never reused across viewers
  const canvas = resetCanvas("#replay-canvas");
  const ui3d = { scrub: $("#replay-scrub"), label: $("#replay-frame"), playBtn: $("#replay-play") };
  const Cap = window.CapsuleViewer3D;   // WebGL capsule viewer (module); falls back to canvas
  state.viewer = (Cap && Cap.supported())
    ? new Cap(canvas, replay, ui3d)
    : new Replay3D(canvas, replay, ui3d);

  wireReplaySync(vid, replay);
  renderRecentRail();
  loadRenderView(state.selectedId, isUpload);
  showTab("plain");
}

/* ---- detected-event markers under the overlay video (click to seek) ---- */
const EVENT_LABELS = {
  address: "Address", toe_up: "Toe-up", mid_backswing: "Mid-backswing", top: "Top",
  mid_downswing: "Mid-downswing", impact: "Impact",
  mid_follow_through: "Follow-through", finish: "Finish",
};
function renderEventMarkers(vid, metrics, replay, isUpload) {
  const bar = $("#event-markers");
  const events = metrics.events || {};
  const total = replay && replay.frames ? replay.frames.length : 0;
  const fps = (replay && replay.fps) || 30;
  const keys = Object.keys(events);
  // uploads play the raw local file — the demo clip's frame numbers don't apply
  if (isUpload || !keys.length || !total) { bar.hidden = true; bar.innerHTML = ""; return; }
  bar.hidden = false;
  bar.innerHTML = keys.map(k => {
    const frame = Number(events[k]);
    if (!Number.isFinite(frame)) return "";
    const name = esc(EVENT_LABELS[k] || k);
    const pct = Math.min(100, Math.max(0, (frame / total) * 100));
    return `<button type="button" class="ev-tick" style="left:${pct}%"
              data-t="${(frame / fps).toFixed(3)}" title="${name} · frame ${frame}">
              <span class="ev-label">${name}</span></button>`;
  }).join("") + `<div class="ev-track"></div>`;
  bar.querySelectorAll(".ev-tick").forEach(b =>
    b.addEventListener("click", () => { vid.currentTime = +b.dataset.t; vid.pause(); }));
}

/* ---- "Lock to video": one timeline drives the video AND the 3D replay ---- */
function wireReplaySync(vid, replay) {
  const box = $("#replay-sync");
  box.checked = false;
  const fps = (replay && replay.fps) || 30;
  const total = replay && replay.frames ? replay.frames.length : 0;
  const onTime = () => {
    if (!box.checked || !state.viewer || !total) return;
    state.viewer.playing = false;
    const f = Math.min(total - 1, Math.round(vid.currentTime * fps));
    state.viewer.frame = f;
    if (state.viewer.ui && state.viewer.ui.scrub) state.viewer.ui.scrub.value = f;
    if (typeof state.viewer.draw === "function") state.viewer.draw();
    else if (typeof state.viewer._pose === "function") state.viewer._pose(f);
  };
  vid.removeEventListener("timeupdate", vid._mcSync || (() => {}));
  vid._mcSync = onTime;
  vid.addEventListener("timeupdate", onTime);
  box.onchange = () => { if (box.checked) { onTime(); } else if (state.viewer) state.viewer.playing = true; };
}

/* ---- recent-swings rail: hop between swings without leaving results ---- */
function renderRecentRail() {
  const rail = $("#recent-rail"), cards = $("#rail-cards");
  const others = state.clips.filter(c => c.id !== state.selectedId);
  if (state.mode === "upload" || !others.length) { rail.hidden = true; return; }
  rail.hidden = false;
  cards.innerHTML = "";
  for (const c of others) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "rail-card";
    b.innerHTML = `<video src="assets/${esc(c.id)}/raw.mp4" preload="metadata" muted playsinline></video>
                   <span>${esc(c.title)}</span>`;
    b.addEventListener("click", () => openSwing(c.id));
    cards.appendChild(b);
  }
}

/* ---- optional "3D Swing View": precomputed headless-Blender render stills ----
 * Presence-driven off the clips.json manifest: the tab appears only for clips whose
 * entry carries a `render.phases` list (no per-clip fetch/404). Uploads never show it.
 *
 * TODO(mixste-swap): clip 0's render is currently generated from the MotionBERT-full
 * lift (the mocap JSON that exists today), NOT the golfpose3d/MixSTE production lifter.
 * Once the model bundle lands, regenerate Data/handoff/0/0_mocap.json through the
 * golfpose3d (MixSTE) path and re-run Scripts/blender_mocap.py (unisex mode) to
 * overwrite the PNGs in assets/0/blender/ in place — no front-end change needed (same
 * filenames = same manifest). The caption is deliberately lifter-neutral so it stays
 * true after the swap. To add a NEW clip's render: drop its PNGs under
 * assets/<id>/blender/ and add a `render.phases` block to that clip in clips.json.
 */
function loadRenderView(clipId, isUpload) {
  const tab = $("#tab-render"), panel = $("#render-phases");
  tab.hidden = true;                       // default: no render for this clip
  if (isUpload) return;                    // uploads use a sample clip; never claim a render
  const clip = state.clips.find(c => c.id === clipId);
  const render = clip && clip.render;
  if (!render || !Array.isArray(render.phases) || !render.phases.length) return;
  // optional avatar animation above the stills (presence-driven, like the phases)
  const vid = render.video
    ? `<video class="render-video" src="assets/${esc(clipId)}/blender/${esc(render.video)}"
         controls muted loop playsinline preload="metadata"></video>`
    : "";
  panel.innerHTML = vid + render.phases.map(p =>
    `<figure class="render-phase">
       <img src="assets/${esc(clipId)}/blender/${esc(p.src)}" alt="${esc(p.label)} — rendered 3D pose" loading="lazy">
       <figcaption>${esc(p.label)}</figcaption>
     </figure>`).join("");
  tab.hidden = false;
}

function renderPlain(explanation) {
  const el = $("#view-plain");
  el.innerHTML = "";
  for (const sec of explanation.sections) {
    const tagLabel = { good: "Good", watch: "Watch", low: "Low confidence" }[sec.tag] || sec.tag;
    const chips = (sec.chips || []).map(key => {
      const m = metricByKey(key);
      return m ? `<span class="chip">${esc(m.label)} · <b>${esc(m.you_display)}</b></span>` : "";
    }).join("");
    const div = document.createElement("div");
    div.className = "eval-section";
    div.innerHTML = `
      <div class="eval-tagrow">
        <span class="eval-tag ${esc(sec.tag)}">${esc(tagLabel)}</span>
        <h3>${esc(sec.title)}</h3>
      </div>
      <p>${esc(sec.body)}</p>
      <div class="chips">${chips}</div>`;
    el.appendChild(div);
  }
}

/* "Listen" — the swing explanation read by the ElevenLabs coach voice
 * (POST /tts, mp3 via CloudFront; the server caches by content hash so
 * replaying a swing's narration is instant). Live backend only. */
let listenAudio = null;
function initListen(explanation, swingId) {
  const btn = $("#btn-listen");
  if (!btn) return;
  btn.hidden = !window.API_BASE;
  if (btn.hidden) return;
  if (listenAudio) { listenAudio.pause(); listenAudio = null; }
  btn.textContent = "🔊 Listen";
  btn.disabled = false;
  btn.onclick = async () => {
    if (listenAudio && !listenAudio.paused) {
      listenAudio.pause(); listenAudio = null;
      btn.textContent = "🔊 Listen";
      return;
    }
    btn.disabled = true; btn.textContent = "Preparing…";
    try {
      const narration = [explanation.headline,
        ...explanation.sections.map(s => `${s.title}. ${s.body}`)]
        .filter(Boolean).join("\n").slice(0, 2400);
      const r = await fetch(`${window.API_BASE}/tts`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ narration, swing_id: swingId }),
      });
      if (!r.ok) throw new Error("tts " + r.status);
      const d = await r.json();
      if (!d.audio_url) throw new Error("tts: no audio_url");
      listenAudio = new Audio(d.audio_url);
      listenAudio.onended = () => { listenAudio = null; btn.textContent = "🔊 Listen"; };
      await listenAudio.play();
      btn.textContent = "⏹ Stop";
    } catch (e) {
      btn.textContent = "Audio unavailable";
      setTimeout(() => { btn.textContent = "🔊 Listen"; }, 2500);
    } finally { btn.disabled = false; }
  };
}

function renderNumbers(metrics, cmp = null) {
  const el = $("#metrics-table");
  el.innerHTML = "";
  const cmpBy = {};
  if (cmp) for (const m of cmp.metrics) cmpBy[m.key] = m;
  for (const m of metrics) {
    const other = cmpBy[m.key];
    // widen the axis so a compared value can never fall off the chart
    const lo = other ? Math.min(m.axis[0], other.you) : m.axis[0];
    const hi = other ? Math.max(m.axis[1], other.you) : m.axis[1];
    const pct = v => Math.min(100, Math.max(0, (v - lo) / (hi - lo) * 100));
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `
      <div class="metric-name">
        <i class="dot dot-${esc(m.status)}"></i>
        <button type="button" class="metric-label" title="What is this?"><strong>${esc(m.label)}</strong></button>
      </div>
      <div class="axis">
        <div class="axis-track"></div>
        <div class="axis-band" style="left:${pct(m.band[0])}%; width:${pct(m.band[1]) - pct(m.band[0])}%"></div>
        <div class="axis-tour" style="left:${pct(m.tour)}%" title="tour median ${esc(m.tour_display)}"></div>
        ${other ? `<div class="axis-cmp" style="left:${pct(other.you)}%" title="compared swing: ${esc(other.you_display)}"></div>` : ""}
        <div class="axis-you ${esc(m.status)}" style="left:${pct(m.you)}%" title="you: ${esc(m.you_display)}"></div>
      </div>
      <div class="metric-vals">
        <div class="you-val">${esc(m.you_display)}</div>
        <div class="tour-val">${other ? `vs ${esc(other.you_display)}` : `tour ${esc(m.tour_display)}`}</div>
      </div>
      <p class="metric-blurb">${esc(m.blurb)}</p>
      ${m.why ? `<p class="metric-why" hidden>${esc(m.why)}</p>` : ""}`;
    const why = row.querySelector(".metric-why");
    if (why) row.querySelector(".metric-label").addEventListener("click",
      () => { why.hidden = !why.hidden; });
    el.appendChild(row);
  }
}

/* compare selector on the numbers tab — line this swing up against another */
function populateNumbersCompare() {
  const sel = $("#numbers-compare");
  sel.innerHTML = `<option value="">— none —</option>` +
    state.clips.filter(c => c.id !== state.selectedId)
      .map(c => `<option value="${esc(c.id)}">${esc(c.title)}</option>`).join("");
  sel.onchange = async () => {
    if (!sel.value) { renderNumbers(state.bundle.metrics.metrics); return; }
    try {
      const cmp = await fetch(`assets/${sel.value}/metrics.json`).then(r => r.json());
      renderNumbers(state.bundle.metrics.metrics, cmp);
    } catch (e) { renderNumbers(state.bundle.metrics.metrics); }
  };
}

function showTab(which) {
  $("#tab-plain").classList.toggle("active", which === "plain");
  $("#tab-numbers").classList.toggle("active", which === "numbers");
  $("#tab-chat").classList.toggle("active", which === "chat");
  $("#tab-render").classList.toggle("active", which === "render");
  $("#view-plain").hidden = which !== "plain";
  $("#view-numbers").hidden = which !== "numbers";
  $("#view-chat").hidden = which !== "chat";
  $("#view-render").hidden = which !== "render";
}

/* =========================================================================
 * Replay3D — tiny dependency-free 3D skeleton viewer (canvas 2D).
 * Data: { fps, joint_names, bones:[{a,b,color}], frames:[T][J][3] } in
 * h36m-camera axes (y is DOWN) — flipped to y-up for display.
 * Drag to orbit (yaw/pitch); play/pause + frame scrub.
 * ========================================================================= */
class Replay3D {
  constructor(canvas, data, ui) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = data;
    this.ui = ui;
    this.frame = 0;
    this.playing = true;
    this.yaw = -0.6;
    this.pitch = 0.12;
    this._destroyed = false;
    this._lastT = 0;
    this._acc = 0;

    // fit: center on the mid-hips averaged over the clip, y flipped
    const pts = data.frames.flat();
    let maxR = 1e-6;
    const c = [0, 0, 0];
    const hipIdx = data.joint_names.indexOf("hip_center");
    for (const fr of data.frames) {
      const h = fr[hipIdx >= 0 ? hipIdx : 0];
      c[0] += h[0]; c[1] += h[1]; c[2] += h[2];
    }
    c[0] /= data.frames.length; c[1] /= data.frames.length; c[2] /= data.frames.length;
    this.center = c;
    for (const p of pts) {
      const r = Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
      if (r > maxR) maxR = r;
    }
    this.radius = maxR;

    // hi-dpi
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    this.w = w; this.h = h;
    this.ctx.scale(dpr, dpr);

    // interactions
    this._onDown = e => { this.drag = { x: e.clientX, y: e.clientY }; };
    this._onMove = e => {
      if (!this.drag) return;
      this.yaw += (e.clientX - this.drag.x) * 0.01;
      this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + (e.clientY - this.drag.y) * 0.01));
      this.drag = { x: e.clientX, y: e.clientY };
      if (!this.playing) this.draw();
    };
    this._onUp = () => { this.drag = null; };
    canvas.addEventListener("pointerdown", this._onDown);
    window.addEventListener("pointermove", this._onMove);
    window.addEventListener("pointerup", this._onUp);

    ui.scrub.max = data.frames.length - 1;
    ui.scrub.value = 0;
    this._onScrub = () => { this.playing = false; this.frame = +ui.scrub.value; this.draw(); };
    this._onPlay = () => { this.playing = !this.playing; };
    ui.scrub.addEventListener("input", this._onScrub);
    ui.playBtn.addEventListener("click", this._onPlay);

    requestAnimationFrame(t => this._loop(t));
  }

  destroy() {
    this._destroyed = true;
    this.canvas.removeEventListener("pointerdown", this._onDown);
    window.removeEventListener("pointermove", this._onMove);
    window.removeEventListener("pointerup", this._onUp);
    this.ui.scrub.removeEventListener("input", this._onScrub);
    this.ui.playBtn.removeEventListener("click", this._onPlay);
  }

  _loop(t) {
    if (this._destroyed) return;
    if (this.playing) {
      if (this._lastT) {
        this._acc += (t - this._lastT) / 1000;
        const spf = 1 / (this.data.fps || 30);
        while (this._acc >= spf) {
          this._acc -= spf;
          this.frame = (this.frame + 1) % this.data.frames.length;
        }
      }
      this.ui.scrub.value = this.frame;
      this.draw();
    }
    this._lastT = t;
    requestAnimationFrame(tt => this._loop(tt));
  }

  _project(p) {
    // recenter, flip y (h36m y is down), orbit, mild perspective
    let x = p[0] - this.center[0];
    let y = -(p[1] - this.center[1]);
    let z = p[2] - this.center[2];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    [x, z] = [x * cy + z * sy, -x * sy + z * cy];
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    [y, z] = [y * cp - z * sp, y * sp + z * cp];
    const persp = 1 / (1 + (z / this.radius) * 0.18);
    const s = (Math.min(this.w, this.h) * 0.40) / this.radius;
    return [this.w / 2 + x * s * persp, this.h / 2 - y * s * persp, persp];
  }

  draw() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = "#10231a";
    ctx.fillRect(0, 0, this.w, this.h);

    // ground grid: grounded replays declare the floor exactly (data y=0 =
    // leveled stance line, y-down); legacy assets fall back to the lowest
    // point of frame 0
    const groundY = this.data.grounded === true
      ? (typeof this.data.floor_y === "number" ? this.data.floor_y : 0)
      : this.center[1] + this.radius * 0.02 +
        Math.max(...this.data.frames[0].map(p => p[1] - this.center[1]));
    ctx.strokeStyle = "rgba(127,227,172,.14)";
    ctx.lineWidth = 1;
    const G = this.radius * 0.9, N = 6;
    for (let i = 0; i <= N; i++) {
      const u = -G + (2 * G * i) / N;
      let a = this._project([this.center[0] + u, groundY, this.center[2] - G]);
      let b = this._project([this.center[0] + u, groundY, this.center[2] + G]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      a = this._project([this.center[0] - G, groundY, this.center[2] + u]);
      b = this._project([this.center[0] + G, groundY, this.center[2] + u]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }

    const fr = this.data.frames[this.frame];
    const proj = fr.map(p => this._project(p));

    // club bones (either end = estimated clubhead 17) draw as a muted thin
    // two-pole placeholder. DELIBERATE: we do not track the club — the clubhead
    // is a forearm extrapolation (web_artifacts.py) — so this stays an honest
    // schematic rather than implying tracking we don't have.
    // TODO(v2-club-tracking): faithful club once measured (DEPLOYMENT_PLAN.md).
    const isClub = (b) => b.a === 17 || b.b === 17;

    // bones (rainbow map from the design spec), far bones first
    const bones = [...this.data.bones].sort((p, q) =>
      Math.min(proj[p.a][2], proj[p.b][2]) - Math.min(proj[q.a][2], proj[q.b][2]));
    for (const bone of bones) {
      const a = proj[bone.a], b = proj[bone.b];
      ctx.strokeStyle = isClub(bone) ? "#8e9089" : bone.color;
      ctx.lineWidth = (isClub(bone) ? 2.0 : 3.5) * Math.min(a[2], b[2]);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    // joints
    ctx.fillStyle = "#faf9f5";
    for (const p of proj) {
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.4 * p[2], 0, Math.PI * 2); ctx.fill();
    }

    this.ui.label.textContent = `${this.frame + 1} / ${this.data.frames.length}`;
  }
}

/* ================================ wiring ================================ */
async function init() {
  state.clips = await loadManifest();
  renderGallery();

  // Prefetch every clip's metrics.json (small) so the coach chat can talk about
  // any swing and compare across them — same source the "numbers" tab renders.
  const metricsByClip = {};
  await Promise.all(state.clips.map(async (c) => {
    try { metricsByClip[c.id] = await fetch(`assets/${c.id}/metrics.json`).then(r => r.json()); }
    catch (e) { /* a clip without metrics just won't be chat-enabled */ }
  }));
  Chat.setLibrary(state.clips, metricsByClip);
  Chat.init({
    stream: $("#chat-stream"), input: $("#chat-q"), send: $("#chat-send"),
    compare: $("#chat-compare"), active: $("#chat-active"), mode: $("#chat-mode"),
    mic: $("#chat-mic"), voice: $("#chat-voice"), voicePick: $("#chat-voice-pick"),
    chips: [...document.querySelectorAll("#view-chat .chip-btn")],
  });

  // upload: file picker or straight-to-camera on phones
  $("#btn-upload").addEventListener("click", () => $("#file-input").click());
  $("#btn-record").addEventListener("click", () => $("#camera-input").click());
  $("#file-input").addEventListener("change", (e) => onFileChosen(e.target.files[0]));
  $("#camera-input").addEventListener("change", (e) => onFileChosen(e.target.files[0]));

  const goHome = () => {
    if (state.viewer) { state.viewer.destroy(); state.viewer = null; }
    $("#overlay-video").pause();
    setHash(null);
    goto("pick");
  };
  $("#btn-restart").addEventListener("click", goHome);
  // brand = back to home; it's a real link (works without JS), but in-page
  // reset is smoother than a full reload when the app is already running
  $("#btn-home").addEventListener("click", (e) => { e.preventDefault(); goHome(); });

  // deep links + persistent library (reconcile: refresh kills polling)
  window.addEventListener("hashchange", onHashChange);
  reconcileLibrary();
  $("#tab-plain").addEventListener("click", () => showTab("plain"));
  $("#tab-numbers").addEventListener("click", () => showTab("numbers"));
  $("#tab-chat").addEventListener("click", () => showTab("chat"));
  $("#tab-render").addEventListener("click", () => showTab("render"));

  // prototype account: render the signed-in dashboard on any auth change
  if (window.Auth) Auth.init();
  document.addEventListener("mc-auth-change", renderDashboard);
  renderDashboard();

  // resume any still-processing jobs from a previous visit
  if (window.RESULTS_BASE) {
    for (const it of libLoad()) {
      if (it.status && it.status.startsWith("processing")) pollJob(it.jobId, 8, 15000);
    }
  }

  onHashChange();               // honor a #swing=<id> deep link on first load
  if (!location.hash) goto("pick");
}

/* ---- persistent "Your swings" library (localStorage; Aurora later) ---- */
function renderLibrary() {
  const box = $("#library"), wrap = $("#library-cards");
  if (!box) return;
  const items = libLoad();
  box.hidden = !items.length;
  if (!items.length) return;
  wrap.innerHTML = items.map(it => {
    const date = new Date(it.date).toLocaleDateString();
    const ready = it.status === "ready";
    const badge = ready ? `<span class="upload-badge ok">Ready</span>`
      : it.status === "upload failed" ? `<span class="upload-badge err">Upload failed</span>`
      : `<span class="upload-badge">Processing<span class="pulse">…</span></span>`;
    return `<div class="dash-swing" data-job="${esc(it.jobId)}" data-ready="${ready}">
      <span class="dash-swing-name" title="${esc(it.name)}">${esc(it.name)}</span>
      <span class="dash-swing-date muted small">${esc(date)}</span>${badge}
      ${ready ? `<button type="button" class="btn-ghost small lib-open">Open result</button>` : ""}
      <button type="button" class="btn-ghost small lib-remove" title="Remove from this list"
              aria-label="Remove ${esc(it.name)} from this list">✕</button>
    </div>`;
  }).join("");
  wrap.querySelectorAll(".lib-open").forEach(b =>
    b.addEventListener("click", () => openJob(b.closest(".dash-swing").dataset.job)));
  wrap.querySelectorAll(".lib-remove").forEach(b =>
    b.addEventListener("click", () => libRemove(b.closest(".dash-swing").dataset.job)));
}

/* open a processed upload's results from RESULTS_BASE (same shape as demo clips) */
async function openJob(jobId) {
  if (!window.RESULTS_BASE) return;
  const seq = ++navSeq;
  try {
    state.mode = "demo";                       // real results — no prototype banner
    if (state.pendingJob === jobId) state.pendingJob = null;
    state.selectedId = jobId;
    state.upload = null;
    setHash(null);                             // job ids aren't hash-routable (yet)
    const bundle = await loadClipBundle(jobId, `${window.RESULTS_BASE}/${jobId}`);
    // full scorecard (published since image v10; absent on older jobs) — the
    // ball-flight card nudges by its hand-speed indicator
    bundle.scorecard = await fetch(`${window.RESULTS_BASE}/${jobId}/scorecard.json`)
      .then(r => r.json()).catch(() => null);
    if (seq !== navSeq) return;
    state.bundle = bundle;
    renderResults();
    goto("results");
  } catch (e) {
    if (seq === navSeq) alert("This swing's results aren't ready yet — check back in a minute.");
  }
}

/* ---- signed-in "My swings" dashboard (prototype) ---- */
function renderDashboard() {
  const dash = $("#dashboard");
  if (!dash) return;
  const signedIn = window.Auth && Auth.isSignedIn();
  dash.hidden = !signedIn;
  if (!signedIn) return;
  const user = Auth.currentUser();
  $("#dash-name").textContent = user.displayName || user.username;
  const wrap = $("#dash-swings");
  if (!state.uploads.length) {
    wrap.innerHTML = `<p class="muted small">No swings yet. Upload a swing video above to get started — analysis is prototype-only in this demo.</p>`;
    return;
  }
  wrap.innerHTML = state.uploads.map(u =>
    `<div class="dash-swing"><span class="dash-swing-name" title="${esc(u.name)}">${esc(u.name)}</span>` +
    `<span class="upload-badge">Prototype</span></div>`).join("");
}

init();
