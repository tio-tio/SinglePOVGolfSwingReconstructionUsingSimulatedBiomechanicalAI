/* MotionCaddie portal — the product entry point. Real cloud pipeline only:
 * upload → presigned S3 → EventBridge/SQS → processing Lambda → results under
 * RESULTS_BASE/<job_id>/. No demo clips, no illustrative fallbacks — every
 * result on this page was measured from the user's own video.
 *
 * Shares config.js / auth.js / chat.js / replay3d.js / styles.css with the
 * demo site (index.html + app.js) so the two front ends can't drift apart.
 */
"use strict";

/* Per-page-load salt for media URLs. Chrome takes a disk-cache lock per media
 * URL; a stalled load in ANY tab (e.g. yesterday's zombie tab) then blocks the
 * same video everywhere — the "video never loads" bug. Unique URLs per load
 * sidestep the lock; the 03_outputs behavior is CachingDisabled anyway. */
const MEDIA_BUST = "s=" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);

/* ---- data access: a processed job's bundle (same shape as demo assets) ---- */
async function loadJobBundle(jobId) {
  const base = `${window.RESULTS_BASE}/${jobId}`;
  const [metrics, explanation, replay, scorecard, ball] = await Promise.all([
    fetch(`${base}/metrics.json`).then(r => r.json()),
    fetch(`${base}/explanation.json`).then(r => r.json()),
    fetch(`${base}/replay_3d.json`).then(r => r.json()),
    // full scorecard (published since image v10) — carries the hand-speed
    // indicator the ball-flight card nudges by; absent on older jobs
    fetch(`${base}/scorecard.json`).then(r => r.json()).catch(() => null),
    // measured ball track + flight fit (image v12+) — absent on older jobs
    fetch(`${base}/ball_3d.json`).then(r => r.ok ? r.json() : null).catch(() => null),
  ]);
  return { metrics, explanation, replay, scorecard, ball,
           overlayUrl: `${base}/overlay.mp4?${MEDIA_BUST}` };
}

/* ================= swing library (persistent, this browser) ===============
 * SAME storage key as the demo page — a swing uploaded on either page shows
 * up in both libraries. */
const LIB_KEY = "mc_library_v1";

function libLoad() {
  // scrub corrupt entries (a sync bug once wrote jobId: undefined) so
  // poisoned libraries heal themselves on every load
  try { return (JSON.parse(localStorage.getItem(LIB_KEY)) || []).filter(i => i && i.jobId && i.jobId !== "undefined"); }
  catch (e) { return []; }
}
function libSave(items) { localStorage.setItem(LIB_KEY, JSON.stringify(items)); }
function libAdd(entry) {
  const items = libLoad().filter(i => i.jobId !== entry.jobId);
  items.unshift(entry);
  libSave(items.slice(0, 50));
  renderLibrary();
}
function libSetStatus(jobId, status) {
  const items = libLoad();
  const it = items.find(i => i.jobId === jobId);
  if (it) { it.status = status; libSave(items); renderLibrary(); }
}
/* removed-swing tombstones: ✕ hides a swing from THIS browser's list, and the
 * team-library sync must not resurrect it — without these, a removed job is
 * indistinguishable from a teammate's upload we've never seen, so every sync
 * pushed it straight back (the "delete doesn't work" bug). Shared with the
 * demo page (app.js) via the same storage key. */
const HIDDEN_KEY = "mc_library_hidden_v1";
function hiddenLoad() {
  try { return JSON.parse(localStorage.getItem(HIDDEN_KEY)) || []; }
  catch (e) { return []; }
}
function hiddenAdd(jobId) {
  const ids = hiddenLoad().filter(id => id !== jobId);
  ids.unshift(jobId);
  localStorage.setItem(HIDDEN_KEY, JSON.stringify(ids.slice(0, 500)));
}

function libRemove(jobId) {
  hiddenAdd(jobId);              // survive the next team-library sync
  libSave(libLoad().filter(i => i.jobId !== jobId));
  renderLibrary();
}
function libName(jobId) {
  const it = libLoad().find(i => i.jobId === jobId);
  return it ? it.name : null;
}

/* ============================== app state =============================== */
const state = {
  selectedId: null,   // job id currently on the results screen
  upload: null,       // { name, file } picked but not yet uploaded
  pendingJob: null,   // job id of the upload currently processing
  bundle: null,
  viewer: null,
  jobMetrics: {},     // jobId -> metrics.json (for compare + chat grounding)
  analyzePreview: null,  // sample-swing viewer on the analyze screen
  analyzeYaw: null,      // yaw-drift interval for its canvas fallback
};
window.__MC_STATE__ = state;

const $ = (sel) => document.querySelector(sel);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let navSeq = 0;

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
  if (screen !== "analyze") stopAnalyzePreview();   // free the GL context on any nav away
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

/* ---- deep links: #job=<id> opens a processed result directly ---- */
let suppressHash = false;
function setHash(jobId) {
  suppressHash = true;
  location.hash = jobId != null ? `job=${jobId}` : "";
  setTimeout(() => { suppressHash = false; }, 0);
}
function onHashChange() {
  if (suppressHash) return;
  const m = location.hash.match(/^#job=([\w-]+)$/);
  if (m) openJob(m[1]);
  else if (!location.hash) goto("pick");
}

/* ======================= private-pilot access gate ======================
 * The swings analyzed here are of real people. Their results are locked at
 * the EDGE (CloudFront mc-results-gate: 03_outputs/* + job audio need the
 * mc_dev access cookie), and this page keeps upload/library/results behind
 * the dev sign-in so the lock is visible, not mysterious. */
const devOnly = () => !!(window.Auth && Auth.isDev && Auth.isDev());
let pendingDeepLink = null;   // a #job= link seen while locked; retried on sign-in

function showPilotNotice() {
  // Be explicit about WHICH account and WHY — a bare "sign in" prompt on the
  // upload button reads as "the upload button is broken".
  const u = window.Auth && Auth.isSignedIn() ? Auth.currentUser() : null;
  const who = u && u.username !== Auth.DEV_USER
    ? `<p>You're signed in as <strong>${esc(u.displayName || u.username)}</strong>, but uploads
       and results run through the shared team account during the pilot.</p>`
    : "";
  statusBox().innerHTML = who +
    `<p><strong>Uploading needs the team account.</strong> The swings here are of real
     people, so uploads and results sit behind one shared login:
     <code>${esc((window.Auth && Auth.DEV_USER) || "dev@motioncaddie.dev")}</code>, with the
     team access code as the password (ask Banjot if you don't have it).</p>
     <p><button type="button" class="btn-primary small" id="pilot-signin">Sign in to upload</button></p>`;
  const b = $("#pilot-signin");
  if (b) b.addEventListener("click", () => Auth.openSignIn(true));
}

/* the access code is validated by the edge, not by JS — probe it once so a
 * wrong code locks the UI immediately instead of failing on every fetch */
async function verifyDevAccess() {
  if (!devOnly() || !window.RESULTS_BASE) return;
  try {
    const r = await fetch(`${window.RESULTS_BASE}/__access_check__`, { cache: "no-store" });
    if (r.status !== 204) throw new Error("denied " + r.status);
    if (pendingDeepLink) { const j = pendingDeepLink; pendingDeepLink = null; openJob(j); }
    syncTeamLibrary();          // pull every teammate's uploads into "Your swings"
  } catch (e) {
    Auth.clearDevCookie();
    renderLibrary();
    statusBox().innerHTML =
      `<p class="err">That access code wasn't accepted — results stay locked.
       Check the code and sign in again.</p>`;
  }
}

/* ---- shared team library: the dev account sees EVERY processed upload ----
 * The browser's localStorage library only knows this browser's uploads, so a
 * teammate's swings were invisible. GET /jobs (access-code-gated Lambda) lists
 * every job straight from S3; merge them in, keeping any richer local entry
 * (a locally-uploaded swing already has its filename even after the uploads
 * bucket's 30-day TTL erases it server-side). */
async function syncTeamLibrary() {
  if (!window.API_BASE || !devOnly()) return;
  try {
    const code = (document.cookie.match(/(?:^|;\s*)mc_dev=([^;]+)/) || [])[1] || "";
    const r = await fetch(`${window.API_BASE}/jobs`,
                          { headers: { "x-mc-access": decodeURIComponent(code) } });
    if (!r.ok) throw new Error("jobs http " + r.status);
    const { jobs } = await r.json();
    const local = libLoad();
    const byId = Object.fromEntries(local.map(i => [i.jobId, i]));
    const hidden = new Set(hiddenLoad());
    let changed = false;
    for (const j of jobs) {
      if (!j.ready) continue;
      if (hidden.has(j.job_id)) continue;   // user removed it from this browser
      const mine = byId[j.job_id];   // the /jobs Lambda speaks snake_case
      if (mine) {   // upgrade placeholder names/stale status, keep local names
        if ((mine.name === "shared swing" || !mine.name) && j.name) { mine.name = j.name; changed = true; }
        if (mine.status !== "ready") { mine.status = "ready"; changed = true; }
      } else {
        local.push({ jobId: j.job_id, name: j.name, date: j.date, status: "ready" });
        changed = true;
      }
    }
    if (changed) {
      local.sort((a, b) => new Date(b.date) - new Date(a.date));
      libSave(local.slice(0, 200));
      renderLibrary();
    }
  } catch (e) {
    console.warn("team library sync failed:", e);   // local library still works
  }
}

/* ============================ upload flow ============================== */
function statusBox() { const b = $("#upload-status"); b.hidden = false; return b; }

function onFileChosen(file) {
  if (!file) return;
  if (file.size > 200 * 1024 * 1024) {
    statusBox().innerHTML =
      `<p class="err">That video is over the 200 MB upload cap — try a shorter clip.</p>`;
    return;
  }
  if (!window.UPLOAD_URL || !window.RESULTS_BASE) {
    statusBox().innerHTML =
      `<p class="err">The analysis backend isn't reachable right now — please try again later.</p>`;
    return;
  }
  state.upload = { name: file.name, file };
  statusBox().innerHTML = `
    <div class="upload-file">
      <span class="upload-file-name" title="${esc(file.name)}">${esc(file.name)}</span>
      <span class="upload-badge">Ready to analyze</span>
    </div>
    <button id="btn-analyze-upload" class="btn-primary" type="button">Analyze swing</button>`;
  $("#btn-analyze-upload").addEventListener("click", startUpload);
}

async function startUpload() {
  const upload = state.upload;
  if (!upload) return;
  const seq = ++navSeq;
  startCloudAnalyzeUI(seq);
  try {
    // 1 · presigned slot
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
    // 3 · track + wait for the pipeline (~2-3 min; poll up to 8)
    state.pendingJob = slot.job_id;
    libAdd({ jobId: slot.job_id, name: upload.name,
             date: new Date().toISOString(), status: "processing" });
    const ready = await pollJob(slot.job_id, 96, 5000);
    if (seq !== navSeq) return;              // user navigated away meanwhile
    if (ready) { openJob(slot.job_id); return; }
    // still grinding — send them home; the library flips to Ready on its own
    goto("pick");
    statusBox().innerHTML =
      `<p><strong>Still processing.</strong> Your swing is taking longer than usual —
       it will flip to <em>Ready</em> under “Your swings” below when it lands.</p>`;
  } catch (e) {
    console.warn("upload failed:", e);
    if (seq !== navSeq) return;
    goto("pick");
    statusBox().innerHTML =
      `<p class="err"><strong>Upload failed.</strong> Please check your connection and
       try again.</p>`;
  }
}

/* ---- analyze-screen sample preview (canned swing loop) --------------------
 * While the cloud pipeline runs, loop the deployed demo swing (assets/269) as
 * a rotating clay-mannequin preview. Clearly captioned as a SAMPLE — it is
 * NOT the user's upload being reconstructed live. Viewer selection mirrors
 * the results screen: WebGL capsule viewer (slow auto-rotate) with the
 * canvas skeleton as fallback (gentle yaw drift). Decorative only — any
 * failure just leaves the plain screen. */
function stopAnalyzePreview() {
  if (state.analyzeYaw) { clearInterval(state.analyzeYaw); state.analyzeYaw = null; }
  if (state.analyzePreview) {
    if (typeof state.analyzePreview.destroy === "function") state.analyzePreview.destroy();
    state.analyzePreview = null;
  }
  const panel = document.querySelector(".analyze-panel");
  if (panel) panel.classList.remove("analyze-live");
}

async function startAnalyzePreview(seq) {
  stopAnalyzePreview();
  const panel = document.querySelector(".analyze-panel");
  if (!panel || !document.getElementById("analyze-preview")) return;
  panel.classList.add("analyze-live");
  try {
    const data = await fetch("assets/269/replay_3d.json").then(r => r.json());
    if (seq !== navSeq || !panel.classList.contains("analyze-live")) return;
    const canvas = resetCanvas("#analyze-preview");
    const Cap = window.CapsuleViewer3D;
    if (Cap && Cap.supported()) {
      const v = new Cap(canvas, data, {});
      v.controls.autoRotate = true;             // slow turntable (caller-side config)
      v.controls.autoRotateSpeed = 0.8;
      state.analyzePreview = v;
    } else {
      const ui = { scrub: document.createElement("input"),
                   playBtn: document.createElement("button"),
                   label: document.createElement("span") };
      const v = new Replay3D(canvas, data, ui);
      state.analyzePreview = v;
      state.analyzeYaw = setInterval(() => { v.yaw += 0.004; }, 33);
    }
  } catch (e) { /* decorative — never block the analyze screen */ }
}

/* analyze screen paced for the real pipeline */
function startCloudAnalyzeUI(seq) {
  goto("analyze");
  const note = $("#analyze-cloud-note"), elapsed = $("#analyze-elapsed");
  if (note) note.hidden = false;
  const items = [...document.querySelectorAll("#analyze-steps li")];
  items.forEach(li => li.classList.remove("doing", "done"));
  const line = $("#analyze-step-line");
  startAnalyzePreview(seq);
  const t0 = Date.now();
  const timer = setInterval(() => {
    // NB: no stopAnalyzePreview() here — a NEWER analyze run may own the
    // preview by now; every real navigation away goes through goto(), which stops it.
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
      if (line) {                                  // one quiet advancing line (same copy)
        const strong = items[i].querySelector("strong");
        const small = items[i].querySelector("div > span");  // NOT "div span": scoped
        // selectors match against the document, so the outer panel div would
        // make the empty .check span the first hit
        line.classList.remove("show");
        void line.offsetWidth;                     // restart the fade transition
        line.textContent = (strong ? strong.textContent : "") +
                           (small ? " — " + small.textContent : "");
        line.classList.add("show");
      }
      i += 1;
      if (i < items.length) setTimeout(tick, 15000);   // ~90s across 7 real steps
    }
  };
  tick();
}
state.debugAnalyze = () => startCloudAnalyzeUI(++navSeq);  // console/QA hook (same spirit as __MC_STATE__)

/* ---- readiness: ALL result files must exist behind CloudFront ----
 * NB: CloudFront rewrites S3 403s to 200/index.html, so require a non-HTML
 * content-type too. */
async function jobIsReady(jobId) {
  const files = ["metrics.json", "explanation.json", "replay_3d.json"];
  const okReal = (r) =>
    r.ok && !(r.headers.get("content-type") || "").includes("html");
  const oks = await Promise.all([
    ...files.map(f =>
      fetch(`${window.RESULTS_BASE}/${jobId}/${f}`, { cache: "no-store" })
        .then(okReal).catch(() => false)),
    fetch(`${window.RESULTS_BASE}/${jobId}/overlay.mp4`,
          { method: "HEAD", cache: "no-store" })
      .then(okReal).catch(() => false),
  ]);
  return oks.every(Boolean);
}

async function pollJob(jobId, tries = 40, delayMs = 15000) {
  for (let i = 0; i < tries; i++) {
    try {
      if (await jobIsReady(jobId)) { libSetStatus(jobId, "ready"); return true; }
    } catch (e) { /* keep polling */ }
    await new Promise(res => setTimeout(res, delayMs));
  }
  libSetStatus(jobId, "processing (check back)");
  return false;
}

/* refresh-proofing: prune dead failures, re-check every non-ready job */
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

/* ====================== library card rendering ========================= */
function renderLibrary() {
  const box = $("#library"), wrap = $("#library-cards");
  if (!box) return;
  const items = libLoad();
  box.hidden = !items.length;
  if (!items.length) return;
  if (!devOnly()) {
    wrap.innerHTML = `<p class="muted small">🔒 ${items.length} analyzed swing${items.length > 1 ? "s" : ""}
      — private during the pilot. Sign in with the dev account to view.</p>`;
    return;
  }
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
      <button type="button" class="btn-ghost small lib-rename" title="Rename this swing"
              aria-label="Rename ${esc(it.name)}">✎</button>
      <button type="button" class="btn-ghost small lib-remove" title="Remove from this list"
              aria-label="Remove ${esc(it.name)} from this list">✕</button>
    </div>`;
  }).join("");
  wrap.querySelectorAll(".lib-open").forEach(b =>
    b.addEventListener("click", () => openJob(b.closest(".dash-swing").dataset.job)));
  wrap.querySelectorAll(".lib-rename").forEach(b =>
    b.addEventListener("click", () => libRename(b.closest(".dash-swing").dataset.job)));
  wrap.querySelectorAll(".lib-remove").forEach(b =>
    b.addEventListener("click", () => libRemove(b.closest(".dash-swing").dataset.job)));
}

/* rename a swing (persists in localStorage across sessions; the team-library
 * sync never overwrites a custom name — it only fills placeholders) */
function libRename(jobId) {
  const items = libLoad();
  const it = items.find(i => i.jobId === jobId);
  if (!it) return;
  const next = prompt("Rename this swing:", it.name || "");
  if (next == null) return;                 // cancelled
  const name = next.trim().slice(0, 80);
  if (!name || name === it.name) return;
  it.name = name;
  libSave(items);
  renderLibrary();
  if (state.selectedId === jobId) {
    $("#results-clip-label").textContent =
      `Your swing · ${name} · analyzed by the real pipeline`;
  }
}

/* ========================== results screen ============================= */
function metricByKey(key) {
  return state.bundle.metrics.metrics.find(m => m.key === key);
}

/* every ready job except the active one — for compare + the rail */
function otherReadyJobs() {
  return libLoad().filter(i => i.status === "ready" && i.jobId !== state.selectedId);
}

/* chat + compare need metrics; fetch once per job and cache */
async function jobMetricsFor(jobId) {
  if (!state.jobMetrics[jobId]) {
    state.jobMetrics[jobId] =
      await fetch(`${window.RESULTS_BASE}/${jobId}/metrics.json`).then(r => r.json());
  }
  return state.jobMetrics[jobId];
}

/* register the active job (and its ready siblings) with the grounded chat */
async function activateChat(jobId) {
  const entries = [{ jobId, name: libName(jobId) || "this swing" },
                   // cap chat compare prefetch — one metrics.json per sibling
                   ...otherReadyJobs().slice(0, 12).map(i => ({ jobId: i.jobId, name: i.name }))];
  const clips = [], byId = {};
  for (const e of entries) {
    try {
      byId[e.jobId] = await jobMetricsFor(e.jobId);
      clips.push({ id: e.jobId, title: `your swing (${e.name})`, club: "", view: "" });
    } catch (err) { /* a job without metrics just won't be chat-enabled */ }
  }
  Chat.setLibrary(clips, byId);
  Chat.activate(jobId);
}

async function openJob(jobId) {
  if (!window.RESULTS_BASE) return;
  if (!devOnly()) {
    pendingDeepLink = jobId;      // reopened automatically after dev sign-in
    goto("pick");
    showPilotNotice();
    Auth.openSignIn(true);
    return;
  }
  const seq = ++navSeq;
  try {
    if (state.pendingJob === jobId) state.pendingJob = null;
    state.selectedId = jobId;
    state.upload = null;
    setHash(jobId);
    const bundle = await loadJobBundle(jobId);
    if (seq !== navSeq) return;
    state.bundle = bundle;
    state.jobMetrics[jobId] = bundle.metrics;
    // a shared/deep-linked job may not be in this browser's library yet
    if (!libLoad().some(i => i.jobId === jobId)) {
      libAdd({ jobId, name: "shared swing", date: new Date().toISOString(), status: "ready" });
    } else {
      libSetStatus(jobId, "ready");
    }
    renderResults();
    goto("results");
    activateChat(jobId);          // async: chat panel fills in as metrics land
  } catch (e) {
    if (seq !== navSeq) return;
    goto("pick");
    // Don't guess — check WHY it failed. An expired/wrong access cookie makes
    // every artifact fetch fail exactly like a missing job would.
    let locked = false;
    try {
      const probe = await fetch(`${window.RESULTS_BASE}/__access_check__`, { cache: "no-store" });
      locked = probe.status !== 204;
    } catch (e2) { /* network trouble — fall through to the generic message */ }
    if (locked) {
      Auth.clearDevCookie();
      renderLibrary();
      statusBox().innerHTML =
        `<p class="err"><strong>Your dev access has expired or the code changed.</strong>
         Sign in again with the access code to view swings.</p>`;
      Auth.openSignIn(true);
    } else {
      console.warn("openJob failed for", jobId, e);
      statusBox().innerHTML =
        `<p class="err">That swing's results aren't ready yet — check back in a minute.</p>`;
    }
  }
}

function renderResults() {
  const { metrics, explanation, overlayUrl, replay } = state.bundle;
  const name = libName(state.selectedId) || "your upload";

  $("#results-banner").hidden = true;
  $("#results-clip-label").textContent = `Your swing · ${name} · analyzed by the real pipeline`;
  $("#results-headline").textContent = explanation.headline;

  // ball-flight estimate card (simulated; same engine + hand-speed nudge rule
  // as the coach). Uploads don't record a club, so it defaults to driver at
  // amateur launch numbers with a picker.
  if (window.BallFlight) {
    const hs = ((state.bundle.scorecard || {}).indicators || {}).hand_speed_impact_bs;
    BallFlight.mount({ club: null, tier: "amateur", metricsRows: metrics.metrics,
                       handSpeed: hs ? { value: hs.value, median: hs.pro_median } : null,
                       ball: state.bundle.ball });
  }

  renderPlain(explanation);
  initListen(explanation, String(state.selectedId));
  renderNumbers(metrics.metrics);
  populateNumbersCompare();

  // share: #job=<id> deep link works in any browser (results are public by link)
  $("#btn-share").onclick = async () => {
    const url = `${location.origin}${location.pathname}#job=${state.selectedId}`;
    try { await navigator.clipboard.writeText(url); $("#btn-share").textContent = "Link copied ✓"; }
    catch (e) { prompt("Copy this link:", url); }
    setTimeout(() => { $("#btn-share").textContent = "Share link"; }, 1800);
  };

  const vid = $("#overlay-video");
  vid.src = overlayUrl;
  vid.load();
  renderEventMarkers(vid, metrics, replay);

  if (state.viewer) state.viewer.destroy();
  const canvas = resetCanvas("#replay-canvas");
  const ui3d = { scrub: $("#replay-scrub"), label: $("#replay-frame"), playBtn: $("#replay-play") };
  const Cap = window.CapsuleViewer3D;
  state.viewer = (Cap && Cap.supported())
    ? new Cap(canvas, replay, ui3d)
    : new Replay3D(canvas, replay, ui3d);

  wireReplaySync(vid, replay);
  renderRecentRail();
  showTab("plain");
}

/* ---- detected-event markers under the overlay video (click to seek) ---- */
const EVENT_LABELS = {
  address: "Address", toe_up: "Toe-up", mid_backswing: "Mid-backswing", top: "Top",
  mid_downswing: "Mid-downswing", impact: "Impact",
  mid_follow_through: "Follow-through", finish: "Finish",
};
function renderEventMarkers(vid, metrics, replay) {
  const bar = $("#event-markers");
  const events = metrics.events || {};
  const total = replay && replay.frames ? replay.frames.length : 0;
  const fps = (replay && replay.fps) || 30;
  const keys = Object.keys(events);
  if (!keys.length || !total) { bar.hidden = true; bar.innerHTML = ""; return; }
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

/* ---- rail: hop between your ready swings without leaving results ---- */
function renderRecentRail() {
  const rail = $("#recent-rail"), cards = $("#rail-cards");
  // cap the rail: every card is a <video> fetching metadata, and with the
  // shared team library (~45 swings) an uncapped rail saturates the browser's
  // per-host connections and starves the MAIN overlay video (it never loads)
  const others = otherReadyJobs().slice(0, 8);
  if (!others.length) { rail.hidden = true; return; }
  rail.hidden = false;
  cards.innerHTML = "";
  for (const it of others) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "rail-card";
    b.innerHTML = `<video src="${esc(window.RESULTS_BASE)}/${esc(it.jobId)}/overlay.mp4?${MEDIA_BUST}"
                     preload="metadata" muted playsinline></video>
                   <span>${esc(it.name)}</span>`;
    b.addEventListener("click", () => openJob(it.jobId));
    cards.appendChild(b);
  }
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

/* "Listen" — the explanation read by the studio coach voice (POST /tts) */
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

/* compare selector on the numbers tab — your other ready swings */
function populateNumbersCompare() {
  const sel = $("#numbers-compare");
  const others = otherReadyJobs();
  sel.parentElement.hidden = !others.length;
  sel.innerHTML = `<option value="">— none —</option>` +
    others.map(i => `<option value="${esc(i.jobId)}">${esc(i.name)}</option>`).join("");
  sel.onchange = async () => {
    if (!sel.value) { renderNumbers(state.bundle.metrics.metrics); return; }
    try {
      const cmp = await jobMetricsFor(sel.value);
      renderNumbers(state.bundle.metrics.metrics, cmp);
    } catch (e) { renderNumbers(state.bundle.metrics.metrics); }
  };
}

function showTab(which) {
  $("#tab-plain").classList.toggle("active", which === "plain");
  $("#tab-numbers").classList.toggle("active", which === "numbers");
  $("#tab-chat").classList.toggle("active", which === "chat");
  $("#view-plain").hidden = which !== "plain";
  $("#view-numbers").hidden = which !== "numbers";
  $("#view-chat").hidden = which !== "chat";
}

/* =========================================================================
 * Replay3D — tiny dependency-free 3D skeleton viewer (canvas 2D) — the
 * fallback when the WebGL capsule viewer isn't supported. Same as app.js.
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

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.width, h = canvas.clientHeight || canvas.height;
    canvas.width = w * dpr; canvas.height = h * dpr;
    this.w = w; this.h = h;
    this.ctx.scale(dpr, dpr);

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

    const bones = [...this.data.bones].sort((p, q) =>
      Math.min(proj[p.a][2], proj[p.b][2]) - Math.min(proj[q.a][2], proj[q.b][2]));
    for (const bone of bones) {
      const a = proj[bone.a], b = proj[bone.b];
      ctx.strokeStyle = isClub(bone) ? "#8e9089" : bone.color;
      ctx.lineWidth = (isClub(bone) ? 2.0 : 3.5) * Math.min(a[2], b[2]);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    ctx.fillStyle = "#faf9f5";
    for (const p of proj) {
      ctx.beginPath(); ctx.arc(p[0], p[1], 2.4 * p[2], 0, Math.PI * 2); ctx.fill();
    }

    this.ui.label.textContent = `${this.frame + 1} / ${this.data.frames.length}`;
  }
}

/* ================================ wiring ================================ */
function fillDevCurl() {
  const pre = $("#dev-curl");
  if (!pre || !window.API_BASE) return;
  pre.textContent =
`# 1 · request an upload slot
curl -s -X POST ${window.API_BASE}/upload-url \\
  -H 'content-type: application/json' \\
  -d '{"filename":"swing.mp4","content_type":"video/mp4"}'
# -> {"job_id":"...","url":"https://...s3...","fields":{...}}

# 2 · send the video straight to storage (form fields, then the file)
curl -s -X POST <url> -F key=<fields.key> ... -F file=@swing.mp4

# 3 · poll for results (private during the pilot — needs the dev access code)
curl -sI "${window.RESULTS_BASE}/<job_id>/metrics.json?t=<access-code>"`;
}

function renderHeroGreeting() {
  const sub = $("#hero-sub");
  if (!sub) return;
  const u = window.Auth && Auth.isSignedIn() ? Auth.currentUser() : null;
  if (u && devOnly()) {
    sub.textContent = `Welcome back, ${u.displayName || u.username}. Upload a swing to add ` +
      `a new analysis — everything you've measured is under “Your swings” below.`;
  } else if (u) {
    sub.textContent = `Hi ${u.displayName || u.username} — MotionCaddie is in a private ` +
      `pilot with real players, so uploads and results are limited to the dev account for now.`;
  } else {
    sub.textContent = "Upload one phone video. MotionCaddie runs the full 3D pipeline in " +
      "the cloud and returns a stabilized pose overlay, a 3D replay you can spin, " +
      "tour-reference metrics, and a coach you can ask about any of it.";
  }
  // say the quiet part BEFORE the click: the buttons are gated during the pilot
  const hint = $("#upload-hint");
  if (hint) {
    hint.innerHTML = devOnly()
      ? "MP4 or MOV, up to 200&nbsp;MB. Analysis runs on the real cloud pipeline and usually takes 2–3 minutes."
      : "🔒 Private pilot — uploading needs the shared team account " +
        `(<code>${esc((window.Auth && Auth.DEV_USER) || "dev@motioncaddie.dev")}</code> + team access code).`;
  }
}

function init() {
  Chat.init({
    stream: $("#chat-stream"), input: $("#chat-q"), send: $("#chat-send"),
    compare: $("#chat-compare"), active: $("#chat-active"), mode: $("#chat-mode"),
    mic: $("#chat-mic"), voice: $("#chat-voice"), voicePick: $("#chat-voice-pick"),
    chips: [...document.querySelectorAll("#view-chat .chip-btn")],
  });

  $("#btn-upload").addEventListener("click", () => {
    if (!devOnly()) { showPilotNotice(); Auth.openSignIn(true); return; }
    $("#file-input").click();
  });
  $("#btn-record").addEventListener("click", () => {
    if (!devOnly()) { showPilotNotice(); Auth.openSignIn(true); return; }
    $("#camera-input").click();
  });
  $("#file-input").addEventListener("change", (e) => onFileChosen(e.target.files[0]));
  $("#camera-input").addEventListener("change", (e) => onFileChosen(e.target.files[0]));

  const goHome = () => {
    if (state.viewer) { state.viewer.destroy(); state.viewer = null; }
    $("#overlay-video").pause();
    setHash(null);
    goto("pick");
  };
  $("#btn-restart").addEventListener("click", goHome);
  // brand is a plain link to the site home (index) — no handler needed

  window.addEventListener("hashchange", onHashChange);
  reconcileLibrary();
  $("#tab-plain").addEventListener("click", () => showTab("plain"));
  $("#tab-numbers").addEventListener("click", () => showTab("numbers"));
  $("#tab-chat").addEventListener("click", () => showTab("chat"));

  if (window.Auth) Auth.init();
  document.addEventListener("mc-auth-change", () => {
    renderHeroGreeting();
    renderLibrary();
    verifyDevAccess();            // edge-validates the code; retries deep links
  });
  renderHeroGreeting();
  fillDevCurl();
  verifyDevAccess();

  // resume any still-processing jobs from a previous visit
  if (window.RESULTS_BASE) {
    for (const it of libLoad()) {
      if (it.status && it.status.startsWith("processing")) pollJob(it.jobId, 8, 15000);
    }
  }

  onHashChange();               // honor a #job=<id> deep link on first load
  if (!location.hash || location.hash === "#dev-login") goto("pick");
}

init();
