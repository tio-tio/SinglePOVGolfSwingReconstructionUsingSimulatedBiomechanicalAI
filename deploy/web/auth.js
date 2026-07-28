/* MotionCaddie — PROTOTYPE account + consent (front-end illusion only).
 *
 * There is NO backend and NO real authentication here. This module stores a small
 * demo profile in localStorage so the UI can show a signed-in state. It NEVER stores
 * or transmits a password — the password field exists only for demo realism and its
 * value is read, ignored, and discarded.
 *
 * When a real pipeline/auth is wired later, replace the store + the two submit
 * handlers; the consent wording below (CONSENT_COPY) is already the single source of
 * truth, and the stored consentVersion lets a real deployment know which text a user
 * agreed to (and honor anyone who opted out of the model-improvement use).
 */
"use strict";

/* ======================================================================== *
 *  CONSENT_COPY — the ONE place to edit consent + privacy wording.
 *  "may / can" phrasing is deliberate: it grants permission for a possible
 *  future use, which is honest both now (nothing is uploaded yet) and after
 *  the real pipeline ships. This stays honest ONLY because the model-
 *  improvement use is a genuine, separate opt-in (off by default), not a
 *  bundled mandatory checkbox — see the two entries below.
 * ======================================================================== */
const CONSENT_COPY = {
  version: "2026-07-05",
  microcopy: "Prototype account only — no real authentication yet.",

  // Required to use the product: permission to analyze the user's own swing.
  analysis: {
    label: "I agree that videos I upload may be used to analyze my swing.",
    note: "Required to use MotionCaddie.",
  },
  // Genuinely OPTIONAL, off by default: permission to use the data to improve models.
  model: {
    label: "I also agree that my uploaded videos and the pose data derived from them may be used to help improve future models.",
    note: "Optional — you can use MotionCaddie without this, and can withdraw it later.",
  },

  privacyLinkText: "Data & Privacy",
  // One-line summary shown next to the checkboxes; the full detail lives in privacy.html.
  privacySummary:
    "MotionCaddie may use the swing videos you upload — and the pose data derived from them — to analyze your swing. If you opt in, that same data may also be used to help improve future models. The Data & Privacy page explains what is collected, how long it is kept, and how to withdraw consent or request deletion.",

  // Full privacy detail, rendered by privacy.html (kept here so wording is one-place).
  privacy: {
    title: "Data & Privacy",
    updated: "2026-07-05",
    intro:
      "This is a prototype. Today MotionCaddie runs entirely in your browser on demo data — nothing you select or type is uploaded anywhere, and there is no real account system. This page describes how data WOULD be handled once the hosted analysis pipeline is connected, so the consent you give now is informed.",
    sections: [
      { h: "What may be collected",
        p: "The swing video you upload, the pose data derived from it (2D and 3D joint positions per frame), the swing metrics computed from that pose, and the small profile you enter when creating an account (a display name and an email-like username). Passwords are never stored." },
      { h: "How it may be used",
        p: "Your video and derived pose data may be used to analyze your swing and show you results — this is the core product and is required to use it. Separately, and only if you opt in, the same data may be used as training data to help improve future models. That model-improvement use is optional: you can use MotionCaddie without it." },
      { h: "The optional model-improvement use",
        p: "If (and only if) you tick the optional consent box, your uploaded videos and derived pose data may be added to datasets used to train and evaluate future versions of the models. If you leave it unticked, your data is used only to analyze your own swing. This choice is stored with your profile so it can be honored." },
      { h: "Retention",
        p: "In this prototype nothing is retained server-side because nothing is uploaded. In a live deployment, uploaded videos and derived data would be retained only as long as needed to provide results and, where opted in, to improve models — and deleted on request." },
      { h: "Withdrawing consent or requesting deletion",
        p: "You can withdraw the optional model-improvement consent at any time from your account, or request deletion of your data. In this prototype, clearing your browser's site data removes the locally stored profile and consent record. A live deployment would provide an in-product control and a deletion request path." },
      { h: "Contact",
        p: "This is a student capstone prototype (MotionCaddie). Questions about data handling can be directed to the project team." },
    ],
  },
};
if (typeof window !== "undefined") window.CONSENT_COPY = CONSENT_COPY;

/* ============================== Auth module ============================= */
const Auth = (() => {
  const PROFILE_KEY = "mc_profile";     // persisted demo account (localStorage)
  const SESSION_KEY = "mc_session";     // "1" while signed in (localStorage — persists until explicit sign-out)

  /* ---- dev access (REAL, edge-enforced) ----
   * Uploaded swings are of real people, so their results (03_outputs/* and
   * job audio) are locked at CloudFront by the mc-results-gate function: it
   * requires an `mc_dev` cookie holding the access code. Signing in as the
   * dev account stores the entered code in that cookie — the page never
   * validates or embeds the code; the EDGE does. Wrong code => every private
   * fetch 403s and the UI stays locked. */
  const DEV_USER = "dev@motioncaddie.dev";
  const DEV_COOKIE = "mc_dev";
  function setDevCookie(code) {
    document.cookie = `${DEV_COOKIE}=${encodeURIComponent(code)}; path=/; secure; samesite=Lax; max-age=2592000`;
  }
  function clearDevCookie() {
    document.cookie = `${DEV_COOKIE}=; path=/; secure; samesite=Lax; max-age=0`;
  }
  function hasDevCookie() {
    return document.cookie.split(/;\s*/).some(c => c.startsWith(DEV_COOKIE + "=") && c.length > DEV_COOKIE.length + 1);
  }
  function isDev() {
    const u = currentUser();
    return !!u && u.username === DEV_USER && hasDevCookie();
  }

  const q = (id) => document.getElementById(id);

  function getProfile() {
    try { return JSON.parse(localStorage.getItem(PROFILE_KEY) || "null"); }
    catch (e) { return null; }
  }
  function isSignedIn() { return localStorage.getItem(SESSION_KEY) === "1" && !!getProfile(); }
  function currentUser() { return isSignedIn() ? getProfile() : null; }

  function emit() { document.dispatchEvent(new CustomEvent("mc-auth-change")); }

  function saveProfile(p) { localStorage.setItem(PROFILE_KEY, JSON.stringify(p)); }

  /* ---- account actions (NO password ever stored) ---- */
  function createAccount({ displayName, username, consentGiven, modelUseConsent }) {
    const profile = {
      displayName: (displayName || "").trim() || username,
      username: (username || "").trim(),
      consentGiven: !!consentGiven,               // analysis use (required)
      modelUseConsent: !!modelUseConsent,         // optional model-improvement opt-in
      consentTimestamp: new Date().toISOString(),
      consentVersion: CONSENT_COPY.version,
    };
    saveProfile(profile);
    localStorage.setItem(SESSION_KEY, "1");
    emit();
  }
  function signIn({ username, password }) {
    let p = getProfile();
    if (!p || p.username !== (username || "").trim()) {
      // returning-user demo: no stored match, mint a minimal session profile.
      p = { displayName: (username || "").trim(), username: (username || "").trim(),
            consentGiven: false, modelUseConsent: false, consentTimestamp: null,
            consentVersion: CONSENT_COPY.version };
      saveProfile(p);
    }
    // dev account: the password IS the results access code — it goes into the
    // edge-checked cookie (never into localStorage) and nowhere else.
    if ((username || "").trim() === DEV_USER && (password || "").length) {
      setDevCookie(password);
    }
    localStorage.setItem(SESSION_KEY, "1");
    emit();
  }
  function signOut() {
    localStorage.removeItem(SESSION_KEY);
    clearDevCookie();                       // locking the UI also drops edge access
    emit();
  }

  /* ------------------------------ modals ------------------------------ */
  let lastFocus = null;
  function openModal(id) {
    const m = q(id); if (!m) return;
    lastFocus = document.activeElement;
    m.hidden = false; document.body.classList.add("modal-open");
    const f = m.querySelector("input, button"); if (f) f.focus();
  }
  function closeModals() {
    document.querySelectorAll(".modal-backdrop").forEach(m => { m.hidden = true; });
    document.body.classList.remove("modal-open");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  /* ---- render the topbar account area ---- */
  function renderAccountArea() {
    const area = q("account-area"); if (!area) return;
    if (isSignedIn()) {
      const u = currentUser();
      area.innerHTML =
        `<span class="acct-name" title="${u.username}">${u.displayName}</span>` +
        `<button class="btn-text" id="btn-signout" type="button">Sign out</button>`;
      const so = q("btn-signout"); if (so) so.addEventListener("click", signOut);
    } else {
      area.innerHTML =
        `<button class="btn-text" id="btn-signin" type="button">Sign in</button>` +
        `<button class="btn-secondary btn-sm" id="btn-create" type="button">Create account</button>`;
      const bi = q("btn-signin"); if (bi) bi.addEventListener("click", () => openModal("modal-signin"));
      const bc = q("btn-create"); if (bc) bc.addEventListener("click", () => openModal("modal-create"));
    }
  }

  /* ---- inject consent wording from CONSENT_COPY (single source) ---- */
  function fillConsentCopy() {
    const set = (id, txt) => { const n = q(id); if (n) n.textContent = txt; };
    set("consent-analysis-label", CONSENT_COPY.analysis.label);
    set("consent-analysis-note", CONSENT_COPY.analysis.note);
    set("consent-model-label", CONSENT_COPY.model.label);
    set("consent-model-note", CONSENT_COPY.model.note);
    set("consent-summary", CONSENT_COPY.privacySummary);
    document.querySelectorAll(".consent-microcopy").forEach(n => { n.textContent = CONSENT_COPY.microcopy; });
    document.querySelectorAll(".privacy-link").forEach(n => { n.textContent = CONSENT_COPY.privacyLinkText; });
  }

  function wire() {
    // close controls
    document.querySelectorAll("[data-close-modal]").forEach(b => b.addEventListener("click", closeModals));
    document.querySelectorAll(".modal-backdrop").forEach(m =>
      m.addEventListener("click", e => { if (e.target === m) closeModals(); }));
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeModals(); });

    // create-account: gate submit on the required analysis checkbox
    const createForm = q("form-create");
    const analysisBox = q("ca-consent-analysis");
    const createBtn = q("ca-submit");
    if (createForm && analysisBox && createBtn) {
      const sync = () => { createBtn.disabled = !analysisBox.checked; };
      analysisBox.addEventListener("change", sync); sync();
      createForm.addEventListener("submit", e => {
        e.preventDefault();
        if (!analysisBox.checked) return;   // hard guard, never create without it
        createAccount({
          displayName: q("ca-name") ? q("ca-name").value : "",
          username: q("ca-username") ? q("ca-username").value : "",
          consentGiven: analysisBox.checked,
          modelUseConsent: q("ca-consent-model") ? q("ca-consent-model").checked : false,
        });
        createForm.reset(); sync(); closeModals();
      });
    }

    // sign-in (password is ignored for demo accounts; for the dev account it
    // is the results access code and feeds the edge-checked cookie)
    const signinForm = q("form-signin");
    if (signinForm) {
      signinForm.addEventListener("submit", e => {
        e.preventDefault();
        signIn({ username: q("si-username") ? q("si-username").value : "",
                 password: q("si-password") ? q("si-password").value : "" });
        signinForm.reset(); closeModals();
      });
    }

    // switch between the two modals
    const toCreate = q("switch-to-create"); if (toCreate) toCreate.addEventListener("click", () => { closeModals(); openModal("modal-create"); });
    const toSignin = q("switch-to-signin"); if (toSignin) toSignin.addEventListener("click", () => { closeModals(); openModal("modal-signin"); });
  }

  /* dev quick-login: open the site at #dev-login. Real swing results are now
   * edge-gated behind the dev access code, so this no longer signs anyone in
   * by itself — it opens the sign-in form with the dev username prefilled and
   * the access code left for the human to type. */
  function devLogin() {
    history.replaceState(null, "", location.pathname + location.search);
    openModal("modal-signin");
    const u = q("si-username"); if (u) u.value = DEV_USER;
    const p = q("si-password"); if (p) p.focus();
  }

  function init() {
    fillConsentCopy();
    wire();
    renderAccountArea();
    document.addEventListener("mc-auth-change", renderAccountArea);
    // defer so app.js has registered its own mc-auth-change listeners
    if (location.hash === "#dev-login") setTimeout(devLogin, 0);
  }

  return { init, isSignedIn, currentUser, signOut, isDev, clearDevCookie,
           DEV_USER, openCreate: () => openModal("modal-create"),
           /* prefill=true drops the dev username in so the pilot gate only
            * ever asks the human for the access code */
           openSignIn: (prefill) => {
             openModal("modal-signin");
             if (prefill) {
               const u = q("si-username"); if (u) u.value = DEV_USER;
               const p = q("si-password"); if (p) p.focus();
             }
           } };
})();
if (typeof window !== "undefined") window.Auth = Auth;
