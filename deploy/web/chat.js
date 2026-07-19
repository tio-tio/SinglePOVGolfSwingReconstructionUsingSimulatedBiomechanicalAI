/* MotionCaddie — coaching Q&A chat (folded in from Banjot's coaching_chatbot_demo.html).
 *
 * The grounded-answer ENGINE (indicator copy, tool mocks, keyword router, refusal
 * taxonomy, grounding verifier) is ported verbatim in spirit from that page so the
 * offline demo behaves like the real agent. What changed for this integration:
 *   - per-clip VALUES / BAND / STATUS are read from each clip's assets/<id>/metrics.json
 *     (the SAME source the "The numbers" tab renders) so chat and table never disagree.
 *   - META below is copy ONLY (plain names, in/low/high phrasing, glossary).
 *
 * TODO(live-backend): flip to the real grounded agent by setting `window.API_BASE`
 * (e.g. a Lambda Function URL, or serve_demo.py's origin). askChat() then POSTs
 *   {clip_id, question, history[], compare_clip_id?}  ->  {answer, grounded, violations, tool_log}
 * and nothing else changes. Unset => the in-page mock below answers. This is the
 * single chat swap seam, mirroring loadClipBundle() for data.
 */
"use strict";

/* ---- indicator COPY (bands/values come from metrics.json, not here) ---- */
const CHAT_META = {
  tempo_ratio: { plain: "the rhythm of the swing", unit: "",
    in: "the swing rhythm is close to the tour-typical pace",
    lo: "the backswing-to-downswing rhythm is quicker than the roughly 3-to-1 tour norm",
    hi: "the backswing is longer relative to the downswing than the roughly 3-to-1 tour norm" },
  shoulder_turn_top_deg: { plain: "how far the shoulders rotate at the top", unit: "°",
    in: "the shoulders rotate about as far as a typical tour swing at the top",
    lo: "the shoulders rotate less than most tour swings at the top",
    hi: "the shoulders rotate further than most tour swings at the top" },
  hip_turn_top_deg: { plain: "how far the hips rotate at the top", unit: "°",
    in: "the hips rotate about as far as a typical tour swing at the top",
    lo: "the hips rotate less than most tour swings at the top",
    hi: "the hips rotate further than most tour swings at the top" },
  x_factor_top_deg: { plain: "the gap between shoulder turn and hip turn at the top", unit: "°",
    in: "the separation between shoulders and hips is about typical for a tour swing",
    lo: "there is less separation between the shoulders and hips than in most tour swings",
    hi: "there is more separation between the shoulders and hips than in most tour swings" },
  hip_turn_impact_deg: { plain: "how open the hips are at impact", unit: "°",
    in: "the hips are about as open at impact as a typical tour swing",
    lo: "the hips are less open at impact than most tour swings",
    hi: "the hips are more open at impact than most tour swings" },
  spine_tilt_address_deg: { plain: "how much the upper body leans at setup", unit: "°",
    in: "the upper body leans about as much as a typical tour swing at setup",
    lo: "the upper body leans less than most tour swings at setup",
    hi: "the upper body leans more than most tour swings at setup" },
  posture_loss_deg: { plain: "how much the spine angle changes from setup to impact", unit: "°",
    in: "the posture holds about as steadily from setup to impact as a typical tour swing",
    lo: "",
    hi: "the spine angle changes more from setup to impact than in most tour swings, meaning the posture shifts during the swing" },
  head_sway_max_pct: { plain: "how much the head drifts sideways", unit: "%",
    in: "the head stays about as steady side-to-side as a typical tour swing",
    lo: "the head moves less side-to-side than most tour swings",
    hi: "the head drifts further side-to-side than most tour swings" },
  hip_lateral_shift_pct: { plain: "how much the hips slide toward the target", unit: "%",
    in: "the hips slide toward the target about as much as a typical tour swing",
    lo: "the hips slide toward the target less than most tour swings",
    hi: "the hips slide sideways more than most tour swings" },
  left_arm_bend_top_deg: { plain: "how straight the lead arm is at the top", unit: "°",
    in: "the lead arm is about as extended at the top as a typical tour swing",
    lo: "the lead arm is more bent at the top than most tour swings", hi: "" },
};
const GLOSSARY = {
  "tour range": "the typical spread of values seen across professional tour swings",
  "tour median": "the middle value among professional tour swings",
  "address": "the setup position, just before the swing starts",
  "top": "the top of the backswing, where the club changes direction",
  "impact": "the moment the club meets the ball",
  "x-factor": "the difference between how far the shoulders and the hips have turned at the top — a measure of coil",
};

/* ============================== chat module ============================== */
const Chat = (() => {
  let ACTIVE = null, COMPARE = null;
  const CLIPDATA = {};       // id -> { name, club, view, ind:{key:{label,value,unit,band,tour,status,tier,plain,in,lo,hi}} }
  const HIST = {};           // per-clip [{role,content}] for the live backend
  let dom = {};

  /* ---- build a clip's indicator record from its metrics.json rows ---- */
  function buildFromMetrics(clip, rows) {
    const ind = {};
    for (const r of rows) {
      const m = CHAT_META[r.key] || {};
      ind[r.key] = {
        label: r.label, value: r.you, unit: m.unit || "",
        band: r.band, tour: r.tour, status: r.status,      // "in" | "watch" | "low"
        tier: r.status === "low" ? "low" : "med",
        plain: m.plain || r.label, in: m.in || "", lo: m.lo || "", hi: m.hi || "",
      };
    }
    CLIPDATA[clip.id] = { name: clip.title, club: clip.club, view: clip.view, ind };
  }

  /* ---- accessors ---- */
  const IND = (c) => CLIPDATA[c].ind;
  const keys = (c) => Object.keys(IND(c));
  const val = (k, c) => IND(c)[k].value;
  const unit = (k, c) => IND(c)[k].unit;
  const band = (k, c) => IND(c)[k].band;
  const tour = (k, c) => IND(c)[k].tour;
  const tier = (k, c) => IND(c)[k].tier;
  const label = (k, c) => IND(c)[k].label;
  const inRange = (k, c) => IND(c)[k].status === "in";
  const reliableKeys = (c) => keys(c).filter(k => tier(k, c) !== "low");
  const clipName = (c) => (CLIPDATA[c] ? CLIPDATA[c].name : "swing " + c);

  const cap = s => s.charAt(0).toUpperCase() + s.slice(1);
  const low = s => s.charAt(0).toLowerCase() + s.slice(1);
  const listEng = a => a.length <= 1 ? (a[0] || "") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];

  /* ---- tools (mirror coaching_chat.py; read ACTIVE, COMPARE) ---- */
  function t_list() { return { indicators: keys(ACTIVE).map(k => ({ key: k, label: label(k, ACTIVE), reliable: tier(k, ACTIVE) !== "low" })), count: keys(ACTIVE).length }; }
  function t_get(k) {
    if (!IND(ACTIVE)[k]) return { measured: false, key: k, note: "not measured for this swing" };
    const o = { measured: true, key: k, label: label(k, ACTIVE), value: val(k, ACTIVE), unit: unit(k, ACTIVE),
      tour_median: tour(k, ACTIVE), pro_band: band(k, ACTIVE), in_tour_range: inRange(k, ACTIVE),
      confidence_tier: tier(k, ACTIVE), reliable: tier(k, ACTIVE) !== "low" };
    if (!o.reliable) o.note = "LOW CONFIDENCE from a single camera — do not state or judge the value";
    return o;
  }
  function t_compare(k) {
    if (!IND(ACTIVE)[k]) return { comparable: false, key: k };
    if (COMPARE == null) return { comparable: false, key: k, note: "no comparison swing selected" };
    if (tier(k, ACTIVE) === "low") return { comparable: false, key: k, confidence_tier: "low", note: "low-confidence; not reliable enough to compare" };
    const a = val(k, ACTIVE), b = (IND(COMPARE)[k] ? val(k, COMPARE) : null);
    if (b == null) return { comparable: false, key: k, note: "not measured on the comparison swing" };
    const d = Math.round((a - b) * 100) / 100;
    return { comparable: true, key: k, earlier_value: b, current_value: a, delta: d,
      direction: d > 0 ? "increased" : d < 0 ? "decreased" : "unchanged",
      current_in_tour_range: inRange(k, ACTIVE), earlier_in_tour_range: inRange(k, COMPARE) };
  }
  function t_flagged() { const f = reliableKeys(ACTIVE).filter(k => !inRange(k, ACTIVE)); return { flagged: f.map(k => ({ key: k, label: label(k, ACTIVE) })), count: f.length }; }
  function t_summary() { const rel = reliableKeys(ACTIVE); return { n_indicators: keys(ACTIVE).length, n_reliable: rel.length, n_in_tour_range: rel.filter(k => inRange(k, ACTIVE)).length, n_flagged: rel.filter(k => !inRange(k, ACTIVE)).length }; }
  function t_term(term) { const t = term.toLowerCase().trim(); return GLOSSARY[t] ? { term: t, in_glossary: true, gloss: GLOSSARY[t] } : { term: t, in_glossary: false, note: "not in glossary" }; }

  /* ---- keyword routing ---- */
  const UNMEASURED = ["ball", "distance", "how far", "carry", "yard", "slice", "hook", "grip", "club face", "clubface", "face open", "face closed", "swing plane", "plane", "hinge", "clubhead", "club head", "speed", "mph", "which club", "what club", "launch", "spin", "draw", "fade"];
  const LOWCONF = ["lead arm", "left arm", "trail arm", "right arm", "arm straight", "arm bend", "arm extension", "elbow"];
  const FIX = ["what should i", "how do i fix", "how can i fix", "what do i fix", "fix my", "should i work on", "drill", "how to improve my", "what to improve", "help me fix", "what can i do to"];
  const PROG = ["improve", "improved", "progress", "better than", "since last", "last time", "before", "earlier", "used to", "weeks ago", "week ago", "changed", "change since", "compared", "compare", "comparison", "versus", "differ", "against", "getting better", "any better"];
  const SUMMARY = /takeaway|summar|overall|rundown|overview|biggest|key (point|thing)|main (thing|point)|how did i do|break it down|how (was|were) my swing|tell me about my swing|in general|the gist|highlight|how('?s| is) my swing|report/;
  const REF = /all of the above|all of them|all of it|all of my|all the metrics|all metrics|everything|every metric|full (breakdown|rundown|report)|list them all|show me all|each of (them|those)|go through (them|all)/;
  const has = (arr, ql) => arr.some(w => ql.includes(w));

  function matchMetrics(ql) {
    const present = new Set(keys(ACTIVE));
    const f = [], add = k => { if (present.has(k) && !f.includes(k)) f.push(k); };
    if (/\btempo\b|rhythm|\bpace\b/.test(ql)) add("tempo_ratio");
    if (/x[-\s]?factor|separation|shoulder[-\s]hip/.test(ql)) add("x_factor_top_deg");
    if (/\bshoulder/.test(ql) && !/shoulder[-\s]hip/.test(ql)) add("shoulder_turn_top_deg");
    if (/\bhip/.test(ql)) { if (/impact/.test(ql)) add("hip_turn_impact_deg"); else if (!/weight|lateral|shift|transfer/.test(ql)) add("hip_turn_top_deg"); }
    if (/weight shift|weight transfer|\btransfer\b|lateral|slide toward|\bweight\b/.test(ql)) add("hip_lateral_shift_pct");
    if (/posture|hold my angle|spine angle change|stay down|keep.*posture|hold.*posture/.test(ql)) add("posture_loss_deg");
    if (/\bspine\b|setup lean|posture at setup|at address/.test(ql) && !/change/.test(ql)) add("spine_tilt_address_deg");
    if (/head sway|head move|side to side|head still|\bhead\b/.test(ql)) add("head_sway_max_pct");
    return f;
  }
  const findKey = ql => matchMetrics(ql)[0] || null;

  function rundown(tc, call, ks, cmp) {
    const lines = [cmp ? `Here's this swing next to ${clipName(COMPARE)}'s:` : "Here's a quick rundown:"];
    ks.forEach(k => {
      const L = cap(low(label(k, ACTIVE)));
      if (tier(k, ACTIVE) === "low") { call("get_indicator", { key: k }, t_get(k)); lines.push(`• ${L}: measured from a single camera, so it's low-confidence — not reliable enough to call.`); return; }
      if (cmp) { const r = call("compare_indicator", { key: k }, t_compare(k)); if (!r.comparable) { lines.push(`• ${L}: no comparison available.`); return; } lines.push(`• ${L}: ${r.current_value}${unit(k, ACTIVE)} here vs ${r.earlier_value}${unit(k, ACTIVE)} for ${clipName(COMPARE)} (${r.current_in_tour_range ? "in range" : "outside range"}).`); }
      else { const r = call("get_indicator", { key: k }, t_get(k)); lines.push(`• ${L}: ${r.value}${unit(k, ACTIVE)} — ${r.in_tour_range ? "inside the tour range" : "outside the tour range"}.`); }
    });
    return { toolCalls: tc, answer: lines.join("\n") };
  }
  function summarize(tc, call, cmp) {
    const sum = call("get_swing_summary", {}, t_summary());
    const fl = call("get_flagged_observations", {}, t_flagged());
    const inR = reliableKeys(ACTIVE).filter(k => inRange(k, ACTIVE));
    const notable = ["tempo_ratio", "shoulder_turn_top_deg", "x_factor_top_deg", "hip_lateral_shift_pct"].filter(k => inR.includes(k)).slice(0, 3);
    const strong = (notable.length ? notable : inR.slice(0, 3));
    strong.forEach(k => call("get_indicator", { key: k }, t_get(k)));
    const lines = [`Here are the biggest takeaways from ${clipName(ACTIVE)}'s swing:`,
      `• Of ${sum.n_reliable} reliably-measured things, ${sum.n_in_tour_range} are tour-like.`,
      `• Working well: ${listEng(strong.map(k => `${low(label(k, ACTIVE))} (${val(k, ACTIVE)}${unit(k, ACTIVE)})`))}.`];
    lines.push(fl.count ? `• Outside the tour range: ${listEng(fl.flagged.map(f => low(f.label)))}.` : "• Nothing fell outside the tour range on this swing.");
    if (cmp) { const diff = reliableKeys(ACTIVE).filter(k => IND(COMPARE)[k] && inRange(k, ACTIVE) && !inRange(k, COMPARE)); diff.forEach(k => call("compare_indicator", { key: k }, t_compare(k))); if (diff.length) lines.push(`• In range here but not in ${clipName(COMPARE)}'s: ${listEng(diff.map(k => low(label(k, ACTIVE))))}.`); }
    return { toolCalls: tc, answer: lines.join("\n") };
  }

  /* ---- the offline "brain" ---- */
  function respond(q) {
    const ql = q.toLowerCase();
    const tc = []; const call = (name, input, result) => { tc.push({ name, input, result }); return result; };
    const cmp = (COMPARE != null);
    if (/what('| i)?s |what does |what is |explain |meaning of |define /.test(ql) && !has(PROG, ql)) {
      const m = ql.match(/(?:what(?:'s| is| does)?|explain|define|meaning of)\s+(?:an?\s+|the\s+|my\s+)?([a-z\- ]+?)(?:\s+mean)?[\?\.]?$/);
      let term = m ? m[1].trim() : "";
      if (term === "x factor") term = "x-factor";
      if (GLOSSARY[term]) { const r = call("explain_term", { term }, t_term(term)); if (r.in_glossary) return { toolCalls: tc, answer: `${cap(term)} is ${r.gloss}.` }; }
    }
    if (has(UNMEASURED, ql) && !findKey(ql)) { call("list_indicators", {}, { count: t_list().count, matched: false }); return { toolCalls: tc, refuse: true, answer: "That isn't something this swing analysis measures — it works from body motion, not ball flight, club, or contact. So I can't tell you that from what I have." }; }
    if (has(LOWCONF, ql) && matchMetrics(ql).length === 0) { call("get_indicator", { key: "left_arm_bend_top_deg" }, t_get("left_arm_bend_top_deg")); return { toolCalls: tc, refuse: true, answer: "That one comes from a single camera, so the measurement isn't reliable enough to judge — I'd rather not call it either way than guess." }; }
    if (has(FIX, ql) && !has(PROG, ql) && !SUMMARY.test(ql)) { return { toolCalls: tc, refuse: true, answer: "I can describe what your swing did, but I don't give fixes or drills — that's outside what this tool is meant to do. Want me to walk through what stood out instead?" }; }
    if (REF.test(ql)) return rundown(tc, call, reliableKeys(ACTIVE), has(PROG, ql) && cmp);
    if (SUMMARY.test(ql) && matchMetrics(ql).length < 2) return summarize(tc, call, cmp);
    { const ms = matchMetrics(ql); if (ms.length >= 2 && (/[,&]|\band\b|\bplus\b/.test(ql) || ms.length >= 3)) return rundown(tc, call, ms, has(PROG, ql) && cmp); }
    // library-wide superlative: "which swing had the best tempo?" (offline engine;
    // the live backend is per-swing today, so this answers from the same
    // metrics.json data the numbers tab renders)
    if (/\bwhich\b.*\bswing|\bbest\b|\bclosest\b|\bworst\b|\bfurthest\b/.test(ql)) {
      const k = findKey(ql);
      const ids = Object.keys(CLIPDATA).filter(id => IND(id) && IND(id)[k]);
      if (k && ids.length > 1 && tier(k, ACTIVE) !== "low") {
        const ranked = ids
          .filter(id => tier(k, id) !== "low")
          .map(id => {
            const r = { key: k, clip: id, value: val(k, id), tour_median: tour(k, id) };
            tc.push({ name: "get_indicator", input: { key: k, clip: id }, result: r });
            return { id, value: r.value, dist: Math.abs(r.value - r.tour_median) };
          })
          .sort((a, b) => a.dist - b.dist);
        if (ranked.length > 1) {
          const worst = /\bworst\b|\bfurthest\b/.test(ql);
          const pick = worst ? ranked[ranked.length - 1] : ranked[0];
          return { toolCalls: tc, answer:
            `${worst ? "Furthest from" : "Closest to"} the tour median for ${low(label(k, ACTIVE))}: ` +
            `${clipName(pick.id)} at ${pick.value}${unit(k, ACTIVE)}. Across the library: ` +
            `${ranked.map(r => `${clipName(r.id)} ${r.value}${unit(k, ACTIVE)}`).join(", ")}.` };
        }
      }
    }
    if (has(PROG, ql)) {
      if (!cmp) return { toolCalls: tc, refuse: true, answer: "Pick a swing to compare against — use the “Compare with” selector above the chat — and I'll line the two up." };
      const k = findKey(ql);
      if (k) {
        if (tier(k, ACTIVE) === "low") { call("compare_indicator", { key: k }, t_compare(k)); return { toolCalls: tc, refuse: true, answer: `${label(k, ACTIVE)} is a low-confidence measurement, so I can't reliably compare it across swings.` }; }
        const r = call("compare_indicator", { key: k }, t_compare(k));
        if (!r.comparable) return { toolCalls: tc, answer: `I don't have ${low(label(k, ACTIVE))} on ${clipName(COMPARE)}'s swing to compare against.` };
        let s = `For ${low(label(k, ACTIVE))}, this swing is ${r.current_value}${unit(k, ACTIVE)} (${r.current_in_tour_range ? "in the tour range" : "outside the tour range"}) versus ${r.earlier_value}${unit(k, ACTIVE)} for ${clipName(COMPARE)} (${r.earlier_in_tour_range ? "in range" : "outside range"}).`;
        if (Math.abs(r.delta) > 0) s += ` A ${Math.abs(r.delta)}${unit(k, ACTIVE)} difference.`;
        return { toolCalls: tc, answer: s };
      }
      call("get_swing_summary", {}, t_summary());
      const diff = reliableKeys(ACTIVE).filter(k => IND(COMPARE)[k] && inRange(k, ACTIVE) && !inRange(k, COMPARE));
      diff.forEach(k => call("compare_indicator", { key: k }, t_compare(k)));
      if (diff.length) return { toolCalls: tc, answer: `Compared with ${clipName(COMPARE)}, this swing is inside the tour range on ${listEng(diff.map(k => low(label(k, ACTIVE))))} where ${clipName(COMPARE)}'s isn't. Ask about any one for the numbers.` };
      return { toolCalls: tc, answer: `This swing and ${clipName(COMPARE)}'s land inside the tour range on the same metrics — no in/out-of-range differences to call out.` };
    }
    if (/stood out|stand out|flag|issue|problem|wrong|off|notable|notice/.test(ql)) {
      const r = call("get_flagged_observations", {}, t_flagged());
      if (!r.count) return { toolCalls: tc, answer: "Nothing stood out — every reliably-measured metric is inside the tour range on this swing." };
      const names = r.flagged.map(f => low(f.label));
      return { toolCalls: tc, answer: `On this swing, ${listEng(names)} ${r.count > 1 ? "were" : "was"} outside the typical tour range. Everything else measured sat inside it.` };
    }
    const k = findKey(ql);
    if (k) {
      if (tier(k, ACTIVE) === "low") { call("get_indicator", { key: k }, t_get(k)); return { toolCalls: tc, refuse: true, answer: "That measurement isn't reliable enough from a single camera, so I can't assess it." }; }
      const r = call("get_indicator", { key: k }, t_get(k));
      const phr = r.in_tour_range ? IND(ACTIVE)[k].in : (r.value < r.pro_band[0] ? IND(ACTIVE)[k].lo : IND(ACTIVE)[k].hi);
      let s = `Your ${low(label(k, ACTIVE))} is ${r.value}${unit(k, ACTIVE)}`;
      s += r.in_tour_range ? `, inside the tour range (${r.pro_band[0]}–${r.pro_band[1]})${phr ? " — " + phr : ""}.` : `${phr ? " — " + phr : ""} (tour range ${r.pro_band[0]}–${r.pro_band[1]}).`;
      return { toolCalls: tc, answer: s };
    }
    call("list_indicators", {}, { count: t_list().count });
    return { toolCalls: tc, answer: "I can break down the measured parts of your swing — tempo, shoulder and hip turn, X-factor, posture, spine angle, head sway, and weight shift. Ask about any of those, list a few, or say “give me the biggest takeaways” for a summary." };
  }

  /* ---- grounding verifier (mirrors verify_chat_grounding), for the mock ---- */
  function verify(res) {
    const a = (res.answer || "").toLowerCase(); const v = [];
    const lc = new Set();
    (res.toolCalls || []).forEach(e => { const r = e.result; if (!r) return; if ((e.name === "get_indicator" || e.name === "compare_indicator") && (r.reliable === false || r.confidence_tier === "low")) lc.add(r.key); });
    lc.forEach(k => { if (IND(ACTIVE)[k] && a.includes(String(val(k, ACTIVE)))) v.push("low_confidence_leak"); });
    if (!res.refuse && /\byou should\b|\btry to\b|\bwork on\b|\bdrill\b|\bpractice\b/.test(a)) v.push("prescriptive");
    return { grounded: v.length === 0, violations: v };
  }

  /* ======= askChat seam: live backend if window.API_BASE set, else mock =======
   * The live /chat Lambda knows the curated demo clips (ALLOWED_CLIPS, integer
   * ids) AND processed uploads (32-hex job ids — it fetches that job's
   * scorecard from 03_outputs). Anything else falls back to the in-page engine
   * grounded in the swing's own metrics.json. */
  const liveFor = (id) => !!window.API_BASE &&
    (/^[0-9a-f]{32}$/.test(String(id)) ||
     (/^\d+$/.test(String(id)) &&
      (!Array.isArray(window.DEMO_CLIPS) || window.DEMO_CLIPS.includes(Number(id)))));
  async function askChat(q) {
    if (liveFor(ACTIVE)) {
      const hist = HIST[ACTIVE] || (HIST[ACTIVE] = []);
      // demo clips post an int id; processed uploads post their hex job id
      const wireId = (id) => (/^\d+$/.test(String(id)) ? Number(id) : String(id));
      const body = { clip_id: wireId(ACTIVE), question: q, history: hist.slice() };
      if (COMPARE != null) body.compare_clip_id = wireId(COMPARE);
      // uploaded swings are private: pass the dev access code (mc_dev cookie,
      // set by dev sign-in) so the backend will talk about them
      const devCode = (document.cookie.match(/(?:^|;\s*)mc_dev=([^;]+)/) || [])[1];
      if (devCode) body.access_token = decodeURIComponent(devCode);
      const r = await fetch(`${window.API_BASE}/chat`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (!r.ok) throw new Error("http " + r.status);
      const d = await r.json();
      hist.push({ role: "user", content: q }, { role: "assistant", content: d.answer || "" });
      const tcs = (d.tool_log && d.tool_log.length) ? d.tool_log.map(e => ({ name: e.name, input: e.input, result: e.result })) : (d.tools_used || []).map(n => ({ name: n }));
      return { answer: d.answer, toolCalls: tcs, grounded: d.grounded, violations: d.violations, refuse: false };
    }
    const res = respond(q);
    await new Promise(r => setTimeout(r, 380 + Math.random() * 260));  // feel of "thinking"
    return res;
  }

  /* ====================== voice (Web Speech API) ======================
   * Dictation: SpeechRecognition fills the input with live interim text; the
   * user reviews and sends (no auto-send — a mis-transcription would put the
   * wrong question to the grounded coach). Spoken replies: while the header
   * toggle is on, the live default is the ElevenLabs studio coach voice
   * (POST /tts, Scottish — see deploy/tts/), with the browser's
   * speechSynthesis voices as picker alternatives and as the offline/error
   * fallback. Dictation works identically in offline-mock and live modes.
   */
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const TTS = "speechSynthesis" in window;
  const COACH_ID = "__coach_elevenlabs__";          // studio narration via POST /tts
  const coachAvailable = () => !!window.API_BASE;
  let rec = null, recOn = false, speakOn = false, voice = null, useCoach = false;
  let coachAudio = null, speakGen = 0;              // gen guards stale /tts fetches after hush()

  /* Rank voices by how human they sound: Edge's neural "Natural" set and
   * Chrome's Google server voices are near-human; Safari's enhanced/Siri
   * voices are good; the legacy Windows SAPI set (David/Mark/Zira) is the
   * robotic last resort. The header dropdown lets the user override the
   * default; the choice is remembered in localStorage. */
  const VOICE_LS = "mc_chat_voice";
  function rankedVoices() {
    const vs = speechSynthesis.getVoices().filter(v => /^en([-_]|$)/i.test(v.lang));
    const score = v => {
      const n = v.name.toLowerCase(); let s = 0;
      if (/natural|neural/.test(n)) s += 8;
      if (/google/.test(n)) s += 6;
      if (/premium|enhanced|siri/.test(n)) s += 5;
      if (/online/.test(n)) s += 3;
      if (/aria|jenny|ava|emma|andrew|samantha/.test(n)) s += 2;  // pick of each vendor's set
      if (!v.localService) s += 1;                                // server voices sound better
      if (/^en[-_]us/i.test(v.lang)) s += 1;
      if (/david|espeak|compact/.test(n)) s -= 3;                 // stiffest of the legacy set
      if (/zira/.test(n)) s -= 2;
      return s;
    };
    return vs.sort((a, b) => score(b) - score(a));
  }
  function pickVoice() {
    const saved = localStorage.getItem(VOICE_LS);
    useCoach = coachAvailable() && (saved === COACH_ID || !saved);  // studio voice is the live default
    const vs = TTS ? rankedVoices() : [];
    return vs.find(v => v.name === saved) || vs[0] || null;
  }
  const shortName = n => n.replace(/^(Microsoft|Google|Apple)\s+/i, "")
    .replace(/\s+-\s+English\b.*$/i, "").replace(/\s+English\b.*$/i, "").trim();
  function fillVoicePick() {
    if (!dom.voicePick) return;
    const vs = TTS ? rankedVoices() : [];
    dom.voicePick.hidden = vs.length + (coachAvailable() ? 1 : 0) < 2;
    dom.voicePick.innerHTML = "";
    if (coachAvailable()) {
      const o = document.createElement("option");
      o.value = COACH_ID; o.textContent = "Coach · Scottish (studio)";
      dom.voicePick.append(o);
    }
    vs.forEach(v => {
      const o = document.createElement("option");
      o.value = v.name; o.textContent = shortName(v.name);
      dom.voicePick.append(o);
    });
    dom.voicePick.value = useCoach ? COACH_ID : (voice ? voice.name : "");
  }

  function speak(text, force) {
    if ((!speakOn && !force) || !text) return;
    hush();
    const lines = text.split("\n")
      .map(l => l.replace(/^[•\-*]\s+/, "")            // list markers
                 .replace(/\*\*?([^*]+)\*\*?/g, "$1")  // markdown emphasis (live backend uses it)
                 .replace(/`([^`]+)`/g, "$1").trim())
      .filter(Boolean);
    if (!lines.length) return;
    if (useCoach) speakCoach(lines);
    else if (TTS) speakBrowser(lines);
  }
  // one utterance per line: bullets get a natural pause, and short utterances
  // dodge Chrome's habit of cutting speech off around the 15-second mark
  function speakBrowser(lines) {
    lines.forEach(line => {
      const u = new SpeechSynthesisUtterance(line);
      if (voice) u.voice = voice;
      u.rate = 1.02;
      speechSynthesis.speak(u);
    });
  }
  /* Studio narration: POST /tts renders the reply with the ElevenLabs coach
   * voice (server caches by content hash, so repeats are instant) and we play
   * the mp3 from CloudFront. Any failure falls back to the browser voice. */
  async function speakCoach(lines) {
    const gen = ++speakGen;
    try {
      const r = await fetch(`${window.API_BASE}/tts`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ narration: lines.join("\n").slice(0, 2400),
                               swing_id: String(ACTIVE != null ? ACTIVE : "chat") })
      });
      if (!r.ok) throw new Error("tts " + r.status);
      const d = await r.json();
      if (!d.audio_url) throw new Error("tts: no audio_url");
      if (gen !== speakGen) return;                   // hushed or superseded meanwhile
      if (!coachAudio) coachAudio = new Audio();
      coachAudio.src = d.audio_url;
      await coachAudio.play();
    } catch (e) {
      if (gen === speakGen && TTS) speakBrowser(lines);
    }
  }
  const hush = () => {
    speakGen++;
    if (TTS) speechSynthesis.cancel();
    if (coachAudio) { coachAudio.pause(); coachAudio.removeAttribute("src"); }
  };

  function initVoiceOut() {
    if (!dom.voice) return;
    if (!TTS && !coachAvailable()) { dom.voice.hidden = true; return; }
    voice = pickVoice(); fillVoicePick();
    if (TTS) speechSynthesis.addEventListener("voiceschanged", () => { voice = pickVoice(); fillVoicePick(); });
    if (dom.voicePick) dom.voicePick.addEventListener("change", () => {
      localStorage.setItem(VOICE_LS, dom.voicePick.value);
      voice = pickVoice();
      speak("Hi — this is how your coach will sound.", true);   // audition even if replies are off
    });
    dom.voice.addEventListener("click", () => {
      speakOn = !speakOn;
      dom.voice.setAttribute("aria-pressed", String(speakOn));
      dom.voice.textContent = speakOn ? "🔊 Voice replies: on" : "🔊 Voice replies: off";
      if (!speakOn) hush();
      else speak("Voice replies are on.");
    });
  }

  function initDictation() {
    if (!dom.mic) return;
    if (!SR) { dom.mic.hidden = true; return; }   // e.g. Firefox
    rec = new SR();
    rec.lang = "en-US"; rec.interimResults = true; rec.continuous = false;
    const stopUI = () => { recOn = false; dom.mic.classList.remove("rec"); };
    rec.onresult = e => {
      let t = ""; for (const r of e.results) t += r[0].transcript;
      dom.input.value = t;
    };
    rec.onend = () => { stopUI(); if (dom.input.value.trim()) dom.input.focus(); };
    rec.onerror = e => {
      stopUI();
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        dom.input.placeholder = "Microphone blocked — allow mic access to ask by voice";
        setTimeout(() => { dom.input.placeholder = "Ask about this swing…"; }, 4000);
      }
    };
    dom.mic.addEventListener("click", () => {
      if (recOn) { rec.stop(); return; }
      hush();                                     // don't transcribe our own TTS
      dom.input.value = "";
      try { rec.start(); recOn = true; dom.mic.classList.add("rec"); } catch (e) { stopUI(); }
    });
  }

  /* ================================ UI ================================ */
  const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
  const scroll = () => { dom.stream.scrollTop = dom.stream.scrollHeight; };
  function addUser(q) { const w = el("div", "chat-turn"); w.append(el("div", "who", "You"), el("div", "msg user", q)); dom.stream.append(w); scroll(); }
  function addTyping() { const t = el("div", "typing"); t.innerHTML = (window.API_BASE ? "Running the coach" : "Checking the measurements") + ' <span class="dot"></span><span class="dot"></span><span class="dot"></span>'; dom.stream.append(t); scroll(); return t; }
  const grade = res => (res.grounded !== undefined) ? { grounded: res.grounded, violations: res.violations || [] } : verify(res);
  function addBot(res) {
    const tcs = res.toolCalls || [];
    if (tcs.length) {
      const d = el("details", "trace"); const sum = el("summary", null, `${tcs.length} tool call${tcs.length > 1 ? "s" : ""} — click to inspect`);
      const bodyd = el("div", "trace-body");
      tcs.forEach(e => {
        const r = el("div", "tc");
        const argStr = e.input ? JSON.stringify(e.input).replace(/^{}$/, "") : "";
        let rs = e.result ? JSON.stringify(e.result) : "";
        if (rs.length > 340) rs = rs.slice(0, 340) + " … (" + rs.length + " chars)";
        const esc = s => s.replace(/[<>&]/g, c => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]));
        r.innerHTML = `<span class="call">${esc(e.name)}</span>(<span class="arg">${esc(argStr)}</span>)` + (rs ? `<span class="res">-&gt; ${esc(rs)}</span>` : "");
        bodyd.append(r);
      });
      d.append(sum, bodyd); dom.stream.append(d);
    }
    const w = el("div", "chat-turn"); w.append(el("div", "who", "MotionCaddie"));
    const m = el("div", "msg bot" + (res.refuse ? " refuse" : ""), res.answer);
    const flight = tcs.map(e => e.result).find(r => r && Array.isArray(r._ui_trajectory) && r._ui_trajectory.length > 2);
    if (flight) m.append(flightArc(flight));
    const g = grade(res); const gd = el("span", "grade" + (g.grounded ? "" : " warn"));
    const vtxt = (g.violations || []).map(v => typeof v === "string" ? v
      : v.type + (v.value != null ? ` (${v.value})` : "")).join(", ");
    gd.textContent = g.grounded ? "Grounded in fetched data"
      : `Grounding check: ${vtxt || "a claim couldn't be verified against the fetched data"}`;
    m.append(document.createElement("br"), gd); w.append(m); dom.stream.append(w); scroll();
    speak(res.answer);
  }
  /* Simulated ball-flight arc (side view) from estimate_ball_flight's UI-only
   * trajectory ([downrange_yd, height_yd, side_yd] points). Pure SVG, themed
   * off currentColor so it works in both palettes. */
  function flightArc(r) {
    const pts = r._ui_trajectory;
    const W = 300, H = 96, PAD = 10, GY = H - 16;            // ground baseline
    const maxX = Math.max(...pts.map(p => p[0])) || 1;
    const maxY = Math.max(...pts.map(p => p[1])) || 1;
    const sx = x => PAD + (x / maxX) * (W - 2 * PAD);
    const sy = y => GY - (y / maxY) * (GY - PAD);
    const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
    const land = pts[pts.length - 1];
    // get_ball_flight results carry a quality tier: "measured" = fitted to the
    // ball actually tracked in the video; "partial" = measured launch, assumed
    // speed; anything else = pure simulation (the pre-ball-tracking behavior).
    const q = r.quality;
    const label = q === "measured" ? "measured from your video"
      : q === "partial" ? "measured launch, estimated distance" : "simulated flight";
    const box = el("div", "flight-arc");
    box.style.cssText = "margin-top:8px;opacity:.9";
    box.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:340px;display:block" role="img"` +
      ` aria-label="Ball flight (${label}): about ${Math.round(land[0])} yards carry, apex ${Math.round(maxY)} yards">` +
      `<line x1="${PAD}" y1="${GY}" x2="${W - PAD}" y2="${GY}" stroke="currentColor" stroke-opacity=".25"/>` +
      `<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-opacity=".8"/>` +
      `<circle cx="${sx(land[0])}" cy="${sy(land[1])}" r="2.6" fill="currentColor"/>` +
      `<text x="${PAD}" y="${H - 3}" font-size="9" fill="currentColor" fill-opacity=".65">` +
      `${label} — ≈${Math.round(land[0])} yd carry, apex ${Math.round(maxY)} yd</text></svg>`;
    return box;
  }

  async function ask(q) {
    hush();
    addUser(q); const typing = addTyping();
    let res;
    try { res = await askChat(q); }
    catch (e) { res = respond(q); res.answer += "  (couldn't reach the live coach — showing the offline read)"; }
    typing.remove(); addBot(res);
  }
  function greet() {
    const c = CLIPDATA[ACTIVE];
    const w = el("div", "chat-turn"); w.append(el("div", "who", "MotionCaddie"));
    const intro = c.club
      ? `You're looking at ${c.name}'s swing (${c.club}, ${c.view}).`
      : `You're looking at ${c.name} — measured by the real pipeline.`;
    w.append(el("div", "msg bot", `${intro} Ask me anything about it — a metric like tempo or weight shift, the biggest takeaways, or pick a second swing to compare.`));
    dom.stream.append(w); scroll();
  }
  function populateCompare() {
    if (!dom.compare) return;
    dom.compare.innerHTML = '<option value="">— none —</option>';
    Object.keys(CLIPDATA).filter(c => c !== ACTIVE).forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = CLIPDATA[c].name; dom.compare.append(o); });
    dom.compare.value = COMPARE || "";
  }

  /* ---- public API ---- */
  return {
    init(nodes) {
      dom = nodes;
      dom.send.addEventListener("click", () => { const v = dom.input.value.trim(); if (v) { ask(v); dom.input.value = ""; } });
      dom.input.addEventListener("keydown", e => { if (e.key === "Enter") { const v = dom.input.value.trim(); if (v) { ask(v); dom.input.value = ""; } } });
      (dom.chips || []).forEach(c => c.addEventListener("click", () => ask(c.textContent.trim())));
      if (dom.compare) dom.compare.addEventListener("change", e => { COMPARE = e.target.value || null; });
      if (dom.mode) dom.mode.textContent = window.API_BASE ? "live · real coach" : "offline preview · in-page coach";
      initVoiceOut(); initDictation();
    },
    setLibrary(clips, metricsByClipId) {
      for (const clip of clips) { const rows = (metricsByClipId[clip.id] || {}).metrics; if (rows) buildFromMetrics(clip, rows); }
    },
    /* a processed UPLOAD: same registration as a demo clip, from its metrics.json */
    registerJob(jobId, metricsJson) {
      const rows = (metricsJson || {}).metrics;
      if (!rows) return false;
      buildFromMetrics({ id: String(jobId), title: "Your uploaded swing",
                         club: "your club", view: "uploaded video" }, rows);
      return true;
    },
    activate(clipId) {
      hush();
      ACTIVE = String(clipId); COMPARE = null;
      if (!CLIPDATA[ACTIVE]) return;
      if (dom.stream) dom.stream.innerHTML = "";
      const c = CLIPDATA[ACTIVE];
      if (dom.active) dom.active.textContent = c.view ? `${c.name} · ${c.view}` : c.name;
      if (dom.mode) dom.mode.textContent = liveFor(ACTIVE) ? "live · real coach"
        : (window.API_BASE ? "grounded · this swing's measured data"
                           : "offline preview · in-page coach");
      populateCompare(); greet();
    },
  };
})();
