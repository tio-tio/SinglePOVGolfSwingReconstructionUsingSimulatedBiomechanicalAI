/* MotionCaddie — topographic contour-line background.
 *
 * Fixed full-viewport canvas behind the page (#hills-canvas, z-index 0;
 * content sits at z-index 1 — see styles.css). Nested open contour lines
 * suggesting rolling terrain sit in a height-capped band anchored to the
 * bottom of the viewport, drifting slowly. Thin 1px strokes, no fill,
 * ramping through the site's greens — faint at the top of the band,
 * deeper toward the bottom.
 *
 * The profile comes from a seeded sum of integer harmonics over a fixed
 * WRAP period, so it is deterministic across reloads and wraps seamlessly
 * as it drifts. One dpr-aware rAF loop; honors prefers-reduced-motion:
 * reduce by drawing a single static frame and never starting the loop.
 */
(function () {
  "use strict";

  var canvas = document.getElementById("hills-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");

  /* ── Tuneable constants ────────────────────────────────────────────── */
  var SEED       = 20260718;  // deterministic profile across reloads
  var WRAP       = 1920;      // terrain period in CSS px — seamless wrap
  var N_LINES    = 7;         // sparse beats dense
  var BAND_MAX   = 220;       // capped band height, CSS px
  var BAND_FRAC  = 0.32;      // never taller than this × viewport height
  var BASE_SPEED = 6;         // CSS px/s drift; +1.2/line toward the bottom
  var HARMONICS  = 3;         // sine components per line
  // color ramp endpoints (site greens): light/faint top → deep bottom
  var TOP = { r: 207, g: 230, b: 212, a: 0.28 };   // #cfe6d4
  var BOT = { r:  94, g: 168, b: 127, a: 0.60 };   // deepened #7fbb93

  var W, H, dpr;

  /* ── Seeded PRNG (mulberry32) ──────────────────────────────────────── */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* seeded periodic terrain: sum of integer harmonics over WRAP → [-1, 1] */
  function makeTerrain(rand) {
    var waves = [], norm = 0;
    for (var j = 0; j < HARMONICS; j++) {
      var w = 1 / (j + 1.5);
      norm += w;
      waves.push({ k: j + 1 + Math.floor(rand() * 2), w: w, phase: rand() * Math.PI * 2 });
    }
    return function (x) {
      var v = 0;
      for (var j = 0; j < waves.length; j++) {
        var wv = waves[j];
        v += wv.w * Math.sin((Math.PI * 2 * wv.k * x) / WRAP + wv.phase);
      }
      return v / norm;
    };
  }

  var rand = mulberry32(SEED);
  var lines = [];
  for (var i = 0; i < N_LINES; i++) {
    var f = i / (N_LINES - 1);               // 0 = top line, 1 = bottom line
    lines.push({
      f: f,
      terrain: makeTerrain(rand),
      speed: BASE_SPEED + i * 1.2,
      color: "rgba(" + Math.round(TOP.r + (BOT.r - TOP.r) * f) + ","
                     + Math.round(TOP.g + (BOT.g - TOP.g) * f) + ","
                     + Math.round(TOP.b + (BOT.b - TOP.b) * f) + ","
                     + (TOP.a + (BOT.a - TOP.a) * f).toFixed(3) + ")",
    });
  }

  /* ── Canvas setup ──────────────────────────────────────────────────── */
  function resize() {
    dpr = window.devicePixelRatio || 1;
    W   = window.innerWidth;
    H   = window.innerHeight;
    canvas.width  = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width  = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ── Draw one frame; t = drift time in seconds ─────────────────────── */
  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    var band = Math.min(BAND_MAX, H * BAND_FRAC);
    var spacing = band / (N_LINES + 1);
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      var baseY = H - band + spacing * (i + 1);
      // amplitude grows toward the bottom but stays under the spacing
      // delta, so adjacent rings never cross — reads as nested contours
      var amp = spacing * (0.55 + 0.4 * ln.f);
      var offset = t * ln.speed;
      ctx.strokeStyle = ln.color;
      ctx.beginPath();
      for (var x = -8; x <= W + 8; x += 6) {
        var y = baseY - amp * ln.terrain(((x + offset) % WRAP + WRAP) % WRAP);
        if (x === -8) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  /* ── Single rAF loop, time-based; static under reduced motion ──────── */
  var rafId = null, elapsed = 0, lastTs = null;

  function frame(ts) {
    if (lastTs != null) elapsed += Math.min(ts - lastTs, 100) / 1000;
    lastTs = ts;
    draw(elapsed);
    rafId = requestAnimationFrame(frame);
  }
  function start() {
    if (rafId == null) { lastTs = null; rafId = requestAnimationFrame(frame); }
  }
  function stop() {
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  }

  var motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  function applyMotionPref() {
    if (motionQuery.matches) { stop(); draw(0); }   // static contours, no loop
    else start();
  }
  if (motionQuery.addEventListener) motionQuery.addEventListener("change", applyMotionPref);

  document.addEventListener("visibilitychange", function () {
    if (motionQuery.matches) return;
    if (document.hidden) stop(); else start();
  });

  window.addEventListener("resize", function () { resize(); if (motionQuery.matches) draw(0); });

  resize();
  applyMotionPref();
}());
