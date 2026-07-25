/* MotionCaddie — ball-flight estimate card for the results page.
 *
 * Browser port of Scripts/ball_flight.py (McNally et al., CVPR 2023W):
 *  - Phys-NN engine: the paper's MLP aerodynamic coefficients, trained on real
 *    TrackMan shots (1.84 yd held-out landing error). Weights are fetched
 *    lazily from assets/ball_flight_nn.json and the NN only runs inside its
 *    training envelope.
 *  - Phys-Q fallback: published polynomial coefficients (Ferguson 2022) for
 *    launch conditions outside the envelope, or if the weights fail to load.
 *
 * The card lets the golfer pick club + skill tier and renders a simulated
 * side-view arc + carry/apex numbers. When the swing's measured hand speed is
 * available (metrics.json row hand_speed_impact_bs) the assumed ball speed is
 * nudged by it, capped ±12% — same rule as the coach's chat tool, so the card
 * and the coach never disagree.
 *
 * Shared by app.js (demo site) and portal.js: BallFlight.mount(opts).
 */
"use strict";

const BallFlight = (() => {
  /* ---------------- physics constants (mirror ball_flight.py) ------------- */
  const M = 0.04593, R = 0.021335, A = Math.PI * R * R, I = 0.4 * M * R * R;
  const RHO = 1.225, G = 9.81;
  const MPH = 0.44704, RPM = 2 * Math.PI / 60, YD = 1.0936133;

  // (ball speed mph, launch deg, backspin rpm) — TrackMan published averages
  const CLUBS = {
    "driver":  { tour: [167.0, 10.9, 2686], lpga: [140.0, 13.2, 2611], amateur: [133.0, 12.6, 3275] },
    "3 wood":  { tour: [158.0, 9.2, 3655],  lpga: [132.0, 11.2, 2704], amateur: [125.0, 11.5, 3600] },
    "hybrid":  { tour: [146.0, 10.2, 4437], lpga: [122.0, 12.5, 4501], amateur: [118.0, 12.0, 4500] },
    "5 iron":  { tour: [132.0, 12.1, 5361], lpga: [112.0, 14.8, 5081], amateur: [108.0, 13.5, 5500] },
    "7 iron":  { tour: [120.0, 16.3, 7097], lpga: [104.0, 19.0, 6699], amateur: [98.0, 18.0, 7000] },
    "9 iron":  { tour: [109.0, 20.4, 8647], lpga: [93.0, 23.9, 7589],  amateur: [88.0, 23.0, 8200] },
    "wedge":   { tour: [102.0, 24.2, 9304], lpga: [86.0, 25.6, 8403],  amateur: [80.0, 26.0, 8700] },
  };
  const CLUB_ALIASES = { iron: "7 iron", wood: "3 wood" };  // scorecard vocabulary

  /* ------------------------- Phys-Q polynomial sim ------------------------ */
  function simulatePhysQ(ballMph, launchDeg, backRpm) {
    const dt = 0.01;
    let v = ballMph * MPH;
    const la = launchDeg * Math.PI / 180;
    let vy = v * Math.cos(la), vz = v * Math.sin(la);
    let wx = backRpm * RPM;                      // backspin only (straight shot)
    let y = 0, z = 0.01, apex = z, t = 0;
    const traj = [[0, 0]];
    while (t < 15) {
      const vmag = Math.hypot(vy, vz);
      const S = R * Math.abs(wx) / vmag;
      const CD = 0.1304 + 0.9287 * S - 0.8259 * S * S;
      const CL = 0.0504 + 1.2031 * S - 1.1490 * S * S;
      const CM = 0.01 * S;
      const q = 0.5 * RHO * vmag * vmag;
      // Magnus lift = (w x v)/|w x v|: with w=+x and v in the y-z plane this is
      // (-vz, vy)/vmag in (y, z)
      const FL = CL * q * A, FD = CD * q * A;
      const fy = FL * (-vz / vmag) - FD * (vy / vmag);
      const fz = FL * (vy / vmag) - FD * (vz / vmag) - M * G;
      wx -= dt * (CM * q * 2 * R * A) / I * Math.sign(wx);
      vy += dt * fy / M; vz += dt * fz / M;
      const zp = z;
      y += dt * vy; z += dt * vz;
      apex = Math.max(apex, z);
      t += dt;
      if (z <= 0 && t > 0.5) {
        const f = zp / (zp - z);
        y = traj.length ? (y - dt * vy) + f * dt * vy : y;
        traj.push([y * YD, 0]);
        break;
      }
      traj.push([y * YD, z * YD]);
    }
    return { carry: y * YD, apex: apex * YD, time: t, traj, engine: "phys_q" };
  }

  /* --------------------------- Phys-NN engine ----------------------------- */
  let nnPromise = null;     // lazy singleton: {layers, pM, envelope} | null
  function loadNN() {
    if (!nnPromise) {
      nnPromise = fetch("assets/ball_flight_nn.json")
        .then(r => { if (!r.ok) throw new Error("http " + r.status); return r.json(); })
        .catch(() => null);
    }
    return nnPromise;
  }
  function inEnvelope(nn, ballMph, launchDeg, backRpm) {
    if (!nn || !nn.envelope) return false;
    const e = nn.envelope;
    const within = (k, v) => v >= e[k][0] && v <= e[k][1];
    return within("ball_speed_mph", ballMph) && within("launch_angle_deg", launchDeg)
      && within("spin_rpm", backRpm) && within("spin_axis_deg", 0);
  }
  function matvec(Wb, x, relu) {
    const [W, b] = Wb, out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = b[i]; const row = W[i];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = relu ? Math.max(s, 0) : s;
    }
    return out;
  }
  function simulateNN(nn, ballMph, launchDeg, backRpm) {
    const dt = 0.1;                              // the dt the network was trained with
    let v = ballMph * MPH;
    const la = launchDeg * Math.PI / 180;
    let vy = v * Math.cos(la), vz = v * Math.sin(la);
    let wx = backRpm * RPM;
    let y = 0, z = 0.01, apex = z, t = 0;
    const traj = [[0, 0]];
    while (t < 10) {
      const vmag = Math.hypot(vy, vz);
      const S = R * Math.abs(wx) / vmag;
      // inputs are the (v, w) VECTORS normalized — here vx=0, wy=wz=0
      const x = [0, vy / 89.4, vz / 89.4, wx / 2094.0, 0, 0];
      const h1 = matvec(nn.layers[0], x, true);
      const h2 = matvec(nn.layers[1], h1, true);
      const o = matvec(nn.layers[2], h2, false).map(u => 1 / (1 + Math.exp(-u)));
      const CL = o[0] / 2, CD = o[1] / 2, CQ = o[2];
      const CM = nn.pM * S;
      const q = CQ * RHO * vmag * vmag;          // paper's cQ variant
      const FL = CL * q * A, FD = CD * q * A;
      const fy = FL * (-vz / vmag) - FD * (vy / vmag);
      const fz = FL * (vy / vmag) - FD * (vz / vmag) - M * G;
      wx -= dt * (CM * q * 2 * R * A) / I * Math.sign(wx);
      vy += dt * fy / M; vz += dt * fz / M;
      const zp = z, yp = y;
      y += dt * vy; z += dt * vz;
      apex = Math.max(apex, z);
      t += dt;
      if (z <= 0 && t > 0.5) {
        const f = zp / (zp - z);
        y = yp + f * (y - yp);
        traj.push([y * YD, 0]);
        break;
      }
      traj.push([y * YD, z * YD]);
    }
    return { carry: y * YD, apex: apex * YD, time: t, traj, engine: "phys_nn" };
  }

  /* ------------------------------ estimate -------------------------------- */
  function normClub(club) {
    const c = String(club || "").trim().toLowerCase().replace(/[-_]/g, " ");
    return CLUBS[c] ? c : (CLUB_ALIASES[c] || null);
  }
  /* hand-speed nudge (same ±12% cap as the chat tool). Sources, in order:
   * an explicit {value, median} pair (read from the job's scorecard.json),
   * else this swing's metrics rows. */
  function speedScaleFrom(metricsRows, handSpeed) {
    let you, median;
    if (handSpeed && handSpeed.value && handSpeed.median) {
      you = handSpeed.value; median = handSpeed.median;
    } else {
      const row = (metricsRows || []).find(r => r.key === "hand_speed_impact_bs");
      if (row) { you = row.you; median = row.tour; }
    }
    if (!you || !median) return { scale: 1, pct: 0 };
    const scale = Math.min(Math.max(you / median, 0.88), 1.12);
    return { scale, pct: Math.round((scale - 1) * 100) };
  }
  async function estimate(club, tier, metricsRows, handSpeed) {
    const key = normClub(club) || "driver";
    const [bs0, la, spin] = CLUBS[key][tier] || CLUBS[key].tour;
    const nudge = speedScaleFrom(metricsRows, handSpeed);
    const bs = bs0 * nudge.scale;
    const nn = await loadNN();
    const r = (nn && inEnvelope(nn, bs, la, spin))
      ? simulateNN(nn, bs, la, spin) : simulatePhysQ(bs, la, spin);
    return { ...r, club: key, tier, ballMph: bs, launchDeg: la, spinRpm: spin, nudge };
  }

  /* ------------------------------- card UI -------------------------------- */
  function arcSVG(traj, carry, apex) {
    const W = 300, H = 96, PAD = 10, GY = H - 16;
    const maxX = Math.max(...traj.map(p => p[0])) || 1;
    const maxY = Math.max(...traj.map(p => p[1])) || 1;
    const sx = x => PAD + (x / maxX) * (W - 2 * PAD);
    const sy = y => GY - (y / maxY) * (GY - PAD);
    const path = traj.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
    const land = traj[traj.length - 1];
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:360px;display:block" role="img"
      aria-label="Simulated ball flight: about ${Math.round(carry)} yards carry, apex ${Math.round(apex)} yards">
      <line x1="${PAD}" y1="${GY}" x2="${W - PAD}" y2="${GY}" stroke="currentColor" stroke-opacity=".25"/>
      <path d="${path}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-opacity=".8"/>
      <circle cx="${sx(land[0])}" cy="${sy(land[1])}" r="2.6" fill="currentColor"/>
    </svg>`;
  }

  let mounted = null;   // avoid duplicate cards on re-render

  /* opts: { aside: ".results-media", club, tier, metricsRows, sourceLabel } */
  function mount(opts) {
    const aside = document.querySelector(opts.aside || ".results-media");
    if (!aside) return;
    if (mounted) mounted.remove();
    const card = document.createElement("div");
    card.className = "card media-card";
    card.id = "ballflight-card";
    const clubOpts = Object.keys(CLUBS).map(c =>
      `<option value="${c}"${c === (normClub(opts.club) || "driver") ? " selected" : ""}>${c}</option>`).join("");
    const tiers = [["tour", "Tour (PGA)"], ["lpga", "Tour (LPGA)"], ["amateur", "Amateur"]];
    const tierOpts = tiers.map(([v, l]) =>
      `<option value="${v}"${v === (opts.tier || "amateur") ? " selected" : ""}>${l}</option>`).join("");
    card.innerHTML = `
      <h3>Ball flight <span class="muted small">— simulated</span></h3>
      <div class="bf-controls" style="display:flex;gap:8px;margin:6px 0 8px;flex-wrap:wrap">
        <label class="small muted">Club <select id="bf-club">${clubOpts}</select></label>
        <label class="small muted">Launch like <select id="bf-tier">${tierOpts}</select></label>
      </div>
      <div id="bf-arc" style="opacity:.9"></div>
      <p id="bf-line" class="small" style="margin:6px 0 2px"></p>
      <p id="bf-note" class="muted small" style="margin:2px 0 0"></p>`;
    aside.appendChild(card);
    mounted = card;

    const render = async () => {
      const club = card.querySelector("#bf-club").value;
      const tier = card.querySelector("#bf-tier").value;
      const e = await estimate(club, tier, opts.metricsRows, opts.handSpeed);
      card.querySelector("#bf-arc").innerHTML = arcSVG(e.traj, e.carry, e.apex);
      card.querySelector("#bf-line").innerHTML =
        `<strong>≈${Math.round(e.carry)} yd carry</strong> · apex ${Math.round(e.apex)} yd · ` +
        `${e.time.toFixed(1)} s flight`;
      const nudgeTxt = e.nudge.pct
        ? ` Ball speed nudged ${e.nudge.pct > 0 ? "+" : ""}${e.nudge.pct}% for this swing's measured hand speed.` : "";
      card.querySelector("#bf-note").textContent =
        `Physics simulation (${e.engine === "phys_nn" ? "deep-learning aerodynamics" : "published aerodynamics"}) ` +
        `from typical ${tiers.find(t => t[0] === tier)[1]} launch numbers for a ${club} — an estimate, ` +
        `not a measurement of the ball in the video.${nudgeTxt} Ask the coach for more detail.`;
    };
    card.querySelector("#bf-club").addEventListener("change", render);
    card.querySelector("#bf-tier").addEventListener("change", render);
    render();
  }

  function unmount() { if (mounted) { mounted.remove(); mounted = null; } }

  return { mount, unmount, estimate };
})();
if (typeof window !== "undefined") window.BallFlight = BallFlight;
