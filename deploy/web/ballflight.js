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

  /* ------------------ playing conditions (mirror ball_flight.py) ----------
   * airDensity(): ideal gas + humid-air vapor correction; defaults reproduce
   * the historic RHO constant. Wind here is the ALONG-LINE component only
   * (tail/head) — this card's sim is 2D (downrange x height); crosswind curve
   * is the coach's 3D tool's job. */
  function airDensity(tempC = 15, pressureHpa = null, humidityPct = 0, elevM = 0) {
    const t = Math.min(Math.max(tempC, -20), 50);
    const p = (pressureHpa != null ? pressureHpa
      : 1013.25 * Math.pow(1 - 2.25577e-5 * Math.min(Math.max(elevM, -100), 4000), 5.25588)) * 100;
    const rh = Math.min(Math.max(humidityPct, 0), 100) / 100;
    const pSat = 610.94 * Math.exp(17.625 * t / (t + 243.04));
    const pv = rh * pSat, pd = p - pv, tK = t + 273.15;
    return pd / (287.058 * tK) + pv / (461.495 * tK);
  }
  /* conditions: { tempC?, pressureHpa?, humidityPct?, elevM?, windMph?,
   *               windDirDeg? } — windDirDeg is where the wind blows TOWARD
   * (0 = tailwind, 180 = headwind); only its along-line component acts here. */
  function condPhysics(conditions) {
    const c = conditions || {};
    const rho = (c.tempC != null || c.pressureHpa != null || c.humidityPct != null || c.elevM != null)
      ? airDensity(c.tempC ?? 15, c.pressureHpa ?? null, c.humidityPct ?? 0, c.elevM ?? 0)
      : RHO;
    const windY = c.windMph
      ? Math.min(Math.max(c.windMph, 0), 40) * MPH * Math.cos((c.windDirDeg ?? 0) * Math.PI / 180)
      : 0;
    return { rho, windY, nonstandard: !!windY || Math.abs(rho - RHO) > 0.001 };
  }

  /* ------------------------- Phys-Q polynomial sim ------------------------ */
  function simulatePhysQ(ballMph, launchDeg, backRpm, conditions) {
    const dt = 0.01;
    const { rho, windY } = condPhysics(conditions);
    let v = ballMph * MPH;
    const la = launchDeg * Math.PI / 180;
    let vy = v * Math.cos(la), vz = v * Math.sin(la);
    let wx = backRpm * RPM;                      // backspin only (straight shot)
    let y = 0, z = 0.01, apex = z, t = 0;
    const traj = [[0, 0]];
    while (t < 15) {
      // aerodynamics act on the air-relative velocity (ground velocity - wind)
      const ay = vy - windY, az = vz;
      const vmag = Math.hypot(ay, az) || 1e-9;
      const S = R * Math.abs(wx) / vmag;
      const CD = 0.1304 + 0.9287 * S - 0.8259 * S * S;
      const CL = 0.0504 + 1.2031 * S - 1.1490 * S * S;
      const CM = 0.01 * S;
      const q = 0.5 * rho * vmag * vmag;
      // Magnus lift = (w x v)/|w x v|: with w=+x and v in the y-z plane this is
      // (-az, ay)/vmag in (y, z)
      const FL = CL * q * A, FD = CD * q * A;
      const fy = FL * (-az / vmag) - FD * (ay / vmag);
      const fz = FL * (ay / vmag) - FD * (az / vmag) - M * G;
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
  async function estimate(club, tier, metricsRows, handSpeed, conditions) {
    const key = normClub(club) || "driver";
    const [bs0, la, spin] = CLUBS[key][tier] || CLUBS[key].tour;
    const nudge = speedScaleFrom(metricsRows, handSpeed);
    const bs = bs0 * nudge.scale;
    const cond = condPhysics(conditions);
    const nn = await loadNN();
    // the NN was trained at standard density with no wind — non-standard
    // conditions force the Phys-Q integrator (same rule as ball_flight.py)
    const r = (nn && inEnvelope(nn, bs, la, spin) && !cond.nonstandard)
      ? simulateNN(nn, bs, la, spin) : simulatePhysQ(bs, la, spin, conditions);
    return { ...r, club: key, tier, ballMph: bs, launchDeg: la, spinRpm: spin, nudge,
             conditions: cond.nonstandard ? conditions : null };
  }

  /* ------------------------------- card UI -------------------------------- */
  /* One scale for BOTH axes, so the drawing has the flight's real proportions.
   * Scaling x and y independently stretched every trajectory to fill the box,
   * which drew a towering rainbow for a flat 190 yd drive and an identical
   * rainbow for a 60 yd apex wedge — the shape carried no information at all. */
  function arcSVG(traj, carry, apex, label) {
    const W = 300, H = 96, PAD = 10, GY = H - 16;
    const maxX = Math.max(...traj.map(p => p[0])) || 1;
    const maxY = Math.max(...traj.map(p => p[1])) || 1;
    const k = Math.min((W - 2 * PAD) / maxX, (GY - PAD) / maxY);
    const sx = x => PAD + x * k;
    const sy = y => GY - y * k;
    const path = traj.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
    const land = traj[traj.length - 1];
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:360px;display:block" role="img"
      aria-label="${label || "Simulated"} ball flight drawn to scale: about ${Math.round(carry)} yards carry, apex ${Math.round(apex)} yards">
      <line x1="${PAD}" y1="${GY}" x2="${W - PAD}" y2="${GY}" stroke="currentColor" stroke-opacity=".25"/>
      <path d="${path}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-opacity=".8"/>
      <circle cx="${sx(land[0])}" cy="${sy(land[1])}" r="2.6" fill="currentColor"/>
    </svg>`;
  }

  let mounted = null;   // avoid duplicate cards on re-render

  /* measured panel (ball_3d.json quality measured/partial) — the real numbers
   * from tracking the ball in THIS video; the simulator demotes to a what-if. */
  function measuredHTML(ball) {
    const q = ball.quality, fit = ball.fit || {}, fl = ball.flight || {};
    const az = fit.azimuth_deg || 0;
    const dir = az > 3 ? `${Math.abs(az).toFixed(0)}° right`
      : az < -3 ? `${Math.abs(az).toFixed(0)}° left` : "straight";
    const ci = (ball.ci_10_90 || {}).carry_yd;
    // Show the range on BOTH tiers: a single-camera fit pins the flight's shape
    // far better than its scale, and hiding that made the number look surveyed.
    const range = ci ? ` <span class="muted small">(${Math.round(ci[0])}–${Math.round(ci[1])} yd)</span>` : "";
    const carry = q === "measured"
      ? `<strong>${fl.carry_yd} yd carry</strong>${range}`
      : `<strong>≈${fl.carry_yd} yd carry</strong>${range} <span class="muted small">(estimated)</span>`;
    const speed = q === "measured"
      ? `${fit.ball_speed_mph} mph ball speed` : `ball speed assumed (${fit.ball_speed_mph} mph)`;
    return `
      <div id="bf-measured" style="margin:4px 0 8px">
        <div id="bf-measured-arc" style="opacity:.9"></div>
        <p class="small" style="margin:6px 0 2px">${carry} · apex ${fl.apex_yd} yd · ${fl.flight_time_s} s</p>
        <p class="small" style="margin:2px 0 2px">Launch <strong>${fit.launch_deg}°</strong> ·
          started <strong>${dir}</strong> of the camera line · ${speed}</p>
        <p class="muted small" style="margin:2px 0 0">${q === "measured"
          ? `Measured from the ball tracked in your video (${ball.n_track_points} frames), fitted with the physics model.`
          : `Launch angle + direction measured from the ball tracked in your video (${ball.n_track_points} frames); distance simulated from that launch with club-typical speed.`}</p>
      </div>`;
  }

  /* Tracking ran but this swing's flight could not be confirmed. Say so: the
   * golfer otherwise just finds their own ball flight missing, with nothing but
   * the Amateur/PGA/LPGA tiers on screen and no idea whether the feature is
   * broken or their video is. */
  function unmeasuredHTML(ball) {
    const n = ball.n_track_points || 0;
    const why = n >= 5
      ? `We followed a ball for ${n} frames of this swing, but the path did not
         hold together as a flight well enough to measure — so the arc below is
         a simulation, not your shot.`
      : `We could not pick your ball out of this video, so the arc below is a
         simulation, not your shot.`;
    return `
      <p class="muted small" id="bf-unmeasured" style="margin:4px 0 8px">${why}
        Ball tracking wants the ball against open sky: film down the line, keep
        the landing area in frame, and avoid shooting into glare.</p>`;
  }

  /* opts: { aside: ".results-media", club, tier, metricsRows, sourceLabel,
   *         ball: parsed ball_3d.json | null } */
  function mount(opts) {
    const aside = document.querySelector(opts.aside || ".results-media");
    if (!aside) return;
    if (mounted) mounted.remove();
    const card = document.createElement("div");
    card.className = "card media-card";
    card.id = "ballflight-card";
    const ball = opts.ball && ["measured", "partial"].includes(opts.ball.quality) ? opts.ball : null;
    const clubOpts = Object.keys(CLUBS).map(c =>
      `<option value="${c}"${c === (normClub(opts.club) || "driver") ? " selected" : ""}>${c}</option>`).join("");
    const tiers = [["tour", "Tour (PGA)"], ["lpga", "Tour (LPGA)"], ["amateur", "Amateur"]];
    const tierOpts = tiers.map(([v, l]) =>
      `<option value="${v}"${v === (opts.tier || "amateur") ? " selected" : ""}>${l}</option>`).join("");
    const badge = ball
      ? (ball.quality === "measured" ? "— measured from your video" : "— measured launch")
      : "— simulated";
    const tracked = !ball && opts.ball && opts.ball.quality === "simulated";
    card.innerHTML = `
      <h3>Ball flight <span class="muted small">${badge}</span></h3>
      ${ball ? measuredHTML(ball) : ""}
      ${tracked ? unmeasuredHTML(opts.ball) : ""}
      ${ball ? `<p class="muted small" style="margin:8px 0 2px;border-top:1px solid rgba(127,127,127,.25);padding-top:6px">What-if simulator</p>` : ""}
      <div class="bf-controls" style="display:flex;gap:8px;margin:6px 0 8px;flex-wrap:wrap">
        <label class="small muted">Club <select id="bf-club">${clubOpts}</select></label>
        <label class="small muted">Launch like <select id="bf-tier">${tierOpts}</select></label>
      </div>
      <div id="bf-arc" style="opacity:.9"></div>
      <p id="bf-line" class="small" style="margin:6px 0 2px"></p>
      <p id="bf-note" class="muted small" style="margin:2px 0 0"></p>`;
    aside.appendChild(card);
    mounted = card;
    if (ball && Array.isArray(ball.trajectory_world) && ball.trajectory_world.length > 2) {
      const traj = ball.trajectory_world.map(p => [p[0], p[1]]);
      card.querySelector("#bf-measured-arc").innerHTML =
        arcSVG(traj, ball.flight.carry_yd, ball.flight.apex_yd,
               ball.quality === "measured" ? "Measured" : "Measured-launch");
    }

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
