/* MotionCaddie — interactive WebGL "3D replay" (Version B).
 *
 * Renders the reconstructed swing as a lit, orbitable CAPSULE/TUBE mannequin, built
 * live from the same per-frame joint positions the canvas replay uses
 * (assets/<id>/replay_3d.json). Same neutral matte look as the offline Blender stills,
 * but real-time and draggable. No GLB export — capsules are constructed in-browser with
 * three.js CylinderGeometry (tapered) + joint spheres + a torso box, mirroring
 * Scripts/blender_mocap.py's geometry recipe.
 *
 * three.js is vendored locally (deploy/web/vendor/) — no CDN, static-hosting-safe.
 * Attaches window.CapsuleViewer3D. app.js uses it when WebGL is available and falls
 * back to the canvas Replay3D otherwise.
 */
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

// H36M-17 parents (child -> parent), matching Scripts/blender_mocap.py.
const PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15];
// limb bones drawn as tapered cylinders: child -> [r_proximal, r_distal] (metres).
// spine(7)/thorax(8) omitted — the torso box covers them.
const LIMB_R = {
  1: [0.080, 0.070], 4: [0.080, 0.070],
  2: [0.078, 0.052], 5: [0.078, 0.052],
  3: [0.050, 0.034], 6: [0.050, 0.034],
  9: [0.042, 0.034], 10: [0.034, 0.030],
  11: [0.052, 0.046], 14: [0.052, 0.046],
  12: [0.050, 0.038], 15: [0.050, 0.038],
  13: [0.037, 0.028], 16: [0.037, 0.028],
};
const JOINT_R = {
  0: 0.090, 8: 0.080, 1: 0.066, 4: 0.066, 2: 0.058, 5: 0.058, 3: 0.042, 6: 0.042,
  11: 0.054, 14: 0.054, 12: 0.042, 15: 0.042, 13: 0.034, 16: 0.034, 9: 0.036, 10: 0.082,
};
const TARGET_H = 1.78;                 // metric height the radii above assume
const YUP = new THREE.Vector3(0, 1, 0);

function webglSupported() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl")));
  } catch (e) { return false; }
}

class CapsuleViewer3D {
  static supported() { return webglSupported(); }

  constructor(canvas, data, ui) {
    this.canvas = canvas;
    this.ui = ui;
    this.playing = true;
    this.frame = 0;
    this._raf = null;
    this._last = 0;
    this._acc = 0;
    this.fps = data.fps || 30;

    // --- convert joints: h36m-camera (x right, y DOWN, z depth) -> three Y-up.
    // (x,y,z)_cam -> (x, -y, -z): det +1, upright, no mirror. Then metric-scale.
    const raw = data.frames.map(fr => fr.map(p => [p[0], -p[1], -p[2]]));
    const NB = 17;                     // body joints (exclude clubhead at 17)
    // metric scale from median vertical body span
    const spans = raw.map(fr => {
      let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < NB; j++) { lo = Math.min(lo, fr[j][1]); hi = Math.max(hi, fr[j][1]); }
      return hi - lo;
    }).sort((a, b) => a - b);
    const scale = TARGET_H / (spans[spans.length >> 1] || 1);
    // recenter on mean hip so orbit target is the body
    const mh = [0, 0, 0];
    for (const fr of raw) { mh[0] += fr[0][0]; mh[1] += fr[0][1]; mh[2] += fr[0][2]; }
    mh[0] /= raw.length; mh[1] /= raw.length; mh[2] /= raw.length;
    this.frames = raw.map(fr => fr.map(p => new THREE.Vector3(
      (p[0] - mh[0]) * scale, (p[1] - mh[1]) * scale, (p[2] - mh[2]) * scale)));
    this.nJoints = data.frames[0].length;
    this.clubBones = (data.bones || []).filter(b => b.a === 17 || b.b === 17);

    // body-only bounding radius (fixes the clubhead-inflates-framing issue)
    let radius = 1e-6, floorY = Infinity;
    for (const fr of this.frames) for (let j = 0; j < NB; j++) {
      radius = Math.max(radius, fr[j].length());
      floorY = Math.min(floorY, fr[j].y);
    }
    // grounded replays declare the floor exactly (data y=0 = leveled stance
    // line, y-down) — map that plane through the same convert/recenter/scale
    // as the joints instead of guessing from the lowest joint of the clip,
    // which floated one foot whenever the ankles didn't match heights.
    if (data.grounded === true) {
      const fy = typeof data.floor_y === "number" ? data.floor_y : 0;
      floorY = (-fy - mh[1]) * scale;
    }
    this.radius = radius;

    // measured ball trajectory (injected by ball_step): [[t_s, x, y, z], ...]
    // in the same h36m-camera frame as the joints. Rendered at TRUE scale —
    // a drive flies ~200 m, so the arc extends far beyond the body-scale scene;
    // controls' zoom-out range and the ground plane are widened to match, and
    // the ball grows with distance (tracer-style) so it stays visible.
    this.ballTrack = null;
    this.ballImpact = null;
    this.ballExtent = 0;
    if (data.ball && Array.isArray(data.ball.trajectory) && data.ball.trajectory.length > 1) {
      const pts = [];
      for (const [t, x, y, z] of data.ball.trajectory) {
        const v = new THREE.Vector3((x - mh[0]) * scale, (-y - mh[1]) * scale, (-z - mh[2]) * scale);
        pts.push({ t, v });
        this.ballExtent = Math.max(this.ballExtent, v.length());
      }
      if (pts.length > 1) {
        this.ballTrack = pts;
        this.ballImpact = data.ball.impact_frame || 0;
      }
    }

    this._initScene(floorY);
    this._buildBody();
    this._bindUI();
    this._resize();
    this._loop = this._loop.bind(this);
    this._onResize = () => this._resize();
    window.addEventListener("resize", this._onResize);
    this._raf = requestAnimationFrame(this._loop);
  }

  _initScene(floorY) {
    const r = this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;

    const s = this.scene = new THREE.Scene();
    s.background = new THREE.Color(0x0f2019);

    const far = Math.max(100, this.ballExtent * 4);   // keep a long ball arc in frustum
    const cam = this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, far);
    const d = this.radius / Math.sin((cam.fov * Math.PI / 180) / 2) * 1.15;
    cam.position.set(d * 0.35, d * 0.12, d);

    const ctr = this.controls = new OrbitControls(cam, this.canvas);
    ctr.target.set(0, 0, 0);
    ctr.enableDamping = true;
    ctr.dampingFactor = 0.08;
    ctr.enablePan = false;
    ctr.minDistance = this.radius * 1.2;
    // with a measured ball arc, let the user zoom out far enough to see it land
    ctr.maxDistance = Math.max(this.radius * 6, this.ballExtent * 1.25);

    s.add(new THREE.HemisphereLight(0xdfe7ee, 0x2a2f2c, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(2.4, 4.0, 2.2);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    const sc = this.radius * 1.6;
    Object.assign(key.shadow.camera, { left: -sc, right: sc, top: sc, bottom: -sc, near: 0.1, far: 20 });
    s.add(key);

    // subtle ground to catch a shadow (wide enough for the ball arc if present)
    const gsz = Math.max(this.radius * 8, this.ballExtent * 2.6);
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(gsz, gsz),
      new THREE.MeshStandardMaterial({ color: 0x16281f, roughness: 1.0 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = floorY - 0.02;
    ground.receiveShadow = true;
    s.add(ground);
  }

  _buildBody() {
    const skin = new THREE.MeshStandardMaterial({ color: 0xbcb6a8, roughness: 0.85, metalness: 0.0 });
    const clubMat = new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.6, metalness: 0.1 });
    const body = this.body = new THREE.Group();
    this.scene.add(body);
    const add = (mesh) => { mesh.castShadow = true; body.add(mesh); return mesh; };

    // tapered limb cylinders (base height 1 along +Y; scaled per frame)
    this.bones = [];
    for (const j of Object.keys(LIMB_R).map(Number)) {
      const [rp, rd] = LIMB_R[j];
      const g = new THREE.CylinderGeometry(rd, rp, 1, 18, 1);  // top=distal, bottom=proximal
      const m = add(new THREE.Mesh(g, skin));
      this.bones.push({ mesh: m, parent: PARENTS[j], child: j });
    }
    // rounded joint blobs
    this.joints = [];
    for (const j of Object.keys(JOINT_R).map(Number)) {
      const m = add(new THREE.Mesh(new THREE.SphereGeometry(JOINT_R[j], 20, 14), skin));
      this.joints.push({ mesh: m, j });
    }
    // torso box (hip_center 0 -> thorax 8), oriented to the shoulder line each frame
    this.torso = add(new THREE.Mesh(new THREE.BoxGeometry(0.32, 1, 0.20), skin));
    this.torso.matrixAutoUpdate = false;
    // club (thin), excluded from framing
    this.club = [];
    for (const b of this.clubBones) {
      const m = add(new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.014, 1, 10, 1), clubMat));
      m.castShadow = false;
      this.club.push({ mesh: m, a: b.a, b: b.b });
    }
    // measured ball: white sphere animated along the tracked flight + faint arc
    if (this.ballTrack) {
      const arcGeo = new THREE.BufferGeometry().setFromPoints(this.ballTrack.map(p => p.v));
      const arc = new THREE.Line(arcGeo,
        new THREE.LineBasicMaterial({ color: 0xffe27a, transparent: true, opacity: 0.55 }));
      this.scene.add(arc);
      this.ballArc = arc;
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.05, 16, 12),
        new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35 }));
      ball.visible = false;
      this.scene.add(ball);
      this.ballMesh = ball;
    }
    this._pose(0);
  }

  _orient(mesh, a, b) {
    const dir = new THREE.Vector3().subVectors(b, a);
    const len = dir.length() || 1e-6;
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    mesh.scale.set(1, len, 1);
    mesh.quaternion.setFromUnitVectors(YUP, dir.multiplyScalar(1 / len));
  }

  _pose(t) {
    const P = this.frames[t];
    for (const bn of this.bones) this._orient(bn.mesh, P[bn.parent], P[bn.child]);
    for (const jt of this.joints) jt.mesh.position.copy(P[jt.j]);
    for (const cb of this.club) this._orient(cb.mesh, P[cb.a], P[cb.b]);
    // torso: oriented basis (up = hip->thorax, x = shoulder line)
    const a = P[0], b = P[8];
    const y = new THREE.Vector3().subVectors(b, a); const len = y.length() || 1e-6; y.multiplyScalar(1 / len);
    const sx = new THREE.Vector3().subVectors(P[14], P[11]);
    if (sx.lengthSq() < 1e-9) sx.set(1, 0, 0);
    const z = new THREE.Vector3().crossVectors(sx, y).normalize();
    const x = new THREE.Vector3().crossVectors(y, z).normalize();
    const m = new THREE.Matrix4().makeBasis(x, y, z);
    m.scale(new THREE.Vector3(1, len, 1));
    m.setPosition(new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5));
    this.torso.matrix.copy(m);
    if (this.ballMesh) {
      const tf = (t - this.ballImpact) / this.fps;   // seconds since impact
      const trk = this.ballTrack;
      if (tf < 0 || tf > trk[trk.length - 1].t) {
        this.ballMesh.visible = false;
      } else {
        let k = 1;
        while (k < trk.length - 1 && trk[k].t < tf) k++;
        const a2 = trk[k - 1], b2 = trk[k];
        const w = (tf - a2.t) / ((b2.t - a2.t) || 1e-6);
        this.ballMesh.position.lerpVectors(a2.v, b2.v, Math.min(Math.max(w, 0), 1));
        // tracer sizing: keep the ball a few pixels tall however far it flies
        const s = 1 + this.ballMesh.position.length() * 0.28;
        this.ballMesh.scale.set(s, s, s);
        this.ballMesh.visible = true;
      }
    }
    if (this.ui.label) this.ui.label.textContent = `${t + 1} / ${this.frames.length}`;
    if (this.ui.scrub) this.ui.scrub.value = t;
  }

  _bindUI() {
    const u = this.ui;
    if (u.scrub) { u.scrub.min = 0; u.scrub.max = this.frames.length - 1; u.scrub.value = 0;
      this._onScrub = () => { this.playing = false; this.frame = +u.scrub.value; this._pose(this.frame); };
      u.scrub.addEventListener("input", this._onScrub); }
    if (u.playBtn) { this._onPlay = () => { this.playing = !this.playing; this._last = performance.now(); };
      u.playBtn.addEventListener("click", this._onPlay); }
  }

  _resize() {
    const w = this.canvas.clientWidth || this.canvas.width;
    const h = this.canvas.clientHeight || this.canvas.height;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop(now) {
    this._raf = requestAnimationFrame(this._loop);
    if (this.playing) {
      if (!this._last) this._last = now;
      this._acc += (now - this._last) / 1000; this._last = now;
      const dt = 1 / this.fps;
      while (this._acc >= dt) { this._acc -= dt; this.frame = (this.frame + 1) % this.frames.length; this._pose(this.frame); }
    } else { this._last = now; }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    if (this._raf) cancelAnimationFrame(this._raf);
    window.removeEventListener("resize", this._onResize);
    if (this.ui.scrub && this._onScrub) this.ui.scrub.removeEventListener("input", this._onScrub);
    if (this.ui.playBtn && this._onPlay) this.ui.playBtn.removeEventListener("click", this._onPlay);
    if (this.controls) this.controls.dispose();
    this.scene && this.scene.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());
    });
    if (this.renderer) this.renderer.dispose();
  }
}

window.CapsuleViewer3D = CapsuleViewer3D;
