// hud_engines.js — switchable HUD render modes
// Loaded BEFORE index.html's inline script. Exposes window.HUD.
(function () {
  'use strict';

  const CANVAS_SIZE = 480;
  const CX = CANVAS_SIZE / 2;
  const CY = CANVAS_SIZE / 2;

  // Published by index.html's arc-reactor loop every frame; read by
  // the two alternate engines below so all three react to the same
  // audio + interaction state.
  const hudState = {
    audioIntensity: 0,
    interactionForce: 0,
    time: 0,
  };

  // Initialize the shared mode flag so index.html's drawParticles
  // knows it should draw (mode === 'current').
  window.hudMode = 'current';

  function lerp(a, b, t) { return a + (b - a) * t; }
  function damp(a, b, rate, dt) { return lerp(a, b, 1 - Math.exp(-rate * dt)); }

  // ---- 3D helpers --------------------------------------------------
  // Rotate a point around Y, then X, and perspective-project to canvas.
  // depth < 0 means "behind the center", depth > 0 means "in front".
  function project3D(x, y, z, rotX, rotY, fov) {
    fov = fov || 500;
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const y1 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;
    const k = fov / Math.max(0.1, z2 + fov);
    return {
      sx: CX + x1 * k,
      sy: CY + y1 * k,
      depth: z2,
      scale: k,
    };
  }

  // Even distribution of N points on a sphere. The golden-angle spiral
  // is the standard trick for this — no clustering at the poles.
  function fibonacciSphere(n, radius) {
    const pts = [];
    const phi = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (i / (n - 1)) * 2;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = phi * i;
      pts.push({
        x: Math.cos(theta) * r * radius,
        y: y * radius,
        z: Math.sin(theta) * r * radius,
      });
    }
    return pts;
  }

  // =================================================================
  // ENGINE: Particle Sphere
  // =================================================================
  function ParticleSphereEngine(canvas) {
    this.ctx = canvas.getContext('2d');
    this.particles = fibonacciSphere(360, 160);
    this.rotY = 0;
    this.rotX = 0.2;
    this.rafId = null;
    this.lastTs = 0;
    this.elapsed = 0;
  }

  ParticleSphereEngine.prototype.start = function () {
    this.lastTs = performance.now();
    const loop = (now) => {
      const dt = Math.min((now - this.lastTs) / 1000, 0.05);
      this.lastTs = now;
      this.elapsed += dt;
      this.update(dt);
      this.draw();
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  };

  ParticleSphereEngine.prototype.stop = function () {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  };

  ParticleSphereEngine.prototype.update = function (dt) {
    const audio = hudState.audioIntensity;
    const interact = hudState.interactionForce;
    this.rotY += dt * (0.15 + audio * 1.5 + interact * 0.8);
    this.rotX = damp(this.rotX, 0.2 + Math.sin(this.elapsed * 0.4) * 0.15, 2, dt);
  };

  ParticleSphereEngine.prototype.draw = function () {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const audio = hudState.audioIntensity;
    const pulse = 1 + audio * 0.15 + Math.sin(this.elapsed * 1.5) * 0.02;

    // Project all points, then sort far-to-near so near particles
    // paint over far ones (correct depth occlusion for free).
    const projected = [];
    for (const p of this.particles) {
      projected.push(project3D(p.x * pulse, p.y * pulse, p.z * pulse, this.rotX, this.rotY));
    }
    projected.sort((a, b) => a.depth - b.depth);

    for (const p of projected) {
      const depthNorm = (p.depth + 160) / 320;       // 0 far .. 1 near
      const size = 0.8 + depthNorm * 2.4 + audio * 1.5;
      const alpha = 0.12 + depthNorm * 0.8;
      const r = Math.round(depthNorm * 200);
      const g = Math.round(180 + depthNorm * 60);
      const b = 255;

      ctx.save();
      ctx.shadowBlur = 8 + depthNorm * 10;
      ctx.shadowColor = 'rgba(0, 229, 255, ' + (alpha * 0.8) + ')';
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Soft central glow
    const glowSize = 40 + audio * 20;
    const glow = ctx.createRadialGradient(CX, CY, 0, CX, CY, glowSize);
    glow.addColorStop(0, 'rgba(0, 229, 255, ' + (0.25 + audio * 0.3) + ')');
    glow.addColorStop(0.5, 'rgba(0, 180, 255, ' + (0.08 + audio * 0.1) + ')');
    glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(CX, CY, glowSize, 0, Math.PI * 2);
    ctx.fill();
  };

  // =================================================================
  // ENGINE: Mesh Particles (dots + connecting edges)
  // =================================================================
  function MeshParticleEngine(canvas) {
    this.ctx = canvas.getContext('2d');
    this.particles = fibonacciSphere(200, 155);
    this.rotY = 0;
    this.rotX = 0.15;
    this.rafId = null;
    this.lastTs = 0;
    this.elapsed = 0;

    // Precompute neighbor edges on the *rest* sphere. This is
    // O(n²) but n=200 → 20k distance checks, done once. Lines
    // connect points within 42% of the radius, which produces a
    // geodesic-looking net without over-connecting.
    this.edges = [];
    const threshold = 0.42 * 155;
    const t2 = threshold * threshold;
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const a = this.particles[i];
        const b = this.particles[j];
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        if (dx * dx + dy * dy + dz * dz < t2) {
          this.edges.push([i, j]);
        }
      }
    }
  }

  MeshParticleEngine.prototype.start = function () {
    this.lastTs = performance.now();
    const loop = (now) => {
      const dt = Math.min((now - this.lastTs) / 1000, 0.05);
      this.lastTs = now;
      this.elapsed += dt;
      this.update(dt);
      this.draw();
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  };

  MeshParticleEngine.prototype.stop = function () {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    this.ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  };

  MeshParticleEngine.prototype.update = function (dt) {
    const audio = hudState.audioIntensity;
    const interact = hudState.interactionForce;
    this.rotY += dt * (0.12 + audio * 1.2 + interact * 0.6);
    this.rotX = damp(this.rotX, 0.15 + Math.sin(this.elapsed * 0.35) * 0.12, 2, dt);
  };

  MeshParticleEngine.prototype.draw = function () {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    const audio = hudState.audioIntensity;
    const pulse = 1 + audio * 0.1 + Math.sin(this.elapsed * 1.2) * 0.015;

    const projected = this.particles.map((p) =>
      project3D(p.x * pulse, p.y * pulse, p.z * pulse, this.rotX, this.rotY)
    );

    // Edges first — behind the dots
    ctx.lineWidth = 0.6;
    for (let e = 0; e < this.edges.length; e++) {
      const [i, j] = this.edges[e];
      const a = projected[i];
      const b = projected[j];
      const avgDepth = (a.depth + b.depth) / 2;
      const depthNorm = (avgDepth + 160) / 320;
      const alpha = 0.04 + depthNorm * 0.28 + audio * 0.05;
      ctx.strokeStyle = 'rgba(0, 200, 255, ' + alpha + ')';
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    }

    // Dots, far to near
    const order = projected
      .map((p) => ({ depth: p.depth, sx: p.sx, sy: p.sy }))
      .sort((a, b) => a.depth - b.depth);
    for (const p of order) {
      const depthNorm = (p.depth + 160) / 320;
      const size = 1 + depthNorm * 2.2 + audio * 1.2;
      const alpha = 0.2 + depthNorm * 0.7;

      ctx.save();
      ctx.shadowBlur = 6 + depthNorm * 8;
      ctx.shadowColor = 'rgba(0, 229, 255, ' + (alpha * 0.7) + ')';
      ctx.fillStyle = 'rgba(180, 240, 255, ' + alpha + ')';
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Central glow
    const glowSize = 35 + audio * 18;
    const glow = ctx.createRadialGradient(CX, CY, 0, CX, CY, glowSize);
    glow.addColorStop(0, 'rgba(0, 229, 255, ' + (0.22 + audio * 0.25) + ')');
    glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(CX, CY, glowSize, 0, Math.PI * 2);
    ctx.fill();
  };

  // =================================================================
  // MODE MANAGER
  // =================================================================
  const engines = { particles: null, mesh: null };
  let currentMode = 'current';
  let canvasEl = null;

  function setCanvas(canvas) {
    canvasEl = canvas;
  }

  function setMode(mode) {
    if (['current', 'particles', 'mesh'].indexOf(mode) === -1) return;
    if (mode === currentMode) return;

    // Stop any active engine
    if (engines.particles) engines.particles.stop();
    if (engines.mesh) engines.mesh.stop();

    // Tell index.html's arc-reactor loop whether to draw
    window.hudMode = mode;

    // Fade the SVG center-circle / pulse so the particle modes can
    // own the visual center without a big solid dot fighting them.
    document.body.classList.toggle('hud-alt-mode', mode !== 'current');

    if (mode === 'particles') {
      if (!engines.particles) engines.particles = new ParticleSphereEngine(canvasEl);
      engines.particles.start();
    } else if (mode === 'mesh') {
      if (!engines.mesh) engines.mesh = new MeshParticleEngine(canvasEl);
      engines.mesh.start();
    }
    // 'current' → nothing to start; index.html's loop just resumes
    // drawing on the next frame since hudMode is back to 'current'.

    currentMode = mode;
    try { localStorage.setItem('jarvisHUDMode', mode); } catch (e) { /* non-fatal */ }
  }

  function getMode() {
    return currentMode;
  }

  window.HUD = {
    setCanvas: setCanvas,
    setMode: setMode,
    getMode: getMode,
    state: hudState,
  };
})();