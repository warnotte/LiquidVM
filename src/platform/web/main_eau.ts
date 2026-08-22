/**
 * Point d'entrée du chantier EAU (page eau.html — branche `eau`, voir
 * PLAN-EAU.md). ÉTAT : J1 — dam break FLIP/PIC rendu en points.
 * `?bench` relance le banc J0 du scatter atomique.
 * `?selftest` : rapport JSON après 240 frames, titre SELFTEST-OK/FAIL.
 */

import { BenchP2G } from '../../liquid3d/bench_p2g';
import { EAU_DEFAULTS, GRID_EAU, PARTICLES_EAU } from '../../liquid3d/config_eau';
import { FluidEau, type FrameEauInput } from '../../liquid3d/sim_eau';
import { Panel3D } from './panel3d';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const J0_BUDGET_MS = 2.0;
const SELFTEST_FRAMES = 240;

async function runBench(hud: HTMLDivElement, device: GPUDevice, selftest: boolean): Promise<void> {
  hud.textContent = `banc P2G : ${PARTICLES_EAU.toLocaleString('fr')} particules sur ${GRID_EAU}³, mesure en cours…`;
  const bench = await BenchP2G.create(device);
  const result = await bench.measure();
  const pass = result.msSorted < J0_BUDGET_MS;
  const fmt = (xs: readonly number[]): string => xs.map((s) => s.toFixed(2)).join(' · ');
  hud.innerHTML =
    `<b>LiquidVM eau — J0 : banc P2G</b><br>` +
    `${PARTICLES_EAU.toLocaleString('fr')} particules groupées → grille MAC ${GRID_EAU}³<br>` +
    `ordre aléatoire (pire cas) : <b>${result.msRandom.toFixed(2)} ms</b> — ${fmt(result.samplesRandom)}<br>` +
    `ordre trié par cellule : <b>${result.msSorted.toFixed(2)} ms</b> — ${fmt(result.samplesSorted)}<br>` +
    `verdict J0 (cas trié, budget ${J0_BUDGET_MS} ms) : ` +
    `<span class="${pass ? 'pass' : 'fail'}">${pass ? 'PASS' : 'FAIL'}</span>`;
  if (selftest) {
    const report = document.createElement('div');
    report.id = 'selftest';
    report.textContent = JSON.stringify({
      ok: pass,
      msRandom: Number(result.msRandom.toFixed(3)),
      msSorted: Number(result.msSorted.toFixed(3)),
    });
    report.style.display = 'none';
    document.body.appendChild(report);
    document.title = pass ? 'SELFTEST-OK' : 'SELFTEST-FAIL';
  }
}

async function boot(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const selftest = params.has('selftest');
  const hud = document.getElementById('hud-eau') as HTMLDivElement;
  const { device, format } = await acquireDevice();
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      showFatalError(`Le device GPU a été perdu : ${info.message}`);
    }
  });

  if (params.has('bench')) {
    await runBench(hud, device, selftest);
    return;
  }

  const canvas = document.getElementById('canvas-eau') as HTMLCanvasElement;
  const context = canvas.getContext('webgpu');
  if (context === null) {
    throw new Error('Impossible de créer un contexte WebGPU sur le canvas.');
  }
  context.configure({ device, format, alphaMode: 'opaque' });
  const resize = (): void => {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
  };
  resize();
  window.addEventListener('resize', resize);

  const sim = await FluidEau.create(device, format);
  const input: FrameEauInput = {
    dt: 0,
    paused: false,
    reset: false,
    gravity: EAU_DEFAULTS.gravity,
    flipBlend: EAU_DEFAULTS.flipBlend,
    damWidth: params.has('tall') ? 32 : EAU_DEFAULTS.damWidth,
    jacobiIterations: EAU_DEFAULTS.jacobiIterations,
    substeps: EAU_DEFAULTS.substeps,
    timeScale: EAU_DEFAULTS.timeScale,
    pointSize: EAU_DEFAULTS.pointSize,
    exposure: EAU_DEFAULTS.exposure,
    renderPoints: params.has('points') || EAU_DEFAULTS.renderPoints,
    absorption: EAU_DEFAULTS.absorption,
    surfaceIso: EAU_DEFAULTS.surfaceIso,
    debugView: EAU_DEFAULTS.debugView,
    cam: {
      azimuth: EAU_DEFAULTS.camAzimuth,
      elevation: EAU_DEFAULTS.camElevation,
      radius: EAU_DEFAULTS.camRadius,
    },
  };
  if (selftest) {
    (window as unknown as Record<string, unknown>)['__frameEau'] = input;
    (window as unknown as Record<string, unknown>)['__simEau'] = sim;
  }

  // Caméra orbitale : glisser = orbiter, molette = zoom (même feeling que 3d.html).
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // capture refusée : sans gravité
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
      return;
    }
    input.cam.azimuth += (e.clientX - lastX) * 0.008;
    input.cam.elevation = Math.min(
      Math.max(input.cam.elevation + (e.clientY - lastY) * 0.006, -1.25),
      1.45,
    );
    lastX = e.clientX;
    lastY = e.clientY;
  });
  const endDrag = (): void => {
    dragging = false;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      input.cam.radius = Math.min(Math.max(input.cam.radius * Math.exp(e.deltaY * 0.001), 0.9), 4.5);
    },
    { passive: false },
  );
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      input.paused = !input.paused;
    } else if (e.code === 'KeyR') {
      input.reset = true;
    } else if (e.code === 'KeyP') {
      input.renderPoints = !input.renderPoints;
      panel.refresh();
    }
  });

  const panel = new Panel3D(document.body, [
    {
      title: 'eau (J1 — dam break)',
      sliders: [
        { label: 'gravité', min: 0, max: 2000, step: 20, get: () => input.gravity, set: (x) => (input.gravity = x), format: (x) => x.toFixed(0) },
        { label: 'mélange FLIP', min: 0, max: 1, step: 0.01, get: () => input.flipBlend, set: (x) => (input.flipBlend = x) },
        { label: 'itérations Jacobi', min: 10, max: 120, step: 2, get: () => input.jacobiIterations, set: (x) => (input.jacobiIterations = x), format: (x) => x.toFixed(0) },
        { label: 'sous-pas', min: 1, max: 4, step: 1, get: () => input.substeps, set: (x) => (input.substeps = x), format: (x) => x.toFixed(0) },
        { label: 'vitesse du temps', min: 0, max: 1.5, step: 0.05, get: () => input.timeScale, set: (x) => (input.timeScale = x), format: (x) => `×${x.toFixed(2)}` },
      ],
      checks: [
        {
          label: 'colonne haute 32×64 (reset)',
          get: () => input.damWidth === 32,
          set: (v) => {
            input.damWidth = v ? 32 : 64;
            input.reset = true;
          },
        },
      ],
      buttons: [
        { label: '⏯ pause (espace)', action: () => (input.paused = !input.paused) },
        { label: '↺ reset (R)', action: () => (input.reset = true) },
      ],
    },
    {
      title: 'rendu',
      sliders: [
        { label: 'absorption', min: 0, max: 3, step: 0.05, get: () => input.absorption, set: (x) => (input.absorption = x) },
        { label: 'seuil de surface', min: 0.1, max: 0.9, step: 0.05, get: () => input.surfaceIso, set: (x) => (input.surfaceIso = x) },
        { label: 'exposition', min: 0.2, max: 3, step: 0.05, get: () => input.exposure, set: (x) => (input.exposure = x) },
        { label: 'taille des points', min: 0.001, max: 0.01, step: 0.0005, get: () => input.pointSize, set: (x) => (input.pointSize = x), format: (x) => x.toFixed(4) },
        { label: 'vue debug', min: 0, max: 3, step: 1, get: () => input.debugView, set: (x) => (input.debugView = x), format: (x) => ['surface', '—', 'coupe z=0', 'densité max'][Math.round(x)] ?? '' },
      ],
      checks: [
        { label: 'points bruts (P — instrument physique)', get: () => input.renderPoints, set: (v) => (input.renderPoints = v) },
      ],
    },
  ]);

  let last = performance.now();
  let fps = 60;
  let frames = 0;
  let hudTimer = 0;
  const tick = (now: number): void => {
    const dt = Math.min(Math.max((now - last) / 1000, 0), 0.05);
    last = now;
    if (dt > 0) {
      fps = fps * 0.95 + (1 / dt) * 0.05;
    }
    input.dt = dt;
    sim.frame(input, context.getCurrentTexture().createView(), canvas.width, canvas.height);
    input.reset = false;
    frames++;
    hudTimer += dt;
    if (hudTimer > 0.5) {
      hudTimer = 0;
      panel.refresh();
      const [valid, lost, fast] = sim.lastCensus;
      const hist = Array.from(sim.lastCensus.subarray(66, 72), (x) => (x / 1000).toFixed(1) + 'k');
      hud.innerHTML =
        `<b>LiquidVM eau — J1 : dam break</b> · ${GRID_EAU}³ · ` +
        `${PARTICLES_EAU.toLocaleString('fr')} particules · ${input.substeps} sous-pas · ` +
        `Jacobi ${input.jacobiIterations} · ${Math.round(fps)} FPS${input.paused ? ' · ⏸' : ''}<br>` +
        `recensement : <b>${(valid ?? 0).toLocaleString('fr')}</b> valides · ` +
        `${(lost ?? 0).toLocaleString('fr')} perdues · ${(fast ?? 0).toLocaleString('fr')} rapides · ` +
        `cellules 1-3/4-7/<b>8-11</b>/12-23/24+ : ${hist.slice(1).join(' / ')}<br>` +
        'glisser : orbiter · molette : zoom · espace : pause · R : reset · Tab : réglages';
    }
    if (selftest && frames === SELFTEST_FRAMES) {
      const report = document.createElement('div');
      report.id = 'selftest';
      report.setAttribute('data-stage', `frames-${frames}`);
      report.textContent = JSON.stringify({ ok: true, frames, fps: Math.round(fps) });
      report.style.display = 'none';
      document.body.appendChild(report);
      document.title = 'SELFTEST-OK';
    }
    if (selftest) {
      document.getElementById('selftest')?.setAttribute('data-stage', `frames-${frames}`);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

boot().catch((err: unknown) => {
  showFatalError(err instanceof Error ? err.message : String(err));
  document.title = 'SELFTEST-FAIL';
});
