/**
 * Point d'entrée du prototype 3D volumétrique (page 3d.html).
 * Couche plateforme minimale : device, canvas/resize, caméra orbitale à la souris
 * (glisser = orbiter, molette = zoom), espace = pause, R = reset, HUD FPS.
 * Mode ?selftest : rapport JSON dans #selftest + titre SELFTEST-OK/FAIL après 240 frames.
 */

import { GRID3, SIM3_DEFAULTS } from '../../core3d/config3d';
import { FluidSim3D, type Frame3DInput } from '../../core3d/sim3d';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const SELFTEST_FRAMES = 240;

async function boot(): Promise<void> {
  const selftest = new URLSearchParams(location.search).has('selftest');
  const canvas = document.getElementById('canvas3d') as HTMLCanvasElement;
  const hud = document.getElementById('hud3d') as HTMLDivElement;

  const { device, format } = await acquireDevice();
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      showFatalError(`Le device GPU a été perdu : ${info.message}`);
    }
  });

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

  const sim = await FluidSim3D.create(device, format);

  const input: Frame3DInput = {
    dt: 0,
    paused: false,
    reset: false,
    multigrid: SIM3_DEFAULTS.multigrid,
    vcycles: SIM3_DEFAULTS.vcycles,
    jacobiIterations: SIM3_DEFAULTS.jacobiIterations,
    vorticityStrength: SIM3_DEFAULTS.vorticityStrength,
    cam: {
      azimuth: SIM3_DEFAULTS.camAzimuth,
      elevation: SIM3_DEFAULTS.camElevation,
      radius: SIM3_DEFAULTS.camRadius,
    },
    exposure: SIM3_DEFAULTS.exposure,
    raymarchSteps: SIM3_DEFAULTS.raymarchSteps,
  };
  if (selftest) {
    (window as unknown as Record<string, unknown>)['__frame3d'] = input;
  }

  // Caméra orbitale : glisser pour tourner, molette pour zoomer.
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.classList.add('dragging');
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Certains environnements tactiles refusent la capture : sans gravité.
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
    canvas.classList.remove('dragging');
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
    }
  });

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
    sim.frame(input, context.getCurrentTexture().createView(), canvas.width / canvas.height);
    input.reset = false;
    frames++;

    hudTimer += dt;
    if (hudTimer > 0.5) {
      hudTimer = 0;
      const solver = input.multigrid ? `MG ×${input.vcycles}` : `jacobi ${input.jacobiIterations} it.`;
      hud.innerHTML =
        `<b>LiquidVM 3D</b> · ${GRID3}³ · ${solver} · ${Math.round(fps)} FPS` +
        `${input.paused ? ' · ⏸ pause' : ''}<br>` +
        'glisser : orbiter · molette : zoom · espace : pause · R : reset';
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
  const message = err instanceof Error ? err.message : String(err);
  showFatalError(message);
  document.title = 'SELFTEST-FAIL';
});
