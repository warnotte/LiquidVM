/**
 * Point d'entrée web : canvas, boucle requestAnimationFrame, câblage input/overlay.
 * Toute la logique GPU vit dans core/ — cette couche ne fait que fournir device,
 * cible de rendu, dt et input, et c'est elle (uniquement) qu'un portage natif réécrit.
 *
 * Mode `?selftest` : un pilote synthétique remplace la souris (orbite en injectant les
 * trois fluides tour à tour) et publie un rapport JSON dans le DOM après 200 frames —
 * utilisé pour la vérification automatisée en navigateur headless.
 */

import { SUBSTANCE_NAMES } from '../../core/config';
import { FluidSim } from '../../core/simulation';
import type { FrameInput, SubstanceId } from '../../core/types';

const BOUNDARY_LABELS = ['parois', 'périodique', 'ouvert'] as const;
const VIEW_LABELS = [
  'fluides',
  'vélocité',
  'pression',
  'divergence',
  'vorticité',
  'caméra',
] as const;
import { acquireDevice } from './gpu';
import { CameraFlow } from './camera';
import { HandTracking } from './hands';
import { InputController } from './input';
import { Overlay, showFatalError } from './overlay';
import { DebugPanel, TOOL_LABELS } from './panel';
import { MobileToolbar } from './toolbar';

const SELFTEST_REPORT_FRAME = 200;

/**
 * Pilote d'input synthétique du mode selftest : dessine d'abord un mur vertical
 * (frames 10–55, bouton secondaire simulé), puis orbite en injectant les trois fluides
 * tour à tour — l'orbite traverse le mur, ce qui rend la déflexion visible.
 */
class SelftestDriver {
  /** Options de bisection : ?selftest&hold=0|1|2 fige le mode, &nowall saute le mur,
   *  &marble déroule un scénario de marbrure (gouttes puis peigne, bain figé). */
  private readonly holdBoundary: number | null;
  private readonly noWall: boolean;
  private readonly marbleMode: boolean;

  constructor(private readonly frame: FrameInput) {
    const params = new URLSearchParams(location.search);
    const hold = params.get('hold');
    this.holdBoundary = hold === null ? null : Number(hold);
    this.noWall = params.has('nowall');
    this.marbleMode = params.has('marble');
  }

  drive(frameIndex: number): void {
    const p = this.frame.pointer;
    if (this.marbleMode) {
      // Scénario marbrure : bain figé, cinq gouttes (motif « pierre »), puis un coup
      // de peigne vertical qui tire les anneaux en chevrons.
      this.frame.paused = frameIndex > 5;
      const m = this.frame.marble;
      const drops: readonly [number, number, number][] = [
        [30, 0.5, 0.5],
        [55, 0.38, 0.42],
        [80, 0.6, 0.55],
        [105, 0.47, 0.62],
        [130, 0.55, 0.38],
      ];
      for (const [f, x, y] of drops) {
        if (frameIndex === f) {
          this.frame.selectedFluid = ((f / 25) % 3 | 0) as SubstanceId;
          m.pending = true;
          m.tool = 0;
          m.ax = m.bx = x;
          m.ay = m.by = y;
        }
      }
      if (frameIndex >= 170 && frameIndex <= 230) {
        const y = 0.15 + ((frameIndex - 170) / 60) * 0.7;
        m.pending = true;
        m.tool = 2;
        m.ax = 0.5;
        m.ay = y - 0.012;
        m.bx = 0.5;
        m.by = y;
      }
      return;
    }
    if (!this.noWall && frameIndex >= 10 && frameIndex <= 55) {
      const t = (frameIndex - 10) / 45;
      p.x = 0.63;
      p.y = 0.3 + 0.4 * t;
      p.dx = 0;
      p.dy = 0;
      p.down = false;
      p.wall = true;
      p.erase = false;
      return;
    }
    p.wall = false;
    const t = frameIndex / 60;
    const angle = t * 2.6;
    const x = 0.5 + 0.27 * Math.cos(angle);
    const y = 0.5 + 0.27 * Math.sin(angle);
    p.dx = frameIndex > 0 ? x - p.x : 0;
    p.dy = frameIndex > 0 ? y - p.y : 0;
    p.x = x;
    p.y = y;
    p.down = frameIndex > 55;
    // Cycle des trois fluides, puis matière feu à partir de la frame 160.
    this.frame.selectedFluid =
      frameIndex >= 160 ? 3 : ((Math.floor(frameIndex / 60) % 3) as SubstanceId);
    // Exerce les trois modes : parois → périodique (wrap visible) → ouvert (le fluide
    // qui atteint un bord disparaît). `&hold=` fige un mode pour la bisection.
    this.frame.boundaryMode = (this.holdBoundary ??
      (frameIndex >= 150 ? 2 : frameIndex >= 100 ? 1 : 0)) as FrameInput['boundaryMode'];
  }
}

function selftestElement(): HTMLElement {
  let el = document.getElementById('selftest');
  if (el === null) {
    el = document.createElement('div');
    el.id = 'selftest';
    el.style.display = 'none';
    document.body.appendChild(el);
  }
  return el;
}

/** Trace de progression du selftest, lisible dans un dump du DOM headless. */
function selftestStage(enabled: boolean, stage: string): void {
  if (enabled) {
    selftestElement().setAttribute('data-stage', stage);
  }
}

function reportSelftest(ok: boolean, frames: number, errors: readonly string[]): void {
  selftestElement().textContent = JSON.stringify({ ok, frames, errors });
  document.title = ok ? 'SELFTEST-OK' : 'SELFTEST-FAIL';
}

async function boot(): Promise<void> {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const selftest = new URLSearchParams(location.search).has('selftest');
  const gpuErrors: string[] = [];

  try {
    selftestStage(selftest, 'boot');
    const { device, format } = await acquireDevice();
    selftestStage(selftest, 'device-ok');

    device.lost.then((info) => {
      if (info.reason !== 'destroyed') {
        showFatalError(`Périphérique GPU perdu : ${info.message}`);
      }
    });
    device.addEventListener('uncapturederror', (event) => {
      const message = (event as GPUUncapturedErrorEvent).error.message;
      gpuErrors.push(message);
      console.error('[WebGPU]', message);
    });

    const context = canvas.getContext('webgpu');
    if (context === null) {
      throw new Error("Impossible d'obtenir un contexte WebGPU sur le canvas.");
    }
    context.configure({ device, format, alphaMode: 'opaque' });

    // Le canvas suit la fenêtre (et le devicePixelRatio) ; la grille de simulation,
    // elle, reste à taille fixe — le rendu upscale.
    const resize = (): void => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const limit = device.limits.maxTextureDimension2D;
      const w = Math.min(Math.max(Math.round(canvas.clientWidth * dpr), 1), limit);
      const h = Math.min(Math.max(Math.round(canvas.clientHeight * dpr), 1), limit);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };
    resize();
    window.addEventListener('resize', resize);

    const sim = await FluidSim.create(device, { targetFormat: format });
    selftestStage(selftest, 'sim-ok');
    const input = new InputController(canvas);
    const overlay = new Overlay(document.body);
    // Webcam → flux optique : la case du panneau demande la permission ; en cas de
    // refus, cameraFlow reste false et la case se resynchronise au refresh suivant.
    const camera = new CameraFlow();
    const toggleCamera = (on: boolean): void => {
      if (on) {
        camera
          .start()
          .then(() => (input.frame.params.cameraFlow = true))
          .catch((err: unknown) => {
            console.warn('[caméra]', err);
            input.frame.params.cameraFlow = false;
          });
      } else {
        camera.stop();
        input.frame.params.cameraFlow = false;
      }
    };
    // Pilotage par les mains : l'activation démarre la caméra si besoin, puis charge
    // MediaPipe à la demande (~8 Mo depuis le CDN au premier usage).
    const hands = new HandTracking(document.body);
    const toggleHands = (on: boolean): void => {
      if (!on) {
        hands.stop(input);
        return;
      }
      camera
        .start()
        .then(() => hands.start())
        .catch((err: unknown) => {
          console.warn('[mains]', err);
          hands.stop(input);
        });
    };
    // Export PNG : readback ponctuel de la présentation à taille fixe → téléchargement.
    const exportPNG = (): void => {
      void sim
        .exportImage()
        .then(async ({ pixels, width, height }) => {
          const out = new OffscreenCanvas(width, height);
          const ctx = out.getContext('2d');
          if (!ctx) {
            return;
          }
          ctx.putImageData(new ImageData(pixels, width, height), 0, 0);
          const blob = await out.convertToBlob({ type: 'image/png' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `liquidvm-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
          a.click();
          URL.revokeObjectURL(url);
        })
        .catch((err: unknown) => console.warn('[export]', err));
    };
    const panel = new DebugPanel(
      document.body,
      input,
      toggleCamera,
      { get: () => hands.active, set: toggleHands },
      exportPNG,
    );
    const toolbar = new MobileToolbar(document.body, input, () => panel.toggle());
    const driver = selftest ? new SelftestDriver(input.frame) : null;
    if (selftest) {
      // Poignées de debug pour l'outillage CDP : mutation des réglages en plein vol.
      (window as unknown as Record<string, unknown>)['__frame'] = input.frame;
      (window as unknown as Record<string, unknown>)['__hands'] = toggleHands;
      // `&camera` : active le flux optique dès le boot (fausse caméra en test headless).
      if (new URLSearchParams(location.search).has('camera')) {
        toggleCamera(true);
      }
    }

    let last = performance.now();
    let frameCount = 0;
    let fpsFrames = 0;
    let fpsLast = last;

    const tick = (now: number): void => {
      const dt = (now - last) / 1000;
      last = now;
      driver?.drive(frameCount);

      if (input.frame.params.cameraFlow) {
        camera.copyFrame(device, sim.cameraTexture);
      }
      hands.update(now, input, camera.videoElement);
      // L'encart caméra doit rester carré à l'écran quel que soit le format du canvas.
      input.frame.render.aspect = canvas.width / Math.max(canvas.height, 1);
      sim.frame(dt, input.frame, context.getCurrentTexture().createView());
      input.endFrame();
      frameCount++;
      fpsFrames++;

      // Overlay et toolbar : mise à jour régulière.
      if (now - fpsLast >= 500) {
        const fps = (fpsFrames * 1000) / (now - fpsLast);
        overlay.update(
          fps,
          `${SUBSTANCE_NAMES[input.frame.selectedFluid]} (${TOOL_LABELS[input.tool]})`,
          BOUNDARY_LABELS[input.frame.boundaryMode],
          VIEW_LABELS[input.frame.viewMode],
          input.frame.params.multigrid
            ? `MG ×${input.frame.params.vcycles}`
            : `jacobi ${input.frame.pressureIterations} it.`,
          input.frame.paused,
        );
        panel.refresh();
        toolbar.sync();
        fpsFrames = 0;
        fpsLast = now;
      }
      if (selftest && frameCount % 50 === 0) {
        selftestStage(true, `frames-${frameCount}`);
      }
      if (selftest && frameCount === SELFTEST_REPORT_FRAME) {
        reportSelftest(gpuErrors.length === 0, frameCount, gpuErrors);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    showFatalError(message);
    if (selftest) {
      reportSelftest(false, 0, [...gpuErrors, message]);
    }
  }
}

void boot();
