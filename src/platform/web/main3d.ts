/**
 * Point d'entrée du prototype 3D volumétrique (page 3d.html).
 * Couche plateforme minimale : device, canvas/resize, caméra orbitale à la souris
 * (glisser = orbiter, molette = zoom), espace = pause, R = reset, HUD FPS.
 * Mode ?selftest : rapport JSON dans #selftest + titre SELFTEST-OK/FAIL après 240 frames.
 */

import {
  defaultTuning3,
  estimateVram3,
  GRID3,
  GRID3_CHOICES,
  INK_COLORS,
  INK_NAMES,
  setGrid3,
  SIM3_DEFAULTS,
  type Sim3Tuning,
} from '../../core3d/config3d';
import { FluidSim3D, type Frame3DInput } from '../../core3d/sim3d';
import { encodeVdb } from '../../core3d/vdb';
import { Panel3D, Toolbar3D } from './panel3d';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const SELFTEST_FRAMES = 240;

/**
 * Mode DÉMO (touche D ou ?demo) : chorégraphie scriptée du moteur — le showcase.
 * Caméra en orbite lente, fontaine d'encre, nappe de carburant soufflée dans la
 * flamme (embrasement), boule qui brasse les panaches, accalmie, et boucle.
 */
class DemoDriver {
  private t = 0;
  private readonly fired = new Set<number>();

  constructor(
    private readonly input: Frame3DInput,
    private readonly sim: FluidSim3D,
    private readonly toast: (text: string) => void,
  ) {}

  private at(mark: number, action: () => void): void {
    if (this.t >= mark && !this.fired.has(mark)) {
      this.fired.add(mark);
      action();
    }
  }

  tick(dt: number): void {
    this.t += dt;
    const input = this.input;

    // Caméra : orbite lente, élévation et distance qui respirent.
    input.cam.azimuth += dt * 0.1;
    input.cam.elevation = 0.24 + 0.1 * Math.sin(this.t * 0.11);
    input.cam.radius = 2.05 + 0.28 * Math.sin(this.t * 0.07);

    // Acte I : la flamme pilote seule (0–6 s), puis les matières entrent.
    this.at(0.5, () => this.toast('Acte I — la flamme pilote'));
    this.at(6, () => {
      this.sim.addEmitterAt(0.27, 0.08, 0.5, 1);
      this.toast("fontaine d'encre magenta — elle retombe en refroidissant");
    });
    this.at(13, () => {
      this.sim.addEmitterAt(0.73, 0.08, 0.6, 0);
      this.toast('colonne de fumée');
    });
    // Acte II : la nappe de carburant se répand (20 s)…
    this.at(20, () => {
      this.sim.addEmitterAt(0.6, 0.08, 0.33, 2);
      this.toast('Acte II — vapeur de carburant : elle coule et nappe le sol');
    });
    this.at(30, () => this.toast('un souffle pousse la nappe dans la flamme…'));
    // …et un souffle la pousse dans la flamme pilote : embrasement (30–34 s).
    if (this.t > 30 && this.t < 34) {
      input.blow.active = true;
      input.blow.ndcX = 0.2;
      input.blow.ndcY = -0.4;
      input.blow.moveX += -dt * 1.4;
    }
    this.at(34, () => (input.blow.active = false));
    this.at(36, () => this.toast('embrasement — la déflagration court le long de la nappe'));
    // Acte III : la boule brasse les panaches en lemniscate (40–58 s).
    this.at(40, () => this.toast('Acte III — la boule brasse les panaches'));
    if (this.t > 40 && this.t < 58) {
      const u = (this.t - 40) * 0.33;
      this.sim.driveSphere(
        0.5 + 0.3 * Math.sin(u),
        0.4 + 0.17 * Math.sin(u * 0.71),
        0.5 + 0.16 * Math.sin(2.0 * u),
        dt,
      );
    }
    // Acte IV : grand souffle transversal (62–65 s), puis accalmie.
    this.at(62, () => this.toast('Acte IV — grand souffle'));
    if (this.t > 62 && this.t < 65) {
      input.blow.active = true;
      input.blow.ndcX = -0.3;
      input.blow.ndcY = 0.0;
      input.blow.moveX += dt * 1.2;
      input.blow.moveY += dt * 0.3;
    }
    this.at(65, () => (input.blow.active = false));
    this.at(69, () => this.toast('accalmie…'));
    this.at(70, () => (input.removeEmitter = true));
    this.at(72, () => (input.removeEmitter = true));
    // Boucle.
    if (this.t > 82) {
      input.reset = true;
      this.t = 0;
      this.fired.clear();
    }
  }
}

async function boot(): Promise<void> {
  const urlParams = new URLSearchParams(location.search);
  const selftest = urlParams.has('selftest');
  let demoOn = urlParams.has('demo');
  // Résolution à la demande : ?grid=128|256|320 (défaut 256 ; 320 = le plafond
  // mesuré de la machine de référence, toujours à 60 FPS).
  setGrid3(Number(urlParams.get('grid') ?? 256));
  const canvas = document.getElementById('canvas3d') as HTMLCanvasElement;
  const hud = document.getElementById('hud3d') as HTMLDivElement;

  // Retours visuels débrayables : toasts d'événements + fuseau du souffle (core).
  const toast = (text: string): void => {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = text;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1700);
  };

  const { device, format } = await acquireDevice();
  // Garde-fou hautes résolutions : Dawn valide ses staging internes (zéro-init
  // des textures 3D) contre maxBufferSize — s'il est trop bas pour la plus
  // grosse texture (rgba16float pleine grille), chaque submit échouerait EN
  // SILENCE (écran noir, HUD vivant). Erreur claire plutôt que mystère.
  const biggestTexture = GRID3 * GRID3 * GRID3 * 8;
  if (device.limits.maxBufferSize < biggestTexture) {
    throw new Error(
      `La résolution ${GRID3}³ dépasse les capacités de ce GPU ` +
        `(maxBufferSize ${Math.round(device.limits.maxBufferSize / 2 ** 20)} Mio < ` +
        `${Math.round(biggestTexture / 2 ** 20)} Mio requis). Choisir une grille plus petite : ` +
        `?grid=${GRID3_CHOICES.filter((n) => n * n * n * 8 <= device.limits.maxBufferSize).join('|')}`,
    );
  }
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

  // RÉSOLUTION DYNAMIQUE : le ray-marching est payé par pixel — l'échelle de rendu
  // s'ajuste au FPS (hystérésis), imperceptible en mouvement, gros gain en charge.
  let renderScale = 1;
  const resize = (): void => {
    const scale = Math.min(window.devicePixelRatio || 1, 2) * renderScale;
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
    params: defaultTuning3(),
    emitInk: 0,
    cam: {
      azimuth: SIM3_DEFAULTS.camAzimuth,
      elevation: SIM3_DEFAULTS.camElevation,
      radius: SIM3_DEFAULTS.camRadius,
    },
    pointer: { ndcX: 0, ndcY: 0 },
    blow: { active: false, ndcX: 0, ndcY: 0, moveX: 0, moveY: 0 },
    grab: { active: false },
    addEmitter: false,
    removeEmitter: false,
    sphereActive: true,
    feedback: true,
    exposure: SIM3_DEFAULTS.exposure,
    raymarchSteps: SIM3_DEFAULTS.raymarchSteps,
  };
  if (selftest) {
    (window as unknown as Record<string, unknown>)['__frame3d'] = input;
  }

  // Caméra orbitale : glisser (gauche) pour tourner — SAUF si le clic attrape un
  // objet (flamme ou sphère : hitTest), auquel cas l'objet suit le pointeur.
  // Souffle : glisser au clic DROIT — le geste pousse le fluide le long du rayon.
  let dragging = false;
  let blowing = false;
  let lastX = 0;
  let lastY = 0;
  const updatePointer = (e: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect();
    input.pointer.ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    input.pointer.ndcY = 1 - ((e.clientY - rect.top) / rect.height) * 2;
  };
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    updatePointer(e);
    if (e.button === 2) {
      blowing = true;
    } else if (sim.hitTest(input.pointer.ndcX, input.pointer.ndcY)) {
      input.grab.active = true;
    } else {
      dragging = true;
    }
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
    updatePointer(e);
    if (blowing) {
      const rect = canvas.getBoundingClientRect();
      const b = input.blow;
      b.active = true;
      b.ndcX = input.pointer.ndcX;
      b.ndcY = input.pointer.ndcY;
      b.moveX += ((e.clientX - lastX) / rect.width) * 2;
      b.moveY += -((e.clientY - lastY) / rect.height) * 2;
      lastX = e.clientX;
      lastY = e.clientY;
      return;
    }
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
    blowing = false;
    input.blow.active = false;
    input.grab.active = false;
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
  // Export OpenVDB : readback ponctuel + encodage TS pur + téléchargement.
  // Grilles « density » (fumée) et « temperature » (chaleur), boîte de taille 1
  // centrée sur l'origine, Y simulation mappé sur +Z Blender (la fumée monte).
  let exporting = false;
  const exportVdbFile = async (): Promise<void> => {
    if (exporting) {
      return;
    }
    exporting = true;
    try {
      // Readback plafonné par maxBufferSize (demandé au max de l'adapter dans
      // gpu.ts — 2 Gio sur la machine de référence : 512³ passe tout juste).
      const { density, heat } = await sim.exportVolume();
      const data = encodeVdb(
        [
          { name: 'density', values: density },
          { name: 'temperature', values: heat },
        ],
        GRID3,
        1 / GRID3,
      );
      const url = URL.createObjectURL(new Blob([data], { type: 'application/octet-stream' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `liquidvm-${Date.now()}.vdb`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      toast(`export impossible à ${GRID3}³ (limite mémoire GPU) — réduire la résolution`);
      console.warn('export VDB:', err);
    } finally {
      exporting = false;
    }
  };
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      input.paused = !input.paused;
    } else if (e.code === 'KeyR') {
      input.reset = true;
    } else if (e.code === 'KeyE') {
      void exportVdbFile();
    } else if (e.code === 'Digit1' || e.code === 'Numpad1') {
      input.emitInk = 0;
    } else if (e.code === 'Digit2' || e.code === 'Numpad2') {
      input.emitInk = 1;
    } else if (e.code === 'Digit3' || e.code === 'Numpad3') {
      input.emitInk = 2;
    }
    // Lettres via e.key : indépendant de la disposition clavier (AZERTY…).
    const k = e.key.toLowerCase();
    const say = (text: string): void => {
      if (input.feedback) {
        toast(text);
      }
    };
    if (k === 'a') {
      input.addEmitter = true;
      say(`➕ émetteur : ${INK_NAMES[input.emitInk] ?? '?'}`);
    } else if (k === 'x') {
      input.removeEmitter = true;
      say('➖ émetteur');
    } else if (k === 'o') {
      input.sphereActive = !input.sphereActive;
      say(input.sphereActive ? 'boule : présente' : 'boule : retirée');
    } else if (k === 'd') {
      toggleDemo();
      say(demoOn ? 'DÉMO — D pour reprendre la main' : 'à toi de jouer');
    } else if (k === 'f') {
      input.feedback = !input.feedback;
      toast(input.feedback ? 'retours visuels : actifs' : 'retours visuels : coupés');
    }
    if (e.code === 'Digit1' || e.code === 'Digit2' || e.code === 'Digit3') {
      say(`encre : ${INK_NAMES[input.emitInk] ?? '?'}`);
    }
  });

  const demoDriver = new DemoDriver(input, sim, (text) => {
    if (input.feedback) {
      toast(text);
    }
  });

  // INTERFACE déclarative : panneau à sections (Tab) + barre d'outils tactile.
  // Ajouter un réglage futur = une entrée ici, rien d'autre.
  const p = input.params;
  const inkCss = INK_COLORS.map((c) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`);
  const toggleDemo = (): void => {
    demoOn = !demoOn;
    if (!demoOn) {
      input.blow.active = false;
    }
  };
  // PRESETS : jeux de réglages physiques nommés, appliqués À CHAUD sur les params
  // vivants (pas de reset — la flamme change de caractère sous vos yeux). Chaque
  // preset part des DÉFAUTS puis applique ses écarts : déterministe quel que soit
  // l'historique de clics. Exposition/pas de marche non touchés.
  const PRESETS: readonly { label: string; values: Partial<Sim3Tuning> }[] = [
    { label: '↺ défaut', values: {} },
    {
      label: '🕯 bougie',
      values: {
        buoyancy: 80, velocityDissipation: 0.05, emitHeat: 1.6, emitInkRate: 0.7,
        heatCooling: 1.5, inkDissipation: 0.3, burnRate: 2, heatYield: 0.5, expansion: 4,
      },
    },
    {
      label: '🔥 fournaise',
      values: {
        buoyancy: 280, velocityDissipation: 0.02, emitHeat: 6.5, emitInkRate: 3.5,
        heatCooling: 0.55, inkDissipation: 0.15, burnRate: 6, heatYield: 1.05,
        expansion: 24, oxygenRecover: 0.05, blowForce: 320,
      },
    },
    {
      label: '🌫 fumée épaisse',
      values: {
        buoyancy: 110, emitHeat: 1.2, emitInkRate: 5.5,
        heatCooling: 1.2, inkDissipation: 0.04,
      },
    },
  ];
  const panel = new Panel3D(document.body, [
    {
      title: 'presets',
      buttons: PRESETS.map((preset) => ({
        label: preset.label,
        action: (): void => {
          Object.assign(p, defaultTuning3(), preset.values);
          panel.refresh();
          if (input.feedback) {
            toast(`preset : ${preset.label}`);
          }
        },
      })),
    },
    {
      title: 'simulation',
      sliders: [
        { label: 'vitesse du temps', min: 0, max: 1.5, step: 0.05, get: () => p.timeScale, set: (x) => (p.timeScale = x), format: (x) => `×${x.toFixed(2)}` },
        { label: 'poussée thermique', min: 0, max: 400, step: 5, get: () => p.buoyancy, set: (x) => (p.buoyancy = x), format: (x) => x.toFixed(0) },
        { label: 'vorticité', min: 0, max: 30, step: 0.5, get: () => p.vorticityStrength, set: (x) => (p.vorticityStrength = x), format: (x) => x.toFixed(1) },
        { label: 'viscosité', min: 0, max: 0.2, step: 0.005, get: () => p.velocityDissipation, set: (x) => (p.velocityDissipation = x), format: (x) => x.toFixed(3) },
      ],
      checks: [
        { label: 'pression multigrid (sinon Jacobi)', get: () => input.multigrid, set: (v) => (input.multigrid = v) },
      ],
    },
    {
      title: 'matières',
      sliders: [
        { label: 'débit chaleur', min: 0, max: 8, step: 0.1, get: () => p.emitHeat, set: (x) => (p.emitHeat = x) },
        { label: 'débit matière', min: 0, max: 6, step: 0.1, get: () => p.emitInkRate, set: (x) => (p.emitInkRate = x) },
        { label: 'refroidissement', min: 0.2, max: 2, step: 0.05, get: () => p.heatCooling, set: (x) => (p.heatCooling = x) },
        { label: 'dissipation', min: 0, max: 0.6, step: 0.01, get: () => p.inkDissipation, set: (x) => (p.inkDissipation = x) },
      ],
    },
    {
      title: 'combustion',
      sliders: [
        { label: 'taux de réaction', min: 0, max: 8, step: 0.1, get: () => p.burnRate, set: (x) => (p.burnRate = x) },
        { label: 'chaleur dégagée', min: 0, max: 1.5, step: 0.05, get: () => p.heatYield, set: (x) => (p.heatYield = x) },
        { label: 'expansion', min: 0, max: 40, step: 1, get: () => p.expansion, set: (x) => (p.expansion = x), format: (x) => x.toFixed(0) },
        { label: 'retour d’oxygène', min: 0, max: 0.08, step: 0.002, get: () => p.oxygenRecover, set: (x) => (p.oxygenRecover = x), format: (x) => x.toFixed(3) },
      ],
    },
    {
      title: 'interaction & rendu',
      sliders: [
        { label: 'force du souffle', min: 0, max: 800, step: 10, get: () => p.blowForce, set: (x) => (p.blowForce = x), format: (x) => x.toFixed(0) },
        { label: 'exposition', min: 0.4, max: 3, step: 0.05, get: () => input.exposure, set: (x) => (input.exposure = x) },
        { label: 'pas de marche', min: 64, max: 256, step: 16, get: () => input.raymarchSteps, set: (x) => (input.raymarchSteps = x), format: (x) => x.toFixed(0) },
      ],
      checks: [
        { label: 'boule-obstacle (O)', get: () => input.sphereActive, set: (v) => (input.sphereActive = v) },
        { label: 'retours visuels (F)', get: () => input.feedback, set: (v) => (input.feedback = v) },
      ],
      buttons: [
        { label: '⏯ pause (espace)', action: () => (input.paused = !input.paused) },
        { label: '↺ reset (R)', action: () => (input.reset = true) },
        { label: '🎬 démo (D)', action: toggleDemo },
        { label: '⬇ .vdb (E)', action: () => void exportVdbFile() },
      ],
    },
    {
      title: 'résolution (recharge la page)',
      buttons: GRID3_CHOICES.map((n) => ({
        // Estimation VRAM affichée : au-delà du budget du GPU, l'échec est un
        // écran noir silencieux — le chiffre permet de choisir en connaissance.
        label: `${n}³ · ${(estimateVram3(n) / 2 ** 30).toFixed(1)} Go${n === GRID3 ? ' ✓' : ''}`,
        action: (): void => {
          const url = new URL(location.href);
          url.searchParams.set('grid', String(n));
          location.href = url.toString();
        },
      })),
    },
  ]);
  const toolbar = new Toolbar3D(document.body, [
    INK_NAMES.map((name, i) => ({
      label: name,
      color: inkCss[i]!,
      isActive: () => input.emitInk === i,
      action: (): void => {
        input.emitInk = i;
      },
    })),
    [
      { label: '➕ émetteur', action: () => (input.addEmitter = true) },
      { label: '➖', action: () => (input.removeEmitter = true) },
      { label: '⚪ boule', isActive: () => input.sphereActive, action: () => (input.sphereActive = !input.sphereActive) },
    ],
    [
      { label: '🎬 démo', isActive: () => demoOn, action: toggleDemo },
      { label: '⚙ réglages', action: () => panel.toggle() },
      // Retour à la page 2D (URL relative : fonctionne en dev et sous /LiquidVM/ sur Pages).
      { label: '🌊 2D', action: () => (location.href = './') },
    ],
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
    if (demoOn) {
      demoDriver.tick(dt);
    }
    input.dt = dt;
    sim.frame(input, context.getCurrentTexture().createView(), canvas.width / canvas.height);
    input.reset = false;
    input.blow.moveX = 0;
    input.blow.moveY = 0;
    input.addEmitter = false;
    input.removeEmitter = false;
    frames++;

    hudTimer += dt;
    if (hudTimer > 0.5) {
      hudTimer = 0;
      panel.refresh();
      toolbar.refresh();
      if (fps < 50 && renderScale > 0.6) {
        renderScale = Math.max(0.6, renderScale - 0.1);
        resize();
      } else if (fps > 58.5 && renderScale < 1) {
        renderScale = Math.min(1, renderScale + 0.05);
        resize();
      }
      const solver = input.multigrid ? `MG ×${input.vcycles}` : `jacobi ${input.jacobiIterations} it.`;
      hud.innerHTML =
        `<b>LiquidVM 3D</b> · ${GRID3}³ · encre : ${INK_NAMES[input.emitInk] ?? '?'} · ${solver} · ${Math.round(fps)} FPS` +
        `${renderScale < 1 ? ` · rendu ${Math.round(renderScale * 100)} %` : ''}` +
        `${input.paused ? ' · ⏸ pause' : ''}${demoOn ? ' · <b>DÉMO</b> (D : reprendre la main)' : ' · D : démo'}<br>` +
        '1/2/3 : encre · glisser sur la flamme/boule : déplacer · A : + émetteur · X : − émetteur · O : boule · F : retours<br>' +
        'glisser : orbiter · clic droit : souffler · molette : zoom · espace : pause · R : reset · E : export .vdb';
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
