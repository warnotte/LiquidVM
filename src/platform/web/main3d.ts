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
  FIELD_NAMES,
  SIM3_DEFAULTS,
  type Sim3Tuning,
} from '../../core3d/config3d';
import { FluidSim3D, type Frame3DInput } from '../../core3d/sim3d';
import { encodeVdb } from '../../core3d/vdb';
import { Panel3D, Toolbar3D } from './panel3d';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const SELFTEST_FRAMES = 240;

/** Jeux de réglages nommés — partagés par les presets du panneau ET la démo. */
const TUNE_CANDLE: Partial<Sim3Tuning> = {
  buoyancy: 80, velocityDissipation: 0.05, emitHeat: 1.6, emitInkRate: 0.7,
  heatCooling: 1.5, inkDissipation: 0.3, burnRate: 2, heatYield: 0.5, expansion: 4,
};
const TUNE_FURNACE: Partial<Sim3Tuning> = {
  buoyancy: 280, velocityDissipation: 0.02, emitHeat: 6.5, emitInkRate: 3.5,
  heatCooling: 0.55, inkDissipation: 0.15, burnRate: 6, heatYield: 1.05,
  expansion: 24, oxygenRecover: 0.05, blowForce: 320,
};
const TUNE_SMOKE: Partial<Sim3Tuning> = {
  buoyancy: 110, emitHeat: 1.2, emitInkRate: 5.5,
  heatCooling: 1.2, inkDissipation: 0.04,
};
// Le vent est COUPÉ partout ailleurs : ce preset est le seul qui l'allume, pour
// qu'on puisse le voir d'un clic sans qu'il change quoi que ce soit par défaut.
const TUNE_WIND: Partial<Sim3Tuning> = {
  buoyancy: 160, emitHeat: 2.6, emitInkRate: 3.0, heatCooling: 1.0,
  inkDissipation: 0.10, vorticityStrength: 16,
  // 35 CALIBRÉ par captures : à 55 le panache commence à se disperser, à 95 le
  // vent l'écrase au ras de l'émetteur. À 35 il monte haut, penche, et traîne.
  windStrength: 35, windSwing: 40, windPeriod: 8, windHeading: 20,
};

// PRESETS du panneau : appliqués À CHAUD par-dessus les défauts (déterministe
// quel que soit l'historique de clics). Exposition/pas de marche non touchés.
const PRESETS: readonly { label: string; values: Partial<Sim3Tuning> }[] = [
  { label: '↺ défaut', values: {} },
  { label: '🕯 bougie', values: TUNE_CANDLE },
  { label: '🔥 fournaise', values: TUNE_FURNACE },
  { label: '🌫 fumée épaisse', values: TUNE_SMOKE },
  { label: '🌬 vent', values: TUNE_WIND },
];

/**
 * Mode DÉMO (touche D ou ?demo) : chorégraphie scriptée du moteur — le showcase.
 * REMASTER : quatre actes qui traversent les presets (bougie → défaut →
 * fournaise avec braises et lueur poussée → fumée épaisse), caméra scénarisée
 * par cibles lissées, boule en lemniscate dans le brasier. Les réglages de
 * l'utilisateur sont SNAPSHOTÉS au lancement et RESTAURÉS à la sortie (D).
 */
class DemoDriver {
  private t = 0;
  private readonly fired = new Set<number>();
  /** Cible caméra de l'acte courant (vitesse d'orbite, élévation, distance). */
  private camT = { speed: 0.06, elevation: 0.1, radius: 1.55 };
  private saved: {
    params: Sim3Tuning;
    embersOn: boolean;
    glowStrength: number;
    exposure: number;
  } | null = null;

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

  /** Réglages d'acte : les écarts par-dessus les défauts (comme les presets). */
  private apply(values: Partial<Sim3Tuning>): void {
    Object.assign(this.input.params, defaultTuning3(), values);
  }

  private act(cam: { speed: number; elevation: number; radius: number }): void {
    this.camT = cam;
  }

  /** Lancement : snapshot des réglages du spectateur, scène vierge, acte I. */
  start(): void {
    this.saved = {
      params: { ...this.input.params },
      embersOn: this.input.embersOn,
      glowStrength: this.input.glowStrength,
      exposure: this.input.exposure,
    };
    this.t = 0;
    this.fired.clear();
    this.input.reset = true;
  }

  /** Sortie (D) : la main revient au spectateur avec SES réglages. */
  stop(): void {
    if (this.saved !== null) {
      Object.assign(this.input.params, this.saved.params);
      this.input.embersOn = this.saved.embersOn;
      this.input.glowStrength = this.saved.glowStrength;
      this.input.exposure = this.saved.exposure;
      this.saved = null;
    }
    this.input.blow.active = false;
  }

  tick(dt: number): void {
    this.t += dt;
    const input = this.input;

    // Caméra : orbite continue, élévation/distance LISSÉES vers la cible de
    // l'acte + une respiration lente par-dessus.
    const k = 1 - Math.exp(-1.1 * dt);
    input.cam.azimuth += dt * this.camT.speed;
    input.cam.elevation +=
      (this.camT.elevation + 0.05 * Math.sin(this.t * 0.13) - input.cam.elevation) * k;
    input.cam.radius +=
      (this.camT.radius + 0.13 * Math.sin(this.t * 0.09) - input.cam.radius) * k;

    // ACTE I — une bougie dans le noir (caméra proche, flamme intime).
    this.at(0.02, () => {
      this.apply(TUNE_CANDLE);
      input.embersOn = false;
      input.glowStrength = 1.8;
      this.act({ speed: 0.06, elevation: 0.1, radius: 1.55 });
      this.toast('Acte I — une bougie dans le noir');
    });
    // La flamme prend ses aises, la caméra recule.
    this.at(8, () => {
      this.apply({});
      this.act({ speed: 0.09, elevation: 0.22, radius: 1.95 });
      this.toast('la flamme prend ses aises');
    });
    this.at(14, () => {
      this.sim.addEmitterAt(0.27, 0.08, 0.5, 1);
      this.toast("fontaine d'encre magenta — lourde, elle retombe en refroidissant");
    });
    // ACTE II — la nappe de carburant rampe au sol (caméra au ras).
    this.at(21, () => {
      this.sim.addEmitterAt(0.62, 0.08, 0.35, 2);
      this.act({ speed: 0.08, elevation: 0.13, radius: 1.85 });
      this.toast('Acte II — la vapeur de carburant coule et nappe le sol');
    });
    this.at(28, () => this.toast('un souffle la pousse vers la flamme…'));
    if (this.t > 28 && this.t < 32) {
      input.blow.active = true;
      input.blow.ndcX = 0.2;
      input.blow.ndcY = -0.4;
      input.blow.moveX += -dt * 1.4;
    }
    this.at(32, () => (input.blow.active = false));
    // ACTE III — embrasement : fournaise, braises, lueur poussée, caméra ample.
    // Dissipation remontée : la brume des actes précédents se consume, le
    // brasier se détache au lieu de se noyer dans la boîte enfumée.
    this.at(34, () => {
      this.apply({ ...TUNE_FURNACE, inkDissipation: 0.4 });
      input.embersOn = true;
      input.glowStrength = 2.3;
      this.act({ speed: 0.12, elevation: 0.3, radius: 2.3 });
      this.toast('Acte III — FOURNAISE : braises au vent, la fumée boit la lumière');
    });
    this.at(40, () => this.toast('la boule plonge dans le brasier'));
    if (this.t > 40 && this.t < 56) {
      const u = (this.t - 40) * 0.33;
      this.sim.driveSphere(
        0.5 + 0.3 * Math.sin(u),
        0.4 + 0.17 * Math.sin(u * 0.71),
        0.5 + 0.16 * Math.sin(2.0 * u),
        dt,
      );
    }
    this.at(57, () => this.act({ speed: 0.1, elevation: 0.2, radius: 2.1 }));
    if (this.t > 58 && this.t < 61) {
      input.blow.active = true;
      input.blow.ndcX = -0.3;
      input.blow.ndcY = 0.0;
      input.blow.moveX += dt * 1.2;
      input.blow.moveY += dt * 0.3;
    }
    this.at(61, () => (input.blow.active = false));
    // ACTE IV — l'après : la fumée épaisse retombe (caméra haute, lente).
    this.at(64, () => {
      this.apply(TUNE_SMOKE);
      input.embersOn = false;
      input.glowStrength = 1.5;
      this.act({ speed: 0.05, elevation: 0.34, radius: 2.45 });
      this.toast('Acte IV — la fumée épaisse retombe');
    });
    this.at(72, () => (input.removeEmitter = true));
    this.at(74, () => (input.removeEmitter = true));
    this.at(80, () => this.toast('accalmie… (D : reprendre la main)'));
    // Boucle.
    if (this.t > 88) {
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
    addField: false,
    removeField: false,
    fieldType: 0,
    fieldStrength: SIM3_DEFAULTS.fieldVortexStrength,
    fieldRadius: SIM3_DEFAULTS.fieldRadius,
    explode: false,
    explosionRadius: SIM3_DEFAULTS.explosionRadius,
    explosionFuel: SIM3_DEFAULTS.explosionFuel,
    sphereActive: true,
    feedback: true,
    exposure: SIM3_DEFAULTS.exposure,
    raymarchSteps: SIM3_DEFAULTS.raymarchSteps,
    glowStrength: SIM3_DEFAULTS.glowStrength,
    bloomStrength: SIM3_DEFAULTS.bloomStrength,
    embersOn: SIM3_DEFAULTS.embersOn,
    emberStrength: SIM3_DEFAULTS.emberStrength,
  };
  if (selftest) {
    (window as unknown as Record<string, unknown>)['__frame3d'] = input;
    (window as unknown as Record<string, unknown>)['__sim3d'] = sim;
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
      // Une seule touche « supprimer », routée vers la famille de l'objet
      // désigné : c'est ce qu'on VOIT en surbrillance qui disparaît.
      if (sim.selectedIsField) {
        input.removeField = true;
        say('➖ champ de force');
      } else if (sim.emitterCount > 1) {
        input.removeEmitter = true;
        say('➖ émetteur');
      } else {
        say('la flamme pilote ne se retire pas');
      }
    } else if (e.code === 'Escape') {
      sim.deselect();
      say('rien de sélectionné');
    } else if (k === 'g') {
      input.explode = true;
      say('💥 détonation');
    } else if (k === 'o') {
      input.sphereActive = !input.sphereActive;
      say(input.sphereActive ? 'boule : présente' : 'boule : retirée');
    } else if (k === 'b') {
      input.embersOn = !input.embersOn;
      say(input.embersOn ? '✨ braises : actives' : 'braises : coupées');
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
  // Le mode démo scénarise TOUT (presets, braises, lueur, caméra) : snapshot
  // des réglages au lancement, restauration à la sortie.
  const toggleDemo = (): void => {
    demoOn = !demoOn;
    if (demoOn) {
      demoDriver.start();
    } else {
      demoDriver.stop();
    }
  };
  if (demoOn) {
    demoDriver.start();
  }
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
        { label: 'vent', min: 0, max: 250, step: 5, get: () => p.windStrength, set: (x) => (p.windStrength = x), format: (x) => (x === 0 ? 'aucun' : x.toFixed(0)) },
        { label: 'vent : cap', min: 0, max: 360, step: 5, get: () => p.windHeading, set: (x) => (p.windHeading = x), format: (x) => `${x.toFixed(0)}°` },
        { label: 'vent : oscillation', min: 0, max: 90, step: 5, get: () => p.windSwing, set: (x) => (p.windSwing = x), format: (x) => (x === 0 ? 'constant' : `±${x.toFixed(0)}°`) },
        { label: 'vent : période', min: 1, max: 30, step: 1, get: () => p.windPeriod, set: (x) => (p.windPeriod = x), format: (x) => `${x.toFixed(0)} s` },
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
        { label: 'lueur du feu', min: 0, max: 3, step: 0.05, get: () => input.glowStrength, set: (x) => (input.glowStrength = x) },
        { label: 'bloom', min: 0, max: 1, step: 0.05, get: () => input.bloomStrength, set: (x) => (input.bloomStrength = x) },
        { label: 'braises : débit', min: 0, max: 1, step: 0.05, get: () => input.emberStrength, set: (x) => (input.emberStrength = x) },
        { label: 'pas de marche', min: 64, max: 256, step: 16, get: () => input.raymarchSteps, set: (x) => (input.raymarchSteps = x), format: (x) => x.toFixed(0) },
      ],
      checks: [
        { label: 'boule-obstacle (O)', get: () => input.sphereActive, set: (v) => (input.sphereActive = v) },
        { label: 'braises (B)', get: () => input.embersOn, set: (v) => (input.embersOn = v) },
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
      // EXPLOSION : une bouffée de carburant + son amorce, sous le pointeur. La
      // boule de feu, la suie et le souffle sortent de la combustion existante —
      // il n'y a rien ici qu'un calibre.
      title: 'explosion',
      sliders: [
        { label: 'rayon', min: 0.03, max: 0.2, step: 0.005, get: () => input.explosionRadius, set: (x) => (input.explosionRadius = x), format: (x) => x.toFixed(3) },
        { label: 'charge', min: 4, max: 80, step: 1, get: () => input.explosionFuel, set: (x) => (input.explosionFuel = x), format: (x) => x.toFixed(0) },
        // La suie ne dépend pas de l'explosion : elle naît partout où le
        // carburant chaud manque d'air. Mais c'est là qu'on la règle, parce que
        // c'est une explosion qui en fabrique.
        { label: 'suie : rendement', min: 0, max: 40, step: 0.5, get: () => p.sootYield, set: (x) => (p.sootYield = x) },
        { label: 'suie : opacité', min: 0, max: 3, step: 0.05, get: () => p.sootDensity, set: (x) => (p.sootDensity = x) },
        { label: 'suie : évanouis.', min: 0, max: 0.6, step: 0.01, get: () => p.sootFade, set: (x) => (p.sootFade = x) },
        { label: 'suie : rayonnement', min: 0, max: 8, step: 0.1, get: () => p.sootCooling, set: (x) => (p.sootCooling = x) },
      ],
      buttons: [{ label: '💥 détoner (G)', action: () => (input.explode = true) }],
    },
    {
      // MODIFICATEURS : objets posés dans la scène, saisissables comme les
      // émetteurs et la boule. Les réglages ci-dessous s'appliquent au PROCHAIN
      // champ posé et à celui qu'on tient — jamais à distance.
      title: 'champs de force (modificateurs)',
      sliders: [
        {
          label: 'type',
          min: 0,
          max: FIELD_NAMES.length - 1,
          step: 1,
          get: () => input.fieldType,
          set: (x) => {
            input.fieldType = x;
            input.fieldStrength =
              x < 0.5 ? SIM3_DEFAULTS.fieldVortexStrength : SIM3_DEFAULTS.fieldWindStrength;
          },
          format: (x) => FIELD_NAMES[Math.round(x)] ?? '',
        },
        { label: 'force', min: 0, max: 600, step: 10, get: () => input.fieldStrength, set: (x) => (input.fieldStrength = x), format: (x) => x.toFixed(0) },
        { label: 'rayon', min: 0.05, max: 0.4, step: 0.01, get: () => input.fieldRadius, set: (x) => (input.fieldRadius = x), format: (x) => x.toFixed(2) },
      ],
      buttons: [
        { label: '＋ poser un champ', action: () => (input.addField = true) },
        { label: '－ retirer', action: () => (input.removeField = true) },
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
      {
        label: '➖',
        action: () => {
          if (sim.selectedIsField) {
            input.removeField = true;
          } else {
            input.removeEmitter = true;
          }
        },
      },
      { label: '⚪ boule', isActive: () => input.sphereActive, action: () => (input.sphereActive = !input.sphereActive) },
    ],
    [
      { label: '💥 boum', action: () => (input.explode = true) },
      { label: '🎬 démo', isActive: () => demoOn, action: toggleDemo },
      { label: '⚙ réglages', action: () => panel.toggle() },
      // Autres pages (URL relatives : valides en dev et sous /LiquidVM/ sur Pages).
      { label: '🌊 2D', action: () => (location.href = './') },
      { label: '💧 eau', action: () => (location.href = './eau.html') },
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
    // Une exception ici tuerait la boucle SANS RIEN DIRE : le canvas WebGPU
    // cesse d'être présenté et l'écran devient noir, symptôme parfaitement
    // muet (vécu deux fois). On la montre au lieu de la subir.
    try {
      sim.frame(input, context.getCurrentTexture().createView(), canvas.width, canvas.height);
    } catch (err: unknown) {
      showFatalError(
        `La simulation s'est arrêtée : ${err instanceof Error ? err.message : String(err)}`,
      );
      document.title = 'SELFTEST-FAIL';
      return;
    }
    input.reset = false;
    input.blow.moveX = 0;
    input.blow.moveY = 0;
    input.addEmitter = false;
    input.addField = false;
    input.removeField = false;
    input.explode = false;
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
        '1/2/3 : encre · glisser un objet : déplacer · ses poignées : un seul axe · son bouton violet : orienter · A : + émetteur · X : supprimer · G : 💥 · O : boule · B : braises · F : repères<br>' +
        'glisser : orbiter · clic droit : souffler · molette : zoom · Échap : désélectionner · espace : pause · R : reset · E : export .vdb';
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
