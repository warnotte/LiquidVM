/**
 * Pilotage de l'interface par les mains (MediaPipe Hands) : l'index est le pointeur,
 * le pincement pouce-index est le clic — les gestes écrivent dans le même FrameInput
 * abstrait que la souris, donc TOUS les outils existants (injecter, feu, murs,
 * tourbillon…) fonctionnent à la main sans que le core change d'une ligne.
 *
 * Première (et seule) dépendance runtime du projet, chargée À LA DEMANDE : le module
 * JS est un chunk séparé (import dynamique), le wasm et le modèle (~8 Mo) viennent du
 * CDN au premier usage — épinglés sur la version installée. Sans réseau, l'activation
 * échoue proprement et la case se décoche.
 */

import type { HandLandmarker } from '@mediapipe/tasks-vision';
import type { InputController } from './input';

/** Doit correspondre à la version de @mediapipe/tasks-vision dans package.json. */
const MEDIAPIPE_VERSION = '1.0.1';
const WASM_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/** Points de repère MediaPipe utilisés (parmi les 21 du squelette de main). */
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const PINKY_MCP = 17;

/**
 * Filtre One-Euro (Casiez 2012) — le standard des interfaces de pointage : lisse fort
 * quand la main est lente (tue le tremblement du squelette), lisse peu quand elle est
 * rapide (pas de latence perceptible). Bien meilleur qu'une moyenne exponentielle fixe.
 */
class OneEuroFilter {
  private prev = 0;
  private dPrev = 0;
  private initialized = false;

  constructor(
    private readonly minCutoff = 1.2,
    private readonly beta = 0.03,
    private readonly dCutoff = 1.0,
  ) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(x: number, dt: number): number {
    if (!this.initialized) {
      this.initialized = true;
      this.prev = x;
      return x;
    }
    const dx = (x - this.prev) / dt;
    this.dPrev += OneEuroFilter.alpha(this.dCutoff, dt) * (dx - this.dPrev);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dPrev);
    const y = this.prev + OneEuroFilter.alpha(cutoff, dt) * (x - this.prev);
    this.prev = y;
    return y;
  }

  reset(): void {
    this.initialized = false;
    this.dPrev = 0;
  }
}

export class HandTracking {
  private landmarker: HandLandmarker | null = null;
  private readonly cursor: HTMLDivElement;
  private readonly filterX = new OneEuroFilter();
  private readonly filterY = new OneEuroFilter();
  private smoothX = 0.5;
  private smoothY = 0.5;
  private hasPrev = false;
  private pinching = false;
  private lastVideoTime = -1;
  private lastSeen = 0;
  private lastDetect = 0;
  active = false;

  constructor(parent: HTMLElement) {
    this.cursor = document.createElement('div');
    this.cursor.className = 'hand-cursor hidden';
    parent.appendChild(this.cursor);
  }

  /** Charge MediaPipe (une seule fois) et active le pilotage. */
  async start(): Promise<void> {
    if (!this.landmarker) {
      const vision = await import('@mediapipe/tasks-vision');
      const fileset = await vision.FilesetResolver.forVisionTasks(WASM_CDN);
      this.landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 1,
        // Seuils de suivi abaissés : le tracker reste accroché pendant les mouvements
        // rapides au lieu de re-détecter (ce qui fait sursauter le curseur).
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.35,
      });
    }
    this.active = true;
  }

  stop(input: InputController): void {
    this.active = false;
    this.hasPrev = false;
    if (this.pinching) {
      this.pinching = false;
      input.setPress(false);
    }
    this.cursor.classList.add('hidden');
  }

  /** À appeler chaque frame : détecte la main et pilote le pointeur du FrameInput. */
  update(nowMs: number, input: InputController, video: HTMLVideoElement | null): void {
    if (!this.active || !this.landmarker || !video || video.readyState < 2) {
      return;
    }
    // La caméra tourne à ~30 fps : ne détecter que sur les nouvelles images.
    if (video.currentTime === this.lastVideoTime) {
      return;
    }
    this.lastVideoTime = video.currentTime;

    const result = this.landmarker.detectForVideo(video, nowMs);
    const lm = result.landmarks[0];
    if (!lm) {
      // Main perdue : relâcher le geste après un court délai de grâce.
      if (this.pinching && nowMs - this.lastSeen > 250) {
        this.pinching = false;
        input.setPress(false);
      }
      if (nowMs - this.lastSeen > 400) {
        this.cursor.classList.add('hidden');
        this.hasPrev = false;
      }
      return;
    }
    this.lastSeen = nowMs;

    // Index = pointeur (miroir horizontal : la webcam se vit comme un miroir),
    // filtre One-Euro contre le tremblement du squelette, sans latence sur les gestes vifs.
    const dt = Math.min(Math.max((nowMs - this.lastDetect) / 1000, 1 / 120), 0.25);
    this.lastDetect = nowMs;
    const tip = lm[INDEX_TIP]!;
    const x = Math.min(Math.max(1 - tip.x, 0), 1);
    const y = Math.min(Math.max(tip.y, 0), 1);
    if (!this.hasPrev) {
      this.filterX.reset();
      this.filterY.reset();
    }
    this.smoothX = this.filterX.filter(x, dt);
    this.smoothY = this.filterY.filter(y, dt);

    const p = input.frame.pointer;
    if (this.hasPrev) {
      p.dx += this.smoothX - p.x;
      p.dy += this.smoothY - p.y;
    }
    p.x = this.smoothX;
    p.y = this.smoothY;
    this.hasPrev = true;

    // Pincement pouce-index = clic, normalisé par la LARGEUR DE PAUME (base de
    // l'index → base de l'auriculaire) — plus stable que poignet→majeur quand la
    // main s'incline vers la caméra. Hystérésis : pincer < 0,55, relâcher > 0,75.
    const thumb = lm[THUMB_TIP]!;
    const indexMcp = lm[INDEX_MCP]!;
    const pinkyMcp = lm[PINKY_MCP]!;
    const palmWidth = Math.hypot(indexMcp.x - pinkyMcp.x, indexMcp.y - pinkyMcp.y) + 1e-6;
    const pinchRatio = Math.hypot(thumb.x - tip.x, thumb.y - tip.y) / palmWidth;
    const wantPress = this.pinching ? pinchRatio < 0.75 : pinchRatio < 0.55;
    if (wantPress !== this.pinching) {
      this.pinching = wantPress;
      if (wantPress) {
        // Pas d'impulsion parasite au début du geste.
        p.dx = 0;
        p.dy = 0;
      }
      input.setPress(wantPress);
    }

    // Curseur à l'écran : anneau qui se remplit quand on pince.
    this.cursor.classList.remove('hidden');
    this.cursor.classList.toggle('pinch', this.pinching);
    this.cursor.style.left = `${this.smoothX * 100}%`;
    this.cursor.style.top = `${this.smoothY * 100}%`;
  }
}
