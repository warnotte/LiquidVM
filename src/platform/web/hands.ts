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
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;

export class HandTracking {
  private landmarker: HandLandmarker | null = null;
  private readonly cursor: HTMLDivElement;
  private smoothX = 0.5;
  private smoothY = 0.5;
  private hasPrev = false;
  private pinching = false;
  private lastVideoTime = -1;
  private lastSeen = 0;
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
    // lissage exponentiel contre le tremblement du squelette.
    const tip = lm[INDEX_TIP]!;
    const x = Math.min(Math.max(1 - tip.x, 0), 1);
    const y = Math.min(Math.max(tip.y, 0), 1);
    const s = this.hasPrev ? 0.45 : 1;
    this.smoothX += (x - this.smoothX) * s;
    this.smoothY += (y - this.smoothY) * s;

    const p = input.frame.pointer;
    if (this.hasPrev) {
      p.dx += this.smoothX - p.x;
      p.dy += this.smoothY - p.y;
    }
    p.x = this.smoothX;
    p.y = this.smoothY;
    this.hasPrev = true;

    // Pincement pouce-index = clic, normalisé par la taille de la main (distance
    // poignet → base du majeur) pour être indépendant de la distance à la caméra.
    // Hystérésis : on pince à < 0,40, on relâche à > 0,55 — pas de clignotement.
    const thumb = lm[THUMB_TIP]!;
    const wrist = lm[WRIST]!;
    const mcp = lm[MIDDLE_MCP]!;
    const handSize = Math.hypot(wrist.x - mcp.x, wrist.y - mcp.y) + 1e-6;
    const pinchRatio = Math.hypot(thumb.x - tip.x, thumb.y - tip.y) / handSize;
    const wantPress = this.pinching ? pinchRatio < 0.55 : pinchRatio < 0.4;
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
