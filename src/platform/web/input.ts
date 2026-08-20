/**
 * Traduction des events navigateur (pointeur, clavier) vers le FrameInput abstrait
 * du core. L'objet `frame` est muté en place et réutilisé — aucune allocation par frame.
 * Équivalent natif : boucle d'événements GLFW/SDL remplissant le même struct.
 */

import { SIM_DEFAULTS } from '../../core/config';
import {
  BOUNDARY_MODE_COUNT,
  VIEW_MODE_COUNT,
  type BoundaryMode,
  type SubstanceId,
  type FrameInput,
  type ToolId,
  type ViewMode,
} from '../../core/types';

const SUBSTANCE_KEYS: Readonly<Record<string, SubstanceId>> = {
  Digit1: 0,
  Numpad1: 0,
  Digit2: 1,
  Numpad2: 1,
  Digit3: 2,
  Numpad3: 2,
  Digit4: 3,
  Numpad4: 3,
};

export type UITool = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
export const UI_TOOL_COUNT = 9;
export const UI_TOOL_LABELS = [
  'injecter',
  'gommer',
  'tourbillon',
  'souffle',
  'mur',
  'gomme mur',
  'goutte',
  'stylet',
  'peigne',
] as const;
/** Index des outils UI qui ne sont pas des outils fluides du core. */
const UI_TOOL_WALL = 4;
const UI_TOOL_WALL_ERASE = 5;
/** Outils de marbrure (déformations mathématiques, voir marble_warp.wgsl). */
const UI_TOOL_MARBLE_DROP = 6;
const UI_TOOL_MARBLE_TINE = 7;

export class InputController {
  private activePointerId: number | null = null;
  private currentUITool: UITool = 0;
  private marbleDragging = false;

  readonly frame: FrameInput = {
    pointer: { x: 0.5, y: 0.5, dx: 0, dy: 0, down: false, wall: false, erase: false },
    selectedFluid: 1,
    boundaryMode: 0,
    tool: 0,
    reset: false,
    clearWalls: false,
    paused: false,
    stepOnce: false,
    pressureIterations: SIM_DEFAULTS.pressureIterations,
    viewMode: 0,
    marble: { pending: false, tool: 0, ax: 0, ay: 0, bx: 0, by: 0 },
    params: {
      velocityDissipation: SIM_DEFAULTS.velocityDissipation,
      vorticityStrength: SIM_DEFAULTS.vorticityStrength,
      splatForce: SIM_DEFAULTS.splatForce,
      splatRadius: SIM_DEFAULTS.splatRadius,
      splatDensity: SIM_DEFAULTS.splatDensity,
      timeScale: SIM_DEFAULTS.timeScale,
      buoyancyScale: 1,
      macCormack: true,
      multigrid: true,
      vcycles: SIM_DEFAULTS.vcycles,
      particles: true,
      particleIntensity: 1,
      cameraFlow: false,
      flowStrength: 1,
      flowGate: 0.015,
    },
    render: {
      exposure: SIM_DEFAULTS.exposure,
      bloomStrength: SIM_DEFAULTS.bloomStrength,
      paper: false,
      aspect: 1,
    },
  };

  constructor(private readonly canvas: HTMLCanvasElement) {
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('blur', () => {
      this.activePointerId = null;
      this.frame.pointer.down = false;
      this.frame.pointer.wall = false;
      this.frame.pointer.erase = false;
    });
  }

  get tool(): UITool {
    return this.currentUITool;
  }

  setTool(tool: UITool): void {
    this.currentUITool = tool;
    if (tool < UI_TOOL_WALL) {
      this.frame.tool = tool as ToolId;
    }
  }

  /**
   * « Clic » virtuel (pilotage par les mains) : applique la même sémantique d'outil
   * que le bouton principal de la souris — les outils fluides pressent `down`, les
   * outils mur/gomme pressent `wall` — pour que les gestes pilotent toute l'interface.
   */
  setPress(pressed: boolean): void {
    const p = this.frame.pointer;
    if (!pressed) {
      p.down = false;
      p.wall = false;
      p.erase = false;
      return;
    }
    if (this.currentUITool === UI_TOOL_WALL) {
      p.wall = true;
      p.erase = false;
    } else if (this.currentUITool === UI_TOOL_WALL_ERASE) {
      p.wall = true;
      p.erase = true;
    } else {
      p.down = true;
      this.frame.tool = this.currentUITool as ToolId;
    }
  }

  /** À appeler après chaque frame : consomme le delta accumulé et les actions ponctuelles. */
  endFrame(): void {
    this.frame.pointer.dx = 0;
    this.frame.pointer.dy = 0;
    this.frame.reset = false;
    this.frame.clearWalls = false;
    this.frame.stepOnce = false;
    this.frame.marble.pending = false;
  }

  private updatePosition(e: PointerEvent, accumulateDelta: boolean): void {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    const x = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
    const y = Math.min(Math.max((e.clientY - rect.top) / rect.height, 0), 1);
    const p = this.frame.pointer;
    if (accumulateDelta) {
      p.dx += x - p.x;
      p.dy += y - p.y;
    }
    p.x = x;
    p.y = y;
  }

  private readonly onPointerDown = (e: PointerEvent): void => {
    // Si un pointeur est déjà actif, ignorer les touchers secondaires pour un tracé propre
    if (this.activePointerId !== null && this.activePointerId !== e.pointerId) {
      return;
    }
    this.activePointerId = e.pointerId;
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch {
      // Ignorer si la capture échoue sur certains environnements tactiles
    }

    // Pas de delta au posé du doigt : évite une impulsion parasite en début de drag.
    this.updatePosition(e, false);

    if (e.button === 2) {
      // Clic droit souris : toujours pinceau à murs (Maj = gommer).
      this.frame.pointer.down = false;
      this.frame.pointer.wall = true;
      this.frame.pointer.erase = e.shiftKey;
    } else {
      // Clic gauche ou toucher tactile principal
      if (this.currentUITool === UI_TOOL_WALL) {
        this.frame.pointer.down = false;
        this.frame.pointer.wall = true;
        this.frame.pointer.erase = false;
      } else if (this.currentUITool === UI_TOOL_WALL_ERASE) {
        this.frame.pointer.down = false;
        this.frame.pointer.wall = true;
        this.frame.pointer.erase = true;
      } else if (this.currentUITool >= UI_TOOL_MARBLE_DROP) {
        // Outils de marbrure : pas d'injection fluide — la goutte part immédiatement,
        // stylet et peigne génèrent leurs segments au fil du geste (onPointerMove).
        this.frame.pointer.down = false;
        this.frame.pointer.wall = false;
        this.marbleDragging = true;
        if (this.currentUITool === UI_TOOL_MARBLE_DROP) {
          const m = this.frame.marble;
          m.pending = true;
          m.tool = 0;
          m.ax = m.bx = this.frame.pointer.x;
          m.ay = m.by = this.frame.pointer.y;
        }
      } else {
        // Outils fluides (injecter, gommer densité, tourbillon, souffle, feu)
        this.frame.pointer.down = true;
        this.frame.pointer.wall = false;
        this.frame.pointer.erase = false;
        this.frame.tool = this.currentUITool as ToolId;
      }
    }
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (this.activePointerId !== e.pointerId) {
      return;
    }
    const prevX = this.frame.pointer.x;
    const prevY = this.frame.pointer.y;
    this.updatePosition(e, true);
    // Stylet et peigne : un segment de geste par frame (coalescé — le point de départ
    // reste celui du dernier segment appliqué, l'arrivée suit le pointeur).
    if (this.marbleDragging && this.currentUITool >= UI_TOOL_MARBLE_TINE) {
      const m = this.frame.marble;
      if (!m.pending) {
        m.ax = prevX;
        m.ay = prevY;
        m.tool = this.currentUITool === UI_TOOL_MARBLE_TINE ? 1 : 2;
        m.pending = true;
      }
      m.bx = this.frame.pointer.x;
      m.by = this.frame.pointer.y;
    }
    if (
      e.button === 2 ||
      this.currentUITool === UI_TOOL_WALL ||
      this.currentUITool === UI_TOOL_WALL_ERASE
    ) {
      if (this.frame.pointer.wall) {
        this.frame.pointer.erase = this.currentUITool === UI_TOOL_WALL_ERASE || e.shiftKey;
      }
    }
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    if (this.activePointerId !== e.pointerId && e.type !== 'pointercancel') {
      return;
    }
    this.activePointerId = null;
    this.marbleDragging = false;
    this.frame.pointer.down = false;
    this.frame.pointer.wall = false;
    this.frame.pointer.erase = false;
  };

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    // Ne pas voler les touches aux contrôles du panneau de réglages.
    if (e.target instanceof HTMLInputElement) {
      return;
    }
    const substance = SUBSTANCE_KEYS[e.code];
    if (substance !== undefined) {
      this.frame.selectedFluid = substance;
      return;
    }
    switch (e.code) {
      case 'Space':
        e.preventDefault();
        this.frame.paused = !this.frame.paused;
        return;
      case 'KeyR':
        this.frame.reset = true;
        return;
      case 'KeyB':
        this.frame.boundaryMode = ((this.frame.boundaryMode + 1) %
          BOUNDARY_MODE_COUNT) as BoundaryMode;
        return;
      case 'KeyX':
        this.frame.clearWalls = true;
        return;
      case 'KeyV':
        this.frame.viewMode = ((this.frame.viewMode + 1) % VIEW_MODE_COUNT) as ViewMode;
        return;
      case 'KeyN':
        this.frame.stepOnce = true;
        return;
      case 'KeyT':
        this.setTool(((this.currentUITool + 1) % UI_TOOL_COUNT) as UITool);
        return;
      case 'NumpadAdd':
        this.adjustIterations(+SIM_DEFAULTS.pressureIterationsStep);
        return;
      case 'NumpadSubtract':
        this.adjustIterations(-SIM_DEFAULTS.pressureIterationsStep);
        return;
      default:
        break;
    }
    // Les touches +/- varient selon la disposition clavier : on regarde aussi e.key.
    if (e.key === '+' || e.key === '=') {
      this.adjustIterations(+SIM_DEFAULTS.pressureIterationsStep);
    } else if (e.key === '-' || e.key === '_') {
      this.adjustIterations(-SIM_DEFAULTS.pressureIterationsStep);
    }
  };

  private adjustIterations(delta: number): void {
    this.frame.pressureIterations = Math.min(
      Math.max(this.frame.pressureIterations + delta, SIM_DEFAULTS.pressureIterationsMin),
      SIM_DEFAULTS.pressureIterationsMax,
    );
  }
}
