/**
 * Aperçu fantôme de l'outil actif sous le curseur : chaque outil montre sa taille et
 * sa forme réelles AVANT d'agir — cercle du splat à l'échelle (ellipse si le canvas
 * n'est pas carré, fidèle à l'étirement de la grille), teinte de la matière pour
 * l'injection, zone élargie du tourbillon/souffle, pinceau à murs.
 * DOM pur, mis à jour sur les mouvements du pointeur — rien dans la boucle GPU.
 */

import { FLUIDS, GRID_SIZE } from '../../core/config';
import type { InputController } from './input';

const SUBSTANCE_CSS = [
  ...FLUIDS.map((f) => `rgb(${f.color.map((c) => Math.round(c * 255)).join(',')})`),
  'rgb(255, 140, 60)', // feu
] as const;

export class ToolPreview {
  private readonly el: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'tool-preview hidden';
    parent.appendChild(this.el);
  }

  move(clientX: number, clientY: number): void {
    this.el.style.left = `${clientX}px`;
    this.el.style.top = `${clientY}px`;
    this.el.classList.remove('hidden');
  }

  hide(): void {
    this.el.classList.add('hidden');
  }

  /** Recalcule taille/forme depuis l'outil actif et le rayon du splat. */
  refresh(input: InputController, canvas: HTMLCanvasElement): void {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    const frac = input.frame.params.splatRadius / GRID_SIZE;
    const s = this.el.style;
    const substance = SUBSTANCE_CSS[input.frame.selectedFluid] ?? SUBSTANCE_CSS[0];
    const ellipse = (scale: number): void => {
      s.width = `${2 * frac * scale * rect.width}px`;
      s.height = `${2 * frac * scale * rect.height}px`;
    };

    switch (input.tool) {
      case 0: // injecter : cercle du splat, teinte de la matière
        ellipse(1);
        s.border = `2px solid ${substance}`;
        break;
      case 1: // gommer
        ellipse(1);
        s.border = '2px dashed rgba(200, 205, 215, 0.8)';
        break;
      case 2: // tourbillon : zone élargie ×2,5 (cf. forces.wgsl)
      case 3: // souffle
        ellipse(2.5);
        s.border = '2px dotted rgba(140, 175, 255, 0.85)';
        break;
      case 4: // mur : pinceau ×0,8 (cf. paint_walls.wgsl)
        ellipse(0.8);
        s.border = '3px solid rgba(170, 175, 190, 0.9)';
        break;
      case 5: // gomme mur
        ellipse(0.8);
        s.border = '3px dashed rgba(170, 175, 190, 0.9)';
        break;
      default:
        break;
    }
  }
}
