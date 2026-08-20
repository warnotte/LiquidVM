/**
 * Overlay d'information (FPS, grille, fluide actif, itérations) et écran d'erreur fatale.
 * Le texte n'est mis à jour que 2×/seconde par la boucle — jamais à chaque frame.
 */

import { GRID_SIZE } from '../../core/config';

export class Overlay {
  private readonly status: HTMLDivElement;

  constructor(parent: HTMLElement) {
    const root = document.createElement('div');
    root.className = 'overlay';
    this.status = document.createElement('div');
    this.status.className = 'overlay-status';

    const hint1 = document.createElement('div');
    hint1.className = 'hint hint-desktop';
    hint1.textContent = 'glisser : injecter · clic droit : construire un mur · maj+clic droit : gommer';

    const hint2 = document.createElement('div');
    hint2.className = 'hint hint-desktop';
    hint2.textContent =
      '1/2/3/4 : matière (eau/encre/fumée/feu) · T : outil · B : frontières · V : vue debug · espace : pause · R : reset · X : murs off';

    const hintTouch = document.createElement('div');
    hintTouch.className = 'hint hint-touch';
    hintTouch.textContent = 'toucher/glisser : interagir · barre inférieure : outils, fluides et réglages';

    root.append(this.status, hint1, hint2, hintTouch);
    parent.appendChild(root);
  }

  update(
    fps: number,
    fluidName: string,
    boundaryLabel: string,
    viewLabel: string,
    solverLabel: string,
    paused: boolean,
  ): void {
    this.status.textContent =
      `${GRID_SIZE}×${GRID_SIZE} · matière : ${fluidName} · frontières : ${boundaryLabel} · ` +
      `vue : ${viewLabel} · pression : ${solverLabel} · ` +
      `${Math.round(fps)} FPS${paused ? ' · ⏸ pause' : ''}`;
  }
}

/** Message plein écran quand WebGPU est absent, refusé, ou que le device est perdu. */
export function showFatalError(message: string): void {
  const div = document.createElement('div');
  div.className = 'fatal';
  const inner = document.createElement('div');
  inner.textContent = message;
  div.appendChild(inner);
  document.body.appendChild(div);
}
