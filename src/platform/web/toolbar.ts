/**
 * Barre d'outils tactile responsive (Quick Toolbar) pour les écrans tactiles et mobiles.
 * Offre un accès direct au choix du fluide, des outils (dont pinceau et gomme de murs),
 * des modes de frontières, des vues, de la pause et du panneau de réglages sans
 * nécessiter de clavier physique ni de clic droit.
 * DOM vanilla, zéro dépendance.
 */

import { FLUIDS } from '../../core/config';
import {
  BOUNDARY_MODE_COUNT,
  VIEW_MODE_COUNT,
  type BoundaryMode,
  type FluidId,
  type ViewMode,
} from '../../core/types';
import type { InputController, UITool } from './input';
import { UI_TOOL_LABELS } from './input';

const BOUNDARY_NAMES = ['Parois', 'Périodique', 'Ouvert'] as const;
const VIEW_NAMES = ['Fluides', 'Vélocité', 'Pression', 'Divergence', 'Vorticité'] as const;

export class MobileToolbar {
  private readonly root: HTMLDivElement;
  private readonly fluidButtons: HTMLButtonElement[] = [];
  private readonly toolButtons: HTMLButtonElement[] = [];
  private readonly pauseBtn: HTMLButtonElement;
  private readonly boundaryBtn: HTMLButtonElement;
  private readonly viewBtn: HTMLButtonElement;
  private readonly settingsBtn: HTMLButtonElement;
  private readonly toggleBtn: HTMLButtonElement;
  private isCollapsed = false;

  constructor(
    parent: HTMLElement,
    private readonly input: InputController,
    private readonly onToggleSettings: () => void,
  ) {
    this.root = document.createElement('div');
    this.root.className = 'mobile-toolbar';

    // Bouton pour réduire / déployer la barre d'outils
    this.toggleBtn = document.createElement('button');
    this.toggleBtn.className = 'toolbar-toggle-btn';
    this.toggleBtn.type = 'button';
    this.toggleBtn.title = 'Afficher / masquer la barre d’outils';
    this.toggleBtn.innerHTML = '&#9660;'; // Flèche bas
    this.toggleBtn.addEventListener('click', () => this.toggleCollapse());

    const content = document.createElement('div');
    content.className = 'toolbar-content';

    // 1. Sélecteur de fluide
    const fluidGroup = document.createElement('div');
    fluidGroup.className = 'toolbar-group fluid-group';
    FLUIDS.forEach((fluid, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `toolbar-btn fluid-btn fluid-btn-${i}`;
      btn.textContent = fluid.name;
      btn.title = `Sélectionner le fluide : ${fluid.name} (${i + 1})`;
      btn.addEventListener('click', () => {
        this.input.frame.selectedFluid = i as FluidId;
        this.sync();
        btn.blur();
      });
      this.fluidButtons.push(btn);
      fluidGroup.appendChild(btn);
    });
    content.appendChild(fluidGroup);

    // 2. Sélecteur d'outils (6 outils : 4 fluides + 2 murs)
    const toolGroup = document.createElement('div');
    toolGroup.className = 'toolbar-group tool-group';
    UI_TOOL_LABELS.forEach((label, i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'toolbar-btn tool-btn';
      btn.textContent = label;
      btn.title = `Outil : ${label}`;
      btn.addEventListener('click', () => {
        this.input.setTool(i as UITool);
        this.sync();
        btn.blur();
      });
      this.toolButtons.push(btn);
      toolGroup.appendChild(btn);
    });
    content.appendChild(toolGroup);

    // 3. Actions rapides (Pause, Reset, Frontières, Vue, Réglages)
    const actionGroup = document.createElement('div');
    actionGroup.className = 'toolbar-group action-group';

    // Pause / Play
    this.pauseBtn = document.createElement('button');
    this.pauseBtn.type = 'button';
    this.pauseBtn.className = 'toolbar-btn action-btn';
    this.pauseBtn.textContent = '⏸ Pause';
    this.pauseBtn.addEventListener('click', () => {
      this.input.frame.paused = !this.input.frame.paused;
      this.sync();
      this.pauseBtn.blur();
    });
    actionGroup.appendChild(this.pauseBtn);

    // Reset
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'toolbar-btn action-btn';
    resetBtn.textContent = '↺ Reset';
    resetBtn.title = 'Réinitialiser les fluides (R)';
    resetBtn.addEventListener('click', () => {
      this.input.frame.reset = true;
      resetBtn.blur();
    });
    actionGroup.appendChild(resetBtn);

    // Frontières
    this.boundaryBtn = document.createElement('button');
    this.boundaryBtn.type = 'button';
    this.boundaryBtn.className = 'toolbar-btn action-btn';
    this.boundaryBtn.textContent = 'Bords : Parois';
    this.boundaryBtn.title = 'Changer le mode de frontières (B)';
    this.boundaryBtn.addEventListener('click', () => {
      this.input.frame.boundaryMode = ((this.input.frame.boundaryMode + 1) %
        BOUNDARY_MODE_COUNT) as BoundaryMode;
      this.sync();
      this.boundaryBtn.blur();
    });
    actionGroup.appendChild(this.boundaryBtn);

    // Vue
    this.viewBtn = document.createElement('button');
    this.viewBtn.type = 'button';
    this.viewBtn.className = 'toolbar-btn action-btn';
    this.viewBtn.textContent = 'Vue : Fluides';
    this.viewBtn.title = 'Changer la vue de debug (V)';
    this.viewBtn.addEventListener('click', () => {
      this.input.frame.viewMode = ((this.input.frame.viewMode + 1) %
        VIEW_MODE_COUNT) as ViewMode;
      this.sync();
      this.viewBtn.blur();
    });
    actionGroup.appendChild(this.viewBtn);

    // Réglages
    this.settingsBtn = document.createElement('button');
    this.settingsBtn.type = 'button';
    this.settingsBtn.className = 'toolbar-btn action-btn settings-btn';
    this.settingsBtn.textContent = '⚙️ Réglages';
    this.settingsBtn.title = 'Afficher / masquer les réglages (Tab)';
    this.settingsBtn.addEventListener('click', () => {
      this.onToggleSettings();
      this.settingsBtn.blur();
    });
    actionGroup.appendChild(this.settingsBtn);

    content.appendChild(actionGroup);
    this.root.appendChild(this.toggleBtn);
    this.root.appendChild(content);

    parent.appendChild(this.root);
    this.sync();
  }

  private toggleCollapse(): void {
    this.isCollapsed = !this.isCollapsed;
    this.root.classList.toggle('collapsed', this.isCollapsed);
    this.toggleBtn.innerHTML = this.isCollapsed ? '&#9650;' : '&#9660;'; // Flèche haut ou bas
  }

  /** Met à jour les états visuels des boutons (actif/sélectionné, libellés). */
  sync(): void {
    const frame = this.input.frame;
    const currentFluid = frame.selectedFluid;
    this.fluidButtons.forEach((btn, i) => {
      btn.classList.toggle('active', i === currentFluid);
    });

    const currentTool = this.input.tool;
    this.toolButtons.forEach((btn, i) => {
      btn.classList.toggle('active', i === currentTool);
    });

    this.pauseBtn.textContent = frame.paused ? '▶ Play' : '⏸ Pause';
    this.pauseBtn.classList.toggle('active', frame.paused);

    this.boundaryBtn.textContent = `Bords : ${BOUNDARY_NAMES[frame.boundaryMode]}`;
    this.viewBtn.textContent = `Vue : ${VIEW_NAMES[frame.viewMode]}`;
  }
}
