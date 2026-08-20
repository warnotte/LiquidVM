/**
 * Panneau de réglages temps réel (Tab pour afficher/masquer) : sliders et bascules
 * branchés directement sur le FrameInput muté en place — le core lit ces valeurs à
 * chaque frame via l'uniform buffer, aucun autre canal n'est nécessaire.
 * DOM vanilla, aucune dépendance.
 */

import { GRID_SIZE, SIM_DEFAULTS } from '../../core/config';
import type { InputController, UITool } from './input';
import { UI_TOOL_LABELS } from './input';

export const TOOL_LABELS = UI_TOOL_LABELS;

interface SliderSpec {
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly get: () => number;
  readonly set: (value: number) => void;
  readonly format?: (value: number) => string;
}

export class DebugPanel {
  private readonly root: HTMLDivElement;
  private readonly refreshers: (() => void)[] = [];

  constructor(
    parent: HTMLElement,
    input: InputController,
    onCameraToggle?: (on: boolean) => void,
    hands?: { get: () => boolean; set: (on: boolean) => void },
    onExport?: () => void,
    marbleMode?: { get: () => boolean; set: (on: boolean) => void },
  ) {
    const frame = input.frame;
    this.root = document.createElement('div');
    this.root.className = 'panel';

    const header = document.createElement('div');
    header.className = 'panel-header';

    const title = document.createElement('div');
    title.className = 'panel-title';
    title.textContent = 'réglages (Tab)';

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'panel-close-btn';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Fermer le panneau';
    closeBtn.addEventListener('click', () => {
      this.hide();
      closeBtn.blur();
    });

    header.append(title, closeBtn);
    this.root.appendChild(header);

    const p = frame.params;
    const r = frame.render;

    // Outils (4 fluides + 2 murs)
    const tools = document.createElement('div');
    tools.className = 'panel-buttons panel-tools';
    const toolButtons: HTMLButtonElement[] = [];
    const syncTools = (): void => {
      toolButtons.forEach((b, i) => b.classList.toggle('active', input.tool === i));
    };
    UI_TOOL_LABELS.forEach((label, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.addEventListener('click', () => {
        input.setTool(i as UITool);
        syncTools();
        button.blur();
      });
      toolButtons.push(button);
      tools.appendChild(button);
    });
    this.refreshers.push(syncTools);
    syncTools();
    this.root.appendChild(tools);
    const sliders: readonly SliderSpec[] = [
      {
        label: 'viscosité',
        min: 0,
        max: 0.5,
        step: 0.005,
        get: () => p.velocityDissipation,
        set: (x) => (p.velocityDissipation = x),
        format: (x) => x.toFixed(3),
      },
      {
        label: 'vorticité',
        min: 0,
        max: 30,
        step: 0.5,
        get: () => p.vorticityStrength,
        set: (x) => (p.vorticityStrength = x),
        format: (x) => x.toFixed(1),
      },
      {
        label: 'V-cycles (MG)',
        min: 1,
        max: SIM_DEFAULTS.vcyclesMax,
        step: 1,
        get: () => p.vcycles,
        set: (x) => (p.vcycles = x),
        format: (x) => `×${x.toFixed(0)}`,
      },
      {
        label: 'jacobi (it.)',
        min: SIM_DEFAULTS.pressureIterationsMin,
        max: SIM_DEFAULTS.pressureIterationsMax,
        step: SIM_DEFAULTS.pressureIterationsStep,
        get: () => frame.pressureIterations,
        set: (x) => (frame.pressureIterations = x),
        format: (x) => x.toFixed(0),
      },
      {
        label: 'force du splat',
        min: 5,
        max: 150,
        step: 5,
        get: () => p.splatForce,
        set: (x) => (p.splatForce = x),
        format: (x) => x.toFixed(0),
      },
      {
        label: 'rayon du splat',
        min: GRID_SIZE * 0.005,
        max: GRID_SIZE * 0.06,
        step: 1,
        get: () => p.splatRadius,
        set: (x) => (p.splatRadius = x),
        format: (x) => `${x.toFixed(0)} px`,
      },
      {
        label: 'débit de densité',
        min: 5,
        max: 100,
        step: 5,
        get: () => p.splatDensity,
        set: (x) => (p.splatDensity = x),
        format: (x) => x.toFixed(0),
      },
      {
        label: 'vitesse du temps',
        min: 0,
        max: 2,
        step: 0.05,
        get: () => p.timeScale,
        set: (x) => (p.timeScale = x),
        format: (x) => `×${x.toFixed(2)}`,
      },
      {
        label: 'gravité encres',
        min: 0,
        max: 1.5,
        step: 0.05,
        get: () => p.buoyancyScale,
        set: (x) => (p.buoyancyScale = x),
        format: (x) => `×${x.toFixed(2)}`,
      },
      {
        label: 'exposition',
        min: 0.4,
        max: 3,
        step: 0.05,
        get: () => r.exposure,
        set: (x) => (r.exposure = x),
        format: (x) => x.toFixed(2),
      },
      {
        label: 'bloom',
        min: 0,
        max: 2,
        step: 0.05,
        get: () => r.bloomStrength,
        set: (x) => (r.bloomStrength = x),
        format: (x) => x.toFixed(2),
      },
      {
        label: 'particules',
        min: 0,
        max: 2.5,
        step: 0.05,
        get: () => p.particleIntensity,
        set: (x) => (p.particleIntensity = x),
        format: (x) => `×${x.toFixed(2)}`,
      },
      {
        label: 'caméra : force',
        min: 0,
        max: 3,
        step: 0.05,
        get: () => p.flowStrength,
        set: (x) => (p.flowStrength = x),
        format: (x) => `×${x.toFixed(2)}`,
      },
      {
        label: 'caméra : seuil',
        min: 0.005,
        max: 0.05,
        step: 0.001,
        get: () => p.flowGate,
        set: (x) => (p.flowGate = x),
        format: (x) => x.toFixed(3),
      },
    ];
    for (const spec of sliders) {
      this.addSlider(spec);
    }

    this.addCheckbox('correcteur MacCormack', () => p.macCormack, (v) => (p.macCormack = v));
    this.addCheckbox(
      'pression multigrid (sinon Jacobi)',
      () => p.multigrid,
      (v) => (p.multigrid = v),
    );
    this.addCheckbox('particules traceuses', () => p.particles, (v) => (p.particles = v));
    this.addCheckbox('rendu papier (marbrure)', () => r.paper, (v) => (r.paper = v));
    if (onCameraToggle) {
      // L'état réel (p.cameraFlow) n'est vrai qu'une fois la permission accordée et le
      // flux démarré — refresh() resynchronise la case si la demande échoue.
      this.addCheckbox('caméra : le mouvement pousse le fluide', () => p.cameraFlow, onCameraToggle);
    }
    if (hands) {
      this.addCheckbox('mains : index = pointeur, pincer = cliquer', hands.get, hands.set);
    }
    if (marbleMode) {
      // Un seul interrupteur : bain figé + gravité 0 + rendu papier + bain préparé.
      // R = bain neuf. Gouttes, stylet et peigne font le reste.
      this.addCheckbox('mode marbrure (R = bain neuf)', marbleMode.get, marbleMode.set);
    }

    const buttons = document.createElement('div');
    buttons.className = 'panel-buttons';
    this.addButton(buttons, '⏭ pas (N)', () => (frame.stepOnce = true));
    this.addButton(buttons, '↺ reset (R)', () => (frame.reset = true));
    this.addButton(buttons, '▨ murs off (X)', () => (frame.clearWalls = true));
    if (onExport) {
      this.addButton(buttons, '⬇ PNG', onExport);
    }
    this.root.appendChild(buttons);

    parent.appendChild(this.root);

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Tab') {
        e.preventDefault();
        this.toggle();
      }
    });
  }

  toggle(): void {
    this.root.classList.toggle('hidden');
  }

  show(): void {
    this.root.classList.remove('hidden');
  }

  hide(): void {
    this.root.classList.add('hidden');
  }

  isOpen(): boolean {
    return !this.root.classList.contains('hidden');
  }

  /** Resynchronise les curseurs avec l'état (touches +/- notamment). Appelé 2×/s. */
  refresh(): void {
    for (const update of this.refreshers) {
      update();
    }
  }

  private addSlider(spec: SliderSpec): void {
    const row = document.createElement('label');
    row.className = 'panel-row';
    const name = document.createElement('span');
    name.textContent = spec.label;
    const value = document.createElement('span');
    value.className = 'panel-value';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    const fmt = spec.format ?? ((x: number) => x.toFixed(2));
    const sync = (): void => {
      input.value = String(spec.get());
      value.textContent = fmt(spec.get());
    };
    sync();
    input.addEventListener('input', () => {
      spec.set(Number(input.value));
      value.textContent = fmt(spec.get());
    });
    input.addEventListener('change', () => input.blur());
    this.refreshers.push(sync);
    row.append(name, input, value);
    this.root.appendChild(row);
  }

  private addCheckbox(label: string, get: () => boolean, set: (v: boolean) => void): void {
    const row = document.createElement('label');
    row.className = 'panel-row panel-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = get();
    input.addEventListener('change', () => {
      set(input.checked);
      input.blur();
    });
    const name = document.createElement('span');
    name.textContent = label;
    this.refreshers.push(() => (input.checked = get()));
    row.append(input, name);
    this.root.appendChild(row);
  }

  private addButton(parent: HTMLElement, label: string, action: () => void): void {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', () => {
      action();
      button.blur();
    });
    parent.appendChild(button);
  }
}
