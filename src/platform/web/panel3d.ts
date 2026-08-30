/**
 * Interface déclarative du prototype 3D : un panneau de réglages à SECTIONS
 * (Tab) et une barre d'outils tactile — entièrement pilotés par des specs.
 * Ajouter un réglage ou un bouton futur = une entrée dans la spec de main3d,
 * zéro nouveau code d'interface. DOM vanilla, aucune dépendance.
 */

export interface Slider3Spec {
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly get: () => number;
  readonly set: (value: number) => void;
  readonly format?: (value: number) => string;
}

export interface Check3Spec {
  readonly label: string;
  readonly get: () => boolean;
  readonly set: (value: boolean) => void;
}

export interface Button3Spec {
  readonly label: string;
  /** État « actif », resynchronisé au refresh — comme la barre d'outils. C'est
   *  lui qui permet à une rangée de boutons EXCLUSIFS (les environnements) de
   *  montrer lequel est en cours, au lieu de trois boutons muets. */
  readonly isActive?: () => boolean;
  readonly action: () => void;
}

export interface Panel3Section {
  readonly title: string;
  /** Boutons AVANT les curseurs. Pour une section où l'on CHOISIT d'abord et
   *  où l'on affine ensuite (l'environnement : on prend la prise de vue, puis
   *  on règle son soleil) — l'ordre inverse se lit à l'envers. */
  readonly buttonsFirst?: boolean;
  readonly sliders?: readonly Slider3Spec[];
  readonly checks?: readonly Check3Spec[];
  readonly buttons?: readonly Button3Spec[];
}

export class Panel3D {
  private readonly root: HTMLDivElement;
  private readonly refreshers: (() => void)[] = [];

  constructor(parent: HTMLElement, sections: readonly Panel3Section[]) {
    this.root = document.createElement('div');
    this.root.className = 'panel3d';

    const header = document.createElement('div');
    header.className = 'panel3d-title';
    header.textContent = 'réglages (Tab)';
    this.root.appendChild(header);

    for (const section of sections) {
      const title = document.createElement('div');
      title.className = 'panel3d-section';
      title.textContent = section.title;
      this.root.appendChild(title);
      const addButtons = (): void => {
        if (!section.buttons || section.buttons.length === 0) {
          return;
        }
        const row = document.createElement('div');
        row.className = 'panel3d-buttons';
        for (const spec of section.buttons) {
          const button = document.createElement('button');
          button.type = 'button';
          button.textContent = spec.label;
          button.addEventListener('click', () => {
            spec.action();
            button.blur();
          });
          if (spec.isActive) {
            const sync = (): void => {
              button.classList.toggle('active', spec.isActive!());
            };
            this.refreshers.push(sync);
            sync();
          }
          row.appendChild(button);
        }
        this.root.appendChild(row);
      };
      if (section.buttonsFirst === true) {
        addButtons();
      }
      for (const spec of section.sliders ?? []) {
        this.addSlider(spec);
      }
      for (const spec of section.checks ?? []) {
        this.addCheck(spec);
      }
      if (section.buttonsFirst !== true) {
        addButtons();
      }
    }
    parent.appendChild(this.root);

    window.addEventListener('keydown', (e) => {
      if (e.code === 'Tab') {
        e.preventDefault();
        this.root.classList.toggle('hidden');
      }
    });
  }

  toggle(): void {
    this.root.classList.toggle('hidden');
  }

  /** Resynchronise curseurs et cases avec l'état (appelé ~2×/s). */
  refresh(): void {
    for (const update of this.refreshers) {
      update();
    }
  }

  private addSlider(spec: Slider3Spec): void {
    const row = document.createElement('label');
    row.className = 'panel3d-row';
    const name = document.createElement('span');
    name.textContent = spec.label;
    const value = document.createElement('span');
    value.className = 'panel3d-value';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    const fmt = spec.format ?? ((x: number): string => x.toFixed(2));
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

  private addCheck(spec: Check3Spec): void {
    const row = document.createElement('label');
    row.className = 'panel3d-row panel3d-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = spec.get();
    input.addEventListener('change', () => {
      spec.set(input.checked);
      input.blur();
    });
    const name = document.createElement('span');
    name.textContent = spec.label;
    this.refreshers.push(() => (input.checked = spec.get()));
    row.append(input, name);
    this.root.appendChild(row);
  }
}

export interface ToolbarItem {
  readonly label: string;
  /** Libellé RECALCULÉ au refresh, pour un bouton qui affiche un état et le
   *  fait défiler (la prise de vue) au lieu d'en basculer un seul. */
  readonly dynamicLabel?: () => string;
  /** Pastille de couleur (matières). */
  readonly color?: string;
  /** L'état « actif » du bouton, resynchronisé au refresh. */
  readonly isActive?: () => boolean;
  readonly action: () => void;
}

export class Toolbar3D {
  private readonly root: HTMLDivElement;
  private readonly refreshers: (() => void)[] = [];

  constructor(parent: HTMLElement, groups: readonly (readonly ToolbarItem[])[]) {
    this.root = document.createElement('div');
    this.root.className = 'toolbar3d';
    for (const group of groups) {
      const div = document.createElement('div');
      div.className = 'toolbar3d-group';
      for (const item of group) {
        const button = document.createElement('button');
        button.type = 'button';
        if (item.color) {
          const chip = document.createElement('span');
          chip.className = 'toolbar3d-chip';
          chip.style.background = item.color;
          button.appendChild(chip);
        }
        const text = document.createTextNode(item.label);
        button.appendChild(text);
        button.addEventListener('click', () => {
          item.action();
          button.blur();
        });
        if (item.dynamicLabel) {
          const syncLabel = (): void => {
            const next = item.dynamicLabel!();
            if (text.nodeValue !== next) {
              text.nodeValue = next;
            }
          };
          this.refreshers.push(syncLabel);
          syncLabel();
        }
        if (item.isActive) {
          const sync = (): void => {
            button.classList.toggle('active', item.isActive!());
          };
          this.refreshers.push(sync);
          sync();
        }
        div.appendChild(button);
      }
      this.root.appendChild(div);
    }
    parent.appendChild(this.root);
  }

  refresh(): void {
    for (const update of this.refreshers) {
      update();
    }
  }
}
