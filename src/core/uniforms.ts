/**
 * Remplissage CPU de l'uniform buffer de simulation.
 *
 * C'est le SEUL flux CPU→GPU par frame : un buffer de 2 slots de 256 octets
 * (alignement `minUniformBufferOffsetAlignment`), un slot par sous-pas de simulation,
 * sélectionné à l'encodage via un offset dynamique — aucune autre écriture, aucune lecture.
 * La disposition doit rester identique au struct `SimParams` des fichiers WGSL.
 */

import { DYE_SIZE, FLUIDS, GRID_SIZE, SIM_DEFAULTS } from './config';
import type { FrameInput } from './types';

/** Alignement d'offset dynamique garanti par la spec (valeur par défaut de la limite). */
export const UNIFORM_SLOT_BYTES = 256;
const SLOT_FLOATS = UNIFORM_SLOT_BYTES / 4;

/** Taille utile d'un slot : 8 × vec4f = 128 octets. */
export const SIM_PARAMS_BYTES = 128;

export class SimUniformWriter {
  /** Deux slots contigus ; écrits en une seule fois par frame via writeBuffer. */
  private readonly data = new Float32Array(SLOT_FLOATS * SIM_DEFAULTS.maxSubsteps);
  /** Offsets dynamiques pré-créés (zéro allocation par frame) : [0, 256]. */
  readonly dynamicOffsets = new Uint32Array(
    Array.from({ length: SIM_DEFAULTS.maxSubsteps }, (_, i) => i * UNIFORM_SLOT_BYTES),
  );

  /**
   * Remplit le slot d'un sous-pas. Le déplacement du pointeur est fractionné entre les
   * sous-pas par l'appelant (via `deltaScale`) pour que l'impulsion totale soit conservée.
   */
  fillSlot(slot: number, dt: number, input: FrameInput, deltaScale: number): void {
    const o = slot * SLOT_FLOATS;
    const d = this.data;
    // grid: vec4f — xy: taille de grille, zw: 1/taille
    d[o + 0] = GRID_SIZE;
    d[o + 1] = GRID_SIZE;
    d[o + 2] = 1 / GRID_SIZE;
    d[o + 3] = 1 / GRID_SIZE;
    // pointer: vec4f — xy: position (texels), zw: delta de ce sous-pas (texels)
    d[o + 4] = input.pointer.x * GRID_SIZE;
    d[o + 5] = input.pointer.y * GRID_SIZE;
    d[o + 6] = input.pointer.dx * GRID_SIZE * deltaScale;
    d[o + 7] = input.pointer.dy * GRID_SIZE * deltaScale;
    // impulse: vec4f — x: dt, y: bouton, z: fluide sélectionné, w: rayon du splat (texels)
    d[o + 8] = dt;
    d[o + 9] = input.pointer.down ? 1 : 0;
    d[o + 10] = input.selectedFluid;
    d[o + 11] = input.params.splatRadius;
    // misc: vec4f — x: dissipation vélocité (1/s), y: force du splat, z: débit de
    //               densité, w: outil du clic gauche (0 injecter, 1 gommer, 2 tourbillon, 3 souffle)
    d[o + 12] = input.params.velocityDissipation;
    d[o + 13] = input.params.splatForce;
    d[o + 14] = input.params.splatDensity;
    d[o + 15] = input.tool;
    // dissipation: vec4f — xyz: dissipation de densité par fluide (1/s)
    d[o + 16] = FLUIDS[0].dissipation;
    d[o + 17] = FLUIDS[1].dissipation;
    d[o + 18] = FLUIDS[2].dissipation;
    d[o + 19] = 0;
    // buoyancy: vec4f — xyz: poussée par fluide (texels/s², positif = monte)
    d[o + 20] = FLUIDS[0].buoyancy;
    d[o + 21] = FLUIDS[1].buoyancy;
    d[o + 22] = FLUIDS[2].buoyancy;
    d[o + 23] = 0;
    // extra: vec4f — x: frontières (0 parois, 1 périodique, 2 ouvert),
    //               y: pinceau mur (-1|0|1), z: force de vorticité, w: MacCormack (0|1)
    d[o + 24] = input.boundaryMode;
    d[o + 25] = input.pointer.wall ? (input.pointer.erase ? -1 : 1) : 0;
    d[o + 26] = input.params.vorticityStrength;
    d[o + 27] = input.params.macCormack ? 1 : 0;
    // dye: vec4f — xy: taille de la grille de densités, zw: 1/taille
    d[o + 28] = DYE_SIZE;
    d[o + 29] = DYE_SIZE;
    d[o + 30] = 1 / DYE_SIZE;
    d[o + 31] = 1 / DYE_SIZE;
  }

  /** Envoie les `substeps` premiers slots vers le GPU (une seule écriture par frame). */
  upload(queue: GPUQueue, buffer: GPUBuffer, substeps: number): void {
    queue.writeBuffer(buffer, 0, this.data, 0, substeps * SLOT_FLOATS);
  }
}

/**
 * Contenu de l'uniform buffer de rendu (couleurs des fluides + tone-mapping + vue).
 * Écrit à l'init, puis uniquement quand la vue de debug change (jamais par frame).
 * L'index 13 (tone.y) contient le ViewMode courant.
 */
export const RENDER_VIEW_MODE_INDEX = 13;

export function renderUniformData(): Float32Array<ArrayBuffer> {
  const d = new Float32Array(16);
  for (let i = 0; i < 3; i++) {
    const c = FLUIDS[i]!.color;
    d[i * 4 + 0] = c[0];
    d[i * 4 + 1] = c[1];
    d[i * 4 + 2] = c[2];
    d[i * 4 + 3] = 0;
  }
  // tone: x = exposition, y = vue (ViewMode), z = échelle debug vélocité, w = force du bloom
  d[12] = SIM_DEFAULTS.exposure;
  d[RENDER_VIEW_MODE_INDEX] = 0;
  d[14] = SIM_DEFAULTS.debugVelocityScale;
  d[15] = SIM_DEFAULTS.bloomStrength;
  return d;
}
