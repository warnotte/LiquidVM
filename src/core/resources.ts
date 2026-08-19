/**
 * Création de toutes les ressources GPU persistantes : textures de champs (en paires
 * ping-pong), sampler, uniform buffers. Tout est créé ici, une fois, à l'init —
 * l'état de la simulation réside intégralement sur GPU, rien n'est alloué en boucle
 * de frame et rien n'est jamais relu côté CPU.
 */

import {
  BLOOM_MID_SIZE,
  BLOOM_WIDE_SIZE,
  DENSITY_FORMAT,
  DYE_SIZE,
  GRID_SIZE,
  PARTICLE_COUNT,
  SCALAR_FORMAT,
  SCENE_SIZE,
  SIM_DEFAULTS,
  VELOCITY_FORMAT,
} from './config';
import type { Pair } from './types';
import { UNIFORM_SLOT_BYTES } from './uniforms';

/** Paire ping-pong : textures + vues pré-créées (createView alloue, donc jamais en frame). */
export interface PingTextures {
  readonly textures: Pair<GPUTexture>;
  readonly views: Pair<GPUTextureView>;
}

export interface SingleTexture {
  readonly texture: GPUTexture;
  readonly view: GPUTextureView;
}

export interface SimResources {
  readonly velocity: PingTextures;
  readonly density: PingTextures;
  readonly pressure: PingTextures;
  readonly divergence: SingleTexture;
  /** Rotationnel de la vélocité, intermédiaire de la vorticity confinement. */
  readonly curl: SingleTexture;
  /** Champ d'obstacles (1 = mur dessiné, 0 = fluide). Read-write, pas de ping-pong. */
  readonly obstacle: SingleTexture;
  /** Prédicteur MacCormack de la vélocité (φ̂), réécrit à chaque sous-pas. */
  readonly velScratch: SingleTexture;
  /** Prédicteur MacCormack des densités, à la résolution dye. */
  readonly dyeScratch: SingleTexture;
  /** Scène HDR intermédiaire (linéaire, pré-tone-mapping) — cible du rendu des fluides. */
  readonly scene: SingleTexture;
  /** Chaîne de bloom : deux niveaux (512², 256²), chacun en paire pour le flou séparable. */
  readonly bloomMid: Pair<SingleTexture>;
  readonly bloomWide: Pair<SingleTexture>;
  /** Sampler bilinéaire clamp-to-edge : advection/rendu en mode parois. */
  readonly linearSampler: GPUSampler;
  /** Sampler bilinéaire repeat : advection en mode périodique (le wrap est gratuit). */
  readonly repeatSampler: GPUSampler;
  /** Uniform buffer de simulation : maxSubsteps slots de 256 o, offset dynamique. */
  readonly simUniforms: GPUBuffer;
  /** Uniform buffer de rendu (couleurs, tone-mapping) : écrit une fois à l'init. */
  readonly renderUniforms: GPUBuffer;
  /** Particules traceuses : 2 vec4f par particule (posvel + âge/vie), zéro-initialisé —
   *  vie = 0 vaut « non initialisée », la première passe d'advection les disperse. */
  readonly particles: GPUBuffer;
}

function createFieldTexture(
  device: GPUDevice,
  label: string,
  format: GPUTextureFormat,
  size: number,
): GPUTexture {
  return device.createTexture({
    label,
    size: { width: size, height: size },
    format,
    // Lecture en compute/fragment + écriture directe en compute. Les textures fraîchement
    // créées sont garanties zéro-initialisées par la spec WebGPU : pas de clear à l'init.
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });
}

function createSingle(
  device: GPUDevice,
  name: string,
  format: GPUTextureFormat,
  size: number,
): SingleTexture {
  const texture = createFieldTexture(device, name, format, size);
  return { texture, view: texture.createView({ label: `${name}-view` }) };
}

function createPing(
  device: GPUDevice,
  name: string,
  format: GPUTextureFormat,
  size: number,
): PingTextures {
  const a = createFieldTexture(device, `${name}-0`, format, size);
  const b = createFieldTexture(device, `${name}-1`, format, size);
  return {
    textures: [a, b],
    views: [a.createView({ label: `${name}-0-view` }), b.createView({ label: `${name}-1-view` })],
  };
}

export function createResources(device: GPUDevice): SimResources {
  return {
    velocity: createPing(device, 'velocity', VELOCITY_FORMAT, GRID_SIZE),
    // Les densités vivent sur leur propre grille, plus fine (voir DYE_SIZE dans config.ts).
    density: createPing(device, 'density', DENSITY_FORMAT, DYE_SIZE),
    pressure: createPing(device, 'pressure', SCALAR_FORMAT, GRID_SIZE),
    divergence: createSingle(device, 'divergence', SCALAR_FORMAT, GRID_SIZE),
    // rgba16float (pas r32float) : le confinement échantillonne ω bilinéairement
    // aux positions de faces — il faut un format filtrable ET storage-capable.
    curl: createSingle(device, 'curl', DENSITY_FORMAT, GRID_SIZE),
    obstacle: createSingle(device, 'obstacle', SCALAR_FORMAT, GRID_SIZE),
    velScratch: createSingle(device, 'velocity-maccormack-hat', VELOCITY_FORMAT, GRID_SIZE),
    dyeScratch: createSingle(device, 'density-maccormack-hat', DENSITY_FORMAT, DYE_SIZE),
    scene: (() => {
      const texture = device.createTexture({
        label: 'scene-hdr',
        size: { width: SCENE_SIZE, height: SCENE_SIZE },
        format: DENSITY_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      return { texture, view: texture.createView({ label: 'scene-hdr-view' }) };
    })(),
    bloomMid: [
      createSingle(device, 'bloom-mid-0', DENSITY_FORMAT, BLOOM_MID_SIZE),
      createSingle(device, 'bloom-mid-1', DENSITY_FORMAT, BLOOM_MID_SIZE),
    ],
    bloomWide: [
      createSingle(device, 'bloom-wide-0', DENSITY_FORMAT, BLOOM_WIDE_SIZE),
      createSingle(device, 'bloom-wide-1', DENSITY_FORMAT, BLOOM_WIDE_SIZE),
    ],
    linearSampler: device.createSampler({
      label: 'linear-clamp-sampler',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    }),
    repeatSampler: device.createSampler({
      label: 'linear-repeat-sampler',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    }),
    simUniforms: device.createBuffer({
      label: 'sim-uniforms',
      size: UNIFORM_SLOT_BYTES * SIM_DEFAULTS.maxSubsteps,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    renderUniforms: device.createBuffer({
      label: 'render-uniforms',
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    particles: device.createBuffer({
      label: 'particles',
      size: PARTICLE_COUNT * 32, // 2 × vec4f par particule
      usage: GPUBufferUsage.STORAGE,
    }),
  };
}
