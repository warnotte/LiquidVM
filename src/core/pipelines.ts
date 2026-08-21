/**
 * Bind group layouts explicites et utilitaires de création de pipelines.
 *
 * Les layouts sont explicites (pas de `layout: 'auto'`) pour deux raisons :
 * 1. Le groupe 0 (uniforms + sampler) est PARTAGÉ par toutes les passes de simulation —
 *    un seul setBindGroup(0) par sous-pas, les pipelines étant « group-compatible ».
 * 2. Les bind groups des variantes ping-pong sont pré-créés à l'init contre un layout
 *    stable, jamais dérivé d'un pipeline.
 */

import { DENSITY_FORMAT, SCALAR_FORMAT, VELOCITY_FORMAT } from './config';
import { SIM_PARAMS_BYTES } from './uniforms';

export interface SimLayouts {
  /** group(0) de toutes les passes compute : SimParams (offset dynamique) + sampler. */
  readonly simGroup0: GPUBindGroupLayout;
  /** group(1) par passe (advection MacCormack : prédicteur et correcteur séparés). */
  readonly advectVelPredict: GPUBindGroupLayout;
  readonly advectVelCorrect: GPUBindGroupLayout;
  readonly advectDenPredict: GPUBindGroupLayout;
  readonly advectDenCorrect: GPUBindGroupLayout;
  readonly forces: GPUBindGroupLayout;
  readonly divergence: GPUBindGroupLayout;
  readonly jacobi: GPUBindGroupLayout;
  readonly gradient: GPUBindGroupLayout;
  readonly curl: GPUBindGroupLayout;
  /** Multigrid : restriction (résidu/obstacles) et prolongation de la correction. */
  readonly mgRestrict: GPUBindGroupLayout;
  readonly mgProlong: GPUBindGroupLayout;
  /** Vue debug 6 : composition de la mosaïque des niveaux (group(0) autonome). */
  readonly mgDebug: GPUBindGroupLayout;
  readonly vorticity: GPUBindGroupLayout;
  /** Pinceau à murs : storage read-write sur le champ d'obstacles. */
  readonly walls: GPUBindGroupLayout;
  /** Particules : advection compute (buffer rw) et rendu (buffer read-only en vertex). */
  readonly particleAdvect: GPUBindGroupLayout;
  readonly particleDraw: GPUBindGroupLayout;
  /** Flux optique : estimation (caméra + luminance ping-pong) et application au champ. */
  readonly opticalFlow: GPUBindGroupLayout;
  readonly applyFlow: GPUBindGroupLayout;
  /** Layouts des passes de clear (group(0) dédié, formats différents). */
  readonly clearRgba: GPUBindGroupLayout;
  readonly clearScalar: GPUBindGroupLayout;
  /** Rendu : scène HDR, chaîne de bloom, présentation vers le canvas. */
  readonly scene: GPUBindGroupLayout;
  readonly bloom: GPUBindGroupLayout;
  readonly present: GPUBindGroupLayout;
}

const COMPUTE = GPUShaderStage.COMPUTE;

/** Texture float filtrable (vélocité/densité en rgba16float). */
function sampledFloat(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: COMPUTE, texture: { sampleType: 'float' } };
}

/** Texture r32float : non filtrable de base, lue via textureLoad uniquement. */
function sampledScalar(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float' } };
}

function storage(binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry {
  return { binding, visibility: COMPUTE, storageTexture: { access: 'write-only', format } };
}

export function createLayouts(device: GPUDevice): SimLayouts {
  return {
    simGroup0: device.createBindGroupLayout({
      label: 'sim-group0-layout',
      entries: [
        {
          binding: 0,
          visibility: COMPUTE,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: SIM_PARAMS_BYTES },
        },
        { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
      ],
    }),
    advectVelPredict: device.createBindGroupLayout({
      label: 'advect-vel-predict-layout',
      entries: [sampledFloat(0), storage(1, VELOCITY_FORMAT)],
    }),
    advectVelCorrect: device.createBindGroupLayout({
      label: 'advect-vel-correct-layout',
      entries: [sampledFloat(0), sampledFloat(2), storage(3, VELOCITY_FORMAT), sampledScalar(4)],
    }),
    advectDenPredict: device.createBindGroupLayout({
      label: 'advect-den-predict-layout',
      entries: [sampledFloat(0), sampledFloat(1), storage(2, DENSITY_FORMAT)],
    }),
    advectDenCorrect: device.createBindGroupLayout({
      label: 'advect-den-correct-layout',
      entries: [
        sampledFloat(0),
        sampledFloat(1),
        sampledFloat(3),
        storage(4, DENSITY_FORMAT),
        sampledScalar(5),
      ],
    }),
    forces: device.createBindGroupLayout({
      label: 'forces-layout',
      entries: [sampledFloat(0), sampledFloat(1), storage(2, VELOCITY_FORMAT), sampledScalar(3)],
    }),
    divergence: device.createBindGroupLayout({
      label: 'divergence-layout',
      entries: [sampledFloat(0), storage(1, SCALAR_FORMAT), sampledScalar(2)],
    }),
    jacobi: device.createBindGroupLayout({
      label: 'jacobi-layout',
      entries: [sampledScalar(0), sampledScalar(1), storage(2, SCALAR_FORMAT), sampledScalar(3)],
    }),
    gradient: device.createBindGroupLayout({
      label: 'gradient-layout',
      entries: [sampledScalar(0), sampledFloat(1), storage(2, VELOCITY_FORMAT), sampledScalar(3)],
    }),
    curl: device.createBindGroupLayout({
      label: 'curl-layout',
      entries: [sampledFloat(0), storage(1, DENSITY_FORMAT)],
    }),
    mgRestrict: device.createBindGroupLayout({
      label: 'mg-restrict-layout',
      entries: [sampledScalar(0), storage(2, SCALAR_FORMAT)],
    }),
    mgProlong: device.createBindGroupLayout({
      label: 'mg-prolong-layout',
      entries: [sampledScalar(0), sampledScalar(1), storage(2, SCALAR_FORMAT)],
    }),
    mgDebug: device.createBindGroupLayout({
      label: 'mg-debug-layout',
      entries: [sampledScalar(0), sampledScalar(1), storage(2, DENSITY_FORMAT)],
    }),
    vorticity: device.createBindGroupLayout({
      label: 'vorticity-layout',
      entries: [sampledFloat(0), sampledFloat(1), storage(2, VELOCITY_FORMAT), sampledScalar(3)],
    }),
    walls: device.createBindGroupLayout({
      label: 'walls-layout',
      entries: [
        {
          binding: 0,
          visibility: COMPUTE,
          storageTexture: { access: 'read-write', format: SCALAR_FORMAT },
        },
      ],
    }),
    particleAdvect: device.createBindGroupLayout({
      label: 'particle-advect-layout',
      entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'storage' } },
        sampledFloat(1),
        sampledScalar(2),
      ],
    }),
    particleDraw: device.createBindGroupLayout({
      label: 'particle-draw-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'float' } },
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage' },
        },
      ],
    }),
    opticalFlow: device.createBindGroupLayout({
      label: 'optical-flow-layout',
      entries: [
        sampledFloat(0),
        sampledScalar(1),
        storage(2, DENSITY_FORMAT),
        storage(3, SCALAR_FORMAT),
        sampledFloat(4),
        { binding: 5, visibility: COMPUTE, buffer: { type: 'uniform' } },
      ],
    }),
    applyFlow: device.createBindGroupLayout({
      label: 'apply-flow-layout',
      entries: [sampledFloat(0), sampledFloat(1), storage(2, VELOCITY_FORMAT), sampledScalar(3)],
    }),
    clearRgba: device.createBindGroupLayout({
      label: 'clear-rgba-layout',
      entries: [storage(0, VELOCITY_FORMAT)],
    }),
    clearScalar: device.createBindGroupLayout({
      label: 'clear-scalar-layout',
      entries: [{ binding: 1, visibility: COMPUTE, storageTexture: { access: 'write-only', format: SCALAR_FORMAT } }],
    }),
    scene: device.createBindGroupLayout({
      label: 'scene-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    }),
    bloom: device.createBindGroupLayout({
      label: 'bloom-layout',
      entries: [
        { binding: 0, visibility: COMPUTE, sampler: { type: 'filtering' } },
        { binding: 1, visibility: COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: COMPUTE, storageTexture: { access: 'write-only', format: DENSITY_FORMAT } },
      ],
    }),
    present: device.createBindGroupLayout({
      label: 'present-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    }),
  };
}

/**
 * Crée un shader module et échoue explicitement si le WGSL ne compile pas —
 * les messages du compilateur sont remontés dans l'erreur (plutôt qu'un pipeline
 * silencieusement invalide).
 */
export async function createShaderModule(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) {
    const details = errors.map((m) => `${label}:${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
    throw new Error(`Échec de compilation WGSL :\n${details}`);
  }
  return module;
}

/**
 * Exécute `fn` sous un error scope de validation et transforme toute erreur GPU
 * en exception JS avec contexte — utilisé pour toute l'init des pipelines.
 */
export async function withValidation<T>(
  device: GPUDevice,
  label: string,
  fn: () => T | Promise<T>,
): Promise<T> {
  device.pushErrorScope('validation');
  const result = await fn();
  const error = await device.popErrorScope();
  if (error) {
    throw new Error(`Erreur de validation WebGPU (${label}) : ${error.message}`);
  }
  return result;
}

/** Pipeline compute à deux groupes : group(0) sim partagé + group(1) spécifique. */
export function createSimPipeline(
  device: GPUDevice,
  label: string,
  layouts: SimLayouts,
  group1: GPUBindGroupLayout,
  module: GPUShaderModule,
  entryPoint = 'main',
): Promise<GPUComputePipeline> {
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({
      label: `${label}-pipeline-layout`,
      bindGroupLayouts: [layouts.simGroup0, group1],
    }),
    compute: { module, entryPoint },
  });
}
