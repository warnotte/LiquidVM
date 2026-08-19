/**
 * Passes de remise à zéro de tous les champs (reset « R »). Deux pipelines car deux
 * formats de storage texture ; un bind group pré-créé par texture cible. Encodées
 * en tête de la passe compute de la frame où le reset est demandé.
 */

import clearWGSL from '../shaders/clear.wgsl?raw';
import { DISPATCH_SIZE, DYE_DISPATCH_SIZE } from '../config';
import { createShaderModule, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';

/** Cible de clear : bind group + nombre de workgroups (les grilles diffèrent). */
export interface ClearTarget {
  readonly bind: GPUBindGroup;
  readonly dispatch: number;
}

export interface ClearPasses {
  readonly rgbaPipeline: GPUComputePipeline;
  readonly scalarPipeline: GPUComputePipeline;
  /** Cibles rgba16float : vélocité ×2 (grille sim), densité ×2 (grille dye). */
  readonly rgbaTargets: readonly ClearTarget[];
  /** Cibles r32float : pression ×2, divergence. */
  readonly scalarTargets: readonly ClearTarget[];
  /** Les deux textures de pression seules — purgées au changement de mode de frontières
   *  (le warm start d'un mode est faux dans l'opérateur d'un autre). */
  readonly pressureTargets: readonly ClearTarget[];
  /** Le champ d'obstacles est effacé séparément (touche X) — R préserve les murs. */
  readonly obstacleBind: GPUBindGroup;
}

export async function createClearPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<ClearPasses> {
  const module = await createShaderModule(device, 'clear.wgsl', clearWGSL);
  const [rgbaPipeline, scalarPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'clear-rgba16f',
      layout: device.createPipelineLayout({
        label: 'clear-rgba16f-pipeline-layout',
        bindGroupLayouts: [layouts.clearRgba],
      }),
      compute: { module, entryPoint: 'clear_rgba16f' },
    }),
    device.createComputePipelineAsync({
      label: 'clear-r32f',
      layout: device.createPipelineLayout({
        label: 'clear-r32f-pipeline-layout',
        bindGroupLayouts: [layouts.clearScalar],
      }),
      compute: { module, entryPoint: 'clear_r32f' },
    }),
  ]);

  const rgba: readonly [string, GPUTextureView, number][] = [
    ['velocity-0', res.velocity.views[0], DISPATCH_SIZE],
    ['velocity-1', res.velocity.views[1], DISPATCH_SIZE],
    ['density-0', res.density.views[0], DYE_DISPATCH_SIZE],
    ['density-1', res.density.views[1], DYE_DISPATCH_SIZE],
    ['curl', res.curl.view, DISPATCH_SIZE],
  ];
  const makeScalarTarget = (name: string, view: GPUTextureView): ClearTarget => ({
    bind: device.createBindGroup({
      label: `clear-bind-${name}`,
      layout: layouts.clearScalar,
      entries: [{ binding: 1, resource: view }],
    }),
    dispatch: DISPATCH_SIZE,
  });
  const pressureTargets = [
    makeScalarTarget('pressure-0', res.pressure.views[0]),
    makeScalarTarget('pressure-1', res.pressure.views[1]),
  ];

  return {
    rgbaPipeline,
    scalarPipeline,
    rgbaTargets: rgba.map(([name, view, dispatch]) => ({
      bind: device.createBindGroup({
        label: `clear-bind-${name}`,
        layout: layouts.clearRgba,
        entries: [{ binding: 0, resource: view }],
      }),
      dispatch,
    })),
    scalarTargets: [...pressureTargets, makeScalarTarget('divergence', res.divergence.view)],
    pressureTargets,
    obstacleBind: device.createBindGroup({
      label: 'clear-bind-obstacle',
      layout: layouts.clearScalar,
      entries: [{ binding: 1, resource: res.obstacle.view }],
    }),
  };
}
