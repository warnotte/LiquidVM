/**
 * Marbrure mathématique : un warp inverse de la texture d'encres par opération
 * (goutte, stylet, peigne — voir marble_warp.wgsl). L'opération arrive au plus une
 * fois par frame (les segments de geste sont coalescés par la plateforme) et
 * fonctionne bain figé — c'est même le mode de travail du marbreur.
 */

import marbleWarpWGSL from '../shaders/marble_warp.wgsl?raw';
import { createShaderModule, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface MarblePass {
  readonly pipeline: GPUComputePipeline;
  /** Indexé par [source de densité]. */
  readonly bind: Pair<GPUBindGroup>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createMarblePass(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<MarblePass> {
  const module = await createShaderModule(device, 'marble-warp.wgsl', marbleWarpWGSL);
  const pipeline = await device.createComputePipelineAsync({
    label: 'marble-warp',
    layout: device.createPipelineLayout({
      label: 'marble-pipeline-layout',
      bindGroupLayouts: [layouts.marble],
    }),
    compute: { module, entryPoint: 'main' },
  });
  return {
    pipeline,
    bind: pair((d) =>
      device.createBindGroup({
        label: `marble-bind-d${d}`,
        layout: layouts.marble,
        entries: [
          { binding: 0, resource: { buffer: res.marbleUniforms } },
          { binding: 1, resource: res.linearSampler },
          { binding: 2, resource: res.density.views[d] },
          { binding: 3, resource: res.density.views[flip(d)] },
        ],
      }),
    ),
  };
}
