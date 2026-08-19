/**
 * Passe de forces externes : buoyancy par fluide + impulsion souris (voir forces.wgsl).
 * Appliquée après l'advection de vélocité et avant la projection, pour que la
 * projection rende le champ incompressible forces comprises.
 */

import forcesWGSL from '../shaders/forces.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface ForcesPass {
  readonly pipeline: GPUComputePipeline;
  /** Indexé par [source de vélocité][source de densité]. */
  readonly bind: Pair<Pair<GPUBindGroup>>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createForcesPass(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<ForcesPass> {
  const module = await createShaderModule(device, 'forces.wgsl', forcesWGSL);
  const pipeline = await createSimPipeline(device, 'forces', layouts, layouts.forces, module);
  return {
    pipeline,
    bind: pair((v) =>
      pair((d) =>
        device.createBindGroup({
          label: `forces-bind-v${v}-d${d}`,
          layout: layouts.forces,
          entries: [
            { binding: 0, resource: res.velocity.views[v] },
            { binding: 1, resource: res.density.views[d] },
            { binding: 2, resource: res.velocity.views[flip(v)] },
            { binding: 3, resource: res.obstacle.view },
          ],
        }),
      ),
    ),
  };
}
