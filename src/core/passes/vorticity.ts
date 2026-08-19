/**
 * Vorticity confinement en deux passes : rotationnel de la vélocité (curl.wgsl) puis
 * réinjection d'une force qui amplifie les tourbillons existants (vorticity.wgsl).
 * Appliquée après les forces et avant la projection, pour que celle-ci nettoie la
 * légère divergence que la force de confinement peut introduire.
 */

import curlWGSL from '../shaders/curl.wgsl?raw';
import vorticityWGSL from '../shaders/vorticity.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface VorticityPasses {
  readonly curlPipeline: GPUComputePipeline;
  readonly confinePipeline: GPUComputePipeline;
  /** Indexé par [source de vélocité]. */
  readonly curlBind: Pair<GPUBindGroup>;
  /** Indexé par [source de vélocité]. */
  readonly confineBind: Pair<GPUBindGroup>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createVorticityPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<VorticityPasses> {
  const [curlModule, confineModule] = await Promise.all([
    createShaderModule(device, 'curl.wgsl', curlWGSL),
    createShaderModule(device, 'vorticity.wgsl', vorticityWGSL),
  ]);
  const [curlPipeline, confinePipeline] = await Promise.all([
    createSimPipeline(device, 'curl', layouts, layouts.curl, curlModule),
    createSimPipeline(device, 'vorticity-confine', layouts, layouts.vorticity, confineModule),
  ]);

  return {
    curlPipeline,
    confinePipeline,
    curlBind: pair((v) =>
      device.createBindGroup({
        label: `curl-bind-v${v}`,
        layout: layouts.curl,
        entries: [
          { binding: 0, resource: res.velocity.views[v] },
          { binding: 1, resource: res.curl.view },
        ],
      }),
    ),
    confineBind: pair((v) =>
      device.createBindGroup({
        label: `vorticity-bind-v${v}`,
        layout: layouts.vorticity,
        entries: [
          { binding: 0, resource: res.velocity.views[v] },
          { binding: 1, resource: res.curl.view },
          { binding: 2, resource: res.velocity.views[flip(v)] },
          { binding: 3, resource: res.obstacle.view },
        ],
      }),
    ),
  };
}
