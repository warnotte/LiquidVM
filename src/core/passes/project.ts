/**
 * Projection : rend le champ de vélocité incompressible (∇·v = 0) en trois passes —
 * divergence, itérations de Jacobi pour la pression, soustraction du gradient.
 * La pression n'est PAS remise à zéro entre les frames : le champ de la frame
 * précédente sert de « warm start », ce qui améliore nettement la convergence
 * à nombre d'itérations égal (Jacobi ne converge que partiellement à 30 itérations).
 */

import divergenceWGSL from '../shaders/divergence.wgsl?raw';
import gradientWGSL from '../shaders/gradient.wgsl?raw';
import jacobiWGSL from '../shaders/jacobi.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface ProjectPasses {
  readonly divergencePipeline: GPUComputePipeline;
  readonly jacobiPipeline: GPUComputePipeline;
  readonly gradientPipeline: GPUComputePipeline;
  /** Indexé par [source de vélocité]. */
  readonly divergenceBind: Pair<GPUBindGroup>;
  /** Indexé par [source de pression]. */
  readonly jacobiBind: Pair<GPUBindGroup>;
  /** Indexé par [source de pression][source de vélocité]. */
  readonly gradientBind: Pair<Pair<GPUBindGroup>>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createProjectPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<ProjectPasses> {
  const [divModule, jacModule, gradModule] = await Promise.all([
    createShaderModule(device, 'divergence.wgsl', divergenceWGSL),
    createShaderModule(device, 'jacobi.wgsl', jacobiWGSL),
    createShaderModule(device, 'gradient.wgsl', gradientWGSL),
  ]);
  const [divergencePipeline, jacobiPipeline, gradientPipeline] = await Promise.all([
    createSimPipeline(device, 'divergence', layouts, layouts.divergence, divModule),
    createSimPipeline(device, 'jacobi', layouts, layouts.jacobi, jacModule),
    createSimPipeline(device, 'gradient-subtract', layouts, layouts.gradient, gradModule),
  ]);

  return {
    divergencePipeline,
    jacobiPipeline,
    gradientPipeline,
    divergenceBind: pair((v) =>
      device.createBindGroup({
        label: `divergence-bind-v${v}`,
        layout: layouts.divergence,
        entries: [
          { binding: 0, resource: res.velocity.views[v] },
          { binding: 1, resource: res.divergence.view },
          { binding: 2, resource: res.obstacle.view },
        ],
      }),
    ),
    jacobiBind: pair((p) =>
      device.createBindGroup({
        label: `jacobi-bind-p${p}`,
        layout: layouts.jacobi,
        entries: [
          { binding: 0, resource: res.pressure.views[p] },
          { binding: 1, resource: res.divergence.view },
          { binding: 2, resource: res.pressure.views[flip(p)] },
          { binding: 3, resource: res.obstacle.view },
        ],
      }),
    ),
    gradientBind: pair((p) =>
      pair((v) =>
        device.createBindGroup({
          label: `gradient-bind-p${p}-v${v}`,
          layout: layouts.gradient,
          entries: [
            { binding: 0, resource: res.pressure.views[p] },
            { binding: 1, resource: res.velocity.views[v] },
            { binding: 2, resource: res.velocity.views[flip(v)] },
            { binding: 3, resource: res.obstacle.view },
          ],
        }),
      ),
    ),
  };
}
