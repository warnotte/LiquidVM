/**
 * Passes d'advection MacCormack — vélocité (auto-advection) et densités (transport
 * passif sur la grille dye, plus fine, + injection souris). Chaque advection = deux
 * dispatches : prédicteur semi-lagrangien vers une texture scratch, puis correcteur
 * d'ordre 2 clampé (voir advect_velocity.wgsl / advect_density.wgsl pour la physique).
 * Toutes les variantes ping-pong des bind groups sont pré-créées ici, à l'init.
 */

import advectDensityWGSL from '../shaders/advect_density.wgsl?raw';
import advectVelocityWGSL from '../shaders/advect_velocity.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface AdvectPasses {
  readonly velPredictPipeline: GPUComputePipeline;
  readonly velCorrectPipeline: GPUComputePipeline;
  readonly denPredictPipeline: GPUComputePipeline;
  readonly denCorrectPipeline: GPUComputePipeline;
  /** Indexés par [source de vélocité]. */
  readonly velPredictBind: Pair<GPUBindGroup>;
  readonly velCorrectBind: Pair<GPUBindGroup>;
  /** Indexés par [source de vélocité][source de densité]. */
  readonly denPredictBind: Pair<Pair<GPUBindGroup>>;
  readonly denCorrectBind: Pair<Pair<GPUBindGroup>>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createAdvectPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<AdvectPasses> {
  const [velModule, denModule] = await Promise.all([
    createShaderModule(device, 'advect-velocity.wgsl', advectVelocityWGSL),
    createShaderModule(device, 'advect-density.wgsl', advectDensityWGSL),
  ]);
  const [velPredictPipeline, velCorrectPipeline, denPredictPipeline, denCorrectPipeline] =
    await Promise.all([
      createSimPipeline(device, 'advect-vel-predict', layouts, layouts.advectVelPredict, velModule, 'predict'),
      createSimPipeline(device, 'advect-vel-correct', layouts, layouts.advectVelCorrect, velModule, 'correct'),
      createSimPipeline(device, 'advect-den-predict', layouts, layouts.advectDenPredict, denModule, 'predict'),
      createSimPipeline(device, 'advect-den-correct', layouts, layouts.advectDenCorrect, denModule, 'correct'),
    ]);

  return {
    velPredictPipeline,
    velCorrectPipeline,
    denPredictPipeline,
    denCorrectPipeline,
    velPredictBind: pair((v) =>
      device.createBindGroup({
        label: `advect-vel-predict-bind-v${v}`,
        layout: layouts.advectVelPredict,
        entries: [
          { binding: 0, resource: res.velocity.views[v] },
          { binding: 1, resource: res.velScratch.view },
        ],
      }),
    ),
    velCorrectBind: pair((v) =>
      device.createBindGroup({
        label: `advect-vel-correct-bind-v${v}`,
        layout: layouts.advectVelCorrect,
        entries: [
          { binding: 0, resource: res.velocity.views[v] },
          { binding: 2, resource: res.velScratch.view },
          { binding: 3, resource: res.velocity.views[flip(v)] },
          { binding: 4, resource: res.obstacle.view },
        ],
      }),
    ),
    denPredictBind: pair((v) =>
      pair((d) =>
        device.createBindGroup({
          label: `advect-den-predict-bind-v${v}-d${d}`,
          layout: layouts.advectDenPredict,
          entries: [
            { binding: 0, resource: res.velocity.views[v] },
            { binding: 1, resource: res.density.views[d] },
            { binding: 2, resource: res.dyeScratch.view },
          ],
        }),
      ),
    ),
    denCorrectBind: pair((v) =>
      pair((d) =>
        device.createBindGroup({
          label: `advect-den-correct-bind-v${v}-d${d}`,
          layout: layouts.advectDenCorrect,
          entries: [
            { binding: 0, resource: res.velocity.views[v] },
            { binding: 1, resource: res.density.views[d] },
            { binding: 3, resource: res.dyeScratch.view },
            { binding: 4, resource: res.density.views[flip(d)] },
            { binding: 5, resource: res.obstacle.view },
          ],
        }),
      ),
    ),
  };
}
