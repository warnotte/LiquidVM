/**
 * Pinceau à murs : une passe compute qui peint (ou gomme) le champ d'obstacles sous
 * le pointeur (paint_walls.wgsl). Encodée uniquement quand le bouton secondaire est
 * enfoncé — y compris en pause, pour construire tranquillement.
 */

import paintWallsWGSL from '../shaders/paint_walls.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';

export interface WallsPass {
  readonly pipeline: GPUComputePipeline;
  readonly bind: GPUBindGroup;
}

export async function createWallsPass(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<WallsPass> {
  const module = await createShaderModule(device, 'paint-walls.wgsl', paintWallsWGSL);
  const pipeline = await createSimPipeline(device, 'paint-walls', layouts, layouts.walls, module);
  return {
    pipeline,
    bind: device.createBindGroup({
      label: 'paint-walls-bind',
      layout: layouts.walls,
      entries: [{ binding: 0, resource: res.obstacle.view }],
    }),
  };
}
