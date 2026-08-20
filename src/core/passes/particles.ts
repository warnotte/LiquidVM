/**
 * Advection du système de particules traceuses (voir particle_advect.wgsl) : une passe
 * compute 1D par sous-pas, qui lit le champ de vélocité MAC courant et met le storage
 * buffer à jour en place (pas de ping-pong : chaque invocation ne touche que sa
 * particule). Le rendu vit dans render.ts (traînées additives dans la scène HDR).
 */

import particleAdvectWGSL from '../shaders/particle_advect.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import type { Pair, PingIndex } from '../types';

export interface ParticlesPass {
  readonly pipeline: GPUComputePipeline;
  /** Indexé par [source de vélocité]. */
  readonly bind: Pair<GPUBindGroup>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createParticlesPass(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<ParticlesPass> {
  const module = await createShaderModule(device, 'particle-advect.wgsl', particleAdvectWGSL);
  const pipeline = await createSimPipeline(
    device,
    'particle-advect',
    layouts,
    layouts.particleAdvect,
    module,
  );
  return {
    pipeline,
    bind: pair((v) =>
      device.createBindGroup({
        label: `particle-advect-bind-v${v}`,
        layout: layouts.particleAdvect,
        entries: [
          { binding: 0, resource: { buffer: res.particles } },
          { binding: 1, resource: res.velocity.views[v] },
          { binding: 2, resource: res.obstacle.view },
        ],
      }),
    ),
  };
}
