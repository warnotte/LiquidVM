/**
 * Flux optique webcam : deux passes compute — l'estimation du mouvement entre deux
 * images caméra (optical_flow.wgsl, une fois par frame, ping-pong sur la luminance),
 * puis l'application du champ de flux comme force sur la vélocité MAC
 * (apply_flow.wgsl, par sous-pas). Le core ne connaît que des textures : c'est la
 * plateforme qui remplit `res.camera` depuis getUserMedia.
 */

import applyFlowWGSL from '../shaders/apply_flow.wgsl?raw';
import opticalFlowWGSL from '../shaders/optical_flow.wgsl?raw';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

export interface OpticalFlowPasses {
  readonly flowPipeline: GPUComputePipeline;
  readonly applyPipeline: GPUComputePipeline;
  /** Indexé par [luminance précédente]. */
  readonly flowBind: Pair<GPUBindGroup>;
  /** Indexé par [source de vélocité]. */
  readonly applyBind: Pair<GPUBindGroup>;
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export async function createOpticalFlowPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<OpticalFlowPasses> {
  const [flowModule, applyModule] = await Promise.all([
    createShaderModule(device, 'optical-flow.wgsl', opticalFlowWGSL),
    createShaderModule(device, 'apply-flow.wgsl', applyFlowWGSL),
  ]);
  const [flowPipeline, applyPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'optical-flow',
      layout: device.createPipelineLayout({
        label: 'optical-flow-pipeline-layout',
        bindGroupLayouts: [layouts.opticalFlow],
      }),
      compute: { module: flowModule, entryPoint: 'main' },
    }),
    createSimPipeline(device, 'apply-flow', layouts, layouts.applyFlow, applyModule),
  ]);

  return {
    flowPipeline,
    applyPipeline,
    flowBind: pair((l) =>
      device.createBindGroup({
        label: `optical-flow-bind-l${l}`,
        layout: layouts.opticalFlow,
        entries: [
          { binding: 0, resource: res.camera.view },
          { binding: 1, resource: res.flowLum.views[l] },
          { binding: 2, resource: res.flow.view },
          { binding: 3, resource: res.flowLum.views[flip(l)] },
        ],
      }),
    ),
    applyBind: pair((v) =>
      device.createBindGroup({
        label: `apply-flow-bind-v${v}`,
        layout: layouts.applyFlow,
        entries: [
          { binding: 0, resource: res.flow.view },
          { binding: 1, resource: res.velocity.views[v] },
          { binding: 2, resource: res.velocity.views[flip(v)] },
          { binding: 3, resource: res.obstacle.view },
        ],
      }),
    ),
  };
}
