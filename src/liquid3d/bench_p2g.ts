/**
 * J0 — banc du scatter atomique P2G (voir PLAN-EAU.md, critère : < 2 ms).
 * Mesure par onSubmittedWorkDone (temps GPU réel, pas de quantisation vsync) :
 * chaque soumission encode K itérations complètes [clear ×6 → scatter →
 * resolve], la moyenne par itération sort en millisecondes. Zéro readback :
 * la mesure est un horodatage CPU autour de la complétion de la file.
 */

import p2gWGSL from './shaders/p2g_bench.wgsl?raw';
import { FIXED_POINT_SCALE, GRID_EAU, particleDispatch, PARTICLES_EAU, WG_GRID } from './config_eau';
import { createShaderModule, withValidation } from '../core/pipelines';

void FIXED_POINT_SCALE; // (documenté dans config — la valeur vit dans le WGSL)

export interface BenchResult {
  /** ms/itération [clear + scatter + resolve], particules en ordre aléatoire. */
  readonly msRandom: number;
  /** Idem, particules TRIÉES par cellule (l'état qu'un tri périodique maintient). */
  readonly msSorted: number;
  readonly samplesRandom: readonly number[];
  readonly samplesSorted: readonly number[];
}

export class BenchP2G {
  private constructor(
    private readonly device: GPUDevice,
    private readonly pipelines: {
      init: GPUComputePipeline;
      initSorted: GPUComputePipeline;
      scatter: GPUComputePipeline;
      resolve: GPUComputePipeline;
    },
    private readonly bind: GPUBindGroup,
    private readonly atomicBuffers: readonly GPUBuffer[],
  ) {}

  static async create(device: GPUDevice): Promise<BenchP2G> {
    return withValidation(device, 'init-bench-p2g', async () => {
      const particles = device.createBuffer({
        label: 'eau-particles',
        size: PARTICLES_EAU * 32, // le banc J0 garde son layout d'origine (2 vec4)
        usage: GPUBufferUsage.STORAGE,
      });
      const cells = GRID_EAU * GRID_EAU * GRID_EAU;
      const atomicBuffers = ['acc-u', 'acc-v', 'acc-w', 'wgt-u', 'wgt-v', 'wgt-w'].map((name) =>
        device.createBuffer({
          label: `eau-${name}`,
          size: cells * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
      );
      const velTex = device
        .createTexture({
          label: 'eau-vel',
          dimension: '3d',
          size: { width: GRID_EAU, height: GRID_EAU, depthOrArrayLayers: GRID_EAU },
          format: 'rgba16float',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        })
        .createView({ label: 'eau-vel-view' });
      const uniforms = device.createBuffer({
        label: 'eau-bench-uniforms',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(uniforms, 0, new Float32Array([GRID_EAU, PARTICLES_EAU, 12.34, 0]));

      const layout = device.createBindGroupLayout({
        label: 'eau-bench',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
          ...[2, 3, 4, 5, 6, 7].map((binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' as const },
          })),
          {
            binding: 8,
            visibility: GPUShaderStage.COMPUTE,
            storageTexture: {
              access: 'write-only' as const,
              format: 'rgba16float' as const,
              viewDimension: '3d' as const,
            },
          },
        ],
      });
      const module = await createShaderModule(device, 'p2g_bench.wgsl', p2gWGSL);
      const pipelineLayout = device.createPipelineLayout({
        label: 'eau-bench-pl',
        bindGroupLayouts: [layout],
      });
      const compute = (label: string, entryPoint: string): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({ label, layout: pipelineLayout, compute: { module, entryPoint } });
      const [init, initSorted, scatter, resolve] = await Promise.all([
        compute('eau-init', 'init'),
        compute('eau-init-sorted', 'init_sorted'),
        compute('eau-scatter', 'scatter'),
        compute('eau-resolve', 'resolve'),
      ]);
      const bind = device.createBindGroup({
        label: 'eau-bench-bind',
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniforms } },
          { binding: 1, resource: { buffer: particles } },
          ...atomicBuffers.map((buffer, i) => ({ binding: 2 + i, resource: { buffer } })),
          { binding: 8, resource: velTex },
        ],
      });
      return new BenchP2G(device, { init, initSorted, scatter, resolve }, bind, atomicBuffers);
    });
  }

  private async runInit(pipeline: GPUComputePipeline): Promise<void> {
    const encoder = this.device.createCommandEncoder({ label: 'eau-init' });
    const pass = encoder.beginComputePass({ label: 'eau-init' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bind);
    const [pdx, pdy] = particleDispatch(PARTICLES_EAU);
    pass.dispatchWorkgroups(pdx, pdy);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  /** Une soumission de K itérations, mesurée à la complétion GPU. */
  private async timeSubmission(iterations: number): Promise<number> {
    const gridDispatch = Math.ceil(GRID_EAU / WG_GRID);
    const [pdx, pdy] = particleDispatch(PARTICLES_EAU);
    const encoder = this.device.createCommandEncoder({ label: 'eau-bench' });
    for (let k = 0; k < iterations; k++) {
      for (const buffer of this.atomicBuffers) {
        encoder.clearBuffer(buffer);
      }
      const pass = encoder.beginComputePass({ label: 'eau-p2g' });
      pass.setBindGroup(0, this.bind);
      pass.setPipeline(this.pipelines.scatter);
      pass.dispatchWorkgroups(pdx, pdy);
      pass.setPipeline(this.pipelines.resolve);
      pass.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
      pass.end();
    }
    const t0 = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / iterations;
  }

  private async measureCase(
    pipeline: GPUComputePipeline,
    iterations: number,
    submissions: number,
  ): Promise<number[]> {
    await this.runInit(pipeline);
    await this.timeSubmission(8); // warmup (compilation pilote, montée d'horloges)
    await this.timeSubmission(8);
    const samples: number[] = [];
    for (let s = 0; s < submissions; s++) {
      samples.push(await this.timeSubmission(iterations));
    }
    return samples;
  }

  /** Mesure les deux distributions : aléatoire (pire cas) et triée (régime réel). */
  async measure(iterations = 32, submissions = 5): Promise<BenchResult> {
    const avg = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
    const samplesRandom = await this.measureCase(this.pipelines.init, iterations, submissions);
    const samplesSorted = await this.measureCase(this.pipelines.initSorted, iterations, submissions);
    return {
      msRandom: avg(samplesRandom),
      msSorted: avg(samplesSorted),
      samplesRandom,
      samplesSorted,
    };
  }
}
