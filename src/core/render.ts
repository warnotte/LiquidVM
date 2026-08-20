/**
 * Rendu en trois étages, un seul CommandEncoder partagé avec la simulation :
 *   1. Scène — render pass vers une texture HDR fixe (rgba16float) : composition des
 *      fluides avec éclairage par gradient de densité, ou vue de debug (scene.wgsl).
 *   2. Bloom — compute : seuil + downsample ½, flou gaussien séparable, downsample ¼,
 *      flou à nouveau (bloom.wgsl). Encodé uniquement en vue fluides.
 *   3. Présentation — render pass vers le canvas : scène + bloom, tone-mapping, fond,
 *      murs, gamma (present.wgsl). Le canvas peut avoir n'importe quelle taille, la
 *      scène reste à résolution fixe (upscale bilinéaire).
 *
 * Le fragment de scène lit densité, vélocité et pression (toutes en ping-pong) :
 * les 2×2×2 variantes de bind group sont pré-créées et indexées à l'encodage.
 */

import { BLOOM_MID_DISPATCH, BLOOM_WIDE_DISPATCH, PARTICLE_COUNT } from './config';
import bloomWGSL from './shaders/bloom.wgsl?raw';
import particlesDrawWGSL from './shaders/particles_draw.wgsl?raw';
import presentWGSL from './shaders/present.wgsl?raw';
import sceneWGSL from './shaders/scene.wgsl?raw';
import { createShaderModule, type SimLayouts } from './pipelines';
import type { SimResources } from './resources';
import type { Pair, PingIndex, ViewMode } from './types';

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

interface BloomStep {
  readonly pipeline: GPUComputePipeline;
  readonly bind: GPUBindGroup;
  readonly dispatch: number;
}

export class CompositeRenderer {
  /** Descripteurs réutilisés chaque frame (zéro allocation) ; seule la vue canvas change. */
  private readonly sceneDescriptor: GPURenderPassDescriptor;
  private readonly presentAttachment: GPURenderPassColorAttachment;
  private readonly presentDescriptor: GPURenderPassDescriptor;
  private readonly bloomPassDesc: GPUComputePassDescriptor = { label: 'bloom-pass' };

  private constructor(
    private readonly scenePipeline: GPURenderPipeline,
    private readonly presentPipeline: GPURenderPipeline,
    /** Variante de la présentation vers rgba8unorm — export PNG à taille fixe. */
    private readonly exportPipeline: GPURenderPipeline,
    /** Indexé par [densité][vélocité][pression]. */
    private readonly sceneBind: Pair<Pair<Pair<GPUBindGroup>>>,
    private readonly presentBind: GPUBindGroup,
    /** Étapes de la chaîne de bloom, dans l'ordre d'encodage. */
    private readonly bloomSteps: readonly BloomStep[],
    /** Traînées de particules, dessinées dans la scène HDR (indexé par [densité]). */
    private readonly particlePipeline: GPURenderPipeline,
    private readonly particleBind: Pair<GPUBindGroup>,
    sceneView: GPUTextureView,
    placeholderTarget: GPUTextureView,
  ) {
    this.sceneDescriptor = {
      label: 'scene-pass',
      colorAttachments: [
        {
          view: sceneView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };
    this.presentAttachment = {
      view: placeholderTarget, // remplacée à chaque frame avant l'encodage
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    };
    this.presentDescriptor = {
      label: 'present-pass',
      colorAttachments: [this.presentAttachment],
    };
  }

  static async create(
    device: GPUDevice,
    layouts: SimLayouts,
    res: SimResources,
    targetFormat: GPUTextureFormat,
  ): Promise<CompositeRenderer> {
    const [sceneModule, bloomModule, presentModule, particlesModule] = await Promise.all([
      createShaderModule(device, 'scene.wgsl', sceneWGSL),
      createShaderModule(device, 'bloom.wgsl', bloomWGSL),
      createShaderModule(device, 'present.wgsl', presentWGSL),
      createShaderModule(device, 'particles-draw.wgsl', particlesDrawWGSL),
    ]);

    const sceneFormat = res.scene.texture.format;
    const bloomLayout = device.createPipelineLayout({
      label: 'bloom-pipeline-layout',
      bindGroupLayouts: [layouts.bloom],
    });
    const bloomPipeline = (label: string, entryPoint: string): Promise<GPUComputePipeline> =>
      device.createComputePipelineAsync({
        label,
        layout: bloomLayout,
        compute: { module: bloomModule, entryPoint },
      });

    const [scenePipeline, presentPipeline, exportPipeline, brightDown, down, blurH, blurV] =
      await Promise.all([
      device.createRenderPipelineAsync({
        label: 'scene',
        layout: device.createPipelineLayout({
          label: 'scene-pipeline-layout',
          bindGroupLayouts: [layouts.scene],
        }),
        vertex: { module: sceneModule, entryPoint: 'vs_main' },
        fragment: { module: sceneModule, entryPoint: 'fs_main', targets: [{ format: sceneFormat }] },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'present',
        layout: device.createPipelineLayout({
          label: 'present-pipeline-layout',
          bindGroupLayouts: [layouts.present],
        }),
        vertex: { module: presentModule, entryPoint: 'vs_main' },
        fragment: { module: presentModule, entryPoint: 'fs_main', targets: [{ format: targetFormat }] },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        label: 'present-export',
        layout: device.createPipelineLayout({
          label: 'present-export-pipeline-layout',
          bindGroupLayouts: [layouts.present],
        }),
        vertex: { module: presentModule, entryPoint: 'vs_main' },
        fragment: {
          module: presentModule,
          entryPoint: 'fs_main',
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      bloomPipeline('bloom-bright-down', 'bright_down'),
      bloomPipeline('bloom-down', 'down'),
      bloomPipeline('bloom-blur-h', 'blur_h'),
      bloomPipeline('bloom-blur-v', 'blur_v'),
    ]);

    // Traînées de particules : quads instanciés, blending additif dans la scène HDR.
    const particlePipeline = await device.createRenderPipelineAsync({
      label: 'particles-draw',
      layout: device.createPipelineLayout({
        label: 'particles-pipeline-layout',
        bindGroupLayouts: [layouts.particleDraw],
      }),
      vertex: { module: particlesModule, entryPoint: 'vs_main' },
      fragment: {
        module: particlesModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: sceneFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });
    const particleBind: Pair<GPUBindGroup> = [0 as PingIndex, 1 as PingIndex].map((d) =>
      device.createBindGroup({
        label: `particles-bind-d${d}`,
        layout: layouts.particleDraw,
        entries: [
          { binding: 0, resource: { buffer: res.renderUniforms } },
          { binding: 1, resource: res.linearSampler },
          { binding: 2, resource: res.density.views[d as PingIndex] },
          { binding: 3, resource: { buffer: res.particles } },
        ],
      }),
    ) as unknown as Pair<GPUBindGroup>;

    const sceneBind: Pair<Pair<Pair<GPUBindGroup>>> = pair((d) =>
      pair((v) =>
        pair((p) =>
          device.createBindGroup({
            label: `scene-bind-d${d}-v${v}-p${p}`,
            layout: layouts.scene,
            entries: [
              { binding: 0, resource: { buffer: res.renderUniforms } },
              { binding: 1, resource: res.linearSampler },
              { binding: 2, resource: res.density.views[d] },
              { binding: 3, resource: res.velocity.views[v] },
              { binding: 4, resource: res.pressure.views[p] },
              { binding: 5, resource: res.divergence.view },
              { binding: 6, resource: res.curl.view },
              { binding: 7, resource: res.camera.view },
              { binding: 8, resource: res.flow.view },
            ],
          }),
        ),
      ),
    );

    const bloomBind = (label: string, src: GPUTextureView, dst: GPUTextureView): GPUBindGroup =>
      device.createBindGroup({
        label,
        layout: layouts.bloom,
        entries: [
          { binding: 0, resource: res.linearSampler },
          { binding: 1, resource: src },
          { binding: 2, resource: dst },
        ],
      });

    // Séquence : scène → seuil+½ → flou H/V (512) → ¼ → flou H/V (256).
    const mid0 = res.bloomMid[0].view;
    const mid1 = res.bloomMid[1].view;
    const wide0 = res.bloomWide[0].view;
    const wide1 = res.bloomWide[1].view;
    const bloomSteps: readonly BloomStep[] = [
      { pipeline: brightDown, bind: bloomBind('bloom-bright', res.scene.view, mid0), dispatch: BLOOM_MID_DISPATCH },
      { pipeline: blurH, bind: bloomBind('bloom-mid-h', mid0, mid1), dispatch: BLOOM_MID_DISPATCH },
      { pipeline: blurV, bind: bloomBind('bloom-mid-v', mid1, mid0), dispatch: BLOOM_MID_DISPATCH },
      { pipeline: down, bind: bloomBind('bloom-widen', mid0, wide0), dispatch: BLOOM_WIDE_DISPATCH },
      { pipeline: blurH, bind: bloomBind('bloom-wide-h', wide0, wide1), dispatch: BLOOM_WIDE_DISPATCH },
      { pipeline: blurV, bind: bloomBind('bloom-wide-v', wide1, wide0), dispatch: BLOOM_WIDE_DISPATCH },
    ];

    const presentBind = device.createBindGroup({
      label: 'present-bind',
      layout: layouts.present,
      entries: [
        { binding: 0, resource: { buffer: res.renderUniforms } },
        { binding: 1, resource: res.linearSampler },
        { binding: 2, resource: res.scene.view },
        { binding: 3, resource: mid0 },
        { binding: 4, resource: wide0 },
        { binding: 5, resource: res.obstacle.view },
        { binding: 6, resource: res.camera.view },
        { binding: 7, resource: res.flow.view },
      ],
    });

    return new CompositeRenderer(
      scenePipeline,
      presentPipeline,
      exportPipeline,
      sceneBind,
      presentBind,
      bloomSteps,
      particlePipeline,
      particleBind,
      res.scene.view,
      res.scene.view,
    );
  }

  /** Encode la présentation vers la cible d'export (rgba8unorm, taille fixe) —
   *  utilisé par l'export PNG, en dehors de la boucle de frame. */
  encodeExport(encoder: GPUCommandEncoder, target: GPUTextureView): void {
    const pass = encoder.beginRenderPass({
      label: 'export-pass',
      colorAttachments: [
        { view: target, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    pass.setPipeline(this.exportPipeline);
    pass.setBindGroup(0, this.presentBind);
    pass.draw(3);
    pass.end();
  }

  /** Encode scène → [bloom] → présentation vers `target`. */
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    densityIndex: PingIndex,
    velocityIndex: PingIndex,
    pressureIndex: PingIndex,
    viewMode: ViewMode,
    drawParticles: boolean,
  ): void {
    const scene = encoder.beginRenderPass(this.sceneDescriptor);
    scene.setPipeline(this.scenePipeline);
    scene.setBindGroup(0, this.sceneBind[densityIndex][velocityIndex][pressureIndex]);
    scene.draw(3);
    // Traînées de particules par-dessus le fluide (vue fluides uniquement — les vues
    // de debug restent des instruments propres).
    if (drawParticles && viewMode === 0) {
      scene.setPipeline(this.particlePipeline);
      scene.setBindGroup(0, this.particleBind[densityIndex]);
      scene.draw(6, PARTICLE_COUNT);
    }
    scene.end();

    // Le bloom n'a de sens que sur la vue fluides — les vues de debug restent brutes.
    if (viewMode === 0) {
      const bloom = encoder.beginComputePass(this.bloomPassDesc);
      for (const step of this.bloomSteps) {
        bloom.setPipeline(step.pipeline);
        bloom.setBindGroup(0, step.bind);
        bloom.dispatchWorkgroups(step.dispatch, step.dispatch);
      }
      bloom.end();
    }

    this.presentAttachment.view = target;
    const present = encoder.beginRenderPass(this.presentDescriptor);
    present.setPipeline(this.presentPipeline);
    present.setBindGroup(0, this.presentBind);
    present.draw(3);
    present.end();
  }
}
