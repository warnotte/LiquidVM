/**
 * J1 — simulation d'eau FLIP/PIC (PLAN-EAU.md). Une frame :
 *   [× sous-pas] clear atomics → P2G (scatter + resolve → velOld) → densité
 *   floutée → gravité (velTmp) → divergence (cellules d'eau, contrôle de
 *   densité) → Jacobi surface libre (air p = 0) → gradient (velNew) → G2P
 *   (FLIP/PIC + advection RK2 bornée)
 *   [toutes les SORT_INTERVAL frames] tri par blocs 8³ (leçon de J0)
 *   → rendu points (projection inverse-rayons).
 * Doctrine : zéro alloc/frame, un CommandEncoder, zéro readback en boucle,
 * bind groups tous pré-créés (deux sens du ping-pong particules inclus).
 */

import simP2gWGSL from './shaders/sim_p2g.wgsl?raw';
import gridWGSL from './shaders/eau_grid.wgsl?raw';
import g2pWGSL from './shaders/eau_g2p.wgsl?raw';
import sortWGSL from './shaders/eau_sort.wgsl?raw';
import pointsWGSL from './shaders/eau_points.wgsl?raw';
import surfaceWGSL from './shaders/eau_surface.wgsl?raw';
import { EAU_DEFAULTS, GRID_EAU, particleDispatch, PARTICLES_EAU, PARTICLE_STRIDE, SCALE_EAU, SORT_BLOCKS, SORT_INTERVAL, WG_GRID } from './config_eau';
import { createShaderModule, withValidation } from '../core/pipelines';
import { flip, type Pair, type PingIndex } from '../core/types';

export interface FrameEauInput {
  dt: number;
  paused: boolean;
  reset: boolean;
  gravity: number;
  /** true = transfert APIC (J2) ; false = FLIP/PIC (flipBlend). */
  apic: boolean;
  flipBlend: number;
  /** Taux du contrôle de densité (1/s) — 0 = coupé. */
  densityRate: number;
  /** Largeur de la colonne initiale (64 = basse 64×32, 32 = haute 32×64). */
  damWidth: number;
  jacobiIterations: number;
  substeps: number;
  timeScale: number;
  pointSize: number;
  exposure: number;
  /** true = points bruts (instrument physique) sur la boîte ; false = surface. */
  renderPoints: boolean;
  absorption: number;
  /** Seuil d'iso-surface sur la densité floutée (1 = densité de repos). */
  surfaceIso: number;
  /** Mousse : blanchiment de l'eau aérée (gouttes, crêtes, nappes fines). */
  foam: number;
  /** Vue debug du rendu : 0 surface, 2 coupe z=0 de la densité, 3 densité max par rayon. */
  debugView: number;
  /** Boule-obstacle présente (J3). */
  sphereActive: boolean;
  /** Saisie de la boule : la plateforme met active à true tant que le pointeur
   *  reste enfoncé après un hitTest positif (sinon c'est une orbite). */
  grab: { active: boolean; ndcX: number; ndcY: number };
  cam: { azimuth: number; elevation: number; radius: number };
}

const COMPUTE = GPUShaderStage.COMPUTE;
const VERTEX = GPUShaderStage.VERTEX;
const FRAGMENT = GPUShaderStage.FRAGMENT;

function pair<T>(f: (i: PingIndex) => T): Pair<T> {
  return [f(0), f(1)];
}

export class FluidEau {
  private particleIdx: PingIndex = 0;
  private pressureIdx: PingIndex = 0;
  private frameIdx = 0;
  private needsInit = true;
  /** Dernier recensement lu : [valides, perdues (NaN/hors monde), rapides],
   *  [4..67] 8 particules brutes (pos+vel en bits float), [66..71] histogramme
   *  des cellules par occupation (0, 1-3, 4-7, 8-11, 12-23, 24+). */
  readonly lastCensus = new Uint32Array(72);
  private censusInFlight = false;
  private readonly uniformData = new Float32Array(36);
  /** Base caméra du dernier writeUniforms : pos(3), right(3), up(3), fwd(3),
   *  tan(fov/2), aspect — de quoi refaire le rayon du pointeur côté CPU. */
  private readonly camRay = new Float32Array(14);
  private readonly rayO = new Float32Array(3);
  private readonly rayD = new Float32Array(3);
  /** Boule-obstacle : position (voxels), vitesse (voxels/s), état de saisie. */
  private readonly spherePos: [number, number, number] = [
    GRID_EAU * EAU_DEFAULTS.sphereStart[0],
    GRID_EAU * EAU_DEFAULTS.sphereStart[1],
    GRID_EAU * EAU_DEFAULTS.sphereStart[2],
  ];
  private readonly sphereVel: [number, number, number] = [0, 0, 0];
  private grabbed = false;
  private readonly submitList: GPUCommandBuffer[] = [];

  private constructor(
    private readonly device: GPUDevice,
    private readonly uniforms: GPUBuffer,
    private readonly clearTargets: readonly GPUBuffer[],
    private readonly blockCount: GPUBuffer,
    private readonly censusBuf: GPUBuffer,
    private readonly censusStaging: GPUBuffer,
    private readonly pipelines: {
      initDam: GPUComputePipeline;
      scatter: GPUComputePipeline;
      resolve: GPUComputePipeline;
      forces: GPUComputePipeline;
      divergence: GPUComputePipeline;
      jacobi: GPUComputePipeline;
      gradient: GPUComputePipeline;
      clearPressure: GPUComputePipeline;
      g2p: GPUComputePipeline;
      census: GPUComputePipeline;
      histogram: GPUComputePipeline;
      scan: GPUComputePipeline;
      reorder: GPUComputePipeline;
      densityBlur: GPUComputePipeline;
      cellCensus: GPUComputePipeline;
      points: GPURenderPipeline;
      surface: GPURenderPipeline;
    },
    private readonly binds: {
      gridG0: GPUBindGroup; // uniform seul (module grille)
      g2pG0: GPUBindGroup; // uniform + sampler (module G2P)
      p2g: Pair<GPUBindGroup>; // [particules] — tout le module P2G
      forces: GPUBindGroup;
      divergence: GPUBindGroup;
      jacobi: Pair<GPUBindGroup>; // [pression source]
      gradient: Pair<GPUBindGroup>; // [pression finale]
      g2p: Pair<GPUBindGroup>; // [particules]
      sort: Pair<GPUBindGroup>; // [source] → destination = flip
      points: Pair<GPUBindGroup>; // [particules]
      densityBlur: GPUBindGroup;
      cellCensus: GPUBindGroup;
      surface: GPUBindGroup;
    },
  ) {}

  static async create(device: GPUDevice, targetFormat: GPUTextureFormat): Promise<FluidEau> {
    return withValidation(device, 'init-eau', async () => {
      const cells = GRID_EAU * GRID_EAU * GRID_EAU;
      const particles = pair((i) =>
        device.createBuffer({
          label: `eau-particles-${i}`,
          size: PARTICLES_EAU * 16 * PARTICLE_STRIDE,
          usage: GPUBufferUsage.STORAGE,
        }),
      );
      const atomicBuffers = ['acc-u', 'acc-v', 'acc-w', 'wgt-u', 'wgt-v', 'wgt-w'].map((name) =>
        device.createBuffer({
          label: `eau-${name}`,
          size: cells * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
      );
      const cellCount = device.createBuffer({
        label: 'eau-cell-count',
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const blockCount = device.createBuffer({
        label: 'eau-block-count',
        size: SORT_BLOCKS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const blockCursor = device.createBuffer({
        label: 'eau-block-cursor',
        size: SORT_BLOCKS * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      // Recensement (instrument J1) : 16 octets, relevés toutes les ~30 frames
      // par un readback DIAGNOSTIQUE hors chemin critique (documenté PLAN-EAU).
      const censusBuf = device.createBuffer({
        label: 'eau-census',
        size: 288,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const censusStaging = device.createBuffer({
        label: 'eau-census-staging',
        size: 288,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const tex3d = (label: string, format: GPUTextureFormat): GPUTextureView =>
        device
          .createTexture({
            label,
            dimension: '3d',
            size: { width: GRID_EAU, height: GRID_EAU, depthOrArrayLayers: GRID_EAU },
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
          })
          .createView({ label: `${label}-view` });
      const velOld = tex3d('eau-vel-old', 'rgba16float');
      const velTmp = tex3d('eau-vel-tmp', 'rgba16float');
      const velNew = tex3d('eau-vel-new', 'rgba16float');
      const pressure = pair((i) => tex3d(`eau-press-${i}`, 'r32float'));
      const divergence = tex3d('eau-div', 'r32float');
      // Densité floutée pour le rendu de surface (rgba16float : filtrable,
      // format storage de base — r16float n'est ni l'un ni l'autre).
      const density = tex3d('eau-density', 'rgba16float');
      const uniforms = device.createBuffer({
        label: 'eau-uniforms',
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const sampler = device.createSampler({
        label: 'eau-lin',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
      });

      // ---- Layouts (par passe : jamais une même texture en storage ET sampled
      // dans un même bind group — la validation d'usage compte les BINDINGS).
      const uniformEntry = (visibility: number): GPUBindGroupLayoutEntry => ({
        binding: 0,
        visibility,
        buffer: { type: 'uniform' },
      });
      const storageBuf = (binding: number, readOnly = false): GPUBindGroupLayoutEntry => ({
        binding,
        visibility: COMPUTE,
        buffer: { type: readOnly ? 'read-only-storage' : 'storage' },
      });
      const sampled = (binding: number, filterable = true): GPUBindGroupLayoutEntry => ({
        binding,
        visibility: COMPUTE,
        texture: { sampleType: filterable ? 'float' : 'unfilterable-float', viewDimension: '3d' },
      });
      const storageTex = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry => ({
        binding,
        visibility: COMPUTE,
        storageTexture: { access: 'write-only', format, viewDimension: '3d' },
      });

      const L = {
        p2g: device.createBindGroupLayout({
          label: 'eau-p2g',
          entries: [
            uniformEntry(COMPUTE),
            storageBuf(1),
            ...[2, 3, 4, 5, 6, 7].map((b) => storageBuf(b)),
            storageBuf(8),
            storageTex(9, 'rgba16float'),
          ],
        }),
        gridG0: device.createBindGroupLayout({
          label: 'eau-grid-g0',
          entries: [uniformEntry(COMPUTE)],
        }),
        forces: device.createBindGroupLayout({
          label: 'eau-forces',
          entries: [sampled(0), storageTex(1, 'rgba16float')],
        }),
        divergence: device.createBindGroupLayout({
          label: 'eau-div',
          entries: [sampled(0), storageBuf(2, true), storageTex(3, 'r32float'), sampled(7)],
        }),
        jacobi: device.createBindGroupLayout({
          label: 'eau-jacobi',
          entries: [storageBuf(2, true), sampled(4, false), storageTex(5, 'r32float'), sampled(6, false)],
        }),
        gradient: device.createBindGroupLayout({
          label: 'eau-gradient',
          entries: [sampled(0), storageTex(1, 'rgba16float'), storageBuf(2, true), sampled(4, false)],
        }),
        g2pG0: device.createBindGroupLayout({
          label: 'eau-g2p-g0',
          entries: [
            uniformEntry(COMPUTE),
            { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
          ],
        }),
        g2p: device.createBindGroupLayout({
          label: 'eau-g2p',
          entries: [storageBuf(0), sampled(1), sampled(2), storageBuf(3)],
        }),
        sort: device.createBindGroupLayout({
          label: 'eau-sort',
          entries: [storageBuf(0, true), storageBuf(1), storageBuf(2), storageBuf(3)],
        }),
        points: device.createBindGroupLayout({
          label: 'eau-points',
          entries: [
            uniformEntry(VERTEX),
            { binding: 1, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
          ],
        }),
        densityBlur: device.createBindGroupLayout({
          label: 'eau-density-blur',
          entries: [uniformEntry(COMPUTE), storageBuf(2, true), storageTex(3, 'rgba16float')],
        }),
        cellCensus: device.createBindGroupLayout({
          label: 'eau-cell-census',
          entries: [uniformEntry(COMPUTE), storageBuf(2, true), storageBuf(5)],
        }),
        surface: device.createBindGroupLayout({
          label: 'eau-surface',
          entries: [
            uniformEntry(FRAGMENT),
            { binding: 1, visibility: FRAGMENT, sampler: { type: 'filtering' } },
            { binding: 4, visibility: FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
          ],
        }),
      };

      const [p2gM, gridM, g2pM, sortM, pointsM, surfaceM] = await Promise.all([
        createShaderModule(device, 'sim_p2g.wgsl', simP2gWGSL),
        createShaderModule(device, 'eau_grid.wgsl', gridWGSL),
        createShaderModule(device, 'eau_g2p.wgsl', g2pWGSL),
        createShaderModule(device, 'eau_sort.wgsl', sortWGSL),
        createShaderModule(device, 'eau_points.wgsl', pointsWGSL),
        createShaderModule(device, 'eau_surface.wgsl', surfaceWGSL),
      ]);

      const compute = (
        label: string,
        layouts: GPUBindGroupLayout[],
        module: GPUShaderModule,
        entryPoint: string,
      ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
          label,
          layout: device.createPipelineLayout({ label: `${label}-pl`, bindGroupLayouts: layouts }),
          compute: { module, entryPoint },
        });

      const [initDam, scatter, resolve, forces, divergencePipe, jacobi, gradient, clearPressure, g2p, censusPipe, histogram, scan, reorder, densityBlur, cellCensus, points, surface] =
        await Promise.all([
          compute('eau-init-dam', [L.p2g], p2gM, 'init_dam'),
          compute('eau-scatter', [L.p2g], p2gM, 'scatter'),
          compute('eau-resolve', [L.p2g], p2gM, 'resolve'),
          compute('eau-forces', [L.gridG0, L.forces], gridM, 'forces'),
          compute('eau-divergence', [L.gridG0, L.divergence], gridM, 'divergence'),
          compute('eau-jacobi', [L.gridG0, L.jacobi], gridM, 'jacobi'),
          compute('eau-gradient', [L.gridG0, L.gradient], gridM, 'gradient'),
          compute('eau-clear-pressure', [L.gridG0, L.jacobi], gridM, 'clear_pressure'),
          compute('eau-g2p', [L.g2pG0, L.g2p], g2pM, 'g2p'),
          compute('eau-census', [L.g2pG0, L.g2p], g2pM, 'census_pass'),
          compute('eau-histogram', [L.gridG0, L.sort], sortM, 'histogram'),
          compute('eau-scan', [L.gridG0, L.sort], sortM, 'scan'),
          compute('eau-reorder', [L.gridG0, L.sort], sortM, 'reorder'),
          compute('eau-density-blur', [L.densityBlur], surfaceM, 'density_blur'),
          compute('eau-cell-census', [L.cellCensus], surfaceM, 'cell_census'),
          device.createRenderPipelineAsync({
            label: 'eau-points',
            layout: device.createPipelineLayout({ label: 'eau-points-pl', bindGroupLayouts: [L.points] }),
            vertex: { module: pointsM, entryPoint: 'vs_points' },
            fragment: {
              module: pointsM,
              entryPoint: 'fs_points',
              targets: [
                {
                  format: targetFormat,
                  blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                  },
                },
              ],
            },
            primitive: { topology: 'triangle-list' },
          }),
          device.createRenderPipelineAsync({
            label: 'eau-surface',
            layout: device.createPipelineLayout({ label: 'eau-surface-pl', bindGroupLayouts: [L.surface] }),
            vertex: { module: surfaceM, entryPoint: 'vs_full' },
            fragment: { module: surfaceM, entryPoint: 'fs_surface', targets: [{ format: targetFormat }] },
            primitive: { topology: 'triangle-list' },
          }),
        ]);

      const binds = {
        gridG0: device.createBindGroup({
          label: 'eau-grid-g0',
          layout: L.gridG0,
          entries: [{ binding: 0, resource: { buffer: uniforms } }],
        }),
        g2pG0: device.createBindGroup({
          label: 'eau-g2p-g0',
          layout: L.g2pG0,
          entries: [
            { binding: 0, resource: { buffer: uniforms } },
            { binding: 1, resource: sampler },
          ],
        }),
        p2g: pair((i) =>
          device.createBindGroup({
            label: `eau-p2g-${i}`,
            layout: L.p2g,
            entries: [
              { binding: 0, resource: { buffer: uniforms } },
              { binding: 1, resource: { buffer: particles[i] } },
              ...atomicBuffers.map((buffer, k) => ({ binding: 2 + k, resource: { buffer } })),
              { binding: 8, resource: { buffer: cellCount } },
              { binding: 9, resource: velOld },
            ],
          }),
        ),
        forces: device.createBindGroup({
          label: 'eau-forces',
          layout: L.forces,
          entries: [
            { binding: 0, resource: velOld },
            { binding: 1, resource: velTmp },
          ],
        }),
        divergence: device.createBindGroup({
          label: 'eau-div',
          layout: L.divergence,
          entries: [
            { binding: 0, resource: velTmp },
            { binding: 2, resource: { buffer: cellCount } },
            { binding: 3, resource: divergence },
            { binding: 7, resource: density },
          ],
        }),
        jacobi: pair((p) =>
          device.createBindGroup({
            label: `eau-jacobi-${p}`,
            layout: L.jacobi,
            entries: [
              { binding: 2, resource: { buffer: cellCount } },
              { binding: 4, resource: pressure[p] },
              { binding: 5, resource: pressure[flip(p)] },
              { binding: 6, resource: divergence },
            ],
          }),
        ),
        gradient: pair((p) =>
          device.createBindGroup({
            label: `eau-gradient-${p}`,
            layout: L.gradient,
            entries: [
              { binding: 0, resource: velTmp },
              { binding: 1, resource: velNew },
              { binding: 2, resource: { buffer: cellCount } },
              { binding: 4, resource: pressure[p] },
            ],
          }),
        ),
        g2p: pair((i) =>
          device.createBindGroup({
            label: `eau-g2p-${i}`,
            layout: L.g2p,
            entries: [
              { binding: 0, resource: { buffer: particles[i] } },
              { binding: 1, resource: velNew },
              { binding: 2, resource: velOld },
              { binding: 3, resource: { buffer: censusBuf } },
            ],
          }),
        ),
        sort: pair((i) =>
          device.createBindGroup({
            label: `eau-sort-${i}`,
            layout: L.sort,
            entries: [
              { binding: 0, resource: { buffer: particles[i] } },
              { binding: 1, resource: { buffer: particles[flip(i)] } },
              { binding: 2, resource: { buffer: blockCount } },
              { binding: 3, resource: { buffer: blockCursor } },
            ],
          }),
        ),
        points: pair((i) =>
          device.createBindGroup({
            label: `eau-points-${i}`,
            layout: L.points,
            entries: [
              { binding: 0, resource: { buffer: uniforms } },
              { binding: 1, resource: { buffer: particles[i] } },
            ],
          }),
        ),
        densityBlur: device.createBindGroup({
          label: 'eau-density-blur',
          layout: L.densityBlur,
          entries: [
            { binding: 0, resource: { buffer: uniforms } },
            { binding: 2, resource: { buffer: cellCount } },
            { binding: 3, resource: density },
          ],
        }),
        cellCensus: device.createBindGroup({
          label: 'eau-cell-census',
          layout: L.cellCensus,
          entries: [
            { binding: 0, resource: { buffer: uniforms } },
            { binding: 2, resource: { buffer: cellCount } },
            { binding: 5, resource: { buffer: censusBuf } },
          ],
        }),
        surface: device.createBindGroup({
          label: 'eau-surface',
          layout: L.surface,
          entries: [
            { binding: 0, resource: { buffer: uniforms } },
            { binding: 1, resource: sampler },
            { binding: 4, resource: density },
          ],
        }),
      };

      return new FluidEau(
        device,
        uniforms,
        [...atomicBuffers, cellCount],
        blockCount,
        censusBuf,
        censusStaging,
        { initDam, scatter, resolve, forces, divergence: divergencePipe, jacobi, gradient, clearPressure, g2p, census: censusPipe, histogram, scan, reorder, densityBlur, cellCensus, points, surface },
        binds,
      );
    });
  }

  frame(input: FrameEauInput, target: GPUTextureView, width: number, height: number): void {
    const aspect = width / Math.max(height, 1);
    const dt = Math.min(Math.max(input.dt, 0), 1 / 30) * input.timeScale;
    const substeps = Math.max(1, Math.round(input.substeps));
    const running = !input.paused && dt > 0;
    if (input.reset || this.needsInit) {
      this.spherePos[0] = GRID_EAU * EAU_DEFAULTS.sphereStart[0];
      this.spherePos[1] = GRID_EAU * EAU_DEFAULTS.sphereStart[1];
      this.spherePos[2] = GRID_EAU * EAU_DEFAULTS.sphereStart[2];
      this.sphereVel[0] = 0;
      this.sphereVel[1] = 0;
      this.sphereVel[2] = 0;
    }
    // La boule est déplacée AVANT l'écriture des uniforms : le bord mobile de
    // cette frame correspond au mouvement de cette frame.
    this.updateSphere(input, dt);
    this.writeUniforms(input, dt / substeps, aspect);

    const encoder = this.device.createCommandEncoder({ label: 'frame-eau' });
    const gridDispatch = Math.ceil(GRID_EAU / WG_GRID);
    const [pdx, pdy] = particleDispatch(PARTICLES_EAU);

    if (input.reset || this.needsInit) {
      this.needsInit = false;
      const pass = encoder.beginComputePass({ label: 'eau-init' });
      pass.setPipeline(this.pipelines.initDam);
      pass.setBindGroup(0, this.binds.p2g[this.particleIdx]);
      pass.dispatchWorkgroups(pdx, pdy);
      // Purge du warm start de pression : l'ancien champ kickait les
      // particules (> 550 voxels/s) dès la première frame après un reset.
      pass.setPipeline(this.pipelines.clearPressure);
      pass.setBindGroup(0, this.binds.gridG0);
      for (const p of [0, 1] as const) {
        pass.setBindGroup(1, this.binds.jacobi[p]);
        pass.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
      }
      pass.end();
    }

    if (running) {
      for (let s = 0; s < substeps; s++) {
        for (const buffer of this.clearTargets) {
          encoder.clearBuffer(buffer);
        }
        const cp = encoder.beginComputePass({ label: 'eau-substep' });
        // P2G : particules → accumulateurs → velOld.
        cp.setPipeline(this.pipelines.scatter);
        cp.setBindGroup(0, this.binds.p2g[this.particleIdx]);
        cp.dispatchWorkgroups(pdx, pdy);
        cp.setPipeline(this.pipelines.resolve);
        cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
        // Densité floutée : lue par le contrôle de densité ET par le rendu.
        cp.setPipeline(this.pipelines.densityBlur);
        cp.setBindGroup(0, this.binds.densityBlur);
        cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
        // Grille : gravité, divergence, pression, gradient.
        cp.setBindGroup(0, this.binds.gridG0);
        cp.setPipeline(this.pipelines.forces);
        cp.setBindGroup(1, this.binds.forces);
        cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
        cp.setPipeline(this.pipelines.divergence);
        cp.setBindGroup(1, this.binds.divergence);
        cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
        cp.setPipeline(this.pipelines.jacobi);
        const iters = Math.max(4, Math.round(input.jacobiIterations));
        for (let j = 0; j < iters; j++) {
          cp.setBindGroup(1, this.binds.jacobi[this.pressureIdx]);
          cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
          this.pressureIdx = flip(this.pressureIdx);
        }
        cp.setPipeline(this.pipelines.gradient);
        cp.setBindGroup(1, this.binds.gradient[this.pressureIdx]);
        cp.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
        // G2P : FLIP/PIC + advection.
        cp.setPipeline(this.pipelines.g2p);
        cp.setBindGroup(0, this.binds.g2pG0);
        cp.setBindGroup(1, this.binds.g2p[this.particleIdx]);
        cp.dispatchWorkgroups(pdx, pdy);
        cp.end();
      }

      // Tri périodique par blocs (leçon de J0) : histogramme → scan → réordonner.
      this.frameIdx++;
      if (this.frameIdx % SORT_INTERVAL === 0) {
        encoder.clearBuffer(this.blockCount);
        const sp = encoder.beginComputePass({ label: 'eau-sort' });
        sp.setBindGroup(0, this.binds.gridG0);
        sp.setBindGroup(1, this.binds.sort[this.particleIdx]);
        sp.setPipeline(this.pipelines.histogram);
        sp.dispatchWorkgroups(pdx, pdy);
        sp.setPipeline(this.pipelines.scan);
        sp.dispatchWorkgroups(1);
        sp.setPipeline(this.pipelines.reorder);
        sp.dispatchWorkgroups(pdx, pdy);
        sp.end();
        this.particleIdx = flip(this.particleIdx);
      }
    }

    // Recensement diagnostique : passe GPU + readback 16 o toutes les 30 frames
    // (hors chemin critique — l'instrument du jalon, voir PLAN-EAU).
    const doCensus = this.frameIdx % 30 === 0 && !this.censusInFlight;
    if (doCensus) {
      encoder.clearBuffer(this.censusBuf);
      const cpn = encoder.beginComputePass({ label: 'eau-census' });
      cpn.setPipeline(this.pipelines.census);
      cpn.setBindGroup(0, this.binds.g2pG0);
      cpn.setBindGroup(1, this.binds.g2p[this.particleIdx]);
      cpn.dispatchWorkgroups(pdx, pdy);
      cpn.setPipeline(this.pipelines.cellCensus);
      cpn.setBindGroup(0, this.binds.cellCensus);
      cpn.dispatchWorkgroups(gridDispatch, gridDispatch, gridDispatch);
      cpn.end();
      encoder.copyBufferToBuffer(this.censusBuf, 0, this.censusStaging, 0, 288);
    }

    // Boîte + surface (densité floutée du dernier sous-pas — valable aussi en
    // pause) ; en mode points, la boîte seule et les points par-dessus.
    const sp = encoder.beginRenderPass({
      label: 'eau-surface',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    sp.setPipeline(this.pipelines.surface);
    sp.setBindGroup(0, this.binds.surface);
    sp.draw(3);
    sp.end();

    if (input.renderPoints) {
      const rp = encoder.beginRenderPass({
        label: 'eau-points',
        colorAttachments: [{ view: target, loadOp: 'load', storeOp: 'store' }],
      });
      rp.setPipeline(this.pipelines.points);
      rp.setBindGroup(0, this.binds.points[this.particleIdx]);
      rp.draw(6, PARTICLES_EAU);
      rp.end();
    }

    this.submitList[0] = encoder.finish();
    this.device.queue.submit(this.submitList);

    if (doCensus) {
      this.censusInFlight = true;
      this.censusStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          this.lastCensus.set(new Uint32Array(this.censusStaging.getMappedRange()));
          this.censusStaging.unmap();
          this.censusInFlight = false;
        })
        .catch(() => {
          this.censusInFlight = false;
        });
    }
  }

  /** Rayon caméra→scène pour un point NDC : origine en VOXELS, direction unitaire
   *  (même construction que le raymarch, à partir de la base caméra mémorisée). */
  private computeRay(ndcX: number, ndcY: number): void {
    const c = this.camRay;
    const tanf = c[12]!;
    const aspect = c[13]!;
    const dx = c[9]! + c[3]! * ndcX * tanf * aspect + c[6]! * ndcY * tanf;
    const dy = c[10]! + c[4]! * ndcX * tanf * aspect + c[7]! * ndcY * tanf;
    const dz = c[11]! + c[5]! * ndcX * tanf * aspect + c[8]! * ndcY * tanf;
    const dl = Math.hypot(dx, dy, dz) || 1;
    this.rayD[0] = dx / dl;
    this.rayD[1] = dy / dl;
    this.rayD[2] = dz / dl;
    this.rayO[0] = (c[0]! + 0.5) * GRID_EAU;
    this.rayO[1] = (c[1]! + 0.5) * GRID_EAU;
    this.rayO[2] = (c[2]! + 0.5) * GRID_EAU;
  }

  /** Distance du centre de la boule au rayon courant (+∞ si elle est derrière). */
  private ballRayDistance(): number {
    const vx = this.spherePos[0] - this.rayO[0]!;
    const vy = this.spherePos[1] - this.rayO[1]!;
    const vz = this.spherePos[2] - this.rayO[2]!;
    const t = vx * this.rayD[0]! + vy * this.rayD[1]! + vz * this.rayD[2]!;
    if (t <= 0) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.hypot(vx - t * this.rayD[0]!, vy - t * this.rayD[1]!, vz - t * this.rayD[2]!);
  }

  /** Le pointeur est-il sur la boule ? (la plateforme décide saisie vs orbite) */
  hitTest(ndcX: number, ndcY: number): boolean {
    this.computeRay(ndcX, ndcY);
    return this.ballRayDistance() < EAU_DEFAULTS.sphereRadius * GRID_EAU * 1.15;
  }

  /** Saisie : la boule suit le pointeur sur le plan face caméra passant par son
   *  centre, et communique sa vitesse au fluide (bord mobile) — lissée EMA et
   *  plafonnée CFL, amortie dès qu'on lâche (comme la boule du feu). */
  private updateSphere(input: FrameEauInput, dt: number): void {
    if (input.grab.active) {
      this.computeRay(input.grab.ndcX, input.grab.ndcY);
      if (!this.grabbed) {
        this.grabbed = this.ballRayDistance() < EAU_DEFAULTS.sphereRadius * GRID_EAU * 1.15;
      }
    } else {
      this.grabbed = false;
    }
    if (this.grabbed && input.sphereActive) {
      const c = this.camRay;
      const before: [number, number, number] = [this.spherePos[0], this.spherePos[1], this.spherePos[2]];
      const denom = c[9]! * this.rayD[0]! + c[10]! * this.rayD[1]! + c[11]! * this.rayD[2]!;
      if (Math.abs(denom) > 1e-5) {
        const t =
          (c[9]! * (before[0] - this.rayO[0]!) +
            c[10]! * (before[1] - this.rayO[1]!) +
            c[11]! * (before[2] - this.rayO[2]!)) /
          denom;
        if (t > 0) {
          // La boule reste entièrement dans la boîte (rayon + un demi-voxel).
          const m = EAU_DEFAULTS.sphereRadius * GRID_EAU + 0.5;
          for (let a = 0; a < 3; a++) {
            const v = this.rayO[a]! + t * this.rayD[a]!;
            this.spherePos[a] = Math.min(Math.max(v, m), GRID_EAU - m);
          }
        }
      }
      if (dt > 0) {
        const cap = EAU_DEFAULTS.sphereSpeedCap * SCALE_EAU;
        for (let a = 0; a < 3; a++) {
          this.sphereVel[a] = 0.55 * this.sphereVel[a]! + (0.45 * (this.spherePos[a]! - before[a]!)) / dt;
        }
        const mag = Math.hypot(this.sphereVel[0], this.sphereVel[1], this.sphereVel[2]);
        if (mag > cap) {
          const k = cap / mag;
          this.sphereVel[0] *= k;
          this.sphereVel[1] *= k;
          this.sphereVel[2] *= k;
        }
      }
    } else {
      this.sphereVel[0] *= 0.82;
      this.sphereVel[1] *= 0.82;
      this.sphereVel[2] *= 0.82;
    }
  }

  private writeUniforms(input: FrameEauInput, dtSub: number, aspect: number): void {
    const d = this.uniformData;
    d[0] = GRID_EAU;
    d[1] = PARTICLES_EAU;
    d[2] = dtSub;
    // Grandeurs en voxels/s : elles suivent la finesse de la grille.
    d[3] = input.gravity * SCALE_EAU;
    d[4] = input.flipBlend;
    d[5] = input.damWidth;
    d[6] = input.apic ? 1 : 0;
    d[7] = input.densityRate;
    // Caméra orbitale (même construction que le feu).
    const { azimuth, elevation, radius } = input.cam;
    const cy = Math.cos(elevation);
    const px = radius * cy * Math.cos(azimuth);
    const py = radius * Math.sin(elevation);
    const pz = radius * cy * Math.sin(azimuth);
    const fl = Math.hypot(px, py, pz);
    const fx = -px / fl;
    const fy = -py / fl;
    const fz = -pz / fl;
    let rx = -fz;
    let rz = fx;
    const rl = Math.hypot(rx, rz) || 1;
    rx /= rl;
    rz /= rl;
    const ux = -rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy;
    d[8] = px;
    d[9] = py;
    d[10] = pz;
    d[11] = 0.364; // tan(fov/2)
    d[12] = rx;
    d[13] = 0;
    d[14] = rz;
    d[15] = aspect;
    d[16] = ux;
    d[17] = uy;
    d[18] = uz;
    d[19] = input.exposure;
    d[20] = fx;
    d[21] = fy;
    d[22] = fz;
    d[23] = input.pointSize;
    d[24] = input.renderPoints ? 1 : input.debugView;
    d[25] = input.absorption;
    d[26] = input.surfaceIso;
    d[27] = input.foam;
    d[28] = this.spherePos[0];
    d[29] = this.spherePos[1];
    d[30] = this.spherePos[2];
    d[31] = input.sphereActive ? EAU_DEFAULTS.sphereRadius * GRID_EAU : 0;
    d[32] = this.sphereVel[0];
    d[33] = this.sphereVel[1];
    d[34] = this.sphereVel[2];
    // Base caméra mémorisée pour le rayon du pointeur (hitTest / saisie).
    const c = this.camRay;
    c[0] = d[8]!;
    c[1] = d[9]!;
    c[2] = d[10]!;
    c[3] = d[12]!;
    c[4] = d[13]!;
    c[5] = d[14]!;
    c[6] = d[16]!;
    c[7] = d[17]!;
    c[8] = d[18]!;
    c[9] = d[20]!;
    c[10] = d[21]!;
    c[11] = d[22]!;
    c[12] = d[11]!;
    c[13] = aspect;
    this.device.queue.writeBuffer(this.uniforms, 0, d);
  }
}
