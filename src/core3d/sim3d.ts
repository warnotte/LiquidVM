/**
 * Prototype 3D volumétrique : solveur MAC 3D (advection semi-lagrangienne, poussée
 * thermique, projection Jacobi — le multigrid 3D est le chantier suivant) + rendu
 * ray-marching. Même doctrine que le 2D : zéro DOM ici, zéro allocation par frame,
 * zéro lecture GPU→CPU, un seul CommandEncoder, bind groups tous pré-créés.
 * La plateforme fournit device, format cible, dt et l'état caméra abstrait.
 */

import advectVelWGSL from './shaders3d/advect_velocity3d.wgsl?raw';
import advectDenWGSL from './shaders3d/advect_density3d.wgsl?raw';
import forcesWGSL from './shaders3d/forces3d.wgsl?raw';
import vorticityWGSL from './shaders3d/vorticity3d.wgsl?raw';
import projectWGSL from './shaders3d/project3d.wgsl?raw';
import multigridWGSL from './shaders3d/multigrid3d.wgsl?raw';
import raymarchWGSL from './shaders3d/raymarch.wgsl?raw';
import clearWGSL from './shaders3d/clear3d.wgsl?raw';
import {
  DISPATCH3,
  GRID3,
  MG3_COARSE_SMOOTH,
  MG3_COARSEST_SIZE,
  MG3_PRE_SMOOTH,
  MG3_POST_SMOOTH,
  SIM3_DEFAULTS,
  WG3,
} from './config3d';
import { createShaderModule, withValidation } from '../core/pipelines';
import { flip, type Pair, type PingIndex } from '../core/types';

export interface Frame3DInput {
  /** Pas de temps en secondes (déjà borné par la plateforme). */
  dt: number;
  paused: boolean;
  /** Vidange des champs (consommé par la frame courante). */
  reset: boolean;
  /** Solveur de pression : V-cycles multigrid (défaut) ou Jacobi simple. */
  multigrid: boolean;
  vcycles: number;
  jacobiIterations: number;
  /** Force du vorticity confinement ε (0 = physique brute), réglable à chaud. */
  vorticityStrength: number;
  /** Caméra orbitale autour du centre de la boîte. */
  cam: { azimuth: number; elevation: number; radius: number };
  exposure: number;
  raymarchSteps: number;
}

const COMPUTE = GPUShaderStage.COMPUTE;
const FRAGMENT = GPUShaderStage.FRAGMENT;

function tex3d(
  device: GPUDevice,
  label: string,
  format: GPUTextureFormat,
  size = GRID3,
): GPUTextureView {
  return device
    .createTexture({
      label,
      dimension: '3d',
      size: { width: size, height: size, depthOrArrayLayers: size },
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    })
    .createView({ label: `${label}-view` });
}

/** Un niveau de la pyramide multigrid 3D (le niveau 0 réutilise pression/divergence). */
interface MGLevel3 {
  readonly dispatch: number;
  /** Lissage pondéré, indexé par [source de pression du niveau]. */
  readonly smoothBind: Pair<GPUBindGroup>;
  /** r = rhs − A·p → texture résidu. Null au niveau le plus grossier. */
  readonly residualBind: Pair<GPUBindGroup> | null;
  /** Résidu de ce niveau → rhs du suivant. Null au plus grossier. */
  readonly restrictBind: GPUBindGroup | null;
  /** Correction du suivant → ce niveau, [pression fine][pression grossière]. */
  readonly prolongBind: Pair<Pair<GPUBindGroup>> | null;
  /** Remise à zéro de la pression de départ (niveaux > 0). */
  readonly clearBind: GPUBindGroup | null;
}

function sampled3d(binding: number, visibility = COMPUTE): GPUBindGroupLayoutEntry {
  return { binding, visibility, texture: { sampleType: 'float', viewDimension: '3d' } };
}

function sampledScalar3d(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } };
}

function storage3d(binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: COMPUTE,
    storageTexture: { access: 'write-only', format, viewDimension: '3d' },
  };
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export class FluidSim3D {
  private velIdx: PingIndex = 0;
  private denIdx: PingIndex = 0;
  private pressIdx: PingIndex = 0;
  private simTime = 0;

  private readonly simData = new Float32Array(16);
  private readonly renderData = new Float32Array(20);
  private lastRender = new Float32Array(20).fill(Number.NaN);
  private readonly submitList: GPUCommandBuffer[] = [new Uint8Array(0) as unknown as GPUCommandBuffer];

  private constructor(
    private readonly device: GPUDevice,
    private readonly simUniforms: GPUBuffer,
    private readonly renderUniforms: GPUBuffer,
    private readonly group0: GPUBindGroup,
    private readonly pipelines: {
      velPredict: GPUComputePipeline;
      velCorrect: GPUComputePipeline;
      forces: GPUComputePipeline;
      curl: GPUComputePipeline;
      confine: GPUComputePipeline;
      denPredict: GPUComputePipeline;
      denCorrect: GPUComputePipeline;
      divergence: GPUComputePipeline;
      jacobi: GPUComputePipeline;
      gradient: GPUComputePipeline;
      mgSmooth: GPUComputePipeline;
      mgResidual: GPUComputePipeline;
      mgRestrict: GPUComputePipeline;
      mgProlong: GPUComputePipeline;
      clearRgba: GPUComputePipeline;
      clearScalar: GPUComputePipeline;
      render: GPURenderPipeline;
    },
    private readonly binds: {
      velPredict: Pair<GPUBindGroup>; // [vel] → scratch
      velCorrect: Pair<GPUBindGroup>; // [vel] (+scratch) → flip(vel)
      forces: Pair<Pair<GPUBindGroup>>; // [vel][den]
      curl: Pair<GPUBindGroup>; // [vel] → curl
      confine: Pair<GPUBindGroup>; // [vel] (+curl) → flip(vel)
      denPredict: Pair<Pair<GPUBindGroup>>; // [den][vel] → scratch
      denCorrect: Pair<Pair<GPUBindGroup>>; // [den][vel] (+scratch) → flip(den)
      divergence: Pair<GPUBindGroup>; // [vel]
      jacobi: Pair<GPUBindGroup>; // [press]
      gradient: Pair<Pair<GPUBindGroup>>; // [press][vel]
      render: Pair<GPUBindGroup>; // [den]
      clearsRgba: readonly GPUBindGroup[];
      clearsScalar: readonly GPUBindGroup[];
    },
    private readonly mgLevels: readonly MGLevel3[],
  ) {
    this.mgIdx = new Array<number>(mgLevels.length).fill(0);
  }

  /** Index de pression courant par niveau multigrid (tableau réutilisé, zéro alloc). */
  private readonly mgIdx: number[];

  static async create(device: GPUDevice, targetFormat: GPUTextureFormat): Promise<FluidSim3D> {
    return withValidation(device, 'init-3d', async () => {
      const velocity = pair((i) => tex3d(device, `vel3d-${i}`, 'rgba16float'));
      const density = pair((i) => tex3d(device, `den3d-${i}`, 'rgba16float'));
      const pressure = pair((i) => tex3d(device, `press3d-${i}`, 'r32float'));
      const divergence = tex3d(device, 'div3d', 'r32float');
      // MacCormack : prédicteurs φ̂ ; vorticité : rotationnel vectoriel aux centres.
      const velScratch = tex3d(device, 'vel3d-hat', 'rgba16float');
      const denScratch = tex3d(device, 'den3d-hat', 'rgba16float');
      const curlTex = tex3d(device, 'curl3d', 'rgba16float');

      const sampler = device.createSampler({
        label: 'lin3d',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
      });
      const simUniforms = device.createBuffer({
        label: 'sim3d-uniforms',
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const renderUniforms = device.createBuffer({
        label: 'render3d-uniforms',
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const L = {
        group0: device.createBindGroupLayout({
          label: 'g0-3d',
          entries: [
            { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
          ],
        }),
        velPredict: device.createBindGroupLayout({
          label: 'vel-predict-3d',
          entries: [sampled3d(0), storage3d(2, 'rgba16float')],
        }),
        velCorrect: device.createBindGroupLayout({
          label: 'vel-correct-3d',
          entries: [sampled3d(0), sampled3d(1), storage3d(2, 'rgba16float')],
        }),
        forces: device.createBindGroupLayout({
          label: 'forces-3d',
          entries: [sampled3d(0), sampled3d(1), storage3d(2, 'rgba16float')],
        }),
        curl: device.createBindGroupLayout({
          label: 'curl-3d',
          entries: [sampled3d(0), storage3d(1, 'rgba16float')],
        }),
        confine: device.createBindGroupLayout({
          label: 'confine-3d',
          entries: [sampled3d(0), sampled3d(2), storage3d(3, 'rgba16float')],
        }),
        denPredict: device.createBindGroupLayout({
          label: 'den-predict-3d',
          entries: [sampled3d(0), sampled3d(1), storage3d(3, 'rgba16float')],
        }),
        denCorrect: device.createBindGroupLayout({
          label: 'den-correct-3d',
          entries: [sampled3d(0), sampled3d(1), sampled3d(2), storage3d(3, 'rgba16float')],
        }),
        divergence: device.createBindGroupLayout({
          label: 'divergence-3d',
          entries: [sampled3d(0), storage3d(3, 'r32float')],
        }),
        jacobi: device.createBindGroupLayout({
          label: 'jacobi-3d',
          entries: [sampledScalar3d(1), sampledScalar3d(2), storage3d(3, 'r32float')],
        }),
        gradient: device.createBindGroupLayout({
          label: 'gradient-3d',
          entries: [sampled3d(0), sampledScalar3d(1), storage3d(4, 'rgba16float')],
        }),
        mgRestrict: device.createBindGroupLayout({
          label: 'mg-restrict-3d',
          entries: [sampledScalar3d(0), storage3d(3, 'r32float')],
        }),
        mgProlong: device.createBindGroupLayout({
          label: 'mg-prolong-3d',
          entries: [sampledScalar3d(0), storage3d(3, 'r32float'), sampledScalar3d(4)],
        }),
        clearRgba: device.createBindGroupLayout({
          label: 'clear-rgba-3d',
          entries: [storage3d(0, 'rgba16float')],
        }),
        clearScalar: device.createBindGroupLayout({
          label: 'clear-scalar-3d',
          entries: [storage3d(1, 'r32float')],
        }),
        render: device.createBindGroupLayout({
          label: 'render-3d',
          entries: [
            { binding: 0, visibility: FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: FRAGMENT, sampler: { type: 'filtering' } },
            sampled3d(2, FRAGMENT),
          ],
        }),
      };

      const [advectVelM, advectDenM, forcesM, vorticityM, projectM, raymarchM, clearM] =
        await Promise.all([
          createShaderModule(device, 'advect_velocity3d.wgsl', advectVelWGSL),
          createShaderModule(device, 'advect_density3d.wgsl', advectDenWGSL),
          createShaderModule(device, 'forces3d.wgsl', forcesWGSL),
          createShaderModule(device, 'vorticity3d.wgsl', vorticityWGSL),
          createShaderModule(device, 'project3d.wgsl', projectWGSL),
          createShaderModule(device, 'raymarch.wgsl', raymarchWGSL),
          createShaderModule(device, 'clear3d.wgsl', clearWGSL),
        ]);
      const multigridM = await createShaderModule(device, 'multigrid3d.wgsl', multigridWGSL);

      const compute = (
        label: string,
        group1: GPUBindGroupLayout | null,
        module: GPUShaderModule,
        entryPoint: string,
      ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
          label,
          layout: device.createPipelineLayout({
            label: `${label}-pl`,
            bindGroupLayouts: group1 === null ? [L.group0] : [L.group0, group1],
          }),
          compute: { module, entryPoint },
        });

      const [velPredict, velCorrect, forces, curlPipe, confine, denPredict, denCorrect, divergencePipe, jacobi, gradient, mgSmooth, mgResidual, mgRestrictPipe, mgProlongPipe, clearRgba, clearScalar, render] =
        await Promise.all([
          compute('vel-predict-3d', L.velPredict, advectVelM, 'predict'),
          compute('vel-correct-3d', L.velCorrect, advectVelM, 'correct'),
          compute('forces-3d', L.forces, forcesM, 'main'),
          compute('curl-3d', L.curl, vorticityM, 'curl'),
          compute('confine-3d', L.confine, vorticityM, 'confine'),
          compute('den-predict-3d', L.denPredict, advectDenM, 'predict'),
          compute('den-correct-3d', L.denCorrect, advectDenM, 'correct'),
          compute('divergence-3d', L.divergence, projectM, 'divergence'),
          compute('jacobi-3d', L.jacobi, projectM, 'jacobi'),
          compute('gradient-3d', L.gradient, projectM, 'gradient'),
          compute('mg-smooth-3d', L.jacobi, multigridM, 'smooth_jacobi'),
          compute('mg-residual-3d', L.jacobi, multigridM, 'residual'),
          compute('mg-restrict-3d', L.mgRestrict, multigridM, 'restrict_rhs'),
          compute('mg-prolong-3d', L.mgProlong, multigridM, 'prolong_add'),
          device.createComputePipelineAsync({
            label: 'clear-rgba-3d',
            layout: device.createPipelineLayout({ label: 'clear-rgba-3d-pl', bindGroupLayouts: [L.clearRgba] }),
            compute: { module: clearM, entryPoint: 'clear_rgba' },
          }),
          device.createComputePipelineAsync({
            label: 'clear-scalar-3d',
            layout: device.createPipelineLayout({ label: 'clear-scalar-3d-pl', bindGroupLayouts: [L.clearScalar] }),
            compute: { module: clearM, entryPoint: 'clear_scalar' },
          }),
          device.createRenderPipelineAsync({
            label: 'raymarch-3d',
            layout: device.createPipelineLayout({ label: 'raymarch-3d-pl', bindGroupLayouts: [L.render] }),
            vertex: { module: raymarchM, entryPoint: 'vs_main' },
            fragment: { module: raymarchM, entryPoint: 'fs_main', targets: [{ format: targetFormat }] },
            primitive: { topology: 'triangle-list' },
          }),
        ]);

      const group0 = device.createBindGroup({
        label: 'g0-3d-bind',
        layout: L.group0,
        entries: [
          { binding: 0, resource: { buffer: simUniforms } },
          { binding: 1, resource: sampler },
        ],
      });

      const binds = {
        velPredict: pair((v) =>
          device.createBindGroup({
            label: `vel-predict-3d-${v}`,
            layout: L.velPredict,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 2, resource: velScratch },
            ],
          }),
        ),
        velCorrect: pair((v) =>
          device.createBindGroup({
            label: `vel-correct-3d-${v}`,
            layout: L.velCorrect,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 1, resource: velScratch },
              { binding: 2, resource: velocity[flip(v)] },
            ],
          }),
        ),
        forces: pair((v) =>
          pair((d) =>
            device.createBindGroup({
              label: `forces-3d-v${v}-d${d}`,
              layout: L.forces,
              entries: [
                { binding: 0, resource: velocity[v] },
                { binding: 1, resource: density[d] },
                { binding: 2, resource: velocity[flip(v)] },
              ],
            }),
          ),
        ),
        curl: pair((v) =>
          device.createBindGroup({
            label: `curl-3d-${v}`,
            layout: L.curl,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 1, resource: curlTex },
            ],
          }),
        ),
        confine: pair((v) =>
          device.createBindGroup({
            label: `confine-3d-${v}`,
            layout: L.confine,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 2, resource: curlTex },
              { binding: 3, resource: velocity[flip(v)] },
            ],
          }),
        ),
        denPredict: pair((d) =>
          pair((v) =>
            device.createBindGroup({
              label: `den-predict-3d-d${d}-v${v}`,
              layout: L.denPredict,
              entries: [
                { binding: 0, resource: density[d] },
                { binding: 1, resource: velocity[v] },
                { binding: 3, resource: denScratch },
              ],
            }),
          ),
        ),
        denCorrect: pair((d) =>
          pair((v) =>
            device.createBindGroup({
              label: `den-correct-3d-d${d}-v${v}`,
              layout: L.denCorrect,
              entries: [
                { binding: 0, resource: density[d] },
                { binding: 1, resource: velocity[v] },
                { binding: 2, resource: denScratch },
                { binding: 3, resource: density[flip(d)] },
              ],
            }),
          ),
        ),
        divergence: pair((v) =>
          device.createBindGroup({
            label: `divergence-3d-${v}`,
            layout: L.divergence,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 3, resource: divergence },
            ],
          }),
        ),
        jacobi: pair((p) =>
          device.createBindGroup({
            label: `jacobi-3d-${p}`,
            layout: L.jacobi,
            entries: [
              { binding: 1, resource: pressure[p] },
              { binding: 2, resource: divergence },
              { binding: 3, resource: pressure[flip(p)] },
            ],
          }),
        ),
        gradient: pair((p) =>
          pair((v) =>
            device.createBindGroup({
              label: `gradient-3d-p${p}-v${v}`,
              layout: L.gradient,
              entries: [
                { binding: 0, resource: velocity[v] },
                { binding: 1, resource: pressure[p] },
                { binding: 4, resource: velocity[flip(v)] },
              ],
            }),
          ),
        ),
        render: pair((d) =>
          device.createBindGroup({
            label: `render-3d-${d}`,
            layout: L.render,
            entries: [
              { binding: 0, resource: { buffer: renderUniforms } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: density[d] },
            ],
          }),
        ),
        clearsRgba: [velocity[0], velocity[1], density[0], density[1]].map((view, i) =>
          device.createBindGroup({
            label: `clear-rgba-3d-${i}`,
            layout: L.clearRgba,
            entries: [{ binding: 0, resource: view }],
          }),
        ),
        clearsScalar: [pressure[0], pressure[1], divergence].map((view, i) =>
          device.createBindGroup({
            label: `clear-scalar-3d-${i}`,
            layout: L.clearScalar,
            entries: [{ binding: 1, resource: view }],
          }),
        ),
      };

      // Pyramide multigrid : niveau 0 = pression/divergence principales, puis
      // paires de pression + rhs + résidu propres jusqu'à 8³.
      interface LevelTex {
        size: number;
        pressure: Pair<GPUTextureView>;
        rhs: GPUTextureView;
        residual: GPUTextureView;
      }
      const tiers: LevelTex[] = [];
      for (let size = GRID3, l = 0; size >= MG3_COARSEST_SIZE; size /= 2, l++) {
        if (l === 0) {
          tiers.push({
            size,
            pressure,
            rhs: divergence,
            residual: tex3d(device, 'mg3-residual-0', 'r32float', size),
          });
        } else {
          tiers.push({
            size,
            pressure: pair((i) => tex3d(device, `mg3-press-${l}-${i}`, 'r32float', size)),
            rhs: tex3d(device, `mg3-rhs-${l}`, 'r32float', size),
            residual: tex3d(device, `mg3-residual-${l}`, 'r32float', size),
          });
        }
      }
      const lastTier = tiers.length - 1;
      const mgLevels: MGLevel3[] = tiers.map((t, l) => {
        const next = l < lastTier ? tiers[l + 1]! : null;
        return {
          dispatch: Math.ceil(t.size / WG3),
          smoothBind: pair((p) =>
            device.createBindGroup({
              label: `mg3-smooth-l${l}-p${p}`,
              layout: L.jacobi,
              entries: [
                { binding: 1, resource: t.pressure[p] },
                { binding: 2, resource: t.rhs },
                { binding: 3, resource: t.pressure[flip(p)] },
              ],
            }),
          ),
          residualBind:
            next === null
              ? null
              : pair((p) =>
                  device.createBindGroup({
                    label: `mg3-residual-l${l}-p${p}`,
                    layout: L.jacobi,
                    entries: [
                      { binding: 1, resource: t.pressure[p] },
                      { binding: 2, resource: t.rhs },
                      { binding: 3, resource: t.residual },
                    ],
                  }),
                ),
          restrictBind:
            next === null
              ? null
              : device.createBindGroup({
                  label: `mg3-restrict-l${l}`,
                  layout: L.mgRestrict,
                  entries: [
                    { binding: 0, resource: t.residual },
                    { binding: 3, resource: next.rhs },
                  ],
                }),
          prolongBind:
            next === null
              ? null
              : pair((fine) =>
                  pair((coarse) =>
                    device.createBindGroup({
                      label: `mg3-prolong-l${l}-f${fine}-c${coarse}`,
                      layout: L.mgProlong,
                      entries: [
                        { binding: 0, resource: next.pressure[coarse] },
                        { binding: 3, resource: t.pressure[flip(fine)] },
                        { binding: 4, resource: t.pressure[fine] },
                      ],
                    }),
                  ),
                ),
          clearBind:
            l === 0
              ? null
              : device.createBindGroup({
                  label: `mg3-clear-l${l}`,
                  layout: L.clearScalar,
                  entries: [{ binding: 1, resource: t.pressure[0] }],
                }),
        };
      });

      return new FluidSim3D(
        device,
        simUniforms,
        renderUniforms,
        group0,
        {
          velPredict,
          velCorrect,
          forces,
          curl: curlPipe,
          confine,
          denPredict,
          denCorrect,
          divergence: divergencePipe,
          jacobi,
          gradient,
          mgSmooth,
          mgResidual,
          mgRestrict: mgRestrictPipe,
          mgProlong: mgProlongPipe,
          clearRgba,
          clearScalar,
          render,
        },
        binds,
        mgLevels,
      );
    });
  }

  /** Encode et soumet une frame complète : simulation (sauf pause) + rendu. */
  frame(input: Frame3DInput, target: GPUTextureView, aspect: number): void {
    const dt = Math.min(Math.max(input.dt, 0), 1 / 30);
    const running = !input.paused && dt > 0;
    if (running) {
      this.simTime += dt;
      this.writeSimUniforms(dt, input.vorticityStrength);
    }
    this.writeRenderUniforms(input, aspect);

    const encoder = this.device.createCommandEncoder({ label: 'frame3d' });
    if (running || input.reset) {
      const cp = encoder.beginComputePass({ label: 'sim3d' });
      const n = DISPATCH3;
      if (input.reset) {
        cp.setPipeline(this.pipelines.clearRgba);
        for (const bind of this.binds.clearsRgba) {
          cp.setBindGroup(0, bind);
          cp.dispatchWorkgroups(n, n, n);
        }
        cp.setPipeline(this.pipelines.clearScalar);
        for (const bind of this.binds.clearsScalar) {
          cp.setBindGroup(0, bind);
          cp.dispatchWorkgroups(n, n, n);
        }
      }
      if (running) {
        cp.setBindGroup(0, this.group0);
        // Advection MacCormack de la vélocité : prédicteur → scratch, correcteur clampé.
        cp.setPipeline(this.pipelines.velPredict);
        cp.setBindGroup(1, this.binds.velPredict[this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);
        cp.setPipeline(this.pipelines.velCorrect);
        cp.setBindGroup(1, this.binds.velCorrect[this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);
        this.velIdx = flip(this.velIdx);

        cp.setPipeline(this.pipelines.forces);
        cp.setBindGroup(1, this.binds.forces[this.velIdx][this.denIdx]);
        cp.dispatchWorkgroups(n, n, n);
        this.velIdx = flip(this.velIdx);

        // Vorticity confinement : rotationnel vectoriel puis force de renforcement.
        // Passes sautées à ε = 0 (défaut — voir config3d.ts sur le grain de grille).
        if (input.vorticityStrength > 0) {
          cp.setPipeline(this.pipelines.curl);
          cp.setBindGroup(1, this.binds.curl[this.velIdx]);
          cp.dispatchWorkgroups(n, n, n);
          cp.setPipeline(this.pipelines.confine);
          cp.setBindGroup(1, this.binds.confine[this.velIdx]);
          cp.dispatchWorkgroups(n, n, n);
          this.velIdx = flip(this.velIdx);
        }

        cp.setPipeline(this.pipelines.divergence);
        cp.setBindGroup(1, this.binds.divergence[this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);

        // Pression : V-cycles multigrid (défaut) ou Jacobi simple — warm start
        // dans les deux cas, la pression de la frame précédente sert de départ.
        if (input.multigrid) {
          const cycles = Math.max(1, Math.round(input.vcycles));
          for (let k = 0; k < cycles; k++) {
            this.encodeVCycle3(cp);
          }
        } else {
          cp.setPipeline(this.pipelines.jacobi);
          for (let i = 0; i < input.jacobiIterations; i++) {
            cp.setBindGroup(1, this.binds.jacobi[this.pressIdx]);
            cp.dispatchWorkgroups(n, n, n);
            this.pressIdx = flip(this.pressIdx);
          }
        }

        cp.setPipeline(this.pipelines.gradient);
        cp.setBindGroup(1, this.binds.gradient[this.pressIdx][this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);
        this.velIdx = flip(this.velIdx);

        // Advection MacCormack des densités + injection de l'émetteur au correcteur.
        cp.setPipeline(this.pipelines.denPredict);
        cp.setBindGroup(1, this.binds.denPredict[this.denIdx][this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);
        cp.setPipeline(this.pipelines.denCorrect);
        cp.setBindGroup(1, this.binds.denCorrect[this.denIdx][this.velIdx]);
        cp.dispatchWorkgroups(n, n, n);
        this.denIdx = flip(this.denIdx);
      }
      cp.end();
    }

    const rp = encoder.beginRenderPass({
      label: 'raymarch',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    rp.setPipeline(this.pipelines.render);
    rp.setBindGroup(0, this.binds.render[this.denIdx]);
    rp.draw(3);
    rp.end();

    this.submitList[0] = encoder.finish();
    this.device.queue.submit(this.submitList);
  }

  /**
   * Encode un V-cycle multigrid complet sur la pression du niveau 0 (warm start).
   * Descente : lissage pondéré, résidu, restriction ; plus grossier : lissage long ;
   * remontée : prolongation trilinéaire + post-lissage. Les index ping-pong par
   * niveau vivent dans mgIdx ; celui du niveau 0 est resynchronisé avec pressIdx.
   */
  private encodeVCycle3(cp: GPUComputePassEncoder): void {
    const levels = this.mgLevels;
    const last = levels.length - 1;
    const idx = this.mgIdx;
    idx[0] = this.pressIdx;

    // L'équation d'erreur des niveaux grossiers part de zéro à chaque cycle.
    cp.setPipeline(this.pipelines.clearScalar);
    for (let l = 1; l <= last; l++) {
      idx[l] = 0;
      const lev = levels[l]!;
      cp.setBindGroup(0, lev.clearBind!);
      cp.dispatchWorkgroups(lev.dispatch, lev.dispatch, lev.dispatch);
    }
    // Le clear utilise un autre layout de groupe 0 : on rétablit le groupe partagé.
    cp.setBindGroup(0, this.group0);

    // Descente.
    for (let l = 0; l <= last; l++) {
      const lev = levels[l]!;
      const count = l === last ? MG3_COARSE_SMOOTH : MG3_PRE_SMOOTH;
      cp.setPipeline(this.pipelines.mgSmooth);
      for (let i = 0; i < count; i++) {
        cp.setBindGroup(1, lev.smoothBind[idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch, lev.dispatch);
        idx[l] = idx[l]! ^ 1;
      }
      if (l < last) {
        const coarse = levels[l + 1]!;
        cp.setPipeline(this.pipelines.mgResidual);
        cp.setBindGroup(1, lev.residualBind![idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch, lev.dispatch);
        cp.setPipeline(this.pipelines.mgRestrict);
        cp.setBindGroup(1, lev.restrictBind!);
        cp.dispatchWorkgroups(coarse.dispatch, coarse.dispatch, coarse.dispatch);
      }
    }

    // Remontée.
    for (let l = last - 1; l >= 0; l--) {
      const lev = levels[l]!;
      cp.setPipeline(this.pipelines.mgProlong);
      cp.setBindGroup(1, lev.prolongBind![idx[l] as PingIndex][idx[l + 1] as PingIndex]);
      cp.dispatchWorkgroups(lev.dispatch, lev.dispatch, lev.dispatch);
      idx[l] = idx[l]! ^ 1;
      cp.setPipeline(this.pipelines.mgSmooth);
      for (let i = 0; i < MG3_POST_SMOOTH; i++) {
        cp.setBindGroup(1, lev.smoothBind[idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch, lev.dispatch);
        idx[l] = idx[l]! ^ 1;
      }
    }
    this.pressIdx = idx[0] as PingIndex;
  }

  private writeSimUniforms(dt: number, vorticityStrength: number): void {
    const d = this.simData;
    const D = SIM3_DEFAULTS;
    d[0] = dt;
    d[1] = this.simTime;
    d[2] = GRID3;
    d[3] = vorticityStrength;
    // Émetteur : bas de la boîte, centre légèrement mobile.
    d[4] = GRID3 * (0.5 + 0.05 * Math.sin(this.simTime * 0.9));
    d[5] = GRID3 * 0.08;
    d[6] = GRID3 * (0.5 + 0.05 * Math.cos(this.simTime * 0.7));
    d[7] = GRID3 * D.emitterRadius;
    d[8] = D.emitHeat;
    d[9] = D.emitSmoke;
    d[10] = D.emitUpVelocity;
    d[11] = D.emitWobbleVelocity;
    d[12] = D.velocityDissipation;
    d[13] = D.smokeDissipation;
    d[14] = D.heatCooling;
    d[15] = D.buoyancy;
    this.device.queue.writeBuffer(this.simUniforms, 0, d);
  }

  private writeRenderUniforms(input: Frame3DInput, aspect: number): void {
    const d = this.renderData;
    const { azimuth, elevation, radius } = input.cam;
    const cy = Math.cos(elevation);
    const px = radius * cy * Math.cos(azimuth);
    const py = radius * Math.sin(elevation);
    const pz = radius * cy * Math.sin(azimuth);
    // Base orthonormée regardant le centre de la boîte.
    const fl = Math.hypot(px, py, pz);
    const fx = -px / fl;
    const fy = -py / fl;
    const fz = -pz / fl;
    // right = fwd × up_monde, up = right × fwd (base droitière, écran non miroir).
    let rx = -fz;
    const ry = 0;
    let rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl;
    rz /= rl;
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;
    d[0] = px;
    d[1] = py;
    d[2] = pz;
    d[3] = 0.364; // tan(fov/2), fov ≈ 40°
    d[4] = rx;
    d[5] = ry;
    d[6] = rz;
    d[7] = aspect;
    d[8] = ux;
    d[9] = uy;
    d[10] = uz;
    d[11] = input.exposure;
    d[12] = fx;
    d[13] = fy;
    d[14] = fz;
    d[15] = input.raymarchSteps;
    // Lumière directionnelle fixe (normalisée), intensité.
    d[16] = 0.5;
    d[17] = 0.74;
    d[18] = 0.45;
    d[19] = 1.0;
    let dirty = false;
    for (let i = 0; i < d.length; i++) {
      if (d[i] !== this.lastRender[i]) {
        dirty = true;
        break;
      }
    }
    if (dirty) {
      this.lastRender.set(d);
      this.device.queue.writeBuffer(this.renderUniforms, 0, d);
    }
  }
}
