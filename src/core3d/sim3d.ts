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
  SCALE3,
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
  /** Encre émise par l'émetteur : 0 = bleu, 1 = magenta, 2 = ambre. */
  emitInk: number;
  /** Caméra orbitale autour du centre de la boîte. */
  cam: { azimuth: number; elevation: number; radius: number };
  /** Position courante du pointeur en NDC [-1,1] (mise à jour en continu). */
  pointer: { ndcX: number; ndcY: number };
  /** Souffle du pointeur (clic droit + glisser) : position NDC [-1,1] et delta NDC
   *  accumulé depuis la dernière frame (consommé par la plateforme après frame()). */
  blow: { active: boolean; ndcX: number; ndcY: number; moveX: number; moveY: number };
  /** Saisie (clic gauche sur un objet — voir hitTest) : l'objet suit le pointeur
   *  sur le plan face caméra passant par sa position. */
  grab: { active: boolean };
  /** Ajout d'un émetteur sous le pointeur / retrait du dernier (consommés). */
  addEmitter: boolean;
  removeEmitter: boolean;
  /** Sphère-obstacle présente. */
  sphereActive: boolean;
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
      usage:
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    })
    .createView({ label: `${label}-view` });
}

/** Décodage float16 → float32 (le readback d'export lit du rgba16float). */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    return sign * mant * 2 ** -24;
  }
  if (exp === 31) {
    return mant ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
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

  private readonly simData = new Float32Array(52);
  private readonly renderData = new Float32Array(24);
  private lastRender = new Float32Array(24).fill(Number.NaN);

  /** Émetteurs (positions en voxels) — le premier est l'émetteur historique. */
  private readonly emitters: { pos: [number, number, number]; ink: number }[] = [
    { pos: [GRID3 * 0.5, GRID3 * 0.08, GRID3 * 0.5], ink: 0 },
  ];
  private activeEmitter = 0;
  private readonly spherePos: [number, number, number] = [
    GRID3 * SIM3_DEFAULTS.sphereStart[0],
    GRID3 * SIM3_DEFAULTS.sphereStart[1],
    GRID3 * SIM3_DEFAULTS.sphereStart[2],
  ];
  private sphereOn = true;
  /** Cible saisie : -2 = aucune, -1 = sphère, ≥0 = index d'émetteur. */
  private grabbed = -2;
  private readonly rayO: [number, number, number] = [0, 0, 0];
  private readonly rayD: [number, number, number] = [0, 0, 0];
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
    private readonly denTextures: Pair<GPUTexture>,
  ) {
    this.mgIdx = new Array<number>(mgLevels.length).fill(0);
  }

  /** Index de pression courant par niveau multigrid (tableau réutilisé, zéro alloc). */
  private readonly mgIdx: number[];

  static async create(device: GPUDevice, targetFormat: GPUTextureFormat): Promise<FluidSim3D> {
    return withValidation(device, 'init-3d', async () => {
      const velocity = pair((i) => tex3d(device, `vel3d-${i}`, 'rgba16float'));
      // Les textures de densité gardent leur handle : l'export VDB les copie (COPY_SRC).
      const denTextures = pair((i) =>
        device.createTexture({
          label: `den3d-${i}`,
          dimension: '3d',
          size: { width: GRID3, height: GRID3, depthOrArrayLayers: GRID3 },
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC,
        }),
      );
      const density = pair((i) => denTextures[i].createView({ label: `den3d-${i}-view` }));
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
        denTextures,
      );
    });
  }

  /** Encode et soumet une frame complète : simulation (sauf pause) + rendu. */
  frame(input: Frame3DInput, target: GPUTextureView, aspect: number): void {
    const dt = Math.min(Math.max(input.dt, 0), 1 / 30);
    const running = !input.paused && dt > 0;
    // Le rendu d'abord : la saisie et le souffle lisent la base caméra depuis
    // renderData — elle doit être celle de cette frame.
    this.writeRenderUniforms(input, aspect);
    this.processInteraction(input);
    if (running) {
      this.simTime += dt;
      this.writeSimUniforms(dt, input);
    }

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

  /** Rayon caméra→scène pour un point NDC : origine en voxels, direction unitaire. */
  private computeRay(ndcX: number, ndcY: number): void {
    const r = this.renderData;
    const tanf = r[3]!;
    const aspect = r[7]!;
    let dx = r[12]! + r[4]! * ndcX * tanf * aspect + r[8]! * ndcY * tanf;
    let dy = r[13]! + r[5]! * ndcX * tanf * aspect + r[9]! * ndcY * tanf;
    let dz = r[14]! + r[6]! * ndcX * tanf * aspect + r[10]! * ndcY * tanf;
    const dl = Math.hypot(dx, dy, dz) || 1;
    this.rayD[0] = dx / dl;
    this.rayD[1] = dy / dl;
    this.rayD[2] = dz / dl;
    this.rayO[0] = (r[0]! + 0.5) * GRID3;
    this.rayO[1] = (r[1]! + 0.5) * GRID3;
    this.rayO[2] = (r[2]! + 0.5) * GRID3;
  }

  /** Distance point (voxels) → rayon courant ; +∞ si le point est derrière. */
  private rayDistance(p: readonly number[]): number {
    const vx = p[0]! - this.rayO[0];
    const vy = p[1]! - this.rayO[1];
    const vz = p[2]! - this.rayO[2];
    const t = vx * this.rayD[0] + vy * this.rayD[1] + vz * this.rayD[2];
    if (t <= 0) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.hypot(
      vx - t * this.rayD[0],
      vy - t * this.rayD[1],
      vz - t * this.rayD[2],
    );
  }

  /** La cible sous le rayon : -1 = sphère, ≥0 = émetteur, -2 = rien. */
  private pickTarget(): number {
    const limit = SIM3_DEFAULTS.grabRadius * GRID3;
    let best = -2;
    let bestDist = limit;
    if (this.sphereOn) {
      const d = this.rayDistance(this.spherePos);
      if (d < bestDist + SIM3_DEFAULTS.sphereRadius * GRID3 * 0.5) {
        best = -1;
        bestDist = d;
      }
    }
    for (let i = 0; i < this.emitters.length; i++) {
      const d = this.rayDistance(this.emitters[i]!.pos);
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    }
    return best;
  }

  /** Y a-t-il un objet saisissable sous ce point NDC ? (décide saisie vs orbite) */
  hitTest(ndcX: number, ndcY: number): boolean {
    this.computeRay(ndcX, ndcY);
    return this.pickTarget() !== -2;
  }

  /** Saisie, ajout/retrait d'émetteurs, sphère — tout l'état interactif. */
  private processInteraction(input: Frame3DInput): void {
    if (input.reset) {
      this.emitters.length = 1;
      this.emitters[0] = { pos: [GRID3 * 0.5, GRID3 * 0.08, GRID3 * 0.5], ink: input.emitInk };
      this.activeEmitter = 0;
      this.spherePos[0] = GRID3 * SIM3_DEFAULTS.sphereStart[0];
      this.spherePos[1] = GRID3 * SIM3_DEFAULTS.sphereStart[1];
      this.spherePos[2] = GRID3 * SIM3_DEFAULTS.sphereStart[2];
    }
    this.sphereOn = input.sphereActive;
    // L'encre sélectionnée s'applique à l'émetteur actif (dernier ajouté ou saisi).
    this.emitters[this.activeEmitter]!.ink = input.emitInk;

    if (input.addEmitter && this.emitters.length < SIM3_DEFAULTS.maxEmitters) {
      // Nouvel émetteur : rayon du pointeur ∩ plan horizontal des émetteurs.
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      const planeY = GRID3 * 0.08;
      const t = (planeY - this.rayO[1]) / (this.rayD[1] || 1e-6);
      const clampXZ = (v: number): number => Math.min(Math.max(v, GRID3 * 0.08), GRID3 * 0.92);
      const px = t > 0 ? clampXZ(this.rayO[0] + t * this.rayD[0]) : GRID3 * 0.5;
      const pz = t > 0 ? clampXZ(this.rayO[2] + t * this.rayD[2]) : GRID3 * 0.5;
      this.emitters.push({ pos: [px, planeY, pz], ink: input.emitInk });
      this.activeEmitter = this.emitters.length - 1;
    }
    if (input.removeEmitter && this.emitters.length > 1) {
      this.emitters.pop();
      this.activeEmitter = Math.min(this.activeEmitter, this.emitters.length - 1);
    }

    if (input.grab.active) {
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      if (this.grabbed === -2) {
        this.grabbed = this.pickTarget();
        if (this.grabbed >= 0) {
          this.activeEmitter = this.grabbed;
        }
      }
      if (this.grabbed !== -2) {
        // Déplacement sur le plan face caméra passant par la position de l'objet.
        const target = this.grabbed === -1 ? this.spherePos : this.emitters[this.grabbed]!.pos;
        const r = this.renderData;
        const denom =
          r[12]! * this.rayD[0] + r[13]! * this.rayD[1] + r[14]! * this.rayD[2];
        if (Math.abs(denom) > 1e-5) {
          const t =
            (r[12]! * (target[0] - this.rayO[0]) +
              r[13]! * (target[1] - this.rayO[1]) +
              r[14]! * (target[2] - this.rayO[2])) /
            denom;
          if (t > 0) {
            const lo = GRID3 * 0.05;
            const hi = GRID3 * 0.95;
            target[0] = Math.min(Math.max(this.rayO[0] + t * this.rayD[0], lo), hi);
            target[1] = Math.min(Math.max(this.rayO[1] + t * this.rayD[1], GRID3 * 0.04), hi);
            target[2] = Math.min(Math.max(this.rayO[2] + t * this.rayD[2], lo), hi);
          }
        }
      }
    } else {
      this.grabbed = -2;
    }
  }

  /**
   * Lecture ponctuelle du volume de densités pour l'export VDB — la SEULE
   * lecture GPU→CPU du moteur 3D, hors boucle de frame (même exception
   * documentée que l'export PNG du 2D). Alloue un staging buffer par appel.
   */
  async exportVolume(): Promise<{ density: Float32Array; heat: Float32Array }> {
    const n = GRID3;
    const texels = n * n * n;
    const buffer = this.device.createBuffer({
      label: 'export3d-readback',
      size: texels * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder({ label: 'export3d' });
    encoder.copyTextureToBuffer(
      { texture: this.denTextures[this.denIdx] },
      { buffer, bytesPerRow: n * 8, rowsPerImage: n },
      { width: n, height: n, depthOrArrayLayers: n },
    );
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buffer.getMappedRange());
    const density = new Float32Array(texels);
    const heat = new Float32Array(texels);
    for (let i = 0; i < texels; i++) {
      // Densité totale = somme des trois encres (xyz) ; chaleur = w.
      density[i] =
        halfToFloat(halves[i * 4]!) + halfToFloat(halves[i * 4 + 1]!) + halfToFloat(halves[i * 4 + 2]!);
      heat[i] = halfToFloat(halves[i * 4 + 3]!);
    }
    buffer.destroy();
    return { density, heat };
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

  private writeSimUniforms(dt: number, input: Frame3DInput): void {
    const d = this.simData;
    const D = SIM3_DEFAULTS;
    d[0] = dt;
    d[1] = this.simTime;
    d[2] = GRID3;
    d[3] = input.vorticityStrength;
    // Émetteurs : position d'état + petit balancement propre à chacun (déphasé).
    const emitterSlot = (slot: number, i: number): void => {
      const e = this.emitters[i]!;
      const wob = GRID3 * 0.02;
      d[slot] = e.pos[0] + wob * Math.sin(this.simTime * 0.9 + i * 2.1);
      d[slot + 1] = e.pos[1];
      d[slot + 2] = e.pos[2] + wob * Math.cos(this.simTime * 0.7 + i * 1.7);
      d[slot + 3] = GRID3 * D.emitterRadius;
    };
    emitterSlot(4, 0);
    d[8] = D.emitHeat;
    d[9] = D.emitSmoke;
    // Forces absolues calibrées à 128³ → remises à l'échelle de la grille.
    d[10] = D.emitUpVelocity * SCALE3;
    d[11] = D.emitWobbleVelocity * SCALE3;
    d[12] = D.velocityDissipation;
    d[13] = D.smokeDissipation;
    d[14] = D.heatCooling;
    d[15] = D.buoyancy * SCALE3;
    // Souffle du pointeur : rayon caméra→scène + force selon le geste écran.
    const b = input.blow;
    const r = this.renderData;
    if (b.active) {
      this.computeRay(b.ndcX, b.ndcY);
      d[16] = this.rayO[0];
      d[17] = this.rayO[1];
      d[18] = this.rayO[2];
      d[19] = D.blowRadius * GRID3;
      d[20] = this.rayD[0];
      d[21] = this.rayD[1];
      d[22] = this.rayD[2];
      d[23] = 1;
      const s = D.blowForce * GRID3;
      d[24] = (r[4]! * b.moveX + r[8]! * b.moveY) * s;
      d[25] = (r[5]! * b.moveX + r[9]! * b.moveY) * s;
      d[26] = (r[6]! * b.moveX + r[10]! * b.moveY) * s;
    } else {
      d[23] = 0;
    }
    // Sphère-obstacle (voxels ; rayon ≤ 0 = absente).
    d[28] = this.spherePos[0];
    d[29] = this.spherePos[1];
    d[30] = this.spherePos[2];
    d[31] = this.sphereOn ? GRID3 * D.sphereRadius : -1;
    // Émetteurs supplémentaires + encres.
    d[32] = this.emitters.length;
    for (let i = 1; i < 4; i++) {
      if (i < this.emitters.length) {
        emitterSlot(32 + i * 4, i);
      }
    }
    for (let i = 0; i < 4; i++) {
      d[48 + i] = this.emitters[i]?.ink ?? 0;
    }
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
    // Sphère-obstacle en unités monde pour le rendu.
    d[20] = this.spherePos[0] / GRID3 - 0.5;
    d[21] = this.spherePos[1] / GRID3 - 0.5;
    d[22] = this.spherePos[2] / GRID3 - 0.5;
    d[23] = input.sphereActive ? SIM3_DEFAULTS.sphereRadius : -1;
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
