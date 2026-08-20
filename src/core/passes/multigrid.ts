/**
 * Solveur multigrid pour la pression : hiérarchie de niveaux 512² → 8² (facteur 2),
 * chacun avec sa paire de pression ping-pong, son second membre, son résidu et son
 * masque d'obstacles restreint. Tous les bind groups de toutes les variantes sont
 * pré-créés ici ; l'encodage du V-cycle (simulation.ts) ne fait que les indexer.
 * Le niveau 0 réutilise les textures principales (pression, divergence, obstacles) —
 * et les bind groups de lissage du niveau 0 SONT les jacobiBind existants (même layout).
 *
 * Limite connue : un mur plus fin que 2^niveau texels peut « disparaître » du masque
 * grossier (restriction majoritaire) — la correction grossière fuit alors légèrement à
 * travers, mais le post-lissage aux niveaux fins, lui, voit le masque exact.
 */

import multigridWGSL from '../shaders/multigrid.wgsl?raw';
import mgDebugWGSL from '../shaders/mg_debug.wgsl?raw';
import { GRID_SIZE, MG_COARSEST_SIZE, SCALAR_FORMAT, WORKGROUP_SIZE } from '../config';
import { createShaderModule, createSimPipeline, type SimLayouts } from '../pipelines';
import type { SimResources } from '../resources';
import { flip, type Pair, type PingIndex } from '../types';

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export interface MGLevel {
  readonly size: number;
  readonly dispatch: number;
  /** Lissage Jacobi pondéré, indexé par [source de pression]. Niveau 0 : jacobiBind. */
  readonly smoothBind: Pair<GPUBindGroup>;
  /** r = rhs − A·p, indexé par [source de pression]. Null au niveau le plus grossier. */
  readonly residualBind: Pair<GPUBindGroup> | null;
  /** Résidu de ce niveau → rhs du niveau suivant. Null au plus grossier. */
  readonly restrictBind: GPUBindGroup | null;
  /** Correction du niveau suivant → ce niveau, [pression fine][pression grossière]. */
  readonly prolongBind: Pair<Pair<GPUBindGroup>> | null;
  /** Remise à zéro de la pression de départ (niveaux > 0 : l'erreur part de zéro). */
  readonly clearBind: GPUBindGroup | null;
  /** Obstacles de ce niveau → masque du niveau suivant. Null au plus grossier. */
  readonly obstacleRestrictBind: GPUBindGroup | null;
  /** Vue debug 6 : cellule de mosaïque de ce niveau, indexée par [source de pression].
   *  Source « résidu » : le résidu du niveau (le rhs au plus grossier, qui n'en a pas). */
  readonly composeBind: Pair<GPUBindGroup>;
}

export interface MultigridPasses {
  readonly smoothPipeline: GPUComputePipeline;
  readonly residualPipeline: GPUComputePipeline;
  readonly restrictPipeline: GPUComputePipeline;
  readonly prolongPipeline: GPUComputePipeline;
  readonly obstacleRestrictPipeline: GPUComputePipeline;
  /** Composition de la mosaïque debug (group(0) autonome, un dispatch par niveau). */
  readonly composePipeline: GPUComputePipeline;
  readonly levels: readonly MGLevel[];
}

interface LevelTextures {
  size: number;
  pressure: Pair<GPUTextureView>;
  rhs: GPUTextureView;
  residual: GPUTextureView;
  obstacle: GPUTextureView;
}

export async function createMultigridPasses(
  device: GPUDevice,
  layouts: SimLayouts,
  res: SimResources,
): Promise<MultigridPasses> {
  const module = await createShaderModule(device, 'multigrid.wgsl', multigridWGSL);
  const debugModule = await createShaderModule(device, 'mg_debug.wgsl', mgDebugWGSL);
  const [smoothPipeline, residualPipeline, restrictPipeline, prolongPipeline, obstacleRestrictPipeline, composePipeline] =
    await Promise.all([
      createSimPipeline(device, 'mg-smooth', layouts, layouts.jacobi, module, 'smooth_jacobi'),
      createSimPipeline(device, 'mg-residual', layouts, layouts.jacobi, module, 'residual'),
      createSimPipeline(device, 'mg-restrict', layouts, layouts.mgRestrict, module, 'restrict_rhs'),
      createSimPipeline(device, 'mg-prolong', layouts, layouts.mgProlong, module, 'prolong_add'),
      createSimPipeline(device, 'mg-restrict-obstacle', layouts, layouts.mgRestrict, module, 'restrict_obstacle'),
      device.createComputePipelineAsync({
        label: 'mg-debug-compose',
        layout: device.createPipelineLayout({
          label: 'mg-debug-compose-pipeline-layout',
          bindGroupLayouts: [layouts.mgDebug],
        }),
        compute: { module: debugModule, entryPoint: 'compose' },
      }),
    ]);

  const makeTexture = (label: string, size: number): GPUTextureView =>
    device
      .createTexture({
        label,
        size: { width: size, height: size },
        format: SCALAR_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      })
      .createView({ label: `${label}-view` });

  // Textures de la hiérarchie. Niveau 0 : réutilise les ressources principales.
  const textures: LevelTextures[] = [];
  for (let size = GRID_SIZE, l = 0; size >= MG_COARSEST_SIZE; size /= 2, l++) {
    if (l === 0) {
      textures.push({
        size,
        pressure: res.pressure.views,
        rhs: res.divergence.view,
        residual: makeTexture('mg-residual-0', size),
        obstacle: res.obstacle.view,
      });
    } else {
      textures.push({
        size,
        pressure: [makeTexture(`mg-pressure-${l}-0`, size), makeTexture(`mg-pressure-${l}-1`, size)],
        rhs: makeTexture(`mg-rhs-${l}`, size),
        residual: makeTexture(`mg-residual-${l}`, size),
        obstacle: makeTexture(`mg-obstacle-${l}`, size),
      });
    }
  }

  const last = textures.length - 1;
  const levels: MGLevel[] = textures.map((t, l) => {
    const next = l < last ? textures[l + 1]! : null;
    return {
      size: t.size,
      dispatch: Math.ceil(t.size / WORKGROUP_SIZE),
      smoothBind: pair((p) =>
        device.createBindGroup({
          label: `mg-smooth-bind-l${l}-p${p}`,
          layout: layouts.jacobi,
          entries: [
            { binding: 0, resource: t.pressure[p] },
            { binding: 1, resource: t.rhs },
            { binding: 2, resource: t.pressure[flip(p)] },
            { binding: 3, resource: t.obstacle },
          ],
        }),
      ),
      residualBind:
        next === null
          ? null
          : pair((p) =>
              device.createBindGroup({
                label: `mg-residual-bind-l${l}-p${p}`,
                layout: layouts.jacobi,
                entries: [
                  { binding: 0, resource: t.pressure[p] },
                  { binding: 1, resource: t.rhs },
                  { binding: 2, resource: t.residual },
                  { binding: 3, resource: t.obstacle },
                ],
              }),
            ),
      restrictBind:
        next === null
          ? null
          : device.createBindGroup({
              label: `mg-restrict-bind-l${l}`,
              layout: layouts.mgRestrict,
              entries: [
                { binding: 0, resource: t.residual },
                { binding: 2, resource: next.rhs },
              ],
            }),
      prolongBind:
        next === null
          ? null
          : pair((fine) =>
              pair((coarse) =>
                device.createBindGroup({
                  label: `mg-prolong-bind-l${l}-f${fine}-c${coarse}`,
                  layout: layouts.mgProlong,
                  entries: [
                    { binding: 0, resource: next.pressure[coarse] },
                    { binding: 1, resource: t.pressure[fine] },
                    { binding: 2, resource: t.pressure[flip(fine)] },
                  ],
                }),
              ),
            ),
      clearBind:
        l === 0
          ? null
          : device.createBindGroup({
              label: `mg-clear-bind-l${l}`,
              layout: layouts.clearScalar,
              entries: [{ binding: 1, resource: t.pressure[0] }],
            }),
      obstacleRestrictBind:
        next === null
          ? null
          : device.createBindGroup({
              label: `mg-obstacle-restrict-bind-l${l}`,
              layout: layouts.mgRestrict,
              entries: [
                { binding: 0, resource: t.obstacle },
                { binding: 2, resource: next.obstacle },
              ],
            }),
      composeBind: pair((p) =>
        device.createBindGroup({
          label: `mg-debug-compose-bind-l${l}-p${p}`,
          layout: layouts.mgDebug,
          entries: [
            { binding: 0, resource: next === null ? t.rhs : t.residual },
            { binding: 1, resource: t.pressure[p] },
            { binding: 2, resource: res.mgDebug.view },
          ],
        }),
      ),
    };
  });

  return {
    smoothPipeline,
    residualPipeline,
    restrictPipeline,
    prolongPipeline,
    obstacleRestrictPipeline,
    composePipeline,
    levels,
  };
}
