/**
 * Orchestration d'une frame de simulation. C'est le point d'entrée du core :
 * la couche plateforme fournit un GPUDevice, un format de cible, un dt et un
 * FrameInput abstrait — rien d'autre. Aucune référence au DOM ici.
 *
 * Schéma d'une frame (un seul CommandEncoder, une seule soumission) :
 *   [clear ×8 si reset]                     — compute (murs préservés)
 *   [clear murs si demandé]                 — compute
 *   [pinceau à murs si bouton secondaire]   — compute, storage read-write
 *   pour chaque sous-pas (1–2) :
 *     1. advection MacCormack de la vélocité (prédicteur + correcteur) — ping-pong vélocité
 *     2. forces (buoyancy + souris)         — compute, ping-pong vélocité
 *     3. vorticity confinement (curl + force) — compute, ping-pong vélocité
 *     4. divergence                         — compute
 *     5. Jacobi ×N (pression)               — compute, ping-pong pression, warm start
 *     6. soustraction du gradient           — compute, ping-pong vélocité
 *     7. advection MacCormack des densités (grille dye, plus fine) + injection
 *   8. composition vers le canvas (vue fluides ou vue de debug) — render
 *
 * Toutes les dispatches compute partagent UNE seule compute pass : dans une compute
 * pass WebGPU, chaque dispatch forme son propre usage scope, la synchronisation
 * écriture→lecture entre passes successives est donc implicite.
 *
 * Frontières : le mode (parois fermées / périodique) est un paramètre d'uniform lu par
 * les shaders (wrap vs clamp des voisins) + un choix de sampler (repeat vs clamp) fait
 * ici en sélectionnant l'une des deux variantes pré-créées du bind group 0.
 */

import {
  DISPATCH_SIZE,
  DYE_DISPATCH_SIZE,
  FLOW_DISPATCH,
  MG_COARSE_SMOOTH,
  MG_POST_SMOOTH,
  MG_PRE_SMOOTH,
  PARTICLE_DISPATCH,
  SIM_DEFAULTS,
} from './config';
import { createAdvectPasses, type AdvectPasses } from './passes/advect';
import { createClearPasses, type ClearPasses } from './passes/clear';
import { createForcesPass, type ForcesPass } from './passes/forces';
import { createMultigridPasses, type MultigridPasses } from './passes/multigrid';
import { createOpticalFlowPasses, type OpticalFlowPasses } from './passes/opticalflow';
import { createParticlesPass, type ParticlesPass } from './passes/particles';
import { createProjectPasses, type ProjectPasses } from './passes/project';
import { createVorticityPasses, type VorticityPasses } from './passes/vorticity';
import { createWallsPass, type WallsPass } from './passes/walls';
import { createLayouts, withValidation } from './pipelines';
import { CompositeRenderer } from './render';
import { createResources, type SimResources } from './resources';
import {
  flip,
  type BoundaryMode,
  type FrameInput,
  type PingIndex,
  type ViewMode,
} from './types';
import {
  RENDER_VIEW_MODE_INDEX,
  renderUniformData,
  SIM_PARAMS_BYTES,
  SimUniformWriter,
} from './uniforms';

/** Bind groups du groupe 0, un par BoundaryMode (le sampler clamp/repeat diffère). */
type Group0Variants = readonly [GPUBindGroup, GPUBindGroup, GPUBindGroup];

export interface FluidSimOptions {
  /** Format de la cible de rendu (canvas côté web, swapchain côté natif). */
  readonly targetFormat: GPUTextureFormat;
}

export class FluidSim {
  /** Index ping-pong courants : quelle texture de chaque paire contient l'état à jour. */
  private velIdx: PingIndex = 0;
  private denIdx: PingIndex = 0;
  private pressIdx: PingIndex = 0;
  /** Ping-pong de la luminance caméra (flux optique). */
  private lumIdx: PingIndex = 0;

  private readonly uniforms = new SimUniformWriter();
  /** Tableau de soumission réutilisé (zéro allocation par frame). */
  private readonly submitList: GPUCommandBuffer[] = [];
  private readonly computePassDesc: GPUComputePassDescriptor = { label: 'sim-compute-pass' };
  /** Uniforms de rendu : réécrits uniquement quand un réglage de rendu change. */
  private readonly renderData = renderUniformData();
  private lastViewMode: ViewMode = 0;
  // Suivis en nombres JS (comparer à la Float32Array échouerait par perte de précision).
  private lastExposure: number = SIM_DEFAULTS.exposure;
  private lastBloom: number = SIM_DEFAULTS.bloomStrength;
  /** Mode de frontières de la frame précédente — un changement purge la pression. */
  private lastBoundaryMode: BoundaryMode = 0;
  private lastParticleIntensity = 1;
  /** Temps simulé cumulé (s) — graine du hash de respawn des particules. */
  private simTime = 0;

  /** Index de pression courant par niveau multigrid (tableau réutilisé, zéro alloc). */
  private readonly mgIdx: number[];

  private constructor(
    private readonly device: GPUDevice,
    private readonly res: SimResources,
    private readonly advect: AdvectPasses,
    private readonly forces: ForcesPass,
    private readonly vorticity: VorticityPasses,
    private readonly project: ProjectPasses,
    private readonly mg: MultigridPasses,
    private readonly particles: ParticlesPass,
    private readonly opticalFlow: OpticalFlowPasses,
    private readonly walls: WallsPass,
    private readonly clear: ClearPasses,
    private readonly renderer: CompositeRenderer,
    /** Variantes du groupe 0 par BoundaryMode : [clamp, repeat, clamp (ouvert)]. */
    private readonly simGroup0: Group0Variants,
  ) {
    this.mgIdx = new Array<number>(mg.levels.length).fill(0);
    device.queue.writeBuffer(res.renderUniforms, 0, this.renderData);
  }

  /**
   * Init complète sous error scope : toute erreur de validation (WGSL, layouts,
   * formats) devient une exception exploitable au lieu d'un échec silencieux.
   */
  static async create(device: GPUDevice, opts: FluidSimOptions): Promise<FluidSim> {
    return withValidation(device, 'init FluidSim', async () => {
      const layouts = createLayouts(device);
      const res = createResources(device);
      const [advect, forces, vorticity, project, mg, particles, flow, wallsPass, clear, renderer] =
        await Promise.all([
          createAdvectPasses(device, layouts, res),
          createForcesPass(device, layouts, res),
          createVorticityPasses(device, layouts, res),
          createProjectPasses(device, layouts, res),
          createMultigridPasses(device, layouts, res),
          createParticlesPass(device, layouts, res),
          createOpticalFlowPasses(device, layouts, res),
          createWallsPass(device, layouts, res),
          createClearPasses(device, layouts, res),
          CompositeRenderer.create(device, layouts, res, opts.targetFormat),
        ]);
      const makeGroup0 = (mode: BoundaryMode, sampler: GPUSampler): GPUBindGroup =>
        device.createBindGroup({
          label: `sim-group0-mode${mode}`,
          layout: layouts.simGroup0,
          entries: [
            {
              binding: 0,
              resource: { buffer: res.simUniforms, offset: 0, size: SIM_PARAMS_BYTES },
            },
            { binding: 1, resource: sampler },
          ],
        });
      const simGroup0: Group0Variants = [
        makeGroup0(0, res.linearSampler),
        makeGroup0(1, res.repeatSampler),
        // Mode ouvert : clamp aussi — l'annulation hors domaine se fait dans les shaders.
        makeGroup0(2, res.linearSampler),
      ];
      return new FluidSim(
        device,
        res,
        advect,
        forces,
        vorticity,
        project,
        mg,
        particles,
        flow,
        wallsPass,
        clear,
        renderer,
        simGroup0,
      );
    });
  }

  /** Texture caméra que la plateforme remplit (copyExternalImageToTexture) quand le
   *  flux optique est actif — le core ne connaît ni getUserMedia ni la permission. */
  get cameraTexture(): GPUTexture {
    return this.res.camera.texture;
  }

  /**
   * Encode et soumet une frame complète. Seul trafic CPU→GPU : l'uniform buffer
   * (≤ 512 octets) — plus, caméra active, la copie de l'image webcam (256², entrée
   * périphérique assumée). Aucune lecture GPU→CPU, aucune création d'objet GPU.
   */
  frame(dtSeconds: number, input: FrameInput, target: GPUTextureView): void {
    // Uniforms de rendu : réécrits uniquement quand un réglage de rendu change.
    if (
      input.viewMode !== this.lastViewMode ||
      input.render.exposure !== this.lastExposure ||
      input.render.bloomStrength !== this.lastBloom ||
      input.params.particleIntensity !== this.lastParticleIntensity
    ) {
      this.renderData[3] = input.params.particleIntensity;
      this.renderData[12] = input.render.exposure;
      this.renderData[RENDER_VIEW_MODE_INDEX] = input.viewMode;
      this.renderData[15] = input.render.bloomStrength;
      this.device.queue.writeBuffer(this.res.renderUniforms, 0, this.renderData);
      this.lastViewMode = input.viewMode;
      this.lastExposure = input.render.exposure;
      this.lastBloom = input.render.bloomStrength;
      this.lastParticleIntensity = input.params.particleIntensity;
    }
    // dt réel × facteur de temps, clampé ; au-delà de maxDt la frame est découpée en
    // sous-pas égaux. En pause, `stepOnce` avance d'exactement une frame à 1/60 s.
    const stepping = input.paused && input.stepOnce;
    const scaled = dtSeconds * input.params.timeScale;
    const dt = stepping
      ? 1 / 60
      : Math.min(Math.max(scaled, 0), SIM_DEFAULTS.maxDt * SIM_DEFAULTS.maxSubsteps);
    const substeps = stepping
      ? 1
      : input.paused || dt === 0
        ? 0
        : dt > SIM_DEFAULTS.maxDt
          ? SIM_DEFAULTS.maxSubsteps
          : 1;
    const painting = input.pointer.wall;

    // Le pinceau à murs fonctionne aussi en pause : il consomme le slot 0 des uniforms.
    const slots = Math.max(substeps, painting ? 1 : 0);
    if (slots > 0) {
      const stepDt = substeps > 0 ? dt / substeps : 0;
      for (let s = 0; s < slots; s++) {
        this.uniforms.fillSlot(
          s,
          stepDt,
          input,
          substeps > 0 ? 1 / substeps : 1,
          this.simTime + s * stepDt,
        );
      }
      this.uniforms.upload(this.device.queue, this.res.simUniforms, slots);
      this.simTime += substeps * stepDt;
    }

    // Changement de mode de frontières : le warm start de pression du mode précédent
    // est faux dans le nouvel opérateur (Neumann/périodique ↔ Dirichlet) — purge.
    const boundaryChanged = input.boundaryMode !== this.lastBoundaryMode;
    this.lastBoundaryMode = input.boundaryMode;

    const encoder = this.device.createCommandEncoder({ label: 'frame-encoder' });
    if (input.reset || input.clearWalls || painting || boundaryChanged || substeps > 0) {
      const cp = encoder.beginComputePass(this.computePassDesc);
      if (input.reset) {
        this.encodeClear(cp);
      }
      if (boundaryChanged && !input.reset) {
        cp.setPipeline(this.clear.scalarPipeline);
        for (const target of this.clear.pressureTargets) {
          cp.setBindGroup(0, target.bind);
          cp.dispatchWorkgroups(target.dispatch, target.dispatch);
        }
      }
      if (input.clearWalls) {
        cp.setPipeline(this.clear.scalarPipeline);
        cp.setBindGroup(0, this.clear.obstacleBind);
        cp.dispatchWorkgroups(DISPATCH_SIZE, DISPATCH_SIZE);
      }
      if (painting) {
        cp.setBindGroup(0, this.simGroup0[input.boundaryMode], this.uniforms.dynamicOffsets, 0, 1);
        cp.setPipeline(this.walls.pipeline);
        cp.setBindGroup(1, this.walls.bind);
        cp.dispatchWorkgroups(DISPATCH_SIZE, DISPATCH_SIZE);
      }
      // Les murs ont changé → la pyramide d'obstacles du multigrid est restreinte
      // niveau par niveau (événementiel : jamais encodé sur une frame sans changement).
      if (painting || input.clearWalls) {
        cp.setBindGroup(0, this.simGroup0[input.boundaryMode], this.uniforms.dynamicOffsets, 0, 1);
        cp.setPipeline(this.mg.obstacleRestrictPipeline);
        for (let l = 0; l < this.mg.levels.length - 1; l++) {
          const coarse = this.mg.levels[l + 1]!;
          cp.setBindGroup(1, this.mg.levels[l]!.obstacleRestrictBind!);
          cp.dispatchWorkgroups(coarse.dispatch, coarse.dispatch);
        }
      }
      // Flux optique : estimation UNE fois par frame (deux images caméra consécutives),
      // avant les sous-pas qui l'appliqueront. Layout autonome — pas de groupe 0.
      if (input.params.cameraFlow && substeps > 0) {
        cp.setPipeline(this.opticalFlow.flowPipeline);
        cp.setBindGroup(0, this.opticalFlow.flowBind[this.lumIdx]);
        cp.dispatchWorkgroups(FLOW_DISPATCH, FLOW_DISPATCH);
        this.lumIdx = flip(this.lumIdx);
      }
      for (let s = 0; s < substeps; s++) {
        this.encodeStep(cp, s, input);
      }
      cp.end();
    }
    this.renderer.encode(
      encoder,
      target,
      this.denIdx,
      this.velIdx,
      this.pressIdx,
      input.viewMode,
      input.params.particles,
    );

    this.submitList[0] = encoder.finish();
    this.device.queue.submit(this.submitList);
  }

  /** Remise à zéro des 8 textures de champs de fluide — les murs sont préservés. */
  private encodeClear(cp: GPUComputePassEncoder): void {
    cp.setPipeline(this.clear.rgbaPipeline);
    for (const target of this.clear.rgbaTargets) {
      cp.setBindGroup(0, target.bind);
      cp.dispatchWorkgroups(target.dispatch, target.dispatch);
    }
    cp.setPipeline(this.clear.scalarPipeline);
    for (const target of this.clear.scalarTargets) {
      cp.setBindGroup(0, target.bind);
      cp.dispatchWorkgroups(target.dispatch, target.dispatch);
    }
  }

  /** Encode un sous-pas de simulation complet (passes 1–7 du schéma de frame). */
  private encodeStep(cp: GPUComputePassEncoder, substep: number, input: FrameInput): void {
    const boundaryMode = input.boundaryMode;
    const n = DISPATCH_SIZE;
    // Groupe 0 partagé par toutes les passes ; l'offset dynamique sélectionne le slot
    // d'uniforms du sous-pas (Uint32Array pré-créé — pas de tableau temporaire), la
    // variante de bind group sélectionne le sampler clamp/repeat du mode de frontières.
    cp.setBindGroup(0, this.simGroup0[boundaryMode], this.uniforms.dynamicOffsets, substep, 1);

    // 1. Advection MacCormack de la vélocité : prédicteur → scratch, correcteur → ping-pong.
    cp.setPipeline(this.advect.velPredictPipeline);
    cp.setBindGroup(1, this.advect.velPredictBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);
    cp.setPipeline(this.advect.velCorrectPipeline);
    cp.setBindGroup(1, this.advect.velCorrectBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);
    this.velIdx = flip(this.velIdx);

    // 2. Forces : buoyancy + impulsion souris.
    cp.setPipeline(this.forces.pipeline);
    cp.setBindGroup(1, this.forces.bind[this.velIdx][this.denIdx]);
    cp.dispatchWorkgroups(n, n);
    this.velIdx = flip(this.velIdx);

    // 2bis. Flux optique : le mouvement devant la caméra pousse le fluide.
    if (input.params.cameraFlow) {
      cp.setPipeline(this.opticalFlow.applyPipeline);
      cp.setBindGroup(1, this.opticalFlow.applyBind[this.velIdx]);
      cp.dispatchWorkgroups(n, n);
      this.velIdx = flip(this.velIdx);
    }

    // 3. Vorticity confinement : rotationnel puis force de renforcement des tourbillons.
    cp.setPipeline(this.vorticity.curlPipeline);
    cp.setBindGroup(1, this.vorticity.curlBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);
    cp.setPipeline(this.vorticity.confinePipeline);
    cp.setBindGroup(1, this.vorticity.confineBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);
    this.velIdx = flip(this.velIdx);

    // 4. Divergence du champ de vélocité.
    cp.setPipeline(this.project.divergencePipeline);
    cp.setBindGroup(1, this.project.divergenceBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);

    // 5. Résolution de la pression : V-cycles multigrid (défaut) ou Jacobi simple
    //    (bascule de debug dans le panneau, pour comparer la convergence dans la vue 3).
    //    Warm start dans les deux cas : la pression de la frame précédente sert de départ.
    // Le mode ouvert utilise désormais le même opérateur Neumann que les parois
    // (boîte fermée + bande éponge dans l'advection) : le multigrid y est stable.
    if (input.params.multigrid) {
      const cycles = Math.max(1, Math.round(input.params.vcycles));
      for (let k = 0; k < cycles; k++) {
        this.encodeVCycle(cp, substep, boundaryMode);
      }
    } else {
      cp.setPipeline(this.project.jacobiPipeline);
      for (let i = 0; i < input.pressureIterations; i++) {
        cp.setBindGroup(1, this.project.jacobiBind[this.pressIdx]);
        cp.dispatchWorkgroups(n, n);
        this.pressIdx = flip(this.pressIdx);
      }
    }

    // 6. Projection : v ← v − ∇p.
    cp.setPipeline(this.project.gradientPipeline);
    cp.setBindGroup(1, this.project.gradientBind[this.pressIdx][this.velIdx]);
    cp.dispatchWorkgroups(n, n);
    this.velIdx = flip(this.velIdx);

    // 6bis. Divergence résiduelle post-projection — uniquement pour la vue de debug 3
    // (la texture est de toute façon recalculée au sous-pas suivant avant Jacobi).
    // Elle doit rester ≈ 0 : c'est la preuve visuelle que la projection fonctionne.
    cp.setPipeline(this.project.divergencePipeline);
    cp.setBindGroup(1, this.project.divergenceBind[this.velIdx]);
    cp.dispatchWorkgroups(n, n);

    // 7. Advection MacCormack des densités (grille dye) + injection souris.
    const nd = DYE_DISPATCH_SIZE;
    cp.setPipeline(this.advect.denPredictPipeline);
    cp.setBindGroup(1, this.advect.denPredictBind[this.velIdx][this.denIdx]);
    cp.dispatchWorkgroups(nd, nd);
    cp.setPipeline(this.advect.denCorrectPipeline);
    cp.setBindGroup(1, this.advect.denCorrectBind[this.velIdx][this.denIdx]);
    cp.dispatchWorkgroups(nd, nd);
    this.denIdx = flip(this.denIdx);

    // 8. Particules traceuses : advection RK2 sur la vélocité projetée (in-place).
    if (input.params.particles) {
      cp.setPipeline(this.particles.pipeline);
      cp.setBindGroup(1, this.particles.bind[this.velIdx]);
      cp.dispatchWorkgroups(PARTICLE_DISPATCH);
    }
  }

  /**
   * Encode un V-cycle multigrid complet sur la pression du niveau 0 (warm start).
   * Descente : lissage pondéré, résidu, restriction ; niveau le plus grossier : lissage
   * long ; remontée : prolongation de la correction + post-lissage. Les index ping-pong
   * de chaque niveau vivent dans `mgIdx` (tableau réutilisé) ; celui du niveau 0 est
   * synchronisé avec `pressIdx`, que la soustraction du gradient consommera.
   */
  private encodeVCycle(
    cp: GPUComputePassEncoder,
    substep: number,
    boundaryMode: BoundaryMode,
  ): void {
    const levels = this.mg.levels;
    const last = levels.length - 1;
    const idx = this.mgIdx;
    idx[0] = this.pressIdx;

    // L'équation d'erreur des niveaux grossiers part de zéro à chaque cycle.
    cp.setPipeline(this.clear.scalarPipeline);
    for (let l = 1; l <= last; l++) {
      idx[l] = 0;
      const lev = levels[l]!;
      cp.setBindGroup(0, lev.clearBind!);
      cp.dispatchWorkgroups(lev.dispatch, lev.dispatch);
    }
    // Le clear utilise un autre pipeline layout : on rétablit le groupe 0 partagé.
    cp.setBindGroup(0, this.simGroup0[boundaryMode], this.uniforms.dynamicOffsets, substep, 1);

    // Descente.
    for (let l = 0; l <= last; l++) {
      const lev = levels[l]!;
      const smoothCount = l === last ? MG_COARSE_SMOOTH : MG_PRE_SMOOTH;
      cp.setPipeline(this.mg.smoothPipeline);
      for (let i = 0; i < smoothCount; i++) {
        cp.setBindGroup(1, lev.smoothBind[idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch);
        idx[l] = idx[l]! ^ 1;
      }
      if (l < last) {
        const coarse = levels[l + 1]!;
        cp.setPipeline(this.mg.residualPipeline);
        cp.setBindGroup(1, lev.residualBind![idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch);
        cp.setPipeline(this.mg.restrictPipeline);
        cp.setBindGroup(1, lev.restrictBind!);
        cp.dispatchWorkgroups(coarse.dispatch, coarse.dispatch);
      }
    }

    // Remontée.
    for (let l = last - 1; l >= 0; l--) {
      const lev = levels[l]!;
      cp.setPipeline(this.mg.prolongPipeline);
      cp.setBindGroup(1, lev.prolongBind![idx[l] as PingIndex][idx[l + 1] as PingIndex]);
      cp.dispatchWorkgroups(lev.dispatch, lev.dispatch);
      idx[l] = idx[l]! ^ 1;
      cp.setPipeline(this.mg.smoothPipeline);
      for (let i = 0; i < MG_POST_SMOOTH; i++) {
        cp.setBindGroup(1, lev.smoothBind[idx[l] as PingIndex]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatch);
        idx[l] = idx[l]! ^ 1;
      }
    }
    this.pressIdx = idx[0] as PingIndex;
  }
}
