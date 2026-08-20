/**
 * Configuration du prototype 3D volumétrique. Une seule constante de résolution :
 * tout le reste (dispatchs, échelles de forces) en découle. Doctrine identique au 2D :
 * zéro allocation par frame, zéro lecture GPU→CPU, un seul CommandEncoder.
 */

/** Côté de la grille cubique (voxels). 128³ ≈ 2,1 M de cellules. */
export const GRID3 = 128;

/** Workgroups 4×4×4 = 64 threads ; dispatch cubique. */
export const WG3 = 4;
export const DISPATCH3 = Math.ceil(GRID3 / WG3);

/** Multigrid 3D : pyramide GRID3 → 8³, lissages du V-cycle. */
export const MG3_COARSEST_SIZE = 8;
export const MG3_PRE_SMOOTH = 2;
export const MG3_POST_SMOOTH = 2;
export const MG3_COARSE_SMOOTH = 16;

export const SIM3_DEFAULTS = {
  /** Solveur de pression : multigrid par défaut, Jacobi en repli comparatif. */
  multigrid: true,
  vcycles: 2,
  /** Itérations de Jacobi quand le multigrid est coupé. */
  jacobiIterations: 36,
  /** Force du vorticity confinement (ε). Défaut 0 : à 128³, le gradient de |ω| à
   *  ±1 voxel injecte du grain à l'échelle de la grille dès ε≈3 (calibré par
   *  captures EPS-*) — MacCormack seul donne un panache turbulent superbe. À
   *  réactiver quand le gradient sera lissé (piste : flouter |ω| avant ∇). */
  vorticityStrength: 0,
  /** Dissipations (1/s) : vélocité, fumée, refroidissement de la chaleur. */
  velocityDissipation: 0.03,
  smokeDissipation: 0.10,
  heatCooling: 0.9,
  /** Poussée thermique (voxels/s² par unité de chaleur). */
  buoyancy: 150,
  /** Émetteur : rayon (fraction de N), débits (unités/s), impulsions (voxels/s). */
  emitterRadius: 0.055,
  emitHeat: 3.6,
  emitSmoke: 2.4,
  emitUpVelocity: 60,
  emitWobbleVelocity: 26,
  /** Rendu. */
  raymarchSteps: 160,
  exposure: 1.25,
  /** Caméra orbitale initiale. */
  camAzimuth: 0.6,
  camElevation: 0.25,
  camRadius: 2.1,
} as const;
