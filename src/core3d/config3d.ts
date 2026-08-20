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

export const SIM3_DEFAULTS = {
  /** Itérations de Jacobi par frame (multigrid 3D : chantier suivant). */
  jacobiIterations: 36,
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
