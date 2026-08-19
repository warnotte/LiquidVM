/**
 * Configuration centrale de la simulation.
 * Aucune dépendance navigateur — ce module est partagé tel quel par un portage natif.
 */

/** Taille de la grille de simulation (carrée). Seule constante à changer pour redimensionner. */
export const GRID_SIZE = 1024;

/**
 * Les forces absolues (buoyancy, échelle debug) ont été réglées à l'œil sur une grille
 * 512 : ce facteur les remet à l'échelle pour que le comportement NORMALISÉ (fraction
 * du domaine par seconde) reste identique quelle que soit la résolution.
 */
const FORCE_SCALE = GRID_SIZE / 512;

/**
 * Taille de la grille des densités (« dye »), découplée de la vélocité : la résolution
 * visible est celle des densités, alors que le coût dominant (projection Jacobi) dépend
 * de la grille de vélocité. 1024² de dye sur 512² de vélocité = image 4× plus fine pour
 * ~15 % de coût en plus (deux passes d'advection MacCormack à 1024²).
 */
export const DYE_SIZE = 2048;

/**
 * Taille de workgroup 16×16 = 256 invocations : c'est exactement la limite par défaut
 * `maxComputeInvocationsPerWorkgroup` (256) de WebGPU, garantie sur tout adapter sans
 * demander de limite supplémentaire. Un carré 16×16 maximise la localité 2D des accès
 * texture (voisinage partagé dans le cache) là où 8×8 sous-utiliserait les SM modernes.
 * Doit rester synchronisé avec les `@workgroup_size(16, 16)` des fichiers WGSL.
 */
export const WORKGROUP_SIZE = 16;

/** Nombre de workgroups à dispatcher par dimension (grille de vélocité). */
export const DISPATCH_SIZE = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);

/** Nombre de workgroups par dimension pour les passes sur la grille de densités. */
export const DYE_DISPATCH_SIZE = Math.ceil(DYE_SIZE / WORKGROUP_SIZE);

/**
 * Solveur multigrid (V-cycle géométrique) pour la pression : niveaux 512² → 8².
 * Le lissage est un Jacobi pondéré (ω = 0.8 dans multigrid.wgsl) — il tue les hautes
 * fréquences du résidu à chaque niveau, la hiérarchie s'occupe des basses. Deux
 * V-cycles ≈ le coût de 18 itérations de Jacobi, pour une convergence incomparable.
 */
export const MG_COARSEST_SIZE = 8;
export const MG_PRE_SMOOTH = 2;
export const MG_POST_SMOOTH = 2;
export const MG_COARSE_SMOOTH = 16;

/**
 * Résolution (carrée, fixe) de la scène HDR intermédiaire : la composition des fluides
 * y est rendue en linéaire pré-tone-mapping, le bloom y est calculé, puis la passe de
 * présentation étire vers le canvas. Taille fixe = zéro réallocation au resize.
 */
export const SCENE_SIZE = 2048;
export const BLOOM_MID_SIZE = SCENE_SIZE / 2;
export const BLOOM_WIDE_SIZE = SCENE_SIZE / 4;
export const BLOOM_MID_DISPATCH = Math.ceil(BLOOM_MID_SIZE / WORKGROUP_SIZE);
export const BLOOM_WIDE_DISPATCH = Math.ceil(BLOOM_WIDE_SIZE / WORKGROUP_SIZE);

/**
 * Formats des champs :
 * - Vélocité : `rgba16float` et non `rg16float`, car WebGPU n'autorise PAS `rg16float`
 *   comme format de storage texture (liste restreinte de la spec). Les canaux .ba sont
 *   inutilisés ; le surcoût mémoire (2 Mo à 512²) est le prix d'écritures compute directes
 *   sans passer par des render passes de ping-pong.
 * - Densités : les 3 fluides sont PACKÉS dans les canaux .rgb d'une seule `rgba16float`.
 *   Justification : une seule passe d'advection pour les 3 champs, un seul fetch dans le
 *   fragment shader de composition, 3× moins de bind groups — et la dissipation/buoyancy
 *   par fluide reste possible via des paramètres vectoriels dans l'uniform buffer.
 * - Pression / divergence : `r32float`, storage-capable et lu uniquement via textureLoad
 *   (pas de filtrage nécessaire pour Jacobi — r32float est « unfilterable-float » de base).
 */
export const VELOCITY_FORMAT: GPUTextureFormat = 'rgba16float';
export const DENSITY_FORMAT: GPUTextureFormat = 'rgba16float';
export const SCALAR_FORMAT: GPUTextureFormat = 'r32float';

/** Propriétés d'un type de fluide (index = canal de la texture de densité). */
export interface FluidProps {
  readonly name: string;
  /** Couleur RGB linéaire utilisée par la composition. */
  readonly color: readonly [number, number, number];
  /**
   * Poussée verticale en texels/s² par unité de densité.
   * Positif = monte (l'axe y de la grille pointe vers le bas, la poussée est appliquée en -y).
   * Négatif = coule.
   */
  readonly buoyancy: number;
  /** Taux de dissipation exponentielle de la densité, en 1/s. */
  readonly dissipation: number;
}

/** Les trois fluides (touches 1/2/3). Valeurs artistiques, ajustées à l'œil. */
export const FLUIDS: readonly [FluidProps, FluidProps, FluidProps] = [
  { name: 'eau', color: [0.15, 0.55, 1.0], buoyancy: -40 * FORCE_SCALE, dissipation: 0.03 },
  { name: 'encre', color: [0.85, 0.25, 0.95], buoyancy: 4 * FORCE_SCALE, dissipation: 0.12 },
  { name: 'fumée', color: [1.0, 0.87, 0.62], buoyancy: 60 * FORCE_SCALE, dissipation: 0.35 },
];

/** Paramètres par défaut de la simulation (tous réglables à chaud via le panneau Tab). */
export const SIM_DEFAULTS = {
  /** Itérations de Jacobi pour la pression (ajustable avec +/-). */
  pressureIterations: 30,
  pressureIterationsMin: 5,
  pressureIterationsMax: 100,
  pressureIterationsStep: 5,
  /** Dissipation de la vélocité (≈ viscosité effective), en 1/s. */
  velocityDissipation: 0.05,
  /** Rayon du splat souris, en texels. */
  splatRadius: GRID_SIZE * 0.025,
  /** Conversion déplacement souris (texels/step) → impulsion de vélocité (texels/s). ≈ 1/dt à 60 Hz. */
  splatForce: 60,
  /** Débit d'injection de densité au centre du splat, en unités/s. */
  splatDensity: 30,
  /** Facteur de vitesse du temps simulé (1 = temps réel). */
  timeScale: 1,
  /** Nombre de V-cycles multigrid par sous-pas (quand le solveur MG est actif). */
  vcycles: 2,
  vcyclesMax: 4,
  /**
   * dt maximal par pas de simulation. L'advection semi-lagrangienne est inconditionnellement
   * stable (c'est l'intérêt de Stable Fluids), mais un dt trop grand « téléporte » les splats
   * et dégrade la projection ; au-delà de maxDt on découpe la frame en sous-pas (2 max).
   */
  maxDt: 1 / 45,
  maxSubsteps: 2,
  /**
   * Force de la vorticity confinement (Fedkiw 2001) : réinjecte les petits tourbillons
   * que la diffusion numérique dissipe. 0 = désactivé. Avec la grille MAC et MacCormack,
   * le solveur préserve déjà bien les vrais tourbillons — défaut volontairement modéré,
   * à pousser via le panneau de réglages pour un rendu plus nerveux (moins physique).
   */
  vorticityStrength: 8,
  /** Exposition du tone-mapping de la composition. */
  exposure: 1.3,
  /** Échelle d'affichage de la vue debug vélocité : |v| = cette valeur → luminosité max. */
  debugVelocityScale: 300 * FORCE_SCALE,
  /** Intensité du bloom ajouté à la scène (0 = désactivé). */
  bloomStrength: 0.9,
} as const;
