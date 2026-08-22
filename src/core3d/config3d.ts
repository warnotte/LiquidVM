/**
 * Configuration du prototype 3D volumétrique. Une seule constante de résolution :
 * tout le reste (dispatchs, échelles de forces) en découle. Doctrine identique au 2D :
 * zéro allocation par frame, zéro lecture GPU→CPU, un seul CommandEncoder.
 */

/** Côté de la grille cubique (voxels). 256³ ≈ 16,8 M de cellules par défaut ;
 *  choisissable à l'init via setGrid3 (la plateforme lit ?grid=…).
 *  Mesuré sur la machine de référence (RTX 5070 Ti 16 Go) : 256³ et 320³ à
 *  60 FPS. 384³/512³ exigent maxBufferSize relevé au max de l'adapter (gpu.ts) :
 *  les staging buffers internes de Dawn (zéro-init des textures 3D) sont validés
 *  contre cette limite, dont le défaut (256 Mio) faisait échouer chaque submit
 *  EN SILENCE (écran noir, HUD vivant — 320³ passait à 2 % près). En cas d'échec
 *  d'allocation réel, toujours juger l'IMAGE, pas le HUD.
 *  L'export VDB n'est disponible qu'aux multiples de 128 (128/256/384/512). */
export let GRID3 = 256;
export let SCALE3 = GRID3 / 128;

export const GRID3_CHOICES = [128, 256, 320, 384, 512] as const;

/** Estimation VRAM du volume (octets) : ~88 o/voxel plein format (2× vélocité
 *  + 2× densités + 2× espèces + scratchs + rotationnel en rgba16float, pression
 *  ×2 + divergence + résidu en r32float) + pyramide multigrid (~16 o/voxel
 *  répartis sur les niveaux ≥ 1, somme des 1/8^k ≈ ×0,143). */
export function estimateVram3(n: number): number {
  return n * n * n * (88 + 16 * 0.143);
}

/** À appeler AVANT la création de la simulation. Valide et propage les dérivés. */
export function setGrid3(n: number): void {
  if ((GRID3_CHOICES as readonly number[]).includes(n)) {
    GRID3 = n;
    SCALE3 = n / 128;
    DISPATCH3 = Math.ceil(n / WG3);
  }
}

/** Braises : nombre de particules (buffer fixe 32 o/particule ≈ 1 Mo).
 *  L'autorégulation par rejet fait que seule une fraction vit à la fois. */
export const EMBERS3 = 32768;

/** Workgroups 4×4×4 = 64 threads ; dispatch cubique. */
export const WG3 = 4;
export let DISPATCH3 = Math.ceil(GRID3 / WG3);

/** Multigrid 3D : pyramide GRID3 → 8³, lissages du V-cycle. */
export const MG3_COARSEST_SIZE = 8;
export const MG3_PRE_SMOOTH = 2;
export const MG3_POST_SMOOTH = 2;
export const MG3_COARSE_SMOOTH = 16;

/** Palette des trois MATIÈRES (canaux xyz de la densité) : fumée grise (produit de
 *  combustion et encre neutre), encre magenta (lourde, retombe en refroidissant),
 *  CARBURANT ambré (vapeur inflammable — n'émet pas de chaleur, s'embrase au contact
 *  d'une flamme). Dupliquée dans raymarch.wgsl (les shaders sont autonomes). */
export const INK_COLORS = [
  [0.55, 0.6, 0.68],
  [1.0, 0.3, 0.8],
  [1.0, 0.8, 0.45],
] as const;
export const INK_NAMES = ['fumée', 'encre', 'carburant'] as const;

/** Réglages à chaud (panneau) : tous initialisés depuis SIM3_DEFAULTS. */
export interface Sim3Tuning {
  timeScale: number;
  buoyancy: number;
  vorticityStrength: number;
  velocityDissipation: number;
  emitHeat: number;
  emitInkRate: number;
  heatCooling: number;
  inkDissipation: number;
  burnRate: number;
  heatYield: number;
  expansion: number;
  oxygenRecover: number;
  blowForce: number;
}

export function defaultTuning3(): Sim3Tuning {
  return {
    timeScale: 1,
    buoyancy: SIM3_DEFAULTS.buoyancy,
    vorticityStrength: SIM3_DEFAULTS.vorticityStrength,
    velocityDissipation: SIM3_DEFAULTS.velocityDissipation,
    emitHeat: SIM3_DEFAULTS.emitHeat,
    emitInkRate: SIM3_DEFAULTS.emitSmoke,
    heatCooling: SIM3_DEFAULTS.heatCooling,
    inkDissipation: SIM3_DEFAULTS.smokeDissipation,
    burnRate: SIM3_DEFAULTS.burnRate,
    heatYield: SIM3_DEFAULTS.heatYield,
    expansion: SIM3_DEFAULTS.expansion,
    oxygenRecover: SIM3_DEFAULTS.oxygenRecover,
    blowForce: SIM3_DEFAULTS.blowForce,
  };
}

export const SIM3_DEFAULTS = {
  /** Solveur de pression : multigrid par défaut, Jacobi en repli comparatif.
   *  1 V-cycle warm-starté suffit pour le visuel — 2 coûtent ~6 balayages de plus. */
  multigrid: true,
  vcycles: 1,
  /** Itérations de Jacobi quand le multigrid est coupé. */
  jacobiIterations: 36,
  /** Force du vorticity confinement (ε). RÉACTIVÉ (2026-08-22) : |ω| est flouté
   *  (boîte 3³, passe blur_curl) AVANT le gradient — c'était le gradient de la
   *  magnitude brute à ±1 voxel qui injectait le grain de grille (ancien défaut
   *  0). Calibré par captures EPS2-0/8/16/24 à 256³ : 8 = subtil, 16 = riche
   *  encore cohérent, 24 = panache déchiqueté (sans grain — l'échec est devenu
   *  « trop chaotique », plus jamais granuleux). Défaut 12 = vivant avec marge. */
  vorticityStrength: 12,
  /** Dissipations (1/s) : vélocité, fumée, refroidissement de la chaleur. */
  velocityDissipation: 0.03,
  smokeDissipation: 0.20,
  heatCooling: 0.9,
  /** Poussée thermique (voxels/s² par unité de chaleur). */
  buoyancy: 150,
  /** Émetteur : rayon (fraction de N), débits (unités/s), impulsions (voxels/s). */
  emitterRadius: 0.055,
  emitHeat: 3.6,
  emitSmoke: 2.4,
  emitUpVelocity: 60,
  emitWobbleVelocity: 26,
  /** Souffle du pointeur : rayon du tube (fraction du monde) et force
   *  (unités monde/s² par unité de geste NDC). */
  blowRadius: 0.10,
  blowForce: 260,
  /** Saisie au pointeur : distance rayon-objet maximale pour attraper (monde). */
  grabRadius: 0.11,
  /** Sphère-obstacle : rayon (fraction du monde), position de départ. */
  sphereRadius: 0.085,
  sphereStart: [0.5, 0.45, 0.5],
  /** Nombre maximal d'émetteurs simultanés. */
  maxEmitters: 4,
  /** Combustion (modèle Feldman/Fedkiw simplifié) : taux de réaction (1/s au-dessus
   *  de l'ignition), chaleur dégagée par unité de carburant, expansion volumique au
   *  front de flamme (source de divergence, voxels/s — remise à l'échelle SCALE3). */
  burnRate: 3.0,
  heatYield: 0.7,
  expansion: 10,
  /** OXYGÈNE (canal x de la texture d'espèces, initialisé à 1 partout) : la
   *  combustion le consomme (stœchiométrie dans le shader), il revient lentement
   *  (la boîte « fuit ») et le SOUFFLE en apporte — le soufflet de forge. */
  oxygenRecover: 0.015,
  blowOxygen: 2.5,
  /** Poids propre des matières (voxels/s² par unité — remis à l'échelle SCALE3) :
   *  la fumée flotte, l'encre magenta est lourde, la vapeur de carburant coule
   *  doucement (elle s'accumule et s'étale tant qu'elle ne brûle pas). */
  inkWeights: [0, 55, 16],
  /** Rendu. Lueur = in-scattering du volume de lueur (glow3d.wgsl) ; bloom =
   *  halo HDR de la présentation (post3d.wgsl). */
  raymarchSteps: 160,
  exposure: 1.25,
  glowStrength: 1.8,
  bloomStrength: 0.35,
  /** Braises : interrupteur (défaut COUPÉ — utilité en cours d'évaluation par
   *  l'utilisateur, touche B) et débit quand actives. Coupées = les passes
   *  update et draw sont entièrement sautées, coût strictement nul. */
  embersOn: false,
  emberStrength: 0.7,
  /** Caméra orbitale initiale. */
  camAzimuth: 0.6,
  camElevation: 0.25,
  camRadius: 2.1,
} as const;
