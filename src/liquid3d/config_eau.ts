/**
 * Configuration du chantier EAU (voir PLAN-EAU.md — à lire avant tout code ici).
 * Doctrine identique au reste du moteur : zéro DOM dans liquid3d/, zéro
 * allocation par frame, un CommandEncoder, zéro readback en boucle.
 */

/** Côté de la grille MAC (voxels). 128³ tant que J1–J4 ne sont pas verts. */
export const GRID_EAU = 128;

/** Nombre de particules : 8 par cellule d'eau initiale, ~1/8 de boîte
 *  (64³ cellules × 8 = exactement 2 M). Buffer fixe 32 o/particule = 64 Mo. */
export const PARTICLES_EAU = 2_097_152;

/** Échelle de virgule fixe des accumulations atomiques P2G (i32) :
 *  vitesses ≤ ~500 voxels/s × poids ≤ 1 × 256 → marge i32 très confortable
 *  même avec ~64 particules contribuant à une même face. */
export const FIXED_POINT_SCALE = 256;

/** Workgroup du scatter (1D sur les particules) et des passes de grille (3D). */
export const WG_PARTICLES = 64;
export const WG_GRID = 4;

/** Tri périodique par BLOC de 8³ voxels (16³ = 4096 blocs) : maintient l'état
 *  « trié » validé par J0 (1,83 ms vs 2,57 ms en désordre). Toutes les
 *  SORT_INTERVAL frames — le désordre croît lentement. */
export const SORT_BLOCK = 8;
export const SORT_BLOCKS = (GRID_EAU / SORT_BLOCK) ** 3;
export const SORT_INTERVAL = 12;

/** Défauts de la simulation (réglables à chaud au panneau). */
export const EAU_DEFAULTS = {
  /** Gravité (voxels/s²) : ~9,81 m/s² pour une boîte de ~1,5 m. */
  gravity: 840,
  /** Mélange FLIP/PIC : 1 = FLIP pur (vif, bruité), 0 = PIC pur (amorti). */
  flipBlend: 0.92,
  /** Balayages Jacobi par sous-pas. LEÇON J1 : Jacobi propage ~1 cellule par
   *  balayage — il ne tient l'hydrostatique que jusqu'à ~32 cellules de
   *  profondeur d'eau (100 it.). Au-delà : explosion d'énergie (le fond se
   *  comprime, la correction de densité surréagit, FLIP accumule). L'eau
   *  PROFONDE attend le multigrid masqué. */
  jacobiIterations: 100,
  /** Sous-pas par frame : CFL ~2-3 voxels/sous-pas aux vitesses de chute. */
  substeps: 2,
  timeScale: 1,
  /** Rendu points. */
  pointSize: 0.0035,
  exposure: 1.0,
  /** Caméra orbitale initiale. */
  camAzimuth: 0.55,
  camElevation: 0.18,
  camRadius: 2.2,
} as const;
