/**
 * Configuration du chantier EAU (voir PLAN-EAU.md — à lire avant tout code ici).
 * Doctrine identique au reste du moteur : zéro DOM dans liquid3d/, zéro
 * allocation par frame, un CommandEncoder, zéro readback en boucle.
 */

/** Côté de la grille MAC (voxels). 128³ tant que J1–J4 ne sont pas verts. */
export const GRID_EAU = 128;

/** Nombre de particules : 8 par cellule d'eau initiale, ~1/8 de boîte
 *  (64³ cellules × 8 = exactement 2 M). Buffer fixe 64 o/particule = 128 Mo. */
export const PARTICLES_EAU = 2_097_152;

/** vec4 par particule (J2 APIC) : [0] pos.xyz + c_w.x · [1] vel.xyz + c_w.y ·
 *  [2] c_u.xyz + c_w.z · [3] c_v.xyz — c_u/c_v/c_w = gradient de chaque
 *  composante MAC de la vitesse au point de la particule (matrice affine C). */
export const PARTICLE_STRIDE = 4;

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
  /** Transfert APIC (J2, Jiang 2015 : vitesse + matrice affine C par
   *  particule — le bruit FLIP en moins, la dissipation PIC en moins) ;
   *  false = FLIP/PIC avec le mélange ci-dessous. */
  apic: true,
  /** Mélange FLIP/PIC : 1 = FLIP pur (vif, bruité), 0 = PIC pur (amorti). */
  flipBlend: 0.92,
  /** Colonne initiale : 64 = basse (64×32×128), 32 = haute (32×64×128). */
  damWidth: 64,
  /** Contrôle de densité (1/s) : expansion des cellules tassées, compression
   *  des cellules intérieures sous-denses (voir eau_grid.wgsl). */
  densityRate: 10,
  /** Balayages Jacobi par sous-pas. LEÇON J1 : Jacobi propage ~1 cellule par
   *  balayage — il ne tient l'hydrostatique que jusqu'à ~32 cellules de
   *  profondeur d'eau (100 it.). Au-delà : explosion d'énergie (le fond se
   *  comprime, la correction de densité surréagit, FLIP accumule). L'eau
   *  PROFONDE attend le multigrid masqué. */
  jacobiIterations: 100,
  /** Sous-pas par frame : CFL ~2-3 voxels/sous-pas aux vitesses de chute. */
  substeps: 2,
  timeScale: 1,
  /** Boule-obstacle (J3) : rayon et position de départ en fraction de grille. */
  sphereRadius: 0.12,
  /** À demi immergée dans le bassin au repos, et HORS de la colonne initiale
   *  (x < 0,5) : la vague de rupture vient la percuter. */
  sphereStart: [0.76, 0.13, 0.5] as const,
  /** Plafond CFL de la vitesse de la boule (voxels/s) et lissage EMA. */
  sphereSpeedCap: 400,
  /** Rendu : surface (défaut) ou points bruts (instrument physique). */
  renderPoints: false,
  /** Absorption de l'eau (×), 1 = bleu-vert d'aquarium. */
  absorption: 1.0,
  /** Seuil d'iso-surface (densité floutée, 1 = repos). */
  surfaceIso: 0.4,
  /** Vue debug du rendu (0 = aucune). */
  debugView: 0,
  pointSize: 0.0035,
  exposure: 1.0,
  /** Caméra orbitale initiale : la boîte entière doit tenir dans le cadre. */
  camAzimuth: 0.55,
  camElevation: 0.32,
  camRadius: 2.4,
} as const;
