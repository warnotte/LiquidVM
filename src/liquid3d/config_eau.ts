/**
 * Configuration du chantier EAU (voir PLAN-EAU.md — à lire avant tout code ici).
 * Doctrine identique au reste du moteur : zéro DOM dans liquid3d/, zéro
 * allocation par frame, un CommandEncoder, zéro readback en boucle.
 */

/** Côté de la grille MAC (voxels), choisissable à l'init via setGridEau (la
 *  plateforme lit `?grid=…`). La boîte garde la MÊME taille physique : monter
 *  la résolution affine les voxels, donc les grandeurs exprimées en voxels/s
 *  (gravité, plafonds CFL, filet float16) sont multipliées par SCALE_EAU —
 *  exactement le rôle de SCALE3 pour le feu et de FORCE_SCALE en 2D. */
export let GRID_EAU = 128;
export let SCALE_EAU = GRID_EAU / 128;

/** Nombre de particules : 8 par cellule de la colonne initiale, laquelle fait
 *  n/2 × n/4 × n cellules — soit exactement n³ particules (2 M à 128³).
 *  Buffer fixe 64 o/particule (position + vitesse + matrice affine APIC). */
export let PARTICLES_EAU = GRID_EAU ** 3;

/** Résolutions proposées. Divisibles par 8 (blocs de tri) et par WG_GRID.
 *  MESURÉ sur la machine de référence (RTX 5070 Ti) — voir le journal
 *  PLAN-EAU : au-delà de 128³ le coût monte en n³ sur DEUX fronts (cellules ET
 *  particules) et le solveur de pression domine. */
export const GRID_EAU_CHOICES = [96, 128, 160, 192] as const;

/** Estimation VRAM (octets) : ~44 o/voxel de textures (3× vélocité rgba16float,
 *  2× pression + divergence r32float, densité floutée rgba16float) + 28 o/voxel
 *  de tampons atomiques (6 accumulateurs + comptage) + 64 o/particule. */
export function estimateVramEau(n: number): number {
  return n * n * n * (44 + 28 + 64);
}

/** À appeler AVANT la création de la simulation. Valide et propage les dérivés. */
export function setGridEau(n: number): void {
  if ((GRID_EAU_CHOICES as readonly number[]).includes(n)) {
    GRID_EAU = n;
    SCALE_EAU = n / 128;
    PARTICLES_EAU = n ** 3;
    SORT_BLOCKS = (n / SORT_BLOCK) ** 3;
  }
}

/** vec4 par particule (J2 APIC) : [0] pos.xyz + c_w.x · [1] vel.xyz + c_w.y ·
 *  [2] c_u.xyz + c_w.z · [3] c_v.xyz — c_u/c_v/c_w = gradient de chaque
 *  composante MAC de la vitesse au point de la particule (matrice affine C). */
export const PARTICLE_STRIDE = 4;

/** Échelle de virgule fixe des accumulations atomiques P2G (i32) :
 *  vitesses ≤ ~500 voxels/s × poids ≤ 1 × 256 → marge i32 très confortable
 *  même avec ~64 particules contribuant à une même face. */
export const FIXED_POINT_SCALE = 256;

/** Workgroup du scatter (sur les particules) et des passes de grille (3D). */
export const WG_PARTICLES = 64;
export const WG_GRID = 4;

/** Les passes de particules dispatchent en 2D : `maxComputeWorkgroupsPerDimension`
 *  vaut 65535, or n³/64 le dépasse dès 192³ (110 592 groupes). Largeur fixe de
 *  1024 groupes = 65 536 invocations par rangée ; l'index global du shader est
 *  `gid.x + gid.y * 65536`. La même constante est écrite en dur dans les WGSL
 *  (PARTICLE_ROW) — les deux doivent rester d'accord. */
export const PARTICLE_DISPATCH_X = 1024;

/** Dispatch 2D pour `n` particules : [x, y] groupes. */
export function particleDispatch(n: number): [number, number] {
  const groups = Math.ceil(n / WG_PARTICLES);
  return [PARTICLE_DISPATCH_X, Math.ceil(groups / PARTICLE_DISPATCH_X)];
}

/** Tri périodique par BLOC de 8³ voxels (16³ = 4096 blocs) : maintient l'état
 *  « trié » validé par J0 (1,83 ms vs 2,57 ms en désordre). Toutes les
 *  SORT_INTERVAL frames — le désordre croît lentement. */
export const SORT_BLOCK = 8;
export let SORT_BLOCKS = (GRID_EAU / SORT_BLOCK) ** 3;
export const SORT_INTERVAL = 12;

/** Défauts de la simulation (réglables à chaud au panneau). */
export const EAU_DEFAULTS = {
  /** Gravité à 128³ (voxels/s²) : ~9,81 m/s² pour une boîte de ~1,5 m.
   *  Multipliée par SCALE_EAU à l'écriture des uniforms. */
  gravity: 840,
  /** Transfert APIC (J2, Jiang 2015 : vitesse + matrice affine C par
   *  particule — le bruit FLIP en moins, la dissipation PIC en moins) ;
   *  false = FLIP/PIC avec le mélange ci-dessous. */
  apic: true,
  /** Mélange FLIP/PIC : 1 = FLIP pur (vif, bruité), 0 = PIC pur (amorti). */
  flipBlend: 0.92,
  /** Colonne initiale, en FRACTION du côté : 0,5 = basse et large (n/2 × n/4),
   *  0,25 = haute et étroite (n/4 × n/2). Profondeur = n dans les deux cas. */
  damWidth: 0.5,
  /** Contrôle de densité (1/s) : expansion des cellules tassées, compression
   *  des cellules intérieures sous-denses (voir eau_grid.wgsl). */
  densityRate: 10,
  /** Multigrid masqué (J4) : le solveur de pression par défaut. Jacobi reste le
   *  repli permanent au panneau — c'est la convention du plan, et le seul moyen
   *  honnête de comparer (A/B à état de simulation égal). */
  multigrid: true,
  /** V-cycles par sous-pas. MESURÉ à 192³ (résidu moyen / FPS) : Jacobi 100 →
   *  0,53 / 27 · Jacobi 30 → 3,11 / 40 · MG ×2 → 1,20 / 44 · MG ×4 → 0,21 / 40.
   *  Quatre cycles dominent Jacobi 100 sur LES DEUX axes, à 128³ comme à 192³. */
  mgCycles: 4,
  /** Niveaux de pyramide (élevé = descend jusqu'au plus grossier disponible). */
  mgLevels: 8,
  /** Restriction des masques permissive (fluide dominant) : RÉFUTÉE par la
   *  mesure (voir le journal J4 de PLAN-EAU) — elle explose quel que soit
   *  l'ancrage. Laissée accessible par `?permissive` pour re-tester, jamais
   *  proposée au panneau : ce serait offrir un bouton qui casse la simulation. */
  mgPermissiveMask: false,
  /** ε de régularisation des niveaux grossiers. Sans effet mesurable sur la
   *  règle conservatrice (leur système n'y est jamais singulier) : c'est une
   *  ASSURANCE contre le seul cas qui l'est vraiment — une poche d'eau
   *  entièrement close par du solide, qui n'a aucune cellule d'air pour fixer
   *  sa constante. Coût mesuré nul. */
  mgAnchor: 0.1,
  /** Balayages Jacobi par sous-pas, quand le multigrid est coupé. LEÇON J1 : Jacobi propage ~1 cellule par
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
  /** Plafond CFL de la vitesse de la boule à 128³ (voxels/s), ×SCALE_EAU. */
  sphereSpeedCap: 400,
  /** Rendu : surface (défaut) ou points bruts (instrument physique). */
  renderPoints: false,
  /** Absorption de l'eau (×), 1 = bleu-vert d'aquarium. */
  absorption: 1.0,
  /** Seuil d'iso-surface (densité floutée, 1 = repos). */
  surfaceIso: 0.4,
  /** Mousse (eau aérée blanchie) : 0 = eau parfaitement claire. */
  foam: 0.7,
  /** Vue debug du rendu (0 = aucune). */
  debugView: 0,
  pointSize: 0.0035,
  exposure: 1.0,
  /** Caméra orbitale initiale : la boîte entière doit tenir dans le cadre. */
  camAzimuth: 0.55,
  camElevation: 0.32,
  camRadius: 2.4,
} as const;
