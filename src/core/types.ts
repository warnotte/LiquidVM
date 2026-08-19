/**
 * Types partagés du cœur de simulation.
 * L'état d'input est volontairement abstrait : la couche plateforme (web aujourd'hui,
 * Dawn/wgpu demain) le remplit depuis ses propres events — le core n'en sait rien.
 */

/** Index de ping-pong (deux textures par champ). */
export type PingIndex = 0 | 1;

/** Identifiant d'un des trois fluides. */
export type FluidId = 0 | 1 | 2;

/**
 * Mode de frontières du domaine :
 * 0 = parois fermées (boîte) ;
 * 1 = périodique (tore — ce qui sort d'un bord rentre par l'autre) ;
 * 2 = ouvert (ce qui sort du domaine disparaît définitivement).
 */
export type BoundaryMode = 0 | 1 | 2;
export const BOUNDARY_MODE_COUNT = 3;

/**
 * Vue de rendu : 0 = fluides, 1 = vélocité (teinte = direction, luminosité = norme),
 * 2 = pression, 3 = divergence (doit rester ~nulle après projection), 4 = vorticité.
 */
export type ViewMode = 0 | 1 | 2 | 3 | 4;
export const VIEW_MODE_COUNT = 5;

/**
 * Outil du clic gauche : 0 = injecter (fluide + impulsion directionnelle),
 * 1 = gommer la densité, 2 = tourbillon (vélocité tangentielle),
 * 3 = souffle (vélocité radiale sortante). La pression ne s'édite pas : elle est
 * entièrement re-résolue à chaque frame par la projection.
 */
export type ToolId = 0 | 1 | 2 | 3;
export const TOOL_COUNT = 4;

/**
 * État du pointeur, en coordonnées normalisées [0,1] relatives à la cible de rendu.
 * Origine en haut à gauche, y vers le bas — même convention que la grille de simulation.
 */
export interface PointerInput {
  x: number;
  y: number;
  /** Déplacement accumulé depuis la frame précédente (unités normalisées). */
  dx: number;
  dy: number;
  /** Bouton principal : injection de fluide + vélocité. */
  down: boolean;
  /** Bouton secondaire : pinceau à murs. */
  wall: boolean;
  /** Modificateur du pinceau : true = gommer au lieu de construire. */
  erase: boolean;
}

/** Paramètres de simulation réglables à chaud (panneau de debug côté plateforme). */
export interface SimTuning {
  /** Dissipation de la vélocité (≈ viscosité effective), en 1/s. */
  velocityDissipation: number;
  /** Force de la vorticity confinement (0 = physique brute, plus = plus nerveux). */
  vorticityStrength: number;
  /** Conversion déplacement souris → impulsion de vélocité. */
  splatForce: number;
  /** Rayon du splat souris, en texels. */
  splatRadius: number;
  /** Débit d'injection de densité, en unités/s. */
  splatDensity: number;
  /** Facteur de vitesse du temps simulé (0 = figé, 1 = temps réel). */
  timeScale: number;
  /** Correcteur MacCormack actif (false = semi-lagrangien pur, pour comparer). */
  macCormack: boolean;
  /** Solveur de pression multigrid (false = Jacobi simple à `pressureIterations`). */
  multigrid: boolean;
  /** Nombre de V-cycles par sous-pas quand le multigrid est actif. */
  vcycles: number;
}

/** Paramètres de rendu réglables à chaud. */
export interface RenderTuning {
  exposure: number;
  bloomStrength: number;
}

/** Tout ce que le core reçoit de l'extérieur à chaque frame. */
export interface FrameInput {
  pointer: PointerInput;
  selectedFluid: FluidId;
  boundaryMode: BoundaryMode;
  /** Outil actif du clic gauche. */
  tool: ToolId;
  /** Vidange des champs de fluide demandée (consommée par la frame courante ; murs conservés). */
  reset: boolean;
  /** Effacement de tous les murs (consommé par la frame courante). */
  clearWalls: boolean;
  /** Simulation figée (le rendu continue, on peut encore construire des murs). */
  paused: boolean;
  /** Avancer d'exactement une frame de simulation pendant la pause (consommé). */
  stepOnce: boolean;
  /** Nombre d'itérations de Jacobi pour cette frame. */
  pressureIterations: number;
  /** Vue de rendu active (fluides ou champs de debug). */
  viewMode: ViewMode;
  /** Réglages à chaud de la simulation et du rendu. */
  params: SimTuning;
  render: RenderTuning;
}

/** Bascule l'index de ping-pong. */
export function flip(i: PingIndex): PingIndex {
  return (i ^ 1) as PingIndex;
}

/** Paire de ressources indexée par PingIndex. */
export type Pair<T> = readonly [T, T];
