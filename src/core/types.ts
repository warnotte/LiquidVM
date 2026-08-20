/**
 * Types partagés du cœur de simulation.
 * L'état d'input est volontairement abstrait : la couche plateforme (web aujourd'hui,
 * Dawn/wgpu demain) le remplit depuis ses propres events — le core n'en sait rien.
 */

/** Index de ping-pong (deux textures par champ). */
export type PingIndex = 0 | 1;

/** Identifiant d'un des trois fluides (index dans FLUIDS). */
export type FluidId = 0 | 1 | 2;

/**
 * MATIÈRE déposée par l'outil « injecter » : les trois fluides, ou le feu (3) —
 * qui n'est pas un fluide transporté mais une injection de température.
 * Le modèle mental de l'interface : la matière dit QUOI déposer, l'outil dit COMMENT agir.
 */
export type SubstanceId = 0 | 1 | 2 | 3;
export const SUBSTANCE_FIRE: SubstanceId = 3;

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
 * 2 = pression, 3 = divergence (doit rester ~nulle après projection), 4 = vorticité,
 * 5 = caméra (image webcam + flux optique en couleurs — l'instrument de debug du flux).
 */
export type ViewMode = 0 | 1 | 2 | 3 | 4 | 5;
export const VIEW_MODE_COUNT = 6;

/**
 * Outil du clic gauche : 0 = injecter (la matière sélectionnée + impulsion du geste),
 * 1 = gommer (densités et chaleur), 2 = tourbillon (vélocité tangentielle),
 * 3 = souffle (jet directionnel). La pression ne s'édite pas : elle est entièrement
 * re-résolue à chaque frame.
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
  /** Particules traceuses affichées et advectées. */
  particles: boolean;
  /** Intensité lumineuse des traînées de particules (0 = invisibles). */
  particleIntensity: number;
  /**
   * Flux optique webcam actif. La plateforme copie la caméra dans la texture exposée
   * par le core (cameraTexture) — c'est elle qui gère getUserMedia et la permission.
   */
  cameraFlow: boolean;
  /** Échelle de la buoyancy des encres (0 = bain de marbrure inerte, 1 = normal). */
  buoyancyScale: number;
  /** Gain du flux optique (multiplie la force appliquée au fluide). */
  flowStrength: number;
  /** Porte de bruit du flux : différence d'intensité minimale considérée comme du
   *  mouvement (à baisser en pièce sombre, où le capteur est timide). */
  flowGate: number;
}

/** Paramètres de rendu réglables à chaud. */
export interface RenderTuning {
  exposure: number;
  bloomStrength: number;
  /** Rendu « papier marbré » : fond crème, encres opaques superposées en pigments
   *  (multiplicatif), sans bloom ni tone-mapping — le look du papier, pas du néon. */
  paper: boolean;
  /** Rapport largeur/hauteur du canvas (mis à jour par la plateforme au resize) —
   *  sert à garder l'encart caméra carré à l'écran. */
  aspect: number;
}

/** Tout ce que le core reçoit de l'extérieur à chaque frame. */
export interface FrameInput {
  pointer: PointerInput;
  /** Matière déposée par l'outil « injecter » (fluides 0-2 ou feu = 3). */
  selectedFluid: SubstanceId;
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
  /**
   * Opération de marbrure de la frame (marbling mathématique — déformations exactes
   * préservant les aires, appliquées à la texture d'encres même en pause) :
   * 0 = goutte (dépose l'encre sélectionnée, déplace le reste en anneaux),
   * 1 = stylet (tire les encres le long du geste), 2 = peigne (n stylets parallèles).
   */
  marble: {
    pending: boolean;
    /** 3 = remplir tout le bain de l'encre sélectionnée (« fond »). */
    tool: 0 | 1 | 2 | 3;
    ax: number;
    ay: number;
    bx: number;
    by: number;
  };
}

/** Bascule l'index de ping-pong. */
export function flip(i: PingIndex): PingIndex {
  return (i ^ 1) as PingIndex;
}

/** Paire de ressources indexée par PingIndex. */
export type Pair<T> = readonly [T, T];
