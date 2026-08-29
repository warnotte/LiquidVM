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

/** HAUTEUR de la grille, en voxels. Le domaine n'est plus forcément un cube :
 *  la grille devient Nx × Ny × Nz avec Nx = Nz = GRID3 et Ny = GRID3Y.
 *
 *  LES CELLULES RESTENT CUBIQUES, et c'est tout le point. Un domaine à cellules
 *  ANISOTROPES (un cube de voxels étiré) aurait exigé un pas de grille par axe
 *  dans le laplacien, la divergence, le gradient et l'advection — donc rouvrir
 *  un solveur mesuré et juste. Ici, l'opérateur de pression, le multigrid et
 *  l'advection ne changent PAS D'UNE LIGNE : seuls le bornage et la taille des
 *  textures deviennent per-axe. Une boîte deux fois plus haute, c'est deux fois
 *  plus de cellules, pas une nouvelle numérique.
 *
 *  Le monde suit : le domaine mesure 1 × HEIGHT3 × 1, centré en x et z, et posé
 *  sur le sol en y. HEIGHT3 = 1 rend exactement le cube d'avant. */
export let GRID3Y = GRID3;
export let HEIGHT3 = 1;

export const GRID3_CHOICES = [128, 256, 320, 384, 512] as const;
/** Hauteurs proposées : ×1 (le cube), ×2, ×3. */
export const STRETCH3_CHOICES = [1, 2, 3] as const;

/** Estimation VRAM du volume (octets) : ~88 o/voxel plein format (2× vélocité
 *  + 2× densités + 2× espèces + scratchs + rotationnel en rgba16float, pression
 *  ×2 + divergence + résidu en r32float) + pyramide multigrid (~16 o/voxel
 *  répartis sur les niveaux ≥ 1, somme des 1/8^k ≈ ×0,143). */
export function estimateVram3(n: number, stretch = HEIGHT3): number {
  return n * n * n * stretch * (88 + 16 * 0.143);
}

/** À appeler AVANT la création de la simulation. Valide et propage les dérivés. */
/** V-cycles nécessaires à une résolution donnée. Depuis le lisseur rouge-noir
 *  (V(1,1), SOR au coarsest), DEUX cycles suffisent à 384³ là où le Jacobi
 *  pondéré en demandait quatre — vérifié sur la jauge du panache couché (scène
 *  par défaut, 24 s). JAMAIS UN SEUL cycle par frame : à ×1 le solve accumule
 *  frame à frame une erreur que le second cycle corrige, et la scène dégénère
 *  (mesuré : boîte vide à 384³ en quelques secondes — et l'ancien Jacobi ×1
 *  couchait déjà le panache, même famille, ampleur moindre). */
export function vcyclesFor(n: number): number {
  // Le second cycle est quasi gratuit à 256³ et au-dessous (mesuré : 59,3 FPS
  // contre 60,3) : on le prend partout plutôt que d'exposer le cas ×1.
  void n;
  return 2;
}

export function setGrid3(n: number, stretch = HEIGHT3): void {
  if ((GRID3_CHOICES as readonly number[]).includes(n)) {
    GRID3 = n;
    SCALE3 = n / 128;
    DISPATCH3 = Math.ceil(n / WG3);
  }
  if ((STRETCH3_CHOICES as readonly number[]).includes(stretch)) {
    HEIGHT3 = stretch;
  }
  GRID3Y = GRID3 * HEIGHT3;
  DISPATCH3Y = Math.ceil(GRID3Y / WG3);
}

/** Braises : nombre de particules (buffer fixe 32 o/particule ≈ 1 Mo).
 *  L'autorégulation par rejet fait que seule une fraction vit à la fois. */
export const EMBERS3 = 32768;

/** Workgroups 4×4×4 = 64 threads ; dispatch cubique. */
export const WG3 = 4;
export let DISPATCH3 = Math.ceil(GRID3 / WG3);
export let DISPATCH3Y = Math.ceil(GRID3Y / WG3);

/** Multigrid 3D : la pyramide descend tant que le côté reste ≥ cette valeur —
 *  le niveau le plus grossier fait donc entre 16 et 31 de côté (24 pour 384,
 *  16 pour les puissances de deux). Elle s'arrêtait à 8³ ; le niveau 12³ de la
 *  grille 384 déclenchait une pathologie de driver mesurée (frame ×5 à
 *  1 V-cycle, quel que soit le lisseur) — et un coarsest de 24³ lissé 16 fois
 *  coûte toujours à peu près rien. L'ancrage suit : COARSEST_SIDE (constante
 *  override des lisseurs) vaut exactement le côté du dernier niveau. */
export const MG3_COARSEST_SIZE = 16;
export const MG3_PRE_SMOOTH = 1;
export const MG3_POST_SMOOTH = 1;
export const MG3_COARSE_SMOOTH = 8;
/** Au-dessous de ce côté, un niveau est « minuscule » : lissé au Jacobi
 *  ping-pong et non au rouge-noir en place — les dispatches read_write sur ces
 *  textures déclenchent une pathologie de driver mesurée (voir multigrid3d). */
export const MG3_TINY = 24;

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
/** Types de champ de force, dans l'ordre de l'uniforme (0, 1). */
export const FIELD_NAMES = ['tourbillon', 'vent local'] as const;

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
  /** VENT horizontal (voxels/s² par unité de matière). 0 = aucun vent, et la
   *  scène est alors STRICTEMENT identique à ce qu'elle était sans la fonction. */
  windStrength: number;
  /** Amplitude d'oscillation du cap (degrés) et période (s). */
  windSwing: number;
  windPeriod: number;
  /** Cap moyen du vent (degrés). */
  windHeading: number;
  /** SUIE : rendement, évanouissement, et densité au RENDU (0 = image d'avant). */
  sootYield: number;
  sootFade: number;
  sootDensity: number;
  sootCooling: number;
  heatCeiling: number;
  openBand: number;
  openStrength: number;
  openCeilBand: number;
  openCeilStrength: number;
  stratStrength: number;
  stratBase: number;
  outdoor: boolean;
  sunHeight: number;
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
    sootYield: SIM3_DEFAULTS.sootYield,
    sootFade: SIM3_DEFAULTS.sootFade,
    sootDensity: SIM3_DEFAULTS.sootDensity,
    sootCooling: SIM3_DEFAULTS.sootCooling,
    heatCeiling: SIM3_DEFAULTS.heatCeiling,
    openBand: SIM3_DEFAULTS.openBand,
    openStrength: SIM3_DEFAULTS.openStrength,
    openCeilBand: SIM3_DEFAULTS.openCeilBand,
    openCeilStrength: SIM3_DEFAULTS.openCeilStrength,
    stratStrength: SIM3_DEFAULTS.stratStrength,
    stratBase: SIM3_DEFAULTS.stratBase,
    outdoor: SIM3_DEFAULTS.outdoor,
    sunHeight: SIM3_DEFAULTS.sunHeight,
    oxygenRecover: SIM3_DEFAULTS.oxygenRecover,
    blowForce: SIM3_DEFAULTS.blowForce,
    windStrength: SIM3_DEFAULTS.windStrength,
    windSwing: SIM3_DEFAULTS.windSwing,
    windPeriod: SIM3_DEFAULTS.windPeriod,
    windHeading: SIM3_DEFAULTS.windHeading,
  };
}

export const SIM3_DEFAULTS = {
  /** Solveur de pression : multigrid par défaut, Jacobi en repli comparatif. */
  multigrid: true,
  /** V-cycles de pression — TOUJOURS 2, jamais 1 (voir `vcyclesFor()` : à ×1 le
   *  solve dégénère frame à frame ; historiquement le symptôme doux en était le
   *  « panache couché » à 384³, qui a longtemps fait croire qu'il fallait ×4). */
  vcycles: 2,
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
  /** Vent : COUPÉ par défaut — la page reste exactement telle qu'elle était.
   *  Le preset « vent » l'allume, le curseur à 0 l'éteint. */
  windStrength: 0,
  windSwing: 35,
  windPeriod: 9,
  windHeading: 0,
  blowRadius: 0.10,
  blowForce: 260,
  /** Saisie au pointeur : distance rayon-objet maximale pour attraper (monde). */
  grabRadius: 0.11,
  /** MANIPULATEURS (poignées d'axe de l'objet sélectionné) : longueur en fraction
   *  de la demi-hauteur de l'écran — donc CONSTANTE à l'écran, elle grandit avec
   *  la distance à la caméra. Une poignée de taille fixe dans le monde devient
   *  invisible de loin et démesurée de près. `handleGrip` : rayon de saisie, en
   *  fraction de cette longueur (constant à l'écran lui aussi). */
  handleScreen: 0.24,
  handleGrip: 0.2,
  /** Départ des flèches, en fraction de leur longueur : elles laissent le CENTRE
   *  libre. Sans ce vide, cliquer au milieu de l'objet tombait dans la zone de
   *  saisie des trois poignées à la fois et donnait un déplacement contraint là
   *  où on voulait le déplacement libre. Transmis au shader (opts.w) plutôt que
   *  recopié : une seule source pour le dessin et pour la saisie. */
  handleInner: 0.3,
  /** BOUTON D'ORIENTATION des champs de force, en fraction de la longueur des
   *  poignées : posé JUSTE AU-DELÀ des flèches de déplacement. Sur l'axe mais
   *  hors de leur portée — un tourbillon a l'axe vertical par défaut, son bouton
   *  tomberait donc pile sur la pointe de la poignée Y et l'une mangerait
   *  l'autre. Il est constant à l'écran pour la même raison qu'elles. */
  handleAim: 1.5,
  /** Sphère-obstacle : rayon (fraction du monde), position de départ. */
  sphereRadius: 0.085,
  sphereStart: [0.5, 0.45, 0.5],
  /** Nombre maximal d'émetteurs simultanés. */
  maxEmitters: 4,
  /** CHAMPS DE FORCE posés dans la scène (« modificateurs ») : objets placés au
   *  pointeur, saisissables comme les émetteurs, qui agissent localement sur le
   *  fluide. Deux types pour l'instant — voir FIELD_NAMES. */
  maxFields: 3,
  fieldRadius: 0.16,
  /** Forces par défaut, calibrées séparément : le tourbillon agit sur l'air
   *  (rotationnel, préservé par la projection), le vent local sur la matière. */
  fieldVortexStrength: 260,
  fieldWindStrength: 120,
  /** EXPLOSION : une bouffée de carburant + son amorce, lâchées d'un coup. Aucune
   *  physique nouvelle — la combustion existante fait la boule de feu, la suie et
   *  le souffle (l'expansion au front de flamme EST une source de divergence, que
   *  la projection convertit en vitesse sortante). Ce qui manquait, c'était de
   *  savoir injecter une BOUFFÉE au lieu d'un débit continu.
   *  L'injection dure `explosionTime` secondes à débit constant, jamais « une
   *  frame » : à débit fixe, une frame donne un résultat qui dépend du framerate.
   *  `explosionSpark` doit dépasser le seuil d'ignition (0,55) pour que la boule
   *  s'allume entière au lieu de couver par le bord. */
  explosionRadius: 0.085,
  explosionFuel: 55,
  explosionSpark: 22,
  explosionTime: 0.05,
  /** CHARGES EN VOL simultanées (2 vec4 d'uniforme par charge, slots 116-147).
   *  Une détonation prend un slot ÉTEINT — les deux fenêtres consumées —, ou à
   *  défaut écrase le plus ancien : un bombardement recycle ses charges mortes
   *  avant d'amputer une charge en vol. */
  maxBursts: 4,
  /** HAUTEUR de la détonation (fraction de N). Au RAS DU SOL par défaut : c'est
   *  ce qui distingue une boule de feu en l'air d'une explosion « au sol », et
   *  c'est la seule position qui produise un pied de champignon. */
  explosionHeight: 0.1,
  /** POUSSIÈRE SOULEVÉE : débit, rayon horizontal et épaisseur (fractions de N).
   *  Le pied d'un champignon n'est PAS fait de la charge — il est fait du sol
   *  arraché par le souffle, que le courant ascendant aspire ensuite derrière la
   *  boule. Sans ce terme, on obtient un chapeau qui flotte sans colonne. */
  /** DURÉE d'arrachement du sol. Elle n'a aucune raison d'être celle de la
   *  charge : la charge est une bouffée, alors que le courant ascendant continue
   *  d'aspirer le sol tant qu'il monte. Injectée une seule fois, la poussière
   *  forme un paquet FINI que la montée étire puis rompt — le pied se détache du
   *  chapeau et on obtient deux nuages séparés au lieu d'un champignon. */
  dustTime: 0.05,
  dustRate: 26,
  dustRadius: 0.22,
  dustHeight: 0.045,
  /** AMORCE en chaleur (1/s) et PLAFOND du canal de chaleur. À 2, on reste dans
   *  le domaine des flammes et la plus grosse charge du monde reste aussi
   *  lumineuse qu'un feu de camp. Au-delà commence le domaine des BOULES DE FEU
   *  (voir `blackbody` dans raymarch.wgsl) : c'est ce qui sépare une détonation
   *  d'un gros feu à l'image. La poussée, elle, sature à 2 — cette chaleur-là
   *  rayonne, elle ne soulève pas. */
  heatCeiling: 2,
  /** SUIE — une flamme qui manque d'air craque son carburant et noircit. Trois
   *  nombres : le rendement (par unité de carburant chaud privé d'air), son
   *  évanouissement, et sa densité AU RENDU (0 = image strictement identique à
   *  ce qu'elle était avant la suie). Une bougie brûle dans l'air libre et reste
   *  claire ; une charge dévore l'oxygène de son propre volume et noircit — le
   *  même terme donne les deux, sans réglage séparé. */
  sootYield: 12,
  sootFade: 0.06,
  sootDensity: 1.0,
  /** Refroidissement PAR la suie (1/s par unité). Ce n'est pas un artifice : les
   *  particules de suie rayonnent bien mieux que les gaz, c'est ce qui rend une
   *  flamme riche lumineuse ET la refroidit vite. Sans ce terme, la boucle reste
   *  ouverte — le nuage devient opaque mais garde de la chaleur, donc il émet en
   *  orange sous une peau désormais impénétrable, et on obtient une braise
   *  géante au lieu d'un nuage de suie. */
  sootCooling: 2.6,
  /** CIEL OUVERT : largeur de la bande éponge (fraction de N) et sa force. La
   *  boîte reste CLOSE — l'opérateur de pression garde donc sa symétrie et le
   *  multigrid sa convergence — mais la matière et la vitesse s'éteignent près
   *  des parois au lieu de s'y empiler. Le SOL est exclu : c'est la seule paroi
   *  qui soit un objet de la scène. `openBand = 0` referme la boîte et rend
   *  exactement le comportement d'avant. */
  openBand: 0.11,
  openStrength: 26,
  /** Bande de PLAFOND, réglée séparément des parois : un panache TRAVERSE la
   *  bande latérale, alors qu'un chapeau de champignon se GARE au plafond. Une
   *  absorption calibrée pour un passage le lamine sur place. */
  openCeilBand: 0.11,
  openCeilStrength: 26,
  /** STRATIFICATION de l'air ambiant : chaleur qu'il gagne entre l'altitude de
   *  base et le plafond, et cette altitude (fraction de la HAUTEUR du domaine).
   *  0 = atmosphère neutre, strictement le comportement d'avant.
   *  C'est ce qui donne son CHAPEAU à un champignon : le panache s'arrête à
   *  l'altitude où sa chaleur rejoint celle du milieu, et s'y étale. Dans une
   *  boîte cubique on s'en passait sans le savoir — le chapeau y était en fait
   *  le PLAFOND. L'illusion n'a pas survécu à une boîte haute. */
  stratStrength: 0,
  stratBase: 0.3,
  /** PRISE DE VUE. `outdoor` remplace l'atelier (boîte de verre, fond gris) par
   *  un ciel, un sol jusqu'à l'horizon et un soleil rasant. Rien n'y touche à la
   *  simulation — mais c'est ce qui donne l'ÉCHELLE, et sans échelle le plus beau
   *  champignon reste une bouffée de fumée dans une boîte. `sunHeight` est la
   *  hauteur du soleil : bas, il sculpte les volutes ; haut, il les aplatit. */
  outdoor: false,
  sunHeight: 0.18,
  /** Combustion (modèle Feldman/Fedkiw simplifié) : taux de réaction (1/s au-dessus
   *  de l'ignition), chaleur dégagée par unité de carburant, expansion volumique au
   *  front de flamme (source de divergence, voxels/s — remise à l'échelle SCALE3). */
  burnRate: 3.0,
  heatYield: 0.7,
  expansion: 20,
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
