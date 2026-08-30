/**
 * Prototype 3D volumétrique : solveur MAC 3D (advection semi-lagrangienne, poussée
 * thermique, projection Jacobi — le multigrid 3D est le chantier suivant) + rendu
 * ray-marching. Même doctrine que le 2D : zéro DOM ici, zéro allocation par frame,
 * zéro lecture GPU→CPU, un seul CommandEncoder, bind groups tous pré-créés.
 * La plateforme fournit device, format cible, dt et l'état caméra abstrait.
 */

import advectVelWGSL from './shaders3d/advect_velocity3d.wgsl?raw';
import advectDenWGSL from './shaders3d/advect_density3d.wgsl?raw';
import vorticityWGSL from './shaders3d/vorticity3d.wgsl?raw';
import projectWGSL from './shaders3d/project3d.wgsl?raw';
import multigridWGSL from './shaders3d/multigrid3d.wgsl?raw';
import raymarchWGSL from './shaders3d/raymarch.wgsl?raw';
import glowWGSL from './shaders3d/glow3d.wgsl?raw';
import postWGSL from './shaders3d/post3d.wgsl?raw';
import gizmoWGSL from './shaders3d/gizmo3d.wgsl?raw';
import embersWGSL from './shaders3d/embers3d.wgsl?raw';
import embersDrawWGSL from './shaders3d/embers_draw.wgsl?raw';
import sparksWGSL from './shaders3d/sparks3d.wgsl?raw';
import sparksDrawWGSL from './shaders3d/sparks_draw.wgsl?raw';
import clearWGSL from './shaders3d/clear3d.wgsl?raw';
import {
  DISPATCH3,
  DISPATCH3Y,
  EMBERS3,
  GRID3,
  GRID3Y,
  HEIGHT3,
  MG3_COARSE_SMOOTH,
  MG3_TINY,
  MG3_COARSEST_SIZE,
  MG3_PRE_SMOOTH,
  MG3_POST_SMOOTH,
  SCALE3,
  SIM3_DEFAULTS,
  SPARKS3,
  WG3,
  type Sim3Tuning,
} from './config3d';
import { createShaderModule, withValidation } from '../core/pipelines';
import { flip, type Pair, type PingIndex } from '../core/types';

export interface Frame3DInput {
  /** Pas de temps en secondes (déjà borné par la plateforme). */
  dt: number;
  paused: boolean;
  /** Vidange des champs (consommé par la frame courante). */
  reset: boolean;
  /** Solveur de pression : V-cycles multigrid (défaut) ou Jacobi simple. */
  multigrid: boolean;
  vcycles: number;
  jacobiIterations: number;
  /** Réglages à chaud de la physique (panneau — voir Sim3Tuning). */
  params: Sim3Tuning;
  /** Encre émise par l'émetteur : 0 = fumée, 1 = encre, 2 = carburant. */
  emitInk: number;
  /** Caméra orbitale autour du centre de la boîte. */
  cam: { azimuth: number; elevation: number; radius: number };
  /** Position courante du pointeur en NDC [-1,1] (mise à jour en continu). */
  pointer: { ndcX: number; ndcY: number };
  /** Souffle du pointeur (clic droit + glisser) : position NDC [-1,1] et delta NDC
   *  accumulé depuis la dernière frame (consommé par la plateforme après frame()). */
  blow: { active: boolean; ndcX: number; ndcY: number; moveX: number; moveY: number };
  /** Saisie (clic gauche sur un objet — voir hitTest) : l'objet suit le pointeur
   *  sur le plan face caméra passant par sa position. */
  grab: { active: boolean };
  /** Ajout d'un émetteur sous le pointeur / retrait du dernier (consommés). */
  addEmitter: boolean;
  removeEmitter: boolean;
  /** CHAMPS DE FORCE (« modificateurs ») : ajout sous le pointeur / retrait du
   *  dernier (consommés), et réglages appliqués au champ ACTIF (le dernier posé
   *  ou celui qu'on tient). */
  addField: boolean;
  removeField: boolean;
  fieldType: number;
  fieldStrength: number;
  fieldRadius: number;
  /** EXPLOSION : détonation sous le pointeur (consommée), et son calibre.
   *  Jusqu'à `maxBursts` charges vivent en même temps — tirer n'ampute plus la
   *  charge précédente. */
  explode: boolean;
  explosionRadius: number;
  explosionFuel: number;
  /** Hauteur de la détonation (fraction de N) et poussière soulevée au sol. */
  explosionHeight: number;
  dustRate: number;
  /** Durée d'arrachement du sol — distincte de celle de la charge. */
  dustTime: number;
  /** Rayon de la galette arrachée (fraction de N) : il doit suivre le calibre de
   *  la charge, sinon une petite bombe pose une nappe de géant. */
  dustRadius: number;
  /** Amorce en chaleur : c'est elle qui décide si la boule est une flamme ou
   *  une boule de feu (avec le plafond `heatCeiling`, côté paramètres). */
  explosionSpark: number;
  /** Sphère-obstacle présente. */
  sphereActive: boolean;
  /** FORME de l'obstacle : index dans OBSTACLE_SHAPES (0 = sphère). La sphère
   *  reste le défaut et le cas exact — voir PLAN-OBSTACLES. */
  obstacleShape: number;
  /** Retours visuels (fuseau du souffle…) — débrayables (touche F). */
  feedback: boolean;
  exposure: number;
  raymarchSteps: number;
  /** Lueur du feu (in-scattering du volume de lueur) et intensité du bloom. */
  glowStrength: number;
  bloomStrength: number;
  /** Braises : interrupteur (coupé = passes entièrement sautées) et débit. */
  embersOn: boolean;
  emberStrength: number;
}

/** Sommets de la passe de gizmos — doit rester d'accord avec gizmo3d.wgsl, dans
 *  le MÊME ORDRE : 3 champs (3 cercles de 64 segments + un axe fléché), puis
 *  4 émetteurs (un anneau de 32 segments + une tige fléchée), puis les 12 arêtes
 *  de la boîte, puis les 3 poignées d'axe de l'objet sélectionné, puis son
 *  bouton d'orientation (un anneau de 24 segments face caméra + sa tige). */
const GIZMO_VERTS = 3 * (64 * 2 * 3 + 6) + 4 * (32 * 2 + 6) + 24 + 3 * 6 + (24 * 2 + 2);

/** Gravité de la fusée TRAÇANTE (fraction de N par s²) — DUPLIQUÉE dans
 *  sparks3d.wgsl (patron 3) : le CPU intègre la même balistique que le shader
 *  pour tirer l'éclat à l'apogée sans readback. */
const ROCKET_G = 0.55;
/** Hauteur de départ de la fusée (fraction de N) : juste au-dessus du sol. */
const ROCKET_Y0 = 0.04;

/** FORMES D'OBSTACLE. Les paramètres sont des RATIOS du rayon : le curseur de
 *  taille de l'utilisateur vaut donc pour toute forme, et le rendu comme la
 *  simulation lisent la même `solid_sd` (PLAN-OBSTACLES). */
export const OBSTACLE_SHAPES: readonly {
  readonly label: string;
  readonly params: readonly [number, number, number];
}[] = [
  { label: '⚪ sphère', params: [1, 1, 1] },
  // Cube de côté 2R — la forme qui montre le mieux qu'un bord n'est plus lisse.
  { label: '⬛ boîte', params: [1, 1, 1] },
  // Tore d'axe vertical : petit rayon 0,38 R. Le panache le TRAVERSE — c'est la
  // preuve à l'image qu'un obstacle peut avoir un trou.
  { label: '🍩 tore', params: [0.38, 0, 0] },
  // CLOCHE de verre : coquille de rayon 3,2 R (le rayon nominal est petit — il
  // faut couvrir la flamme) et d'épaisseur 9 % de ce rayon. Le SOL ferme le bas.
  { label: '🔔 cloche', params: [0.17, 1.5, 0] },
];

const COMPUTE = GPUShaderStage.COMPUTE;
const FRAGMENT = GPUShaderStage.FRAGMENT;
const VERTEX = GPUShaderStage.VERTEX;

function tex3d(
  device: GPUDevice,
  label: string,
  format: GPUTextureFormat,
  size = GRID3,
  // Hauteur INDÉPENDANTE : le domaine n'est plus forcément cubique. Par défaut
  // elle suit le même facteur que la grille fine, de sorte qu'un niveau
  // grossier reste proportionnel au domaine.
  height = Math.round(size * HEIGHT3),
): GPUTextureView {
  return device
    .createTexture({
      label,
      dimension: '3d',
      size: { width: size, height, depthOrArrayLayers: size },
      format,
      usage:
        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    })
    .createView({ label: `${label}-view` });
}

/** Décodage float16 → float32 (le readback d'export lit du rgba16float). */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    return sign * mant * 2 ** -24;
  }
  if (exp === 31) {
    return mant ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Un niveau de la pyramide multigrid 3D (le niveau 0 réutilise pression/divergence). */
interface MGLevel3 {
  readonly dispatch: number;
  /** Dispatch VERTICAL : le domaine n'est plus forcément cubique. */
  readonly dispatchY: number;
  /** Taille du niveau (côté horizontal), pour choisir le lisseur. */
  readonly size: number;
  /** Lissage rouge-noir EN PLACE, indexé par [pression courante du niveau].
   *  Pas de ping-pong : les deux couleurs écrivent la même texture. Utilisé sur
   *  les GRANDS niveaux (> MG3_TINY), où il converge ~2× mieux par balayage. */
  readonly rbBind: Pair<GPUBindGroup>;
  /** Lissage rouge-noir PING-PONG des niveaux MINUSCULES (≤ MG3_TINY), indexé
   *  par [ping source][parité][0 = Gauss-Seidel, 1 = SOR] — huit bind groups
   *  STATIQUES par niveau : le read_write, l'alternance de pipelines et les
   *  offsets dynamiques sont trois poisons de driver, tous mesurés. */
  readonly tinyBind: Pair<Pair<Pair<GPUBindGroup>>>;
  /** Résidu des 8 enfants évalué à la volée → rhs du niveau suivant (une seule
   *  passe, dispatchée à la taille grossière), indexé par [ping de pression
   *  fine]. Null au plus grossier. */
  readonly restrictBind: Pair<GPUBindGroup> | null;
  /** Correction du suivant (toujours son ping 0) AJOUTÉE EN PLACE à ce niveau,
   *  indexé par [pression fine]. Les niveaux grossiers ne basculent plus jamais :
   *  leur pression vit dans le ping 0, le ping 1 ne sert qu'au repli Jacobi du
   *  niveau 0. */
  readonly prolongBind: Pair<GPUBindGroup> | null;
  /** Prolongation PING-PONG (niveaux minuscules : le read_write y est
   *  pathologique) : lit ping 0, écrit ping 1. Null si non minuscule. */
  readonly prolongPPBind: GPUBindGroup | null;
  /** Remise à zéro de la pression de départ (niveaux > 0). */
  readonly clearBind: GPUBindGroup | null;
}

function sampled3d(binding: number, visibility = COMPUTE): GPUBindGroupLayoutEntry {
  return { binding, visibility, texture: { sampleType: 'float', viewDimension: '3d' } };
}

function sampledScalar3d(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '3d' } };
}

function storage3d(binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: COMPUTE,
    storageTexture: { access: 'write-only', format, viewDimension: '3d' },
  };
}

/** Storage READ_WRITE — réservé au lisseur rouge-noir en place (r32float est le
 *  seul format où cet accès est garanti par WebGPU). */
function storageRW3d(binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: COMPUTE,
    storageTexture: { access: 'read-write', format, viewDimension: '3d' },
  };
}

const pair = <T,>(f: (i: PingIndex) => T): Pair<T> => [f(0), f(1)];

export class FluidSim3D {
  private velIdx: PingIndex = 0;
  private denIdx: PingIndex = 0;
  private pressIdx: PingIndex = 0;
  private oxyIdx: PingIndex = 0;
  private simTime = 0;
  /** Première frame : initialiser la texture d'espèces (O₂ = 1 partout). */
  private needSpeciesInit = true;

  // 160 floats = les 640 o du tampon : 148 pour la simulation (2 vec4 par
  // charge aux slots 116-147 ; 92-99, l'ancien slot unique, sont libres) puis
  // TROIS vec4 en queue (148-159) pour l'événement de tir des étincelles.
  private readonly simData = new Float32Array(160);
  // 96 floats : les quatre derniers portent la FORME de l'obstacle (slot #23).
  // PIÈGE PAYÉ : ce tableau faisait exactement 92, et une écriture en 92-95 est
  // SILENCIEUSEMENT IGNORÉE par un Float32Array — la simulation voyait la boîte
  // et le tore, le rendu dessinait toujours une sphère.
  private readonly renderData = new Float32Array(96);
  // MÊME TAILLE que renderData, impérativement : `lastRender.set(d)` LÈVE si la
  // source est plus longue, et la comparaison lirait `undefined` sur la queue.
  private lastRender = new Float32Array(96).fill(Number.NaN);

  /** Émetteurs (positions en voxels) — le premier est l'émetteur historique. */
  private readonly emitters: { pos: [number, number, number]; ink: number }[] = [
    { pos: [GRID3 * 0.5, GRID3 * 0.08, GRID3 * 0.5], ink: 0 },
  ];

  /** Champs de force posés dans la scène. `axis` sert d'axe au tourbillon et de
   *  direction au vent local. */
  private readonly fields: {
    pos: [number, number, number];
    axis: [number, number, number];
    type: number;
    strength: number;
    radius: number;
  }[] = [];

  /** Encodage de `grabbed` : −2 rien · −1 la boule · 0..n émetteur ·
   *  FIELD_TAG+i champ de force. NE PAS tester `grabbed` à la main : un
   *  `grabbed >= 0` naïf attrape aussi les champs (bug de l'écran noir, corrigé
   *  le 2026-08-24 — l'encre était appliquée à `emitters[100]`, l'exception
   *  tuait la boucle de rendu). Passer par les deux accesseurs ci-dessous. */
  private static readonly FIELD_TAG = 100;

  /** Index de l'émetteur tenu, ou −1. */
  private get heldEmitter(): number {
    return this.grabbed >= 0 && this.grabbed < FluidSim3D.FIELD_TAG ? this.grabbed : -1;
  }

  /** Index du champ de force tenu, ou −1. */
  private get heldField(): number {
    return this.grabbed >= FluidSim3D.FIELD_TAG ? this.grabbed - FluidSim3D.FIELD_TAG : -1;
  }
  private readonly spherePos: [number, number, number] = [
    GRID3 * SIM3_DEFAULTS.sphereStart[0],
    GRID3 * SIM3_DEFAULTS.sphereStart[1],
    GRID3 * SIM3_DEFAULTS.sphereStart[2],
  ];
  private sphereOn = true;
  /** Vitesse de la sphère (voxels/s) — condition de bord mobile, lissée EMA. */
  private readonly sphereVel: [number, number, number] = [0, 0, 0];
  /** CHARGES EN VOL (jusqu'à `maxBursts` simultanées) : centre (voxels), rayon
   *  (voxels), et le temps RESTANT de chaque fenêtre — l'injection de la charge
   *  et l'arrachement de la POUSSIÈRE, chacune la sienne (le sol s'arrache tant
   *  que le courant monte, pas seulement pendant la détonation). Injecter
   *  « pendant une frame » donnerait un résultat dépendant du framerate ; on
   *  injecte à débit constant pendant une durée fixe, et la même détonation rend
   *  la même boule à 30 comme à 120 FPS. `frac`/`dustFrac` = fraction de CETTE
   *  frame passée dans la fenêtre (0 à 1) : les débits en sont multipliés, la
   *  charge délivrée vaut exactement débit × durée quel que soit le pas de
   *  temps. Slots pré-alloués (zéro allocation par frame) ; `stamp` = rang de
   *  tir, pour écraser le plus ancien quand tout est en vol. */
  private readonly bursts = Array.from({ length: SIM3_DEFAULTS.maxBursts }, () => ({
    pos: [0, 0, 0] as [number, number, number],
    radius: 0,
    left: 0,
    frac: 0,
    dustLeft: 0,
    dustFrac: 0,
    seed: 0,
    stamp: 0,
    // Multiplicateurs PAR CHARGE des débits d'entrée (1 = la charge au
    // pointeur, inchangée). L'éclat d'un feu d'artifice est une charge de
    // chaleur PRESQUE SANS CARBURANT : petit calibre, carburant réduit — sans
    // toucher aux réglages de l'utilisateur.
    fuelMul: 1,
    sparkMul: 1,
  }));
  private burstStamp = 0;
  /** Graine des grumeaux de la charge : avancée à chaque détonation pour que
   *  deux explosions ne soient pas jumelles, mais AVANCÉE D'UN PAS FIXE — la
   *  même suite de détonations rejoue donc à l'identique, ce qu'on veut d'un
   *  simulateur dont on rebaie les sorties. */
  private burstSeed = 0;
  /** ÉTINCELLES de feu d'artifice — l'ÉVENEMENT DE TIR du pas courant
   *  (consommé en une frame, le patron des charges) et son anneau. Le CURSEUR
   *  attribue les particules par tranches contiguës — déterministe, pas
   *  d'atomics ; la graine avance d'un pas fixe comme celle des charges et
   *  VOYAGE avec l'événement ; l'ÉPOQUE tue au premier update tout ce qui est
   *  né avant un reset. `sparksAlive` (secondes de temps SIMULÉ) garde les
   *  trois passes entièrement sautées quand plus rien ne vit. */
  private readonly sparkEvent = {
    pos: [0, 0, 0] as [number, number, number],
    speed: 0,
    color: [0, 0, 0] as [number, number, number],
    count: 0,
    base: 0,
    pattern: 0,
    window: 3.0,
    seed: 0,
  };
  /** FILE DES TIRS (anneau pré-alloué) : l'uniforme ne porte qu'UN événement
   *  par pas simulé — un tir complet en produit deux (traçante au départ,
   *  éclat à l'apogée) et deux fusées peuvent culminer la même frame. La file
   *  les étale d'un pas au lieu de les écraser ; PLEINE, elle perd le tir
   *  entrant (curseur et graine ont déjà avancé : le rejeu reste identique). */
  private readonly sparkQueue = Array.from({ length: 8 }, () => ({
    pos: [0, 0, 0] as [number, number, number],
    speed: 0,
    color: [0, 0, 0] as [number, number, number],
    count: 0,
    base: 0,
    pattern: 0,
    window: 3.0,
    seed: 0,
  }));
  private sparkQHead = 0;
  private sparkQLen = 0;
  private sparkCursor = 0;
  private sparkSeed = 0;
  private sparkEpoch = 0;
  private sparkFireNow = false;
  private sparksAlive = 0;
  /** Le splat de lueur teintée porte des valeurs : à solder d'un clearBuffer
   *  après la mort du dernier tir (sinon un résidu de lumière colle). */
  private sparkGlowDirty = false;
  /** FUSÉES en vol (le tir complet, jalon J2) : cinématique CPU — même Euler
   *  semi-implicite que la traçante GPU (patron 3), même gravité, AUCUNE
   *  traînée : les deux trajectoires restent confondues sans readback. À
   *  l'apogée (vy ≤ 0) le CPU tire l'éclat exactement là où la traçante meurt
   *  — elle meurt du même critère, côté GPU. Slots pré-alloués, `stamp` pour
   *  écraser la plus ancienne quand tout est en vol (patron des charges). */
  private readonly rockets = Array.from({ length: 4 }, () => ({
    live: false,
    pos: [0, 0, 0] as [number, number, number],
    vy: 0,
    color: [0, 0, 0] as [number, number, number],
    // Seconde teinte (bi-couleur, jalon J3) : x < 0 = tir monochrome.
    color2: [-1, 0, 0] as [number, number, number],
    count: 0,
    speed: 0,
    pattern: 0,
    stamp: 0,
  }));
  private rocketStamp = 0;

  /** Cible saisie : même encodage que `selected` (-2 = aucune). */
  private grabbed = -2;
  /** OBJET SÉLECTIONNÉ (même encodage que `grabbed`) : le dernier posé ou saisi,
   *  toutes familles confondues. UNE seule notion d'« objet courant » — il porte
   *  les poignées, il est mis en évidence, et c'est lui que retire la commande
   *  de suppression. Il y avait auparavant deux curseurs de plus, un par
   *  famille, qui ne servaient qu'à la mise en évidence : trois surbrillances
   *  pour une seule idée. */
  private selected = -2;
  /** Traînée en cours : axe du monde (0/1/2) pour un déplacement contraint,
   *  `AIM_DRAG` pour une réorientation, ou −1 pour le déplacement libre sur le
   *  plan face caméra. La DROITE de contrainte est figée à la saisie
   *  (`dragOrigin`) : la recalculer depuis l'objet qui bouge ferait boucler la
   *  mesure sur elle-même et l'objet dériverait. */
  private dragAxis = -1;
  /** Valeur de `dragAxis` qui signifie « on fait tourner l'axe, pas déplacer ».
   *  Rayon (voxels) de la sphère sur laquelle le bouton d'orientation coulisse,
   *  figé à la saisie comme la droite de contrainte. */
  private static readonly AIM_DRAG = 3;
  private dragAimRadius = 0;
  private readonly dragOrigin: [number, number, number] = [0, 0, 0];
  private dragGrip = 0;
  private readonly rayO: [number, number, number] = [0, 0, 0];
  private readonly rayD: [number, number, number] = [0, 0, 0];
  private readonly submitList: GPUCommandBuffer[] = [new Uint8Array(0) as unknown as GPUCommandBuffer];

  private constructor(
    private readonly device: GPUDevice,
    private readonly simUniforms: GPUBuffer,
    private readonly renderUniforms: GPUBuffer,
    private readonly group0: GPUBindGroup,
    private readonly pipelines: {
      velPredict: GPUComputePipeline;
      velCorrect: GPUComputePipeline;
      curl: GPUComputePipeline;
      blurCurl: GPUComputePipeline;
      confine: GPUComputePipeline;
      denPredict: GPUComputePipeline;
      denCorrect: GPUComputePipeline;
      divergence: GPUComputePipeline;
      jacobi: GPUComputePipeline;
      gradient: GPUComputePipeline;
      mgSmoothRed: GPUComputePipeline;
      mgSmoothBlack: GPUComputePipeline;
      mgTiny: GPUComputePipeline;
      mgProlongPP: GPUComputePipeline;
      mgRestrict: GPUComputePipeline;
      mgProlong: GPUComputePipeline;
      clearOne: GPUComputePipeline;
      clearRgba: GPUComputePipeline;
      clearScalar: GPUComputePipeline;
      render: GPURenderPipeline;
      glowInject: GPUComputePipeline;
      glowBlur: GPUComputePipeline;
      bloomDown: GPUComputePipeline;
      bloomH: GPUComputePipeline;
      bloomV: GPUComputePipeline;
      present: GPURenderPipeline;
      gizmo: GPURenderPipeline;
      embersUpdate: GPUComputePipeline;
      embersOcc: GPUComputePipeline;
      embersDraw: GPURenderPipeline;
      sparksUpdate: GPUComputePipeline;
      sparksOcc: GPUComputePipeline;
      sparksDraw: GPURenderPipeline;
    },
    private readonly binds: {
      velPredict: Pair<GPUBindGroup>; // [vel] → scratch
      velCorrect: Pair<Pair<GPUBindGroup>>; // [vel][den] (+scratch) → flip(vel) — forces fusionnées
      curl: Pair<GPUBindGroup>; // [vel] → curl
      blurCurl: GPUBindGroup; // curl → velScratch (|ω| flouté, ω passthrough)
      confine: Pair<GPUBindGroup>; // [vel] (+scratch flouté) → flip(vel)
      denPredict: Pair<Pair<GPUBindGroup>>; // [den][vel] → scratch
      denCorrect: Pair<Pair<Pair<GPUBindGroup>>>; // [den][vel][oxy] (+scratch) → flip(den)
      divergence: Pair<Pair<Pair<GPUBindGroup>>>; // [vel][den][oxy] (expansion)
      clearsOne: readonly GPUBindGroup[]; // espèces → O₂ = 1
      jacobi: Pair<GPUBindGroup>; // [press]
      gradient: Pair<Pair<GPUBindGroup>>; // [press][vel]
      render: Pair<Pair<GPUBindGroup>>; // [den][espèces] (+glow[1])
      glowInject: Pair<GPUBindGroup>; // [den] → glow[0]
      glowBlurAB: GPUBindGroup; // glow[0] → glow[1]
      glowBlurBA: GPUBindGroup; // glow[1] → glow[0]
      bloomH: GPUBindGroup; // bloomA → bloomB
      bloomV: GPUBindGroup; // bloomB → bloomA
      embersSim: Pair<Pair<GPUBindGroup>>; // [vel][den] + particules + R
      embersDraw: Pair<GPUBindGroup>; // [den] + particules + occlusion précalculée
      embersOcc: Pair<GPUBindGroup>; // [den] — la passe compute d'occlusion
      sparksSim: Pair<GPUBindGroup>; // [vel] + étincelles (naissance par événement)
      sparksDraw: Pair<GPUBindGroup>; // [den] + étincelles + occlusion précalculée
      sparksOcc: Pair<GPUBindGroup>; // [den] — occlusion des étincelles
      clearsRgba: readonly GPUBindGroup[];
      clearsScalar: readonly GPUBindGroup[];
    },
    private readonly mgLevels: readonly MGLevel3[],
    /** Profil GPU par passe (?profile + feature timestamp-query) — null sinon. */
    private readonly prof: {
      readonly qs: GPUQuerySet;
      readonly resolve: GPUBuffer;
      readonly read: readonly GPUBuffer[];
      readonly busy: boolean[];
    } | null,
    private readonly denTextures: Pair<GPUTexture>,
    private readonly post: {
      readonly post2dLayout: GPUBindGroupLayout;
      readonly presentLayout: GPUBindGroupLayout;
      readonly gizmoBind: GPUBindGroup;
      readonly sampler: GPUSampler;
      readonly bloomA: GPUTextureView;
      readonly bloomW: number;
      readonly bloomH: number;
      readonly glowDispatch: number;
      readonly glowDispatchY: number;
      /** Splat de LUEUR TEINTÉE des étincelles (remis à zéro par clearBuffer). */
      readonly sparkGlow: GPUBuffer;
    },
  ) {
  }

  /** Profil GPU (?profile) : ms lissées par section, lisible via __sim3d. */
  readonly profileMs: Record<string, number> = {};
  private readonly profWritten = new Array<boolean>(8).fill(false);

  /** Index de pression courant par niveau multigrid (tableau réutilisé, zéro alloc). */

  /** Cible HDR (raymarch → bloom → présentation) et bind groups qui la
   *  référencent — recréés UNIQUEMENT quand la taille du canvas change
   *  (événementiel : la doctrine zéro-alloc-par-frame reste tenue). */
  private hdrTexture: GPUTexture | null = null;
  private hdrView: GPUTextureView | null = null;
  private hdrW = 0;
  private hdrH = 0;
  private bloomDownBind: GPUBindGroup | null = null;
  private presentBind: GPUBindGroup | null = null;

  static async create(
    device: GPUDevice,
    targetFormat: GPUTextureFormat,
    profile = false,
  ): Promise<FluidSim3D> {
    // Les grosses grilles échouent en OUT-OF-MEMORY, pas en validation — sans ce
    // scope, l'échec est silencieux (écran noir, HUD à 60 FPS). Ici : erreur claire.
    device.pushErrorScope('out-of-memory');
    const sim = await withValidation(device, 'init-3d', async () => {
      const velocity = pair((i) => tex3d(device, `vel3d-${i}`, 'rgba16float'));
      // Les textures de densité gardent leur handle : l'export VDB les copie (COPY_SRC).
      const denTextures = pair((i) =>
        device.createTexture({
          label: `den3d-${i}`,
          dimension: '3d',
          size: { width: GRID3, height: GRID3, depthOrArrayLayers: GRID3 },
          format: 'rgba16float',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC,
        }),
      );
      const density = pair((i) => denTextures[i].createView({ label: `den3d-${i}-view` }));
      const pressure = pair((i) => tex3d(device, `press3d-${i}`, 'r32float'));
      const divergence = tex3d(device, 'div3d', 'r32float');
      // MacCormack : prédicteurs φ̂ ; vorticité : rotationnel vectoriel aux centres.
      const velScratch = tex3d(device, 'vel3d-hat', 'rgba16float');
      const denScratch = tex3d(device, 'den3d-hat', 'rgba16float');
      // Vorticité en MI-RÉSOLUTION : rotationnel et |ω| flouté vivent sur une
      // grille moitié (1/8 du coût — voir le commentaire du shader) ; deux
      // textures ping (le flou lit l'une, écrit l'autre), 2×7 Mo à 384³.
      const curlHalf = pair((i) => tex3d(device, `curl3d-${i}`, 'rgba16float', GRID3 / 2));
      // Espèces transportées : x = oxygène (init 1), yzw réservés.
      const species = pair((i) => tex3d(device, `species3d-${i}`, 'rgba16float'));
      // Volume de LUEUR (grille grossière) : inject → [0], blurs ping-pong,
      // le rendu lit [1] (3 blurs : 0→1→0→1). Coût mémoire négligeable.
      const GLOW3 = Math.max(GRID3 / 8, 16);
      const GLOW3Y = Math.round(GLOW3 * HEIGHT3);
      const glowDispatch = Math.ceil(GLOW3 / WG3);
      const glowDispatchY = Math.ceil(GLOW3Y / WG3);
      const glow = pair((i) => tex3d(device, `glow3d-${i}`, 'rgba16float', GLOW3));
      // Chaîne bloom : taille FIXE (la lueur est basse fréquence), 16:9 approx —
      // l'étirement du noyau sur d'autres aspects est invisible sur un halo.
      const BLOOM_W = 384;
      const BLOOM_H = 216;
      const tex2d = (label: string): GPUTextureView =>
        device
          .createTexture({
            label,
            size: { width: BLOOM_W, height: BLOOM_H },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
          })
          .createView({ label: `${label}-view` });
      const bloomA = tex2d('bloom3d-a');
      const bloomB = tex2d('bloom3d-b');
      // Braises : buffer fixe, zéro-initialisé = toutes mortes (life 0).
      const emberBuffer = device.createBuffer({
        label: 'embers3d',
        size: EMBERS3 * 32,
        usage: GPUBufferUsage.STORAGE,
      });
      // Occlusion par particule (écrite par la passe `occlude`, lue au vertex).
      const emberOcc = device.createBuffer({
        label: 'embers3d-occ',
        size: EMBERS3 * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      // Étincelles de feu d'artifice : même doctrine que les braises (buffer
      // fixe zéro-initialisé = toutes mortes), 48 o/particule — la couleur
      // prescrite et l'époque vivent dans un troisième vec4.
      const sparkBuffer = device.createBuffer({
        label: 'sparks3d',
        size: SPARKS3 * 48,
        usage: GPUBufferUsage.STORAGE,
      });
      const sparkOcc = device.createBuffer({
        label: 'sparks3d-occ',
        size: SPARKS3 * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      // LUEUR TEINTÉE (essai, chantier PLAN-ARTIFICES) : les étincelles
      // splattent leur couleur dans un tampon à la résolution du volume de
      // lueur (3 × u32 par cellule, virgule fixe ×1024, atomicAdd dans la
      // passe update) ; l'injection l'ajoute à l'émission corps noir — le
      // seul endroit du moteur où une lumière VERTE peut naître. COPY_DST :
      // remis à zéro par clearBuffer avant chaque update (et une fois après
      // la mort du dernier tir, pour solder le résidu).
      const sparkGlowBuf = device.createBuffer({
        label: 'sparks3d-glow',
        size: GLOW3 * GLOW3Y * GLOW3 * 3 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const sampler = device.createSampler({
        label: 'lin3d',
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
      });
      const simUniforms = device.createBuffer({
        label: 'sim3d-uniforms',
        // 640 o : 148 floats utilisés depuis les charges multiples.
        size: 640,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const renderUniforms = device.createBuffer({
        label: 'render3d-uniforms',
        // 512 o depuis les gizmos d'émetteurs et de boîte (80 floats utilisés).
        size: 512,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const L = {
        group0: device.createBindGroupLayout({
          label: 'g0-3d',
          entries: [
            { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
          ],
        }),
        velPredict: device.createBindGroupLayout({
          label: 'vel-predict-3d',
          entries: [sampled3d(0), storage3d(2, 'rgba16float')],
        }),
        velCorrect: device.createBindGroupLayout({
          label: 'vel-correct-3d',
          entries: [sampled3d(0), sampled3d(1), storage3d(2, 'rgba16float'), sampled3d(3)],
        }),
        curl: device.createBindGroupLayout({
          label: 'curl-3d',
          entries: [sampled3d(0), storage3d(1, 'rgba16float')],
        }),
        confine: device.createBindGroupLayout({
          label: 'confine-3d',
          entries: [sampled3d(0), sampled3d(2), storage3d(3, 'rgba16float')],
        }),
        denPredict: device.createBindGroupLayout({
          label: 'den-predict-3d',
          entries: [sampled3d(0), sampled3d(1), storage3d(3, 'rgba16float')],
        }),
        denCorrect: device.createBindGroupLayout({
          label: 'den-correct-3d',
          entries: [
            sampled3d(0),
            sampled3d(1),
            sampled3d(2),
            storage3d(3, 'rgba16float'),
            sampled3d(4),
            storage3d(5, 'rgba16float'),
          ],
        }),
        divergence: device.createBindGroupLayout({
          label: 'divergence-3d',
          entries: [sampled3d(0), storage3d(3, 'r32float'), sampled3d(5), sampled3d(6)],
        }),
        jacobi: device.createBindGroupLayout({
          label: 'jacobi-3d',
          entries: [sampledScalar3d(1), sampledScalar3d(2), storage3d(3, 'r32float')],
        }),
        mgSmoothRB: device.createBindGroupLayout({
          label: 'mg-smooth-rb-3d',
          entries: [sampledScalar3d(2), storageRW3d(5, 'r32float')],
        }),
        mgTiny: device.createBindGroupLayout({
          label: 'mg-tiny-3d',
          entries: [
            sampledScalar3d(1),
            sampledScalar3d(2),
            storage3d(3, 'r32float'),
            // Offset STATIQUE par bind group — le dynamique est un poison de
            // plus (mesuré : 4,6 FPS avec offsets dynamiques, 21,6 sans).
            { binding: 6, visibility: COMPUTE, buffer: { type: 'uniform' } },
          ],
        }),
        gradient: device.createBindGroupLayout({
          label: 'gradient-3d',
          entries: [sampled3d(0), sampledScalar3d(1), storage3d(4, 'rgba16float')],
        }),
        mgProlong: device.createBindGroupLayout({
          label: 'mg-prolong-3d',
          entries: [sampledScalar3d(0), storageRW3d(5, 'r32float')],
        }),
        mgProlongPP: device.createBindGroupLayout({
          label: 'mg-prolong-pp-3d',
          entries: [sampledScalar3d(0), storage3d(3, 'r32float'), sampledScalar3d(4)],
        }),
        clearRgba: device.createBindGroupLayout({
          label: 'clear-rgba-3d',
          entries: [storage3d(0, 'rgba16float')],
        }),
        clearScalar: device.createBindGroupLayout({
          label: 'clear-scalar-3d',
          entries: [storage3d(1, 'r32float')],
        }),
        render: device.createBindGroupLayout({
          label: 'render-3d',
          entries: [
            { binding: 0, visibility: FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: FRAGMENT, sampler: { type: 'filtering' } },
            sampled3d(2, FRAGMENT),
            sampled3d(3, FRAGMENT),
            sampled3d(4, FRAGMENT),
          ],
        }),
        // Chaîne bloom (compute 2D) : source échantillonnée + sortie storage.
        post2d: device.createBindGroupLayout({
          label: 'post2d-3d',
          entries: [
            { binding: 0, visibility: COMPUTE, texture: { sampleType: 'float' } },
            {
              binding: 1,
              visibility: COMPUTE,
              storageTexture: { access: 'write-only', format: 'rgba16float' },
            },
          ],
        }),
        // Présentation : scène HDR + bloom → canvas (tone-mapping).
        gizmo: device.createBindGroupLayout({
          label: 'gizmo-3d',
          entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
        }),
        present: device.createBindGroupLayout({
          label: 'present-3d',
          entries: [
            { binding: 0, visibility: FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: FRAGMENT, sampler: { type: 'filtering' } },
            { binding: 2, visibility: FRAGMENT, texture: { sampleType: 'float' } },
            { binding: 3, visibility: FRAGMENT, texture: { sampleType: 'float' } },
          ],
        }),
        // Braises : mise à jour (compute — vel + den + particules + R pour le
        // débit) et tracé (billboards au vertex : R + sampler + den + particules).
        embersSim: device.createBindGroupLayout({
          label: 'embers-sim-3d',
          entries: [
            sampled3d(0),
            sampled3d(1),
            { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: COMPUTE, buffer: { type: 'uniform' } },
          ],
        }),
        embersDraw: device.createBindGroupLayout({
          label: 'embers-draw-3d',
          entries: [
            { binding: 0, visibility: VERTEX, buffer: { type: 'uniform' } },
            { binding: 1, visibility: VERTEX, sampler: { type: 'filtering' } },
            sampled3d(2, VERTEX),
            { binding: 3, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
          ],
        }),
        embersOcc: device.createBindGroupLayout({
          label: 'embers-occ-3d',
          entries: [
            { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
            { binding: 2, visibility: COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
            { binding: 3, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: COMPUTE, buffer: { type: 'storage' } },
          ],
        }),
        // Étincelles : l'update ne lit que la vitesse (naissance par événement,
        // pas de rejet sur la chaleur — donc pas de densités) et SPLATTE la
        // lueur teintée (binding 2, atomics) ; tracé et occlusion clonent le
        // patron des braises.
        sparksSim: device.createBindGroupLayout({
          label: 'sparks-sim-3d',
          entries: [
            sampled3d(0),
            { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
          ],
        }),
        // L'injection de lueur quitte le layout partagé avec curl : elle lit
        // EN PLUS le splat de lueur teintée des étincelles (les blurs, eux,
        // restent sur le layout curl).
        glowInject: device.createBindGroupLayout({
          label: 'glow-inject-3d',
          entries: [
            sampled3d(0),
            storage3d(1, 'rgba16float'),
            { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
          ],
        }),
        sparksDraw: device.createBindGroupLayout({
          label: 'sparks-draw-3d',
          entries: [
            { binding: 0, visibility: VERTEX, buffer: { type: 'uniform' } },
            { binding: 1, visibility: VERTEX, sampler: { type: 'filtering' } },
            sampled3d(2, VERTEX),
            { binding: 3, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: VERTEX, buffer: { type: 'read-only-storage' } },
          ],
        }),
        sparksOcc: device.createBindGroupLayout({
          label: 'sparks-occ-3d',
          entries: [
            { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: COMPUTE, sampler: { type: 'filtering' } },
            { binding: 2, visibility: COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
            { binding: 3, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 5, visibility: COMPUTE, buffer: { type: 'storage' } },
          ],
        }),
      };

      const [advectVelM, advectDenM, vorticityM, projectM, raymarchM, glowM, postM, clearM] =
        await Promise.all([
          createShaderModule(device, 'advect_velocity3d.wgsl', advectVelWGSL),
          createShaderModule(device, 'advect_density3d.wgsl', advectDenWGSL),
          createShaderModule(device, 'vorticity3d.wgsl', vorticityWGSL),
          createShaderModule(device, 'project3d.wgsl', projectWGSL),
          createShaderModule(device, 'raymarch.wgsl', raymarchWGSL),
          createShaderModule(device, 'glow3d.wgsl', glowWGSL),
          createShaderModule(device, 'post3d.wgsl', postWGSL),
          createShaderModule(device, 'clear3d.wgsl', clearWGSL),
        ]);
      const [embersM, embersDrawM, sparksM, sparksDrawM, gizmoM] = await Promise.all([
        createShaderModule(device, 'embers3d.wgsl', embersWGSL),
        createShaderModule(device, 'embers_draw.wgsl', embersDrawWGSL),
        createShaderModule(device, 'sparks3d.wgsl', sparksWGSL),
        createShaderModule(device, 'sparks_draw.wgsl', sparksDrawWGSL),
        createShaderModule(device, 'gizmo3d.wgsl', gizmoWGSL),
      ]);
      const multigridM = await createShaderModule(device, 'multigrid3d.wgsl', multigridWGSL);

      const compute = (
        label: string,
        group1: GPUBindGroupLayout | null,
        module: GPUShaderModule,
        entryPoint: string,
        constants?: Record<string, number>,
      ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
          label,
          layout: device.createPipelineLayout({
            label: `${label}-pl`,
            bindGroupLayouts: group1 === null ? [L.group0] : [L.group0, group1],
          }),
          compute: { module, entryPoint, constants },
        });

      // Côté du niveau le plus grossier de la pyramide : l'ancrage de l'espace
      // nul (dans les lisseurs) ne doit viser que lui.
      let coarsestSide = GRID3;
      while (coarsestSide / 2 >= MG3_COARSEST_SIZE) {
        coarsestSide /= 2;
      }
      const anchor = { COARSEST_SIDE: coarsestSide };
      // Les quatre slots {parité, ω} du lisseur minuscule, à offset dynamique
      // (alignement 256) : rouge/noir en Gauss-Seidel pur (ω = 1, pour LISSER),
      // rouge/noir en SOR (ω optimal du Laplacien N³ — le niveau le plus
      // grossier RÉSOUT, et le SOR y converge en O(N) balayages).
      const omegaSOR = 2 / (1 + Math.sin(Math.PI / coarsestSide));
      const tinyParams = device.createBuffer({
        label: 'mg-tiny-params',
        size: 1024,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      for (const [slot, parity, omega] of [
        [0, 0, 1],
        [1, 1, 1],
        [2, 0, omegaSOR],
        [3, 1, omegaSOR],
      ] as const) {
        device.queue.writeBuffer(tinyParams, slot * 256, new Float32Array([parity, omega, 0, 0]));
      }

      const embersOccPipe = await device.createComputePipelineAsync({
        label: 'embers-occ-3d',
        layout: device.createPipelineLayout({
          label: 'embers-occ-3d-pl',
          bindGroupLayouts: [L.embersOcc],
        }),
        compute: { module: embersDrawM, entryPoint: 'occlude' },
      });
      const sparksOccPipe = await device.createComputePipelineAsync({
        label: 'sparks-occ-3d',
        layout: device.createPipelineLayout({
          label: 'sparks-occ-3d-pl',
          bindGroupLayouts: [L.sparksOcc],
        }),
        compute: { module: sparksDrawM, entryPoint: 'occlude' },
      });
      const [velPredict, velCorrect, curlPipe, blurCurlPipe, confine, denPredict, denCorrect, divergencePipe, jacobi, gradient, mgSmoothRed, mgSmoothBlack, mgTinyPipe, mgRestrictPipe, mgProlongPipe, mgProlongPPPipe, clearOne, clearRgba, clearScalar, render, glowInjectPipe, glowBlurPipe, bloomDownPipe, bloomHPipe, bloomVPipe, presentPipe, gizmoPipe, embersUpdatePipe, embersDrawPipe, sparksUpdatePipe, sparksDrawPipe] =
        await Promise.all([
          compute('vel-predict-3d', L.velPredict, advectVelM, 'predict'),
          compute('vel-correct-3d', L.velCorrect, advectVelM, 'correct'),
          compute('curl-3d', L.curl, vorticityM, 'curl'),
          compute('blur-curl-3d', L.curl, vorticityM, 'blur_curl'),
          compute('confine-3d', L.confine, vorticityM, 'confine'),
          compute('den-predict-3d', L.denPredict, advectDenM, 'predict'),
          compute('den-correct-3d', L.denCorrect, advectDenM, 'correct'),
          compute('divergence-3d', L.divergence, projectM, 'divergence'),
          compute('jacobi-3d', L.jacobi, projectM, 'jacobi'),
          compute('gradient-3d', L.gradient, projectM, 'gradient'),
          compute('mg-smooth-red-3d', L.mgSmoothRB, multigridM, 'smooth_red', anchor),
          compute('mg-smooth-black-3d', L.mgSmoothRB, multigridM, 'smooth_black', anchor),
          compute('mg-tiny-3d', L.mgTiny, multigridM, 'smooth_tiny', anchor),
          compute('mg-restrict-3d', L.jacobi, multigridM, 'restrict_residual'),
          compute('mg-prolong-3d', L.mgProlong, multigridM, 'prolong_add'),
          compute('mg-prolong-pp-3d', L.mgProlongPP, multigridM, 'prolong_add_pp'),
          device.createComputePipelineAsync({
            label: 'clear-one-3d',
            layout: device.createPipelineLayout({
              label: 'clear-one-3d-pl',
              bindGroupLayouts: [L.clearRgba],
            }),
            compute: { module: clearM, entryPoint: 'clear_one' },
          }),
          device.createComputePipelineAsync({
            label: 'clear-rgba-3d',
            layout: device.createPipelineLayout({ label: 'clear-rgba-3d-pl', bindGroupLayouts: [L.clearRgba] }),
            compute: { module: clearM, entryPoint: 'clear_rgba' },
          }),
          device.createComputePipelineAsync({
            label: 'clear-scalar-3d',
            layout: device.createPipelineLayout({ label: 'clear-scalar-3d-pl', bindGroupLayouts: [L.clearScalar] }),
            compute: { module: clearM, entryPoint: 'clear_scalar' },
          }),
          // Le raymarch écrit la scène HDR LINÉAIRE (rgba16float), plus le canvas.
          device.createRenderPipelineAsync({
            label: 'raymarch-3d',
            layout: device.createPipelineLayout({ label: 'raymarch-3d-pl', bindGroupLayouts: [L.render] }),
            vertex: { module: raymarchM, entryPoint: 'vs_main' },
            fragment: { module: raymarchM, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list' },
          }),
          compute('glow-inject-3d', L.glowInject, glowM, 'inject'),
          compute('glow-blur-3d', L.curl, glowM, 'blur'),
          compute('bloom-down-3d', L.post2d, postM, 'bloom_down'),
          compute('bloom-h-3d', L.post2d, postM, 'bloom_h'),
          compute('bloom-v-3d', L.post2d, postM, 'bloom_v'),
          device.createRenderPipelineAsync({
            label: 'present-3d',
            layout: device.createPipelineLayout({ label: 'present-3d-pl', bindGroupLayouts: [L.present] }),
            vertex: { module: postM, entryPoint: 'vs_present' },
            fragment: { module: postM, entryPoint: 'fs_present', targets: [{ format: targetFormat }] },
          }),
          device.createRenderPipelineAsync({
            label: 'gizmo-3d',
            layout: device.createPipelineLayout({ label: 'gizmo-3d-pl', bindGroupLayouts: [L.gizmo] }),
            vertex: { module: gizmoM, entryPoint: 'vs_gizmo' },
            fragment: {
              module: gizmoM,
              entryPoint: 'fs_gizmo',
              targets: [
                {
                  format: targetFormat,
                  blend: {
                    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                  },
                },
              ],
            },
            primitive: { topology: 'line-list' },
          }),
          compute('embers-update-3d', L.embersSim, embersM, 'update'),
          // Braises : billboards ADDITIFS dans la passe HDR (avant le bloom).
          device.createRenderPipelineAsync({
            label: 'embers-draw-3d',
            layout: device.createPipelineLayout({
              label: 'embers-draw-3d-pl',
              bindGroupLayouts: [L.embersDraw],
            }),
            vertex: { module: embersDrawM, entryPoint: 'vs_embers' },
            fragment: {
              module: embersDrawM,
              entryPoint: 'fs_embers',
              targets: [
                {
                  format: 'rgba16float',
                  blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                  },
                },
              ],
            },
            primitive: { topology: 'triangle-list' },
          }),
          compute('sparks-update-3d', L.sparksSim, sparksM, 'update'),
          // Étincelles : mêmes billboards additifs, couleur PRESCRITE.
          device.createRenderPipelineAsync({
            label: 'sparks-draw-3d',
            layout: device.createPipelineLayout({
              label: 'sparks-draw-3d-pl',
              bindGroupLayouts: [L.sparksDraw],
            }),
            vertex: { module: sparksDrawM, entryPoint: 'vs_sparks' },
            fragment: {
              module: sparksDrawM,
              entryPoint: 'fs_sparks',
              targets: [
                {
                  format: 'rgba16float',
                  blend: {
                    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                  },
                },
              ],
            },
            primitive: { topology: 'triangle-list' },
          }),
        ]);

      const group0 = device.createBindGroup({
        label: 'g0-3d-bind',
        layout: L.group0,
        entries: [
          { binding: 0, resource: { buffer: simUniforms } },
          { binding: 1, resource: sampler },
        ],
      });

      const binds = {
        velPredict: pair((v) =>
          device.createBindGroup({
            label: `vel-predict-3d-${v}`,
            layout: L.velPredict,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 2, resource: velScratch },
            ],
          }),
        ),
        velCorrect: pair((v) =>
          pair((d) =>
            device.createBindGroup({
              label: `vel-correct-3d-v${v}-d${d}`,
              layout: L.velCorrect,
              entries: [
                { binding: 0, resource: velocity[v] },
                { binding: 1, resource: velScratch },
                { binding: 2, resource: velocity[flip(v)] },
                { binding: 3, resource: density[d] },
              ],
            }),
          ),
        ),
        curl: pair((v) =>
          device.createBindGroup({
            label: `curl-3d-${v}`,
            layout: L.curl,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 1, resource: curlHalf[0] },
            ],
          }),
        ),
        blurCurl: device.createBindGroup({
          label: 'blur-curl-3d',
          layout: L.curl,
          entries: [
            { binding: 0, resource: curlHalf[0] },
            { binding: 1, resource: curlHalf[1] },
          ],
        }),
        confine: pair((v) =>
          device.createBindGroup({
            label: `confine-3d-${v}`,
            layout: L.confine,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 2, resource: curlHalf[1] },
              { binding: 3, resource: velocity[flip(v)] },
            ],
          }),
        ),
        denPredict: pair((d) =>
          pair((v) =>
            device.createBindGroup({
              label: `den-predict-3d-d${d}-v${v}`,
              layout: L.denPredict,
              entries: [
                { binding: 0, resource: density[d] },
                { binding: 1, resource: velocity[v] },
                { binding: 3, resource: denScratch },
              ],
            }),
          ),
        ),
        denCorrect: pair((d) =>
          pair((v) =>
            pair((o) =>
              device.createBindGroup({
                label: `den-correct-3d-d${d}-v${v}-o${o}`,
                layout: L.denCorrect,
                entries: [
                  { binding: 0, resource: density[d] },
                  { binding: 1, resource: velocity[v] },
                  { binding: 2, resource: denScratch },
                  { binding: 3, resource: density[flip(d)] },
                  { binding: 4, resource: species[o] },
                  { binding: 5, resource: species[flip(o)] },
                ],
              }),
            ),
          ),
        ),
        clearsOne: [species[0], species[1]].map((view, i) =>
          device.createBindGroup({
            label: `clear-one-3d-${i}`,
            layout: L.clearRgba,
            entries: [{ binding: 0, resource: view }],
          }),
        ),
        divergence: pair((v) =>
          pair((dn) =>
            pair((o) =>
              device.createBindGroup({
                label: `divergence-3d-v${v}-d${dn}-o${o}`,
                layout: L.divergence,
                entries: [
                  { binding: 0, resource: velocity[v] },
                  { binding: 3, resource: divergence },
                  { binding: 5, resource: density[dn] },
                  { binding: 6, resource: species[o] },
                ],
              }),
            ),
          ),
        ),
        jacobi: pair((p) =>
          device.createBindGroup({
            label: `jacobi-3d-${p}`,
            layout: L.jacobi,
            entries: [
              { binding: 1, resource: pressure[p] },
              { binding: 2, resource: divergence },
              { binding: 3, resource: pressure[flip(p)] },
            ],
          }),
        ),
        gradient: pair((p) =>
          pair((v) =>
            device.createBindGroup({
              label: `gradient-3d-p${p}-v${v}`,
              layout: L.gradient,
              entries: [
                { binding: 0, resource: velocity[v] },
                { binding: 1, resource: pressure[p] },
                { binding: 4, resource: velocity[flip(v)] },
              ],
            }),
          ),
        ),
        // Deux ping-pong à croiser depuis l'ajout de la suie : densités ET
        // espèces. Les groupes sont pré-créés à l'init, comme tous les autres.
        render: pair((d) =>
          pair((o) =>
            device.createBindGroup({
              label: `render-3d-d${d}-o${o}`,
              layout: L.render,
              entries: [
                { binding: 0, resource: { buffer: renderUniforms } },
                { binding: 1, resource: sampler },
                { binding: 2, resource: density[d] },
                { binding: 3, resource: glow[1] },
                { binding: 4, resource: species[o] },
              ],
            }),
          ),
        ),
        glowInject: pair((d) =>
          device.createBindGroup({
            label: `glow-inject-3d-${d}`,
            layout: L.glowInject,
            entries: [
              { binding: 0, resource: density[d] },
              { binding: 1, resource: glow[0] },
              { binding: 2, resource: { buffer: sparkGlowBuf } },
            ],
          }),
        ),
        glowBlurAB: device.createBindGroup({
          label: 'glow-blur-ab',
          layout: L.curl,
          entries: [
            { binding: 0, resource: glow[0] },
            { binding: 1, resource: glow[1] },
          ],
        }),
        glowBlurBA: device.createBindGroup({
          label: 'glow-blur-ba',
          layout: L.curl,
          entries: [
            { binding: 0, resource: glow[1] },
            { binding: 1, resource: glow[0] },
          ],
        }),
        bloomH: device.createBindGroup({
          label: 'bloom-h-3d',
          layout: L.post2d,
          entries: [
            { binding: 0, resource: bloomA },
            { binding: 1, resource: bloomB },
          ],
        }),
        bloomV: device.createBindGroup({
          label: 'bloom-v-3d',
          layout: L.post2d,
          entries: [
            { binding: 0, resource: bloomB },
            { binding: 1, resource: bloomA },
          ],
        }),
        embersSim: pair((v) =>
          pair((d) =>
            device.createBindGroup({
              label: `embers-sim-3d-v${v}-d${d}`,
              layout: L.embersSim,
              entries: [
                { binding: 0, resource: velocity[v] },
                { binding: 1, resource: density[d] },
                { binding: 2, resource: { buffer: emberBuffer } },
                { binding: 3, resource: { buffer: renderUniforms } },
              ],
            }),
          ),
        ),
        embersDraw: pair((d) =>
          device.createBindGroup({
            label: `embers-draw-3d-${d}`,
            layout: L.embersDraw,
            entries: [
              { binding: 0, resource: { buffer: renderUniforms } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: density[d] },
              { binding: 3, resource: { buffer: emberBuffer } },
              { binding: 4, resource: { buffer: emberOcc } },
            ],
          }),
        ),
        embersOcc: pair((d) =>
          device.createBindGroup({
            label: `embers-occ-3d-${d}`,
            layout: L.embersOcc,
            entries: [
              { binding: 0, resource: { buffer: renderUniforms } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: density[d] },
              { binding: 3, resource: { buffer: emberBuffer } },
              { binding: 5, resource: { buffer: emberOcc } },
            ],
          }),
        ),
        sparksSim: pair((v) =>
          device.createBindGroup({
            label: `sparks-sim-3d-v${v}`,
            layout: L.sparksSim,
            entries: [
              { binding: 0, resource: velocity[v] },
              { binding: 1, resource: { buffer: sparkBuffer } },
              { binding: 2, resource: { buffer: sparkGlowBuf } },
            ],
          }),
        ),
        sparksDraw: pair((d) =>
          device.createBindGroup({
            label: `sparks-draw-3d-${d}`,
            layout: L.sparksDraw,
            entries: [
              { binding: 0, resource: { buffer: renderUniforms } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: density[d] },
              { binding: 3, resource: { buffer: sparkBuffer } },
              { binding: 4, resource: { buffer: sparkOcc } },
            ],
          }),
        ),
        sparksOcc: pair((d) =>
          device.createBindGroup({
            label: `sparks-occ-3d-${d}`,
            layout: L.sparksOcc,
            entries: [
              { binding: 0, resource: { buffer: renderUniforms } },
              { binding: 1, resource: sampler },
              { binding: 2, resource: density[d] },
              { binding: 3, resource: { buffer: sparkBuffer } },
              { binding: 5, resource: { buffer: sparkOcc } },
            ],
          }),
        ),
        clearsRgba: [velocity[0], velocity[1], density[0], density[1]].map((view, i) =>
          device.createBindGroup({
            label: `clear-rgba-3d-${i}`,
            layout: L.clearRgba,
            entries: [{ binding: 0, resource: view }],
          }),
        ),
        clearsScalar: [pressure[0], pressure[1], divergence].map((view, i) =>
          device.createBindGroup({
            label: `clear-scalar-3d-${i}`,
            layout: L.clearScalar,
            entries: [{ binding: 1, resource: view }],
          }),
        ),
      };

      // Pyramide multigrid : niveau 0 = pression/divergence principales, puis
      // paires de pression + rhs propres jusqu'au niveau le plus grossier. (Le
      // RÉSIDU n'a plus de texture : il est évalué à la volée par la passe de
      // restriction fusionnée — 260 Mo de VRAM rendus à 384³.)
      interface LevelTex {
        size: number;
        pressure: Pair<GPUTextureView>;
        rhs: GPUTextureView;
      }
      const tiers: LevelTex[] = [];
      for (let size = GRID3, l = 0; size >= MG3_COARSEST_SIZE; size /= 2, l++) {
        if (l === 0) {
          tiers.push({ size, pressure, rhs: divergence });
        } else {
          tiers.push({
            size,
            pressure: pair((i) => tex3d(device, `mg3-press-${l}-${i}`, 'r32float', size)),
            rhs: tex3d(device, `mg3-rhs-${l}`, 'r32float', size),
          });
        }
      }
      const lastTier = tiers.length - 1;
      const mgLevels: MGLevel3[] = tiers.map((t, l) => {
        const next = l < lastTier ? tiers[l + 1]! : null;
        return {
          size: t.size,
          dispatch: Math.ceil(t.size / WG3),
          dispatchY: Math.ceil(Math.round(t.size * HEIGHT3) / WG3),
          rbBind: pair((p) =>
            device.createBindGroup({
              label: `mg3-rb-l${l}-p${p}`,
              layout: L.mgSmoothRB,
              entries: [
                { binding: 2, resource: t.rhs },
                { binding: 5, resource: t.pressure[p] },
              ],
            }),
          ),
          tinyBind: pair((p) =>
            pair((parity) =>
              pair((sor) =>
                device.createBindGroup({
                  label: `mg3-tiny-l${l}-p${p}-c${parity}-s${sor}`,
                  layout: L.mgTiny,
                  entries: [
                    { binding: 1, resource: t.pressure[p] },
                    { binding: 2, resource: t.rhs },
                    { binding: 3, resource: t.pressure[flip(p)] },
                    {
                      binding: 6,
                      resource: { buffer: tinyParams, offset: sor * 512 + parity * 256, size: 16 },
                    },
                  ],
                }),
              ),
            ),
          ),
          restrictBind:
            next === null
              ? null
              : pair((p) =>
                  device.createBindGroup({
                    label: `mg3-restrict-l${l}-p${p}`,
                    layout: L.jacobi,
                    entries: [
                      { binding: 1, resource: t.pressure[p] },
                      { binding: 2, resource: t.rhs },
                      { binding: 3, resource: next.rhs },
                    ],
                  }),
                ),
          prolongBind:
            next === null
              ? null
              : pair((fine) =>
                  device.createBindGroup({
                    label: `mg3-prolong-l${l}-f${fine}`,
                    layout: L.mgProlong,
                    entries: [
                      { binding: 0, resource: next.pressure[0] },
                      { binding: 5, resource: t.pressure[fine] },
                    ],
                  }),
                ),
          prolongPPBind:
            next === null || t.size > MG3_TINY
              ? null
              : device.createBindGroup({
                  label: `mg3-prolong-pp-l${l}`,
                  layout: L.mgProlongPP,
                  entries: [
                    { binding: 0, resource: next.pressure[0] },
                    { binding: 3, resource: t.pressure[1] },
                    { binding: 4, resource: t.pressure[0] },
                  ],
                }),
          clearBind:
            l === 0
              ? null
              : device.createBindGroup({
                  label: `mg3-clear-l${l}`,
                  layout: L.clearScalar,
                  entries: [{ binding: 1, resource: t.pressure[0] }],
                }),
        };
      });

      // PROFIL GPU PAR PASSE (?profile) : timestamps matériels aux bornes des
      // passes — c'est l'instrument du chantier de perf, et la SEULE mesure qui
      // ne passe pas par les paliers rAF/vsync. Coût nul sans le flag (aucun
      // QuerySet créé) ; le readback est un instrument de dev assumé, hors du
      // contrat zéro-readback de la boucle (même statut que l'export VDB).
      const prof =
        profile && device.features.has('timestamp-query')
          ? {
              qs: device.createQuerySet({ label: 'prof3d', type: 'timestamp', count: 16 }),
              resolve: device.createBuffer({
                label: 'prof3d-resolve',
                size: 16 * 8,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
              }),
              read: [0, 1].map((i) =>
                device.createBuffer({
                  label: `prof3d-read-${i}`,
                  size: 16 * 8,
                  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
              ),
              busy: [false, false],
            }
          : null;
      if (profile && prof === null) {
        console.warn('?profile demandé mais timestamp-query absent de ce GPU.');
      }
      return new FluidSim3D(
        device,
        simUniforms,
        renderUniforms,
        group0,
        {
          velPredict,
          velCorrect,
          curl: curlPipe,
          blurCurl: blurCurlPipe,
          confine,
          denPredict,
          denCorrect,
          divergence: divergencePipe,
          jacobi,
          gradient,
          mgSmoothRed,
          mgSmoothBlack,
          mgTiny: mgTinyPipe,
          mgRestrict: mgRestrictPipe,
          mgProlong: mgProlongPipe,
          mgProlongPP: mgProlongPPPipe,
          clearOne,
          clearRgba,
          clearScalar,
          render,
          glowInject: glowInjectPipe,
          glowBlur: glowBlurPipe,
          bloomDown: bloomDownPipe,
          bloomH: bloomHPipe,
          bloomV: bloomVPipe,
          present: presentPipe,
          gizmo: gizmoPipe,
          embersUpdate: embersUpdatePipe,
          embersOcc: embersOccPipe,
          embersDraw: embersDrawPipe,
          sparksUpdate: sparksUpdatePipe,
          sparksOcc: sparksOccPipe,
          sparksDraw: sparksDrawPipe,
        },
        binds,
        mgLevels,
        prof,
        denTextures,
        {
          post2dLayout: L.post2d,
          presentLayout: L.present,
          gizmoBind: device.createBindGroup({
            label: 'gizmo-3d',
            layout: L.gizmo,
            entries: [{ binding: 0, resource: { buffer: renderUniforms } }],
          }),
          sampler,
          bloomA,
          bloomW: BLOOM_W,
          bloomH: BLOOM_H,
          glowDispatch,
          glowDispatchY,
          sparkGlow: sparkGlowBuf,
        },
      );
    });
    const oom = await device.popErrorScope();
    if (oom) {
      throw new Error(
        `Mémoire GPU insuffisante pour une grille ${GRID3}³ — relancer avec ?grid=256 (ou 320).`,
      );
    }
    return sim;
  }

  /** Recrée la cible HDR et ses bind groups quand la taille du canvas change. */
  private ensureTargets(width: number, height: number): void {
    if (this.hdrView !== null && width === this.hdrW && height === this.hdrH) {
      return;
    }
    this.hdrTexture?.destroy();
    this.hdrW = width;
    this.hdrH = height;
    this.hdrTexture = this.device.createTexture({
      label: 'hdr3d',
      size: { width, height },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdrTexture.createView({ label: 'hdr3d-view' });
    this.bloomDownBind = this.device.createBindGroup({
      label: 'bloom-down-3d',
      layout: this.post.post2dLayout,
      entries: [
        { binding: 0, resource: this.hdrView },
        { binding: 1, resource: this.post.bloomA },
      ],
    });
    this.presentBind = this.device.createBindGroup({
      label: 'present-3d',
      layout: this.post.presentLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniforms } },
        { binding: 1, resource: this.post.sampler },
        { binding: 2, resource: this.hdrView },
        { binding: 3, resource: this.post.bloomA },
      ],
    });
  }

  /** Encode et soumet une frame complète : simulation (sauf pause) + rendu. */
  frame(input: Frame3DInput, target: GPUTextureView, width: number, height: number): void {
    const aspect = width / Math.max(height, 1);
    this.ensureTargets(width, height);
    const dt = Math.min(Math.max(input.dt, 0), 1 / 30) * input.params.timeScale;
    const running = !input.paused && dt > 0;
    // Le rendu d'abord : la saisie et le souffle lisent la base caméra depuis
    // renderData — elle doit être celle de cette frame.
    this.writeRenderUniforms(input, aspect);
    this.processInteraction(input, dt, running);
    if (running) {
      this.simTime += dt;
      this.writeSimUniforms(dt, input);
    }

    const encoder = this.device.createCommandEncoder({ label: 'frame3d' });
    // Bornes de section pour le profil GPU (?profile) : indices (2i, 2i+1) du
    // QuerySet. Sections : 0 amont (advection vitesse, forces, vorticité) ·
    // 1 divergence · 2 aval (gradient, densités+espèces) · 3 lueur · 4 occlusion
    // braises · 5 raymarch · 6 bloom · 7 présentation. La PRESSION se déduit du
    // creux entre la fin de 1 et le début de 2 (le multigrid a ses passes).
    this.profWritten.fill(false);
    const tw = (i: number): GPUComputePassTimestampWrites | undefined => {
      if (this.prof === null) {
        return undefined;
      }
      this.profWritten[i] = true;
      return {
        querySet: this.prof.qs,
        beginningOfPassWriteIndex: 2 * i,
        endOfPassWriteIndex: 2 * i + 1,
      };
    };
    if (running || input.reset) {
      let cp = encoder.beginComputePass({ label: 'sim3d-amont', timestampWrites: tw(0) });
      const n = DISPATCH3;
      const ny = DISPATCH3Y;
      // Initialisation / reset des espèces : O₂ = 1 partout.
      if (this.needSpeciesInit || input.reset) {
        this.needSpeciesInit = false;
        cp.setPipeline(this.pipelines.clearOne);
        for (const bind of this.binds.clearsOne) {
          cp.setBindGroup(0, bind);
          cp.dispatchWorkgroups(n, ny, n);
        }
      }
      if (input.reset) {
        cp.setPipeline(this.pipelines.clearRgba);
        for (const bind of this.binds.clearsRgba) {
          cp.setBindGroup(0, bind);
          cp.dispatchWorkgroups(n, ny, n);
        }
        cp.setPipeline(this.pipelines.clearScalar);
        for (const bind of this.binds.clearsScalar) {
          cp.setBindGroup(0, bind);
          cp.dispatchWorkgroups(n, ny, n);
        }
      }
      if (running) {
        cp.setBindGroup(0, this.group0);
        // Advection MacCormack de la vélocité : prédicteur → scratch, correcteur clampé.
        cp.setPipeline(this.pipelines.velPredict);
        cp.setBindGroup(1, this.binds.velPredict[this.velIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        // Le correcteur porte AUSSI les forces (fusion 2026-08-28) : la passe
        // séparée relisait et réécrivait la vitesse entière pour des termes
        // locaux — un aller-retour de 450 Mo économisé à 384³.
        cp.setPipeline(this.pipelines.velCorrect);
        cp.setBindGroup(1, this.binds.velCorrect[this.velIdx][this.denIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        this.velIdx = flip(this.velIdx);

        // Vorticity confinement : rotationnel, |ω| flouté (le fix anti-grain),
        // puis force de renforcement. Passes sautées à ε = 0.
        if (input.params.vorticityStrength > 0) {
          // Rotationnel et flou en mi-résolution ; le confinement plein grain.
          const nh = Math.ceil(n / 2);
          const nhy = Math.ceil(ny / 2);
          cp.setPipeline(this.pipelines.curl);
          cp.setBindGroup(1, this.binds.curl[this.velIdx]);
          cp.dispatchWorkgroups(nh, nhy, nh);
          cp.setPipeline(this.pipelines.blurCurl);
          cp.setBindGroup(1, this.binds.blurCurl);
          cp.dispatchWorkgroups(nh, nhy, nh);
          cp.setPipeline(this.pipelines.confine);
          cp.setBindGroup(1, this.binds.confine[this.velIdx]);
          cp.dispatchWorkgroups(n, ny, n);
          this.velIdx = flip(this.velIdx);
        }

        cp.end();
        cp = encoder.beginComputePass({ label: 'sim3d-div', timestampWrites: tw(1) });
        cp.setBindGroup(0, this.group0);
        cp.setPipeline(this.pipelines.divergence);
        cp.setBindGroup(1, this.binds.divergence[this.velIdx][this.denIdx][this.oxyIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        cp.end();

        // Pression : V-cycles multigrid (défaut, passes internes) ou Jacobi.
        // Warm start dans les deux cas — la pression précédente sert de départ.
        if (input.multigrid) {
          const cycles = Math.max(1, Math.round(input.vcycles));
          for (let k = 0; k < cycles; k++) {
            this.encodeVCycle3(encoder, k > 0);
          }
        } else {
          cp = encoder.beginComputePass({ label: 'sim3d-jacobi' });
          cp.setBindGroup(0, this.group0);
          cp.setPipeline(this.pipelines.jacobi);
          for (let i = 0; i < input.jacobiIterations; i++) {
            cp.setBindGroup(1, this.binds.jacobi[this.pressIdx]);
            cp.dispatchWorkgroups(n, ny, n);
            this.pressIdx = flip(this.pressIdx);
          }
          cp.end();
        }

        cp = encoder.beginComputePass({ label: 'sim3d-aval', timestampWrites: tw(2) });
        cp.setBindGroup(0, this.group0);
        cp.setPipeline(this.pipelines.gradient);
        cp.setBindGroup(1, this.binds.gradient[this.pressIdx][this.velIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        this.velIdx = flip(this.velIdx);

        // Advection MacCormack des densités + injection de l'émetteur au correcteur.
        cp.setPipeline(this.pipelines.denPredict);
        cp.setBindGroup(1, this.binds.denPredict[this.denIdx][this.velIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        // Le correcteur écrit AUSSI les espèces (oxygène advecté sur la même
        // rétro-trace, chimie, suie) : la passe séparée refaisait la trace et
        // la combustion pour rien — un plein parcours de grille économisé.
        cp.setPipeline(this.pipelines.denCorrect);
        cp.setBindGroup(1, this.binds.denCorrect[this.denIdx][this.velIdx][this.oxyIdx]);
        cp.dispatchWorkgroups(n, ny, n);
        this.denIdx = flip(this.denIdx);
        this.oxyIdx = flip(this.oxyIdx);
      }
      cp.end();
    }

    // Volume de LUEUR : injection (corps noir des densités fraîches) puis
    // 3 diffusions ping-pong — tourne aussi en pause (le rendu le lit toujours).
    {
      const gd = this.post.glowDispatch;
      const gdy = this.post.glowDispatchY;
      // Étincelles : l'update PRÉCÈDE désormais la lueur — il splatte la
      // couleur des étoiles dans le tampon de lueur teintée (remis à zéro
      // ici, au niveau encodeur), que l'injection lit dans la foulée. La
      // vitesse est la même qu'après la section sim : rien d'autre ne bouge.
      if (running && this.sparksAlive > 0) {
        encoder.clearBuffer(this.post.sparkGlow);
        const sp = encoder.beginComputePass({ label: 'sparks-sim' });
        sp.setBindGroup(0, this.group0);
        sp.setPipeline(this.pipelines.sparksUpdate);
        sp.setBindGroup(1, this.binds.sparksSim[this.velIdx]);
        sp.dispatchWorkgroups(SPARKS3 / 64);
        sp.end();
        this.sparkGlowDirty = true;
      } else if (this.sparkGlowDirty && (running || input.reset)) {
        encoder.clearBuffer(this.post.sparkGlow);
        this.sparkGlowDirty = false;
      }
      const gp = encoder.beginComputePass({ label: 'glow3d', timestampWrites: tw(3) });
      gp.setBindGroup(0, this.group0);
      gp.setPipeline(this.pipelines.glowInject);
      gp.setBindGroup(1, this.binds.glowInject[this.denIdx]);
      gp.dispatchWorkgroups(gd, gdy, gd);
      gp.setPipeline(this.pipelines.glowBlur);
      gp.setBindGroup(1, this.binds.glowBlurAB);
      gp.dispatchWorkgroups(gd, gdy, gd);
      gp.setBindGroup(1, this.binds.glowBlurBA);
      gp.dispatchWorkgroups(gd, gdy, gd);
      gp.setBindGroup(1, this.binds.glowBlurAB);
      gp.dispatchWorkgroups(gd, gdy, gd);
      // Braises : mise à jour des particules (naissances autorégulées dans le
      // chaud, portage par le fluide) — sautée si coupées, en pause ou à débit nul.
      if (running && input.embersOn && input.emberStrength > 0) {
        gp.setPipeline(this.pipelines.embersUpdate);
        gp.setBindGroup(1, this.binds.embersSim[this.velIdx][this.denIdx]);
        gp.dispatchWorkgroups(EMBERS3 / 64);
      }
      gp.end();
      // Occlusion des braises, par PARTICULE (voir `occlude`) — aussi en pause :
      // l'orbite de caméra change la ligne de visée, l'occlusion doit suivre.
      if (input.embersOn && input.emberStrength > 0) {
        const op = encoder.beginComputePass({ label: 'embers-occ', timestampWrites: tw(4) });
        op.setPipeline(this.pipelines.embersOcc);
        op.setBindGroup(0, this.binds.embersOcc[this.denIdx]);
        op.dispatchWorkgroups(EMBERS3 / 64);
        op.end();
      }
      // Occlusion des étincelles — même logique, même raison d'exister en pause.
      if (this.sparksAlive > 0) {
        const sp = encoder.beginComputePass({ label: 'sparks-occ' });
        sp.setPipeline(this.pipelines.sparksOcc);
        sp.setBindGroup(0, this.binds.sparksOcc[this.denIdx]);
        sp.dispatchWorkgroups(SPARKS3 / 64);
        sp.end();
      }
    }

    // Raymarch → scène HDR linéaire.
    const rp = encoder.beginRenderPass({
      label: 'raymarch',
      timestampWrites: tw(5),
      colorAttachments: [
        { view: this.hdrView!, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    rp.setPipeline(this.pipelines.render);
    rp.setBindGroup(0, this.binds.render[this.denIdx][this.oxyIdx]);
    rp.draw(3);
    // Braises : billboards additifs par-dessus le volume (occlusion par
    // particule au vertex) — elles hériteront du bloom.
    if (input.embersOn && input.emberStrength > 0) {
      rp.setPipeline(this.pipelines.embersDraw);
      rp.setBindGroup(0, this.binds.embersDraw[this.denIdx]);
      rp.draw(6, EMBERS3);
    }
    // Étincelles : même chemin, couleur prescrite par particule.
    if (this.sparksAlive > 0) {
      rp.setPipeline(this.pipelines.sparksDraw);
      rp.setBindGroup(0, this.binds.sparksDraw[this.denIdx]);
      rp.draw(6, SPARKS3);
    }
    rp.end();

    // Bloom : seuil+réduction, gaussienne séparable (H puis V), taille fixe.
    {
      const bw = Math.ceil(this.post.bloomW / 8);
      const bh = Math.ceil(this.post.bloomH / 8);
      const bp = encoder.beginComputePass({ label: 'bloom3d', timestampWrites: tw(6) });
      bp.setBindGroup(0, this.group0);
      bp.setPipeline(this.pipelines.bloomDown);
      bp.setBindGroup(1, this.bloomDownBind!);
      bp.dispatchWorkgroups(bw, bh);
      bp.setPipeline(this.pipelines.bloomH);
      bp.setBindGroup(1, this.binds.bloomH);
      bp.dispatchWorkgroups(bw, bh);
      bp.setPipeline(this.pipelines.bloomV);
      bp.setBindGroup(1, this.binds.bloomV);
      bp.dispatchWorkgroups(bw, bh);
      bp.end();
    }

    // Présentation : HDR + bloom → canvas (tone-mapping, gamma).
    const pp = encoder.beginRenderPass({
      label: 'present',
      timestampWrites: tw(7),
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pp.setPipeline(this.pipelines.present);
    pp.setBindGroup(0, this.presentBind!);
    pp.draw(3);
    // GIZMOS : lignes tracées APRÈS le tone-mapping, sur la même cible — elles
    // gardent leur couleur exacte et passent devant le volume (voir gizmo3d.wgsl).
    pp.setPipeline(this.pipelines.gizmo);
    pp.setBindGroup(0, this.post.gizmoBind);
    pp.draw(GIZMO_VERTS);
    pp.end();

    // Profil GPU : résoudre les timestamps et, si un buffer de lecture est
    // libre, en copier un instantané. La lecture est asynchrone et lissée (EMA)
    // — un instrument de dev (?profile), hors du contrat zéro-readback.
    if (this.prof !== null) {
      const prof = this.prof;
      encoder.resolveQuerySet(prof.qs, 0, 16, prof.resolve, 0);
      const free = prof.busy.indexOf(false);
      if (free >= 0) {
        prof.busy[free] = true;
        encoder.copyBufferToBuffer(prof.resolve, 0, prof.read[free]!, 0, 16 * 8);
      }
      this.submitList[0] = encoder.finish();
      this.device.queue.submit(this.submitList);
      if (free >= 0) {
        const written = [...this.profWritten];
        void prof.read[free]!.mapAsync(GPUMapMode.READ).then(() => {
          const q = new BigInt64Array(prof.read[free]!.getMappedRange());
          const NOMS = ['amont', 'divergence', 'aval', 'lueur', 'occ braises', 'raymarch', 'bloom', 'présentation'];
          const ema = (nom: string, ms: number): void => {
            // Une résolution de timestamps peut rendre 0 (frame sautée) : ignorer.
            if (ms <= 0 || ms > 1000) {
              return;
            }
            this.profileMs[nom] = (this.profileMs[nom] ?? ms) * 0.9 + ms * 0.1;
          };
          for (let i = 0; i < 8; i++) {
            if (written[i]) {
              ema(NOMS[i]!, Number(q[2 * i + 1]! - q[2 * i]!) / 1e6);
            }
          }
          if (written[1] && written[2]) {
            ema('pression', Number(q[4]! - q[3]!) / 1e6);
          }
          if (written[0] && written[7]) {
            ema('TOTAL GPU', Number(q[15]! - q[0]!) / 1e6);
          }
          prof.read[free]!.unmap();
          prof.busy[free] = false;
        });
      }
      return;
    }
    this.submitList[0] = encoder.finish();
    this.device.queue.submit(this.submitList);
  }

  /** Rayon caméra→scène pour un point NDC : origine en voxels, direction unitaire. */
  private computeRay(ndcX: number, ndcY: number): void {
    const r = this.renderData;
    const tanf = r[3]!;
    const aspect = r[7]!;
    let dx = r[12]! + r[4]! * ndcX * tanf * aspect + r[8]! * ndcY * tanf;
    let dy = r[13]! + r[5]! * ndcX * tanf * aspect + r[9]! * ndcY * tanf;
    let dz = r[14]! + r[6]! * ndcX * tanf * aspect + r[10]! * ndcY * tanf;
    const dl = Math.hypot(dx, dy, dz) || 1;
    this.rayD[0] = dx / dl;
    this.rayD[1] = dy / dl;
    this.rayD[2] = dz / dl;
    this.rayO[0] = (r[0]! + 0.5) * GRID3;
    this.rayO[1] = (r[1]! + 0.5) * GRID3;
    this.rayO[2] = (r[2]! + 0.5) * GRID3;
  }

  /** Distance point (voxels) → rayon courant ; +∞ si le point est derrière.
   *  Prend TROIS NOMBRES et non un vecteur : les appelants calculent souvent un
   *  point dérivé (le bout d'une poignée…) et un littéral de tableau ici serait
   *  une allocation dans un chemin appelé au pointeur. */
  private rayDistance(x: number, y: number, z: number): number {
    const vx = x - this.rayO[0];
    const vy = y - this.rayO[1];
    const vz = z - this.rayO[2];
    const t = vx * this.rayD[0] + vy * this.rayD[1] + vz * this.rayD[2];
    if (t <= 0) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.hypot(
      vx - t * this.rayD[0],
      vy - t * this.rayD[1],
      vz - t * this.rayD[2],
    );
  }

  /** La cible sous le rayon : -1 = sphère, ≥0 = émetteur, -2 = rien. */
  private pickTarget(): number {
    const limit = SIM3_DEFAULTS.grabRadius * GRID3;
    let best = -2;
    let bestDist = limit;
    if (this.sphereOn) {
      const d = this.rayDistance(this.spherePos[0], this.spherePos[1], this.spherePos[2]);
      if (d < bestDist + SIM3_DEFAULTS.sphereRadius * GRID3 * 0.5) {
        best = -1;
        bestDist = d;
      }
    }
    for (let i = 0; i < this.emitters.length; i++) {
      const e = this.emitters[i]!.pos;
      const d = this.rayDistance(e[0], e[1], e[2]);
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    }
    for (let i = 0; i < this.fields.length; i++) {
      const f = this.fields[i]!;
      // La zone de saisie suit le RAYON VISIBLE du champ (même bonus que la
      // boule) : sans ça on cliquait sur le gizmo sans rien attraper.
      const d = this.rayDistance(f.pos[0], f.pos[1], f.pos[2]);
      if (d < bestDist + f.radius * GRID3 * 0.5) {
        best = FluidSim3D.FIELD_TAG + i;
        bestDist = d;
      }
    }
    return best;
  }

  /** Position (voxels) de l'objet SÉLECTIONNÉ, ou `undefined` si rien n'est
   *  sélectionné (ou si c'est la boule et qu'elle a été retirée). */
  private selectedPos(): [number, number, number] | undefined {
    if (this.selected === -1) {
      return this.sphereOn ? this.spherePos : undefined;
    }
    if (this.selected >= FluidSim3D.FIELD_TAG) {
      return this.fields[this.selected - FluidSim3D.FIELD_TAG]?.pos;
    }
    if (this.selected >= 0) {
      return this.emitters[this.selected]?.pos;
    }
    return undefined;
  }

  /** L'axe de l'objet sélectionné, quand il en a un — seuls les champs de force
   *  ont une ORIENTATION (axe de rotation d'un tourbillon, cap d'un vent local).
   *  Un émetteur souffle vers le haut, la boule est une boule. */
  private selectedAxis(): [number, number, number] | undefined {
    return this.selected >= FluidSim3D.FIELD_TAG
      ? this.fields[this.selected - FluidSim3D.FIELD_TAG]?.axis
      : undefined;
  }

  /** Longueur des poignées EN VOXELS. Constante à l'écran : elle grandit avec la
   *  distance à la caméra (voir handleScreen). */
  private handleLength(p: readonly number[]): number {
    const r = this.renderData;
    const dx = p[0]! / GRID3 - 0.5 - r[0]!;
    const dy = p[1]! / GRID3 - 0.5 - r[1]!;
    const dz = p[2]! / GRID3 - 0.5 - r[2]!;
    return SIM3_DEFAULTS.handleScreen * Math.hypot(dx, dy, dz) * r[3]! * GRID3;
  }

  /** Paramètre s du point de la droite `dragOrigin + s·axe` le plus proche du
   *  rayon courant — la mathématique de toute poignée de translation. NaN quand
   *  le rayon est PARALLÈLE à l'axe : le point le plus proche est alors
   *  indéterminé, et le geste n'a de toute façon plus de sens à l'écran. */
  private axisParam(origin: readonly number[], axis: number): number {
    const wx = origin[0]! - this.rayO[0];
    const wy = origin[1]! - this.rayO[1];
    const wz = origin[2]! - this.rayO[2];
    const b = this.rayD[axis]!;
    const along = axis === 0 ? wx : axis === 1 ? wy : wz;
    const e = wx * this.rayD[0] + wy * this.rayD[1] + wz * this.rayD[2];
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-4) {
      return Number.NaN;
    }
    return (b * e - along) / denom;
  }

  /** Distance (voxels) du rayon courant au segment de poignée — qui ne commence
   *  PAS au centre de l'objet (voir handleInner). */
  private handleDistance(p: readonly number[], axis: number, len: number): number {
    const inner = len * SIM3_DEFAULTS.handleInner;
    const raw = this.axisParam(p, axis);
    const s = Number.isFinite(raw) ? Math.min(Math.max(raw, inner), len) : inner;
    const vx = p[0]! - this.rayO[0] + (axis === 0 ? s : 0);
    const vy = p[1]! - this.rayO[1] + (axis === 1 ? s : 0);
    const vz = p[2]! - this.rayO[2] + (axis === 2 ? s : 0);
    const t = vx * this.rayD[0] + vy * this.rayD[1] + vz * this.rayD[2];
    if (t <= 0) {
      return Number.POSITIVE_INFINITY;
    }
    return Math.hypot(vx - t * this.rayD[0], vy - t * this.rayD[1], vz - t * this.rayD[2]);
  }

  /** La poignée sous le rayon : 0/1/2 pour un axe de déplacement, `AIM_DRAG`
   *  pour le bouton d'orientation, −1 pour rien. Testé AVANT les objets : les
   *  poignées se DESSINENT par-dessus tout, elles doivent aussi s'attraper par-
   *  dessus tout — sinon viser la poignée d'un objet revient à saisir l'objet. */
  private pickHandle(): number {
    const p = this.selectedPos();
    if (p === undefined) {
      return -1;
    }
    const len = this.handleLength(p);
    let best = -1;
    let bestDist = len * SIM3_DEFAULTS.handleGrip;
    for (let a = 0; a < 3; a++) {
      const d = this.handleDistance(p, a, len);
      if (d < bestDist) {
        best = a;
        bestDist = d;
      }
    }
    const axis = this.selectedAxis();
    if (axis !== undefined) {
      const reach = len * SIM3_DEFAULTS.handleAim;
      const d = this.rayDistance(
        p[0]! + axis[0] * reach,
        p[1]! + axis[1] * reach,
        p[2]! + axis[2] * reach,
      );
      if (d < bestDist) {
        best = FluidSim3D.AIM_DRAG;
      }
    }
    return best;
  }

  /** Réoriente l'objet sélectionné : l'axe pointe vers le point de la SPHÈRE de
   *  rayon `dragAimRadius` (centrée sur `centre`) que vise le pointeur. Quand le
   *  rayon manque la sphère on prend le point le plus proche de sa surface —
   *  c'est ce qui rend le geste continu au-delà de la silhouette, au lieu de le
   *  faire décrocher net au bord. */
  private aimSelected(centre: readonly number[]): void {
    const axis = this.selectedAxis();
    if (axis === undefined) {
      return;
    }
    const R = this.dragAimRadius;
    const mx = centre[0]! - this.rayO[0];
    const my = centre[1]! - this.rayO[1];
    const mz = centre[2]! - this.rayO[2];
    const t = mx * this.rayD[0] + my * this.rayD[1] + mz * this.rayD[2];
    if (t <= 0 || R <= 0) {
      return; // objet derrière la caméra : le geste n'a plus de sens à l'écran
    }
    // Point du rayon le plus proche du centre, puis écart au centre.
    let hx = this.rayO[0] + t * this.rayD[0] - centre[0]!;
    let hy = this.rayO[1] + t * this.rayD[1] - centre[1]!;
    let hz = this.rayO[2] + t * this.rayD[2] - centre[2]!;
    const h = Math.hypot(hx, hy, hz);
    if (h < R) {
      // Le rayon traverse la sphère : on prend l'intersection AVANT (la face
      // qu'on voit), sinon l'axe basculerait vers l'arrière au moindre tremblé.
      const back = Math.sqrt(R * R - h * h);
      hx -= back * this.rayD[0];
      hy -= back * this.rayD[1];
      hz -= back * this.rayD[2];
    } else if (h > 1e-4) {
      const k = R / h;
      hx *= k;
      hy *= k;
      hz *= k;
    } else {
      return; // pointeur pile sur le centre : direction indéterminée
    }
    const len = Math.hypot(hx, hy, hz);
    if (len < 1e-4) {
      return;
    }
    axis[0] = hx / len;
    axis[1] = hy / len;
    axis[2] = hz / len;
  }

  /** Émetteurs posés. La plateforme s'en sert pour dire la vérité plutôt que
   *  d'annoncer une suppression qui n'aura pas lieu : le dernier émetteur ne se
   *  retire pas, une scène sans flamme pilote n'a rien à montrer. */
  get emitterCount(): number {
    return this.emitters.length;
  }

  /** La sélection est-elle un champ de force ? (la plateforme s'en sert pour
   *  router une même touche « supprimer » vers la bonne famille) */
  get selectedIsField(): boolean {
    return this.selected >= FluidSim3D.FIELD_TAG;
  }

  /** Ne plus rien avoir de sélectionné : les poignées disparaissent. */
  deselect(): void {
    this.selected = -2;
    this.dragAxis = -1;
  }

  /** Y a-t-il quelque chose à saisir sous ce point NDC — poignée d'axe ou objet ?
   *  (décide saisie vs orbite) */
  hitTest(ndcX: number, ndcY: number): boolean {
    this.computeRay(ndcX, ndcY);
    return this.pickHandle() >= 0 || this.pickTarget() !== -2;
  }

  /** Pilotage scripté (mode démo) : déplace la sphère vers (nx,ny,nz) ∈ [0,1]³,
   *  en communiquant sa vitesse au fluide comme une saisie réelle. */
  driveSphere(nx: number, ny: number, nz: number, dt: number): void {
    const clamp01 = (v: number): number => Math.min(Math.max(v, 0.05), 0.95);
    const target = [clamp01(nx) * GRID3, clamp01(ny) * GRID3, clamp01(nz) * GRID3];
    if (dt > 0) {
      for (let a = 0; a < 3; a++) {
        const inst = (target[a]! - this.spherePos[a]!) / dt;
        this.sphereVel[a] = 0.55 * this.sphereVel[a]! + 0.45 * inst;
      }
    }
    this.spherePos[0] = target[0]!;
    this.spherePos[1] = target[1]!;
    this.spherePos[2] = target[2]!;
  }

  /** Pilotage scripté : pose un émetteur à (nx,ny,nz) ∈ [0,1]³ avec son encre. */
  addEmitterAt(nx: number, ny: number, nz: number, ink: number): void {
    if (this.emitters.length >= SIM3_DEFAULTS.maxEmitters) {
      return;
    }
    this.emitters.push({ pos: [nx * GRID3, ny * GRID3, nz * GRID3], ink });
    this.selected = this.emitters.length - 1;
  }

  /** Pilotage scripté : DÉTONATION à (nx,ny,nz) — le pendant de driveSphere et
   *  addEmitterAt. La démo et les bancs posent leurs charges en clair au lieu
   *  de viser au pointeur : un plan de charge BAS vu d'une caméra rasante se
   *  coupe hors boîte et le tir serait rabattu au centre (piège documenté de la
   *  visée). Les fractions sont de N, la LARGEUR — même convention que
   *  `explosionHeight`. Calibre et fenêtres viennent de l'entrée, comme au
   *  pointeur. */
  explodeAt(nx: number, ny: number, nz: number, input: Frame3DInput): void {
    const c = (v: number): number => Math.min(Math.max(v, 0.05), 0.95);
    this.fireBurst(c(nx) * GRID3, c(ny) * GRID3, c(nz) * GRID3, input);
  }

  /** Pilotage scripté : FEU D'ARTIFICE — un éclat d'étincelles à couleur
   *  PRESCRITE en (nx,ny,nz), fractions de N (convention explodeAt). Les
   *  particules naissent par tranche d'anneau au prochain pas SIMULÉ libre —
   *  la FILE étale les tirs à un par pas (en pause ils attendent, un reset les
   *  désarme). Le diamètre de la pivoine se règle par la vitesse radiale
   *  (`speed`, fraction de N par seconde). Patrons : 0 pivoine, 1 saule (vie
   *  longue — fenêtre de passes élargie), 2 éclat bref. */
  launchFirework(
    nx: number,
    ny: number,
    nz: number,
    r: number,
    g: number,
    b: number,
    count = 1500,
    speed = 0.16,
    pattern = 0,
  ): void {
    const c = (v: number): number => Math.min(Math.max(v, 0.05), 0.95);
    const patt = Math.min(Math.max(Math.round(pattern), 0), 2);
    this.enqueueSpark(
      c(nx) * GRID3,
      c(ny) * GRID3,
      c(nz) * GRID3,
      speed * GRID3,
      r,
      g,
      b,
      count,
      patt,
      patt === 1 ? 4.5 : 3.0,
    );
  }

  /** Pilotage scripté : TIR COMPLET (jalon J2) — la fusée traçante part du sol
   *  en (nx,nz) et monte en balistique pure (gravité ROCKET_G, aucune
   *  traînée) jusqu'à culminer vers `apexNy` (fraction de N), où le CPU tire
   *  l'éclat : les étoiles au patron demandé PLUS une charge de chaleur
   *  presque sans carburant (flash, poussée, fumée grise — poussière NULLE :
   *  un éclat en l'air n'arrache pas le sol). La traçante GPU meurt au même
   *  critère (vy ≤ 0) : les deux trajectoires sont le même Euler
   *  semi-implicite — aucun readback. BI-COULEUR (r2 ≥ 0) : la coque garde la
   *  teinte principale, le CŒUR prend la seconde — deux éclats au même point
   *  (65 % à pleine vitesse, 35 % à 0,55×), que la file étale d'un pas
   *  simulé, invisible à l'œil. */
  launchRocket(
    nx: number,
    nz: number,
    apexNy = 0.6,
    r = 1,
    g = 1,
    b = 1,
    count = 1500,
    speed = 0.16,
    pattern = 0,
    r2 = -1,
    g2 = 0,
    b2 = 0,
  ): void {
    const c = (v: number): number => Math.min(Math.max(v, 0.08), 0.92);
    const x = c(nx) * GRID3;
    const z = c(nz) * GRID3;
    const y0 = ROCKET_Y0 * GRID3;
    const apex = Math.max(apexNy * GRID3, y0 + 0.08 * GRID3);
    // v² = 2·g·h — g en voxels/s², l'apogée tombe où on l'a demandée.
    const vy0 = Math.sqrt(2 * ROCKET_G * GRID3 * (apex - y0));
    let slot = this.rockets[0]!; // slots ≥ 1 : jamais vide
    for (const rk of this.rockets) {
      const rFree = !rk.live;
      const sFree = !slot.live;
      if (rFree !== sFree ? rFree : rk.stamp < slot.stamp) slot = rk;
    }
    slot.live = true;
    slot.pos[0] = x;
    slot.pos[1] = y0;
    slot.pos[2] = z;
    slot.vy = vy0;
    slot.color[0] = r;
    slot.color[1] = g;
    slot.color[2] = b;
    slot.color2[0] = r2;
    slot.color2[1] = g2;
    slot.color2[2] = b2;
    slot.count = count;
    slot.speed = speed;
    slot.pattern = Math.min(Math.max(Math.round(pattern), 0), 2);
    slot.stamp = ++this.rocketStamp;
    // La TRAÇANTE : une comète de quelques étoiles d'or qui monte — fenêtre de
    // passes jusqu'à l'apogée (l'éclat ré-armera la sienne).
    const tApex = vy0 / (ROCKET_G * GRID3);
    this.enqueueSpark(x, y0, z, vy0, 1.0, 0.84, 0.55, 24, 3, tApex + 0.6);
  }

  /** TIR VISÉ (l'outil fusée de J4) : l'apogée est posée sous le POINTEUR —
   *  rayon du pointeur ∩ plan horizontal à `apexNy` (fraction de N), même
   *  géométrie que la visée des charges, même piège : hors boîte, on retombe
   *  au centre (une fusée mal visée doit éclater au milieu de la scène, pas
   *  collée à une paroi). La fusée part du sol à la verticale du point visé. */
  launchRocketAt(
    ndcX: number,
    ndcY: number,
    apexNy = 0.6,
    r = 1,
    g = 1,
    b = 1,
    count = 1500,
    speed = 0.16,
    pattern = 0,
    r2 = -1,
    g2 = 0,
    b2 = 0,
  ): void {
    this.computeRay(ndcX, ndcY);
    const planeY = GRID3 * apexNy;
    const t = (planeY - this.rayO[1]) / (this.rayD[1] || 1e-6);
    const rawX = this.rayO[0] + t * this.rayD[0];
    const rawZ = this.rayO[2] + t * this.rayD[2];
    const inside = t > 0 && rawX > 0 && rawX < GRID3 && rawZ > 0 && rawZ < GRID3;
    const clampXZ = (v: number): number => Math.min(Math.max(v, GRID3 * 0.14), GRID3 * 0.86);
    this.launchRocket(
      (inside ? clampXZ(rawX) : GRID3 * 0.5) / GRID3,
      (inside ? clampXZ(rawZ) : GRID3 * 0.5) / GRID3,
      apexNy,
      r,
      g,
      b,
      count,
      speed,
      pattern,
      r2,
      g2,
      b2,
    );
  }

  /** Pousse un événement de tir dans la FILE (positions en VOXELS, vitesse en
   *  voxels/s). Curseur et graine avancent à l'ENQUEUE et voyagent avec
   *  l'événement — une file pleine PERD le tir entrant mais le rejeu reste
   *  identique. */
  private enqueueSpark(
    px: number,
    py: number,
    pz: number,
    speed: number,
    r: number,
    g: number,
    b: number,
    count: number,
    pattern: number,
    window: number,
  ): void {
    const n = Math.min(Math.max(Math.round(count), 1), SPARKS3);
    const base = this.sparkCursor;
    this.sparkCursor = (this.sparkCursor + n) % SPARKS3;
    this.sparkSeed = (this.sparkSeed + 7.31) % 97;
    if (this.sparkQLen >= this.sparkQueue.length) {
      return;
    }
    const ev = this.sparkQueue[(this.sparkQHead + this.sparkQLen) % this.sparkQueue.length]!;
    this.sparkQLen++;
    ev.pos[0] = px;
    ev.pos[1] = py;
    ev.pos[2] = pz;
    ev.speed = speed;
    ev.color[0] = r;
    ev.color[1] = g;
    ev.color[2] = b;
    ev.count = n;
    ev.base = base;
    ev.pattern = pattern;
    ev.window = window;
    ev.seed = this.sparkSeed;
  }

  /** Arme une charge (position en voxels) : slot ÉTEINT de préférence — les
   *  deux fenêtres consumées —, sinon le plus ancien (rang de tir `stamp`) :
   *  un bombardement recycle ses charges mortes avant d'amputer une charge en
   *  vol. */
  private fireBurst(
    x: number,
    y: number,
    z: number,
    input: Frame3DInput,
    opts?: { radius?: number; fuelMul?: number; sparkMul?: number; dustTime?: number },
  ): void {
    let slot = this.bursts[0]!; // maxBursts ≥ 1 : jamais vide
    for (const b of this.bursts) {
      const bFree = b.left <= 0 && b.dustLeft <= 0;
      const sFree = slot.left <= 0 && slot.dustLeft <= 0;
      if (bFree !== sFree ? bFree : b.stamp < slot.stamp) slot = b;
    }
    slot.pos[0] = x;
    slot.pos[1] = y;
    slot.pos[2] = z;
    slot.radius = opts?.radius ?? input.explosionRadius * GRID3;
    slot.left = SIM3_DEFAULTS.explosionTime;
    slot.dustLeft = opts?.dustTime ?? input.dustTime;
    slot.fuelMul = opts?.fuelMul ?? 1;
    slot.sparkMul = opts?.sparkMul ?? 1;
    this.burstSeed = (this.burstSeed + 7.31) % 97;
    slot.seed = this.burstSeed;
    slot.stamp = ++this.burstStamp;
  }

  /** Saisie, ajout/retrait d'émetteurs, sphère — tout l'état interactif. */
  private processInteraction(input: Frame3DInput, dt: number, running: boolean): void {
    if (input.reset) {
      this.emitters.length = 1;
      this.fields.length = 0;
      this.deselect();
      this.emitters[0] = { pos: [GRID3 * 0.5, GRID3 * 0.08, GRID3 * 0.5], ink: input.emitInk };
      this.spherePos[0] = GRID3 * SIM3_DEFAULTS.sphereStart[0];
      this.spherePos[1] = GRID3 * SIM3_DEFAULTS.sphereStart[1];
      this.spherePos[2] = GRID3 * SIM3_DEFAULTS.sphereStart[2];
      // Les fenêtres des charges s'éteignent : une poussière encore en cours
      // (champignon = 5 s) continuerait sinon d'arracher le sol de la scène
      // NEUVE au point de l'ancien impact. Avec le slot unique, le tir suivant
      // écrasait tout ; avec plusieurs slots, il faut éteindre explicitement.
      // Graine et positions restent (inertes) : la suite de détonations rejoue
      // à l'identique.
      for (const b of this.bursts) {
        b.left = 0;
        b.dustLeft = 0;
      }
      // Les étincelles aussi : l'ÉPOQUE avance — le premier update qui suit
      // tue tout ce qui est né avant (le tracé vient après lui dans la frame,
      // aucune étoile rassie ne réapparaît). Modulo 2²¹ : l'époque vit dans
      // tint.w MULTIPLIÉE PAR 8 (les bits bas portent le patron), et 2²¹ × 8
      // reste sous 2²⁴, le dernier entier exact en f32 — un modulo court
      // pouvait boucler et ressusciter des étoiles gelées. Et les TIRS EN
      // ATTENTE sont désarmés (file vidée, fusées tuées), comme les fenêtres
      // des charges : un tir posé pendant la pause ne doit pas exploser en
      // fantôme dans la scène neuve.
      this.sparkEpoch = (this.sparkEpoch + 1) % 2097152;
      this.sparksAlive = 0;
      this.sparkQLen = 0;
      for (const rk of this.rockets) {
        rk.live = false;
      }
    }
    this.sphereOn = input.sphereActive;
    // L'encre sélectionnée s'applique aux FUTURS émetteurs — et à celui qu'on tient
    // en main (manipulation directe). Jamais à un émetteur posé et lâché : changer
    // d'encre ne doit pas éteindre la flamme pilote à distance.
    if (this.heldEmitter >= 0) {
      this.emitters[this.heldEmitter]!.ink = input.emitInk;
    }

    if (input.addEmitter && this.emitters.length < SIM3_DEFAULTS.maxEmitters) {
      // Nouvel émetteur : rayon du pointeur ∩ plan horizontal des émetteurs.
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      const planeY = GRID3 * 0.08;
      const t = (planeY - this.rayO[1]) / (this.rayD[1] || 1e-6);
      const clampXZ = (v: number): number => Math.min(Math.max(v, GRID3 * 0.08), GRID3 * 0.92);
      const px = t > 0 ? clampXZ(this.rayO[0] + t * this.rayD[0]) : GRID3 * 0.5;
      const pz = t > 0 ? clampXZ(this.rayO[2] + t * this.rayD[2]) : GRID3 * 0.5;
      this.emitters.push({ pos: [px, planeY, pz], ink: input.emitInk });
      // Ce qu'on vient de poser est sélectionné : ses poignées sont là tout de
      // suite, précisément quand on veut le relever du plan où il a atterri.
      this.selected = this.emitters.length - 1;
    }
    if (input.removeEmitter && this.emitters.length > 1) {
      // Celui qu'on a DÉSIGNÉ, pas le dernier posé. Retirer toujours le dernier
      // rendait un émetteur du milieu impossible à effacer sans effacer les
      // autres. Repli sur le dernier quand rien n'est sélectionné.
      const i =
        this.selected >= 0 && this.selected < this.emitters.length
          ? this.selected
          : this.emitters.length - 1;
      this.emitters.splice(i, 1);
      // Les index au-delà glissent d'un cran : toute référence par index doit
      // tomber, sinon on manipulerait le voisin sans s'en rendre compte.
      this.deselect();
      this.grabbed = -2;
    }
    // EXPLOSION : sous le pointeur, à mi-hauteur — comme les champs de force.
    if (input.explode) {
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      // Le point de visée est l'intersection du rayon avec le plan de la charge.
      // ATTENTION au plan BAS : vue de presque niveau, une caméra le coupe très
      // loin de la boîte, et rabattre ce point sur la paroi la plus proche pose
      // la bombe dans un COIN. Hors de la boîte, on retombe donc au centre — une
      // détonation mal visée doit rester au milieu de la scène, pas se coller à
      // un mur où elle n'a plus de place pour monter.
      const planeY = GRID3 * input.explosionHeight;
      const t = (planeY - this.rayO[1]) / (this.rayD[1] || 1e-6);
      const rawX = this.rayO[0] + t * this.rayD[0];
      const rawZ = this.rayO[2] + t * this.rayD[2];
      const inside =
        t > 0 && rawX > 0 && rawX < GRID3 && rawZ > 0 && rawZ < GRID3;
      const clampXZ = (v: number): number => Math.min(Math.max(v, GRID3 * 0.14), GRID3 * 0.86);
      this.fireBurst(
        inside ? clampXZ(rawX) : GRID3 * 0.5,
        planeY,
        inside ? clampXZ(rawZ) : GRID3 * 0.5,
        input,
      );
    }
    // TRANCHE de la fenêtre d'injection couverte par cette frame. Le `min` est
    // tout le sujet : sans lui, la charge délivrée dépend du PAS DE TEMPS, donc
    // du framerate, donc de la RÉSOLUTION — une grille plus fine tourne moins
    // vite. Trois régimes fautifs coexistaient : la dernière tranche était
    // perdue (sous-injection), les framerates intermédiaires en ajoutaient une
    // de trop (sur-injection jusqu'à +67 %), et dès que dt atteignait la durée
    // de la bouffée la fenêtre se refermait AVANT la moindre injection — une
    // détonation strictement sans effet, ce qui est le cas des grosses grilles.
    // La fenêtre ne s'écoule pas non plus en PAUSE : sinon la charge se consume
    // pendant qu'on regarde, et repartir donne une explosion amputée.
    for (const b of this.bursts) {
      b.frac = 0;
      b.dustFrac = 0;
      if (running && dt > 1e-6) {
        if (b.left > 0) {
          b.frac = Math.min(b.left, dt) / dt;
          b.left = Math.max(b.left - dt, 0);
        }
        if (b.dustLeft > 0) {
          b.dustFrac = Math.min(b.dustLeft, dt) / dt;
          b.dustLeft = Math.max(b.dustLeft - dt, 0);
        }
      }
    }
    // FUSÉES : la cinématique CPU — le même Euler semi-implicite que la
    // traçante GPU (patron 3), vitesse d'abord, position ensuite. À l'apogée
    // (vy ≤ 0, ou le plafond de la boîte), le CPU tire l'éclat : les étoiles
    // ET la charge de chaleur presque sans carburant (flash + poussée + fumée
    // grise ; POUSSIÈRE NULLE — un éclat en l'air n'arrache pas le sol).
    if (running && dt > 1e-6) {
      for (const rk of this.rockets) {
        if (!rk.live) continue;
        rk.vy -= ROCKET_G * GRID3 * dt;
        rk.pos[1] += rk.vy * dt;
        if (rk.vy <= 0 || rk.pos[1] > GRID3Y - 2) {
          rk.live = false;
          const px = rk.pos[0] / GRID3;
          const py = rk.pos[1] / GRID3;
          const pz = rk.pos[2] / GRID3;
          if (rk.color2[0] >= 0) {
            // BI-COULEUR : la coque à la teinte principale, le cœur (plus
            // lent, donc plus petit) à la seconde — la file étale d'un pas.
            const shell = Math.round(rk.count * 0.65);
            this.launchFirework(px, py, pz, rk.color[0], rk.color[1], rk.color[2], shell, rk.speed, rk.pattern);
            this.launchFirework(px, py, pz, rk.color2[0], rk.color2[1], rk.color2[2], rk.count - shell, rk.speed * 0.55, rk.pattern);
          } else {
            this.launchFirework(px, py, pz, rk.color[0], rk.color[1], rk.color[2], rk.count, rk.speed, rk.pattern);
          }
          this.fireBurst(rk.pos[0], rk.pos[1], rk.pos[2], input, {
            radius: 0.045 * GRID3,
            fuelMul: 0.12,
            sparkMul: 0.8,
            dustTime: 0,
          });
        }
      }
    }
    // ÉTINCELLES : le premier tir de la FILE est consommé à chaque pas SIMULÉ
    // (en pause la file attend — comme les fenêtres des charges), et il arme
    // la fenêtre de vie des passes ; elle décroît en temps simulé et les trois
    // passes sont entièrement sautées dès qu'elle est vide.
    this.sparkFireNow = running && dt > 1e-6 && this.sparkQLen > 0;
    if (this.sparkFireNow) {
      const ev = this.sparkQueue[this.sparkQHead]!;
      this.sparkQHead = (this.sparkQHead + 1) % this.sparkQueue.length;
      this.sparkQLen--;
      const sk = this.sparkEvent;
      sk.pos[0] = ev.pos[0];
      sk.pos[1] = ev.pos[1];
      sk.pos[2] = ev.pos[2];
      sk.speed = ev.speed;
      sk.color[0] = ev.color[0];
      sk.color[1] = ev.color[1];
      sk.color[2] = ev.color[2];
      sk.count = ev.count;
      sk.base = ev.base;
      sk.pattern = ev.pattern;
      sk.window = ev.window;
      sk.seed = ev.seed;
      this.sparksAlive = Math.max(this.sparksAlive, ev.window);
    }
    if (running && this.sparksAlive > 0) {
      this.sparksAlive = Math.max(this.sparksAlive - dt, 0);
    }
    // CHAMPS DE FORCE : posés à mi-hauteur, là où le panache passe.
    if (input.addField && this.fields.length < SIM3_DEFAULTS.maxFields) {
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      const planeY = GRID3 * 0.42;
      const t = (planeY - this.rayO[1]) / (this.rayD[1] || 1e-6);
      const clampXZ = (v: number): number => Math.min(Math.max(v, GRID3 * 0.1), GRID3 * 0.9);
      const px = t > 0 ? clampXZ(this.rayO[0] + t * this.rayD[0]) : GRID3 * 0.5;
      const pz = t > 0 ? clampXZ(this.rayO[2] + t * this.rayD[2]) : GRID3 * 0.5;
      this.fields.push({
        pos: [px, planeY, pz],
        // Tourbillon d'axe vertical, vent local horizontal : les orientations
        // qui parlent le plus pour un panache qui monte.
        axis: input.fieldType < 0.5 ? [0, 1, 0] : [1, 0, 0],
        type: input.fieldType,
        strength: input.fieldStrength,
        radius: input.fieldRadius,
      });
      this.selected = FluidSim3D.FIELD_TAG + this.fields.length - 1;
    }
    if (input.removeField && this.fields.length > 0) {
      const tagged = this.selected - FluidSim3D.FIELD_TAG;
      const i = tagged >= 0 && tagged < this.fields.length ? tagged : this.fields.length - 1;
      this.fields.splice(i, 1);
      this.deselect();
      this.grabbed = -2;
    }
    // Les réglages s'appliquent aux FUTURS champs et à celui qu'on TIENT — jamais
    // à un champ posé et lâché (même piège que l'encre des émetteurs, qui
    // éteignait la flamme pilote à distance).
    if (this.heldField >= 0) {
      const f = this.fields[this.heldField];
      if (f) {
        // L'axe ne revient à sa valeur canonique QUE si le type change. Le
        // remettre à chaque frame effaçait, à la frame suivante, toute
        // orientation donnée à la main par le bouton d'orientation.
        const wasVortex = f.type < 0.5;
        const isVortex = input.fieldType < 0.5;
        if (wasVortex !== isVortex) {
          f.axis[0] = isVortex ? 0 : 1;
          f.axis[1] = isVortex ? 1 : 0;
          f.axis[2] = 0;
        }
        f.type = input.fieldType;
        f.strength = input.fieldStrength;
        f.radius = input.fieldRadius;
      }
    }

    if (input.grab.active) {
      this.computeRay(input.pointer.ndcX, input.pointer.ndcY);
      if (this.grabbed === -2) {
        const axis = this.pickHandle();
        const held = this.selectedPos();
        if (axis === FluidSim3D.AIM_DRAG && held !== undefined) {
          // Réorientation : le bouton coulisse sur une SPHÈRE centrée sur
          // l'objet, dont on fige le rayon — sinon il grandirait à mesure qu'on
          // s'éloigne de la caméra, en plein geste.
          this.grabbed = this.selected;
          this.dragAxis = FluidSim3D.AIM_DRAG;
          this.dragAimRadius = this.handleLength(held) * SIM3_DEFAULTS.handleAim;
        } else if (axis >= 0 && held !== undefined) {
          // Traînée CONTRAINTE : on fige la droite de contrainte et le point de
          // prise, pour que l'objet ne saute pas sous le curseur au premier pas.
          this.grabbed = this.selected;
          this.dragAxis = axis;
          this.dragOrigin[0] = held[0];
          this.dragOrigin[1] = held[1];
          this.dragOrigin[2] = held[2];
          const s = this.axisParam(this.dragOrigin, axis);
          this.dragGrip = Number.isFinite(s) ? s : 0;
        } else {
          this.grabbed = this.pickTarget();
          this.dragAxis = -1;
          if (this.grabbed !== -2) {
            this.selected = this.grabbed;
          }
        }
      }
      if (this.grabbed !== -2) {
        // Déplacement sur le plan face caméra passant par la position de l'objet.
        const target =
          this.grabbed === -1
            ? this.spherePos
            : this.heldField >= 0
              ? this.fields[this.heldField]!.pos
              : this.emitters[this.heldEmitter]!.pos;
        const px = target[0];
        const py = target[1];
        const pz = target[2];
        const r = this.renderData;
        const lo = GRID3 * 0.05;
        const hi = GRID3 * 0.95;
        // Le plafond suit la HAUTEUR du domaine, pas sa largeur : dans une boîte
        // haute, s'arrêter à 0,95 × largeur bloquerait tout objet au tiers de la
        // scène.
        const hiY = GRID3Y - GRID3 * 0.05;
        if (this.dragAxis === FluidSim3D.AIM_DRAG) {
          this.aimSelected(target);
        } else if (this.dragAxis >= 0) {
          // Poignée : un seul degré de liberté, mesuré sur la droite figée.
          const a = this.dragAxis;
          const s = this.axisParam(this.dragOrigin, a);
          if (Number.isFinite(s)) {
            const floor = a === 1 ? GRID3 * 0.04 : lo;
            target[a] = Math.min(
              Math.max(this.dragOrigin[a]! + (s - this.dragGrip), floor),
              a === 1 ? hiY : hi,
            );
          }
        } else {
        const denom =
          r[12]! * this.rayD[0] + r[13]! * this.rayD[1] + r[14]! * this.rayD[2];
        if (Math.abs(denom) > 1e-5) {
          const t =
            (r[12]! * (target[0] - this.rayO[0]) +
              r[13]! * (target[1] - this.rayO[1]) +
              r[14]! * (target[2] - this.rayO[2])) /
            denom;
          if (t > 0) {
            target[0] = Math.min(Math.max(this.rayO[0] + t * this.rayD[0], lo), hi);
            target[1] = Math.min(Math.max(this.rayO[1] + t * this.rayD[1], GRID3 * 0.04), hiY);
            target[2] = Math.min(Math.max(this.rayO[2] + t * this.rayD[2], lo), hi);
          }
        }
        }
        // La sphère saisie communique sa vitesse au fluide (lissée, plafonnée CFL).
        if (this.grabbed === -1 && dt > 0) {
          const cap = 420 * SCALE3;
          for (let a = 0; a < 3; a++) {
            const prev = a === 0 ? px : a === 1 ? py : pz;
            const inst = (this.spherePos[a]! - prev) / dt;
            this.sphereVel[a] = 0.55 * this.sphereVel[a]! + 0.45 * inst;
          }
          const mag = Math.hypot(this.sphereVel[0], this.sphereVel[1], this.sphereVel[2]);
          if (mag > cap) {
            const k = cap / mag;
            this.sphereVel[0] *= k;
            this.sphereVel[1] *= k;
            this.sphereVel[2] *= k;
          }
        }
      }
    } else {
      this.grabbed = -2;
      this.dragAxis = -1;
    }
    // Hors saisie (ou autre cible), la vitesse de la boule s'amortit vite.
    if (this.grabbed !== -1) {
      this.sphereVel[0] *= 0.82;
      this.sphereVel[1] *= 0.82;
      this.sphereVel[2] *= 0.82;
    }
  }

  /**
   * Lecture ponctuelle du volume de densités pour l'export VDB — la SEULE
   * lecture GPU→CPU du moteur 3D, hors boucle de frame (même exception
   * documentée que l'export PNG du 2D). Alloue un staging buffer par appel.
   */
  async exportVolume(): Promise<{ density: Float32Array; heat: Float32Array }> {
    const n = GRID3;
    const texels = n * n * n;
    const buffer = this.device.createBuffer({
      label: 'export3d-readback',
      size: texels * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder({ label: 'export3d' });
    encoder.copyTextureToBuffer(
      { texture: this.denTextures[this.denIdx] },
      { buffer, bytesPerRow: n * 8, rowsPerImage: n },
      { width: n, height: n, depthOrArrayLayers: n },
    );
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buffer.getMappedRange());
    const density = new Float32Array(texels);
    const heat = new Float32Array(texels);
    for (let i = 0; i < texels; i++) {
      // Densité totale = somme des trois encres (xyz) ; chaleur = w.
      density[i] =
        halfToFloat(halves[i * 4]!) + halfToFloat(halves[i * 4 + 1]!) + halfToFloat(halves[i * 4 + 2]!);
      heat[i] = halfToFloat(halves[i * 4 + 3]!);
    }
    buffer.destroy();
    return { density, heat };
  }

  /**
   * Encode un V-cycle multigrid complet sur la pression du niveau 0 (warm start).
   * Descente : balayages rouge-noir, résidu, restriction ; plus grossier :
   * lissage long ; remontée : prolongation trilinéaire ajoutée EN PLACE +
   * post-lissage. Aucun ping-pong : voir le commentaire dans le corps.
   */
  private encodeVCycle3(encoder: GPUCommandEncoder, skipFinePre = false): void {
    const levels = this.mgLevels;
    const last = levels.length - 1;
    // Le cycle gère ses propres passes compute, le segment minuscule isolé
    // dans la sienne (héritage du débogage de la pathologie ci-dessous ; les
    // passes découpées n'ont pas suffi seules, mais elles rendent le cycle
    // lisible dans un profil GPU et ne coûtent rien).
    let cp = encoder.beginComputePass({ label: 'mg3-descente' });
    cp.setBindGroup(0, this.group0);
    const cut = (label: string): void => {
      cp.end();
      cp = encoder.beginComputePass({ label });
      cp.setBindGroup(0, this.group0);
    };
    // ARCHITECTURE, et les deux plaies qui l'ont sculptée (2026-08-27) :
    //
    // · GRANDS niveaux : rouge-noir EN PLACE (read_write) — pas de ping-pong,
    //   pas d'écriture de la moitié inchangée. Le niveau 0 vit dans
    //   pression[pressIdx], les niveaux grossiers dans leur ping 0 : plus une
    //   seule bascule d'index dans le cycle.
    // · Niveaux MINUSCULES (≤ 24³) : rouge-noir en PING-PONG, UN SEUL pipeline,
    //   offsets d'uniforme STATIQUES. Toute autre forme mesurée là — read_write,
    //   deux pipelines alternés, offsets dynamiques — fait passer la frame 384³
    //   de ~50 à ~230 ms, sans exception ni explication propre (pathologie de
    //   driver ; les passes séparées n'y changent rien).
    // · JAMAIS un seul V-cycle par frame (voir vcyclesFor) : à ×1 le solve
    //   dégénère numériquement frame à frame — boîte vide en quelques secondes
    //   à 384³, quel que soit le lisseur — quand ×2, strictement identique par
    //   cycle, reste sain. Le Jacobi d'origine montrait la même maladie en
    //   moins fort (le « panache couché » du journal). Non élucidé au fond ;
    //   contourné en ne prenant jamais ce chemin.
    const p0 = this.pressIdx;
    // HOOK DE BISSECTION (debug seulement) : __mgLast limite la profondeur de la
    // pyramide, __mgSkip saute des passes nommées — pour localiser un coût.
    const dbg = globalThis as unknown as { __mgLast?: number; __mgSkip?: string };
    const lastEff = Math.min(last, dbg.__mgLast ?? last);
    const skip = dbg.__mgSkip ?? '';

    // L'équation d'erreur des niveaux grossiers part de zéro à chaque cycle.
    cp.setPipeline(this.pipelines.clearScalar);
    for (let l = 1; l <= lastEff; l++) {
      const lev = levels[l]!;
      cp.setBindGroup(0, lev.clearBind!);
      cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
    }
    // Le clear utilise un autre layout de groupe 0 : on rétablit le groupe partagé.
    cp.setBindGroup(0, this.group0);

    // Un balayage rouge-noir : deux dispatches EN PLACE sur la même texture
    // (les voisins d'une cellule sont tous de l'autre couleur, et WebGPU ordonne
    // les écritures entre dispatches).
    const rbSweeps = (lev: MGLevel3, l: number, count: number): void => {
      const bind = lev.rbBind[l === 0 ? p0 : 0];
      // Le niveau le plus grossier est toujours minuscule (côté ≤ 24 quel que
      // soit GRID3) : le SOR ne passe donc jamais par ici.
      const red = this.pipelines.mgSmoothRed;
      const black = this.pipelines.mgSmoothBlack;
      for (let i = 0; i < count; i++) {
        cp.setPipeline(red);
        cp.setBindGroup(1, bind);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
        cp.setPipeline(black);
        cp.setBindGroup(1, bind);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
      }
    };
    // Niveaux MINUSCULES : rouge-noir en PING-PONG (2 dispatches par balayage,
    // nombre de dispatches PAIR — la pression du niveau repart du ping `start`
    // et retombe dessus après chaque balayage). Jamais de read_write ici, et UN
    // SEUL pipeline : parité et ω par offset dynamique — deux contournements de
    // driver, tous deux mesurés. (Un niveau minuscule n'est jamais le niveau
    // 0 : la grille fine fait ≥ 128.)
    const tinySweeps = (lev: MGLevel3, count: number, start = 0, sor = false): void => {
      const w = (sor ? 1 : 0) as PingIndex;
      cp.setPipeline(this.pipelines.mgTiny);
      for (let i = 0; i < count; i++) {
        cp.setBindGroup(1, lev.tinyBind[(start & 1) as PingIndex][0][w]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
        cp.setBindGroup(1, lev.tinyBind[((start + 1) & 1) as PingIndex][1][w]);
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
      }
    };
    const sweeps = (lev: MGLevel3, l: number, count: number, sor = false): void => {
      if (lev.size <= MG3_TINY) {
        tinySweeps(lev, count, 0, sor);
      } else {
        rbSweeps(lev, l, count);
      }
    };

    // Descente.
    let tinyOpen = false;
    for (let l = 0; l <= lastEff; l++) {
      const lev = levels[l]!;
      if (!tinyOpen && lev.size <= MG3_TINY) {
        tinyOpen = true;
        cut('mg3-minuscules');
      }
      if (!skip.includes('smooth')) {
        // Au niveau le plus grossier : SOR — il résout, il ne lisse pas.
        // `skipFinePre` (cycles enchaînés) : le pré-lissage du niveau 0 d'un
        // cycle suivant relisse ce que le post du cycle précédent vient de
        // lisser, au même second membre — un balayage fin économisé.
        if (!(skipFinePre && l === 0)) {
          sweeps(lev, l, l === lastEff ? MG3_COARSE_SMOOTH : MG3_PRE_SMOOTH, l === lastEff);
        }
      }
      if (l < lastEff) {
        // Résidu + restriction FUSIONNÉS : une passe à la taille grossière.
        const coarse = levels[l + 1]!;
        cp.setPipeline(this.pipelines.mgRestrict);
        cp.setBindGroup(1, lev.restrictBind![l === 0 ? p0 : 0]);
        cp.dispatchWorkgroups(coarse.dispatch, coarse.dispatchY, coarse.dispatch);
      }
    }

    // Remontée.
    for (let l = lastEff - 1; l >= 0; l--) {
      const lev = levels[l]!;
      const tiny = lev.size <= MG3_TINY;
      if (tinyOpen && !tiny) {
        tinyOpen = false;
        cut('mg3-remontee');
      }
      if (!skip.includes('prolong')) {
        if (tiny) {
          // Cible MINUSCULE : ping-pong (0 → 1), le read_write y est
          // pathologique (mesuré : la prolongation en place vers 24³ suffisait
          // à faire tomber la frame de 46 à 260 ms à 384³).
          cp.setPipeline(this.pipelines.mgProlongPP);
          cp.setBindGroup(1, lev.prolongPPBind!);
        } else {
          cp.setPipeline(this.pipelines.mgProlong);
          cp.setBindGroup(1, lev.prolongBind![l === 0 ? p0 : 0]);
        }
        cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
      }
      if (!skip.includes('post')) {
        if (tiny) {
          // La prolongation ping-pong a laissé la pression sur le ping 1 : les
          // balayages partent de 1, et une DEMI-PASSE finale (rouge seule, qui
          // recopie les noirs) la ramène sur le ping 0 que les binds du parent
          // supposent.
          tinySweeps(lev, MG3_POST_SMOOTH, 1);
          cp.setPipeline(this.pipelines.mgTiny);
          cp.setBindGroup(1, lev.tinyBind[1][0][0]);
          cp.dispatchWorkgroups(lev.dispatch, lev.dispatchY, lev.dispatch);
        } else {
          sweeps(lev, l, MG3_POST_SMOOTH);
        }
      }
    }
    cp.end();
  }


  private writeSimUniforms(dt: number, input: Frame3DInput): void {
    const d = this.simData;
    const D = SIM3_DEFAULTS;
    const p = input.params;
    d[0] = dt;
    d[1] = this.simTime;
    d[2] = GRID3;
    // misc.w porte la HAUTEUR du domaine (Ny/Nx) : c'est le seul slot déclaré
    // par tous les shaders, donc le seul endroit où la forme de la grille est
    // lisible partout sans rallonger le préfixe d'uniforme de chacun. La force
    // de vorticité, qui l'occupait, a déménagé en sphere_vel.w.
    d[3] = HEIGHT3;
    // VORTICITÉ : PAS de mise à l'échelle, et ce n'est pas un oubli.
    // La forme de Fedkiw est f = ε·h·(N̂ × ω) : le facteur h (taille de cellule,
    // 1/N en monde) annule EXACTEMENT la conversion monde→voxels (×N) de
    // l'accélération. Le ε discret est donc déjà indépendant de la résolution.
    // J'ai cru le contraire le 2026-08-26 en lisant la formule discrète sans son
    // h, et j'ai ajouté ×SCALE3 : à 384³ et 512³ le confinement injectait alors
    // assez d'énergie pour faire naître un TOURBILLON PARASITE qui envahissait
    // la boîte au bout de quelques secondes (invisible à 128³ et 256³, où le
    // facteur restait petit). Signalé par Renaud, reproduit, puis isolé en
    // rejouant la même scène avec l'ancienne valeur effective.
    d[55] = p.vorticityStrength;
    // Émetteurs : position d'état + petit balancement propre à chacun (déphasé).
    const emitterSlot = (slot: number, i: number): void => {
      const e = this.emitters[i]!;
      const wob = GRID3 * 0.02;
      d[slot] = e.pos[0] + wob * Math.sin(this.simTime * 0.9 + i * 2.1);
      d[slot + 1] = e.pos[1];
      d[slot + 2] = e.pos[2] + wob * Math.cos(this.simTime * 0.7 + i * 1.7);
      d[slot + 3] = GRID3 * D.emitterRadius;
    };
    emitterSlot(4, 0);
    d[8] = p.emitHeat;
    d[9] = p.emitInkRate;
    // Forces absolues calibrées à 128³ → remises à l'échelle de la grille.
    d[10] = D.emitUpVelocity * SCALE3;
    d[11] = D.emitWobbleVelocity * SCALE3;
    d[12] = p.velocityDissipation;
    d[13] = p.inkDissipation;
    d[14] = p.heatCooling;
    d[15] = p.buoyancy * SCALE3;
    // Souffle du pointeur : rayon caméra→scène + force selon le geste écran.
    const b = input.blow;
    const r = this.renderData;
    if (b.active) {
      this.computeRay(b.ndcX, b.ndcY);
      d[16] = this.rayO[0];
      d[17] = this.rayO[1];
      d[18] = this.rayO[2];
      d[19] = D.blowRadius * GRID3;
      d[20] = this.rayD[0];
      d[21] = this.rayD[1];
      d[22] = this.rayD[2];
      d[23] = 1;
      const s = p.blowForce * GRID3;
      d[24] = (r[4]! * b.moveX + r[8]! * b.moveY) * s;
      d[25] = (r[5]! * b.moveX + r[9]! * b.moveY) * s;
      d[26] = (r[6]! * b.moveX + r[10]! * b.moveY) * s;
    } else {
      d[23] = 0;
    }
    // Sphère-obstacle (voxels ; rayon ≤ 0 = absente).
    d[28] = this.spherePos[0];
    d[29] = this.spherePos[1];
    d[30] = this.spherePos[2];
    d[31] = this.sphereOn ? GRID3 * D.sphereRadius : -1;
    // Émetteurs + combustion (taux, chaleur dégagée, expansion en voxels/s).
    d[32] = this.emitters.length;
    d[33] = p.burnRate;
    d[34] = p.heatYield;
    // EXPANSION : une source de DIVERGENCE, donc en 1/s — surtout pas ×SCALE3.
    // La divergence discrète est la somme des écarts de vitesse d'une face à
    // l'autre, en voxels/s par voxel : c'est DÉJÀ une grandeur en 1/s,
    // indépendante de la résolution. La multiplier par SCALE3 rendait le souffle
    // proportionnel à N — mesuré : à temps simulé égal, la même détonation
    // donnait une colonnette à 128³ et un nuage massif à 384³. C'est ce qui
    // faisait « empirer » les hautes résolutions.
    // (Les accélérations, elles, se mettent bien à l'échelle : une accélération
    //  en monde/s² vaut N fois plus en voxels/s². D'où la poussée, le vent, le
    //  poids des matières et la force des champs, tous ×SCALE3 à juste titre.)
    d[35] = p.expansion;
    for (let i = 1; i < 4; i++) {
      if (i < this.emitters.length) {
        emitterSlot(32 + i * 4, i);
      }
    }
    for (let i = 0; i < 4; i++) {
      d[48 + i] = this.emitters[i]?.ink ?? 0;
    }
    // Vitesse de la sphère (voxels/s) : condition de bord mobile.
    d[52] = this.sphereOn ? this.sphereVel[0] : 0;
    d[53] = this.sphereOn ? this.sphereVel[1] : 0;
    d[54] = this.sphereOn ? this.sphereVel[2] : 0;
    // Poids propre des matières (fumée / encre / carburant).
    d[56] = D.inkWeights[0] * SCALE3;
    d[57] = D.inkWeights[1] * SCALE3;
    d[58] = D.inkWeights[2] * SCALE3;
    // Vent : force en voxels/s² (donc ×SCALE3 comme la poussée), cap et
    // oscillation en radians.
    d[60] = p.windStrength * SCALE3;
    d[61] = (p.windSwing * Math.PI) / 180;
    d[62] = p.windPeriod;
    d[63] = (p.windHeading * Math.PI) / 180;
    // Champs de force : méta (nombre + types) puis deux vec4 par champ.
    d[64] = this.fields.length;
    for (let i = 0; i < SIM3_DEFAULTS.maxFields; i++) {
      const f = this.fields[i];
      d[65 + i] = f?.type ?? 0;
      const o = 68 + i * 8;
      d[o] = f?.pos[0] ?? 0;
      d[o + 1] = f?.pos[1] ?? 0;
      d[o + 2] = f?.pos[2] ?? 0;
      d[o + 3] = (f?.radius ?? 0) * GRID3;
      d[o + 4] = f?.axis[0] ?? 0;
      d[o + 5] = f?.axis[1] ?? 1;
      d[o + 6] = f?.axis[2] ?? 0;
      d[o + 7] = (f?.strength ?? 0) * SCALE3;
    }
    // OBSTACLE : forme et paramètres (slot #23 = floats 92-95, l'ancien slot
    // d'explosion unique). Les dix passes lisent la même `solid_sd`.
    const shape = OBSTACLE_SHAPES[input.obstacleShape] ?? OBSTACLE_SHAPES[0]!;
    d[92] = input.obstacleShape;
    d[93] = shape.params[0];
    d[94] = shape.params[1];
    d[95] = shape.params[2];
    // CHIMIE (slot #24 = floats 96-99) : stœchiométrie de l'oxygène.
    d[96] = p.oxygenBurn;
    // (Slots 97-99 : libres — les charges vivent en 116-147, en fin de tampon.)
    // Suie : rendement et évanouissement (le rendu lit sa densité côté render).
    d[100] = p.sootYield;
    d[101] = p.sootFade;
    d[102] = p.sootCooling;
    // Ciel ouvert : la bande est donnée en VOXELS au shader (fraction × N).
    d[104] = p.openBand * GRID3;
    d[105] = p.openStrength;
    d[106] = p.openCeilBand * GRID3;
    d[107] = p.openCeilStrength;
    // Poussière soulevée : rayon et épaisseur EN VOXELS — communs à toutes les
    // charges. Le DÉBIT, lui, suit chaque charge (fenêtre propre), plus bas.
    d[109] = input.dustRadius * GRID3;
    d[110] = SIM3_DEFAULTS.dustHeight * GRID3;
    d[111] = p.heatCeiling;
    // Stratification : la base est donnée en VOXELS (fraction × hauteur).
    d[112] = p.stratStrength;
    d[113] = p.stratBase * GRID3Y;
    // Oxygène : apport du souffle (blow_force.w) et récupération lente.
    d[27] = D.blowOxygen;
    d[59] = p.oxygenRecover;
    // CHARGES : deux vec4 par charge — centre + rayon, puis les débits et la
    // graine. Les débits portent déjà la fraction de frame de LEUR fenêtre ;
    // une charge éteinte a ses trois débits à zéro et le shader saute ses blocs.
    for (let i = 0; i < this.bursts.length; i++) {
      const b = this.bursts[i];
      if (!b) continue;
      const o = 116 + i * 8;
      d[o] = b.pos[0];
      d[o + 1] = b.pos[1];
      d[o + 2] = b.pos[2];
      d[o + 3] = b.radius;
      d[o + 4] = b.frac * input.explosionFuel * b.fuelMul;
      d[o + 5] = b.frac * input.explosionSpark * b.sparkMul;
      d[o + 6] = b.dustFrac * input.dustRate;
      d[o + 7] = b.seed;
    }
    // ÉTINCELLES : l'événement de tir, trois vec4 en QUEUE de tampon (au-delà
    // des 148 floats de la simulation — le shader y accède par un struct à
    // blocs de padding). Le compte n'est non nul QUE la frame du tir : comme
    // les charges, l'événement se consume tout seul.
    const sk = this.sparkEvent;
    d[148] = sk.pos[0];
    d[149] = sk.pos[1];
    d[150] = sk.pos[2];
    d[151] = sk.speed;
    d[152] = sk.color[0];
    d[153] = sk.color[1];
    d[154] = sk.color[2];
    d[155] = this.sparkFireNow ? sk.count : 0;
    d[156] = sk.base;
    d[157] = sk.seed;
    d[158] = this.sparkEpoch;
    d[159] = sk.pattern;
    this.device.queue.writeBuffer(this.simUniforms, 0, d);
  }

  private writeRenderUniforms(input: Frame3DInput, aspect: number): void {
    const d = this.renderData;
    const { azimuth, elevation, radius } = input.cam;
    const cy = Math.cos(elevation);
    const px = radius * cy * Math.cos(azimuth);
    const py = radius * Math.sin(elevation);
    const pz = radius * cy * Math.sin(azimuth);
    // Base orthonormée regardant le centre de la boîte.
    // La caméra vise le CENTRE DE LA BOÎTE, qui n'est plus l'origine dès que le
    // domaine est plus haut que large (le sol reste à −0,5, la boîte pousse vers
    // le haut). Sans ce recentrage, un domaine ×3 est cadré sur son premier
    // tiers et le reste sort de l'écran.
    const boxMid = -0.5 + HEIGHT3 * 0.5;
    const fl = Math.hypot(px, py, pz);
    const fx = -px / fl;
    const fy = -py / fl;
    const fz = -pz / fl;
    // right = fwd × up_monde, up = right × fwd (base droitière, écran non miroir).
    let rx = -fz;
    const ry = 0;
    let rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1;
    rx /= rl;
    rz /= rl;
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;
    d[0] = px;
    d[1] = py + boxMid;
    d[2] = pz;
    d[3] = 0.364; // tan(fov/2), fov ≈ 40°
    d[4] = rx;
    d[5] = ry;
    d[6] = rz;
    d[7] = aspect;
    d[8] = ux;
    d[9] = uy;
    d[10] = uz;
    d[11] = input.exposure;
    d[12] = fx;
    d[13] = fy;
    d[14] = fz;
    d[15] = input.raymarchSteps;
    // Lumière directionnelle fixe (normalisée), intensité.
    d[16] = 0.5;
    d[17] = 0.74;
    d[18] = 0.45;
    d[19] = 1.0;
    // Sphère-obstacle en unités monde pour le rendu.
    d[20] = this.spherePos[0] / GRID3 - 0.5;
    d[21] = this.spherePos[1] / GRID3 - 0.5;
    d[22] = this.spherePos[2] / GRID3 - 0.5;
    d[23] = input.sphereActive ? SIM3_DEFAULTS.sphereRadius : -1;
    // Retour visuel du souffle : fuseau le long du rayon (unités monde), si actif.
    if (input.feedback && input.blow.active) {
      this.computeRay(input.blow.ndcX, input.blow.ndcY);
      d[24] = this.rayO[0] / GRID3 - 0.5;
      d[25] = this.rayO[1] / GRID3 - 0.5;
      d[26] = this.rayO[2] / GRID3 - 0.5;
      d[27] = SIM3_DEFAULTS.blowRadius;
      d[28] = this.rayD[0];
      d[29] = this.rayD[1];
      d[30] = this.rayD[2];
      d[31] = 1;
    } else {
      d[31] = 0;
    }
    // Style : lueur du feu (raymarch), bloom (présentation), braises (débit),
    // N (conversion voxels → monde du tracé des braises).
    d[32] = input.glowStrength;
    d[33] = input.bloomStrength;
    d[34] = input.embersOn ? input.emberStrength : 0;
    d[35] = GRID3;
    // Gizmos des champs de force : deux vec4 par champ — (centre monde, rayon
    // monde) et (axe unitaire, type + 2 si actif). La boîte étant le cube
    // unité, le rayon en fraction de grille EST le rayon monde. Coupés avec les
    // retours visuels (F), comme le fuseau du souffle.
    for (let i = 0; i < 3; i++) {
      const f = input.feedback ? this.fields[i] : undefined;
      const o = 36 + i * 8;
      d[o] = (f?.pos[0] ?? 0) / GRID3 - 0.5;
      d[o + 1] = (f?.pos[1] ?? 0) / GRID3 - 0.5;
      d[o + 2] = (f?.pos[2] ?? 0) / GRID3 - 0.5;
      d[o + 3] = f?.radius ?? 0;
      d[o + 4] = f?.axis[0] ?? 0;
      d[o + 5] = f?.axis[1] ?? 1;
      d[o + 6] = f?.axis[2] ?? 0;
      const fieldSel = i === this.selected - FluidSim3D.FIELD_TAG;
      d[o + 7] = f === undefined ? 0 : (f.type < 0.5 ? 0 : 1) + (fieldSel ? 2 : 0);
    }
    // Gizmos des ÉMETTEURS : un vec4 chacun — (centre monde, w = −1 si le slot
    // est vide, sinon numéro d'encre + 4 si c'est l'émetteur actif). Un émetteur
    // posé loin de la flamme n'a AUCUNE trace visible tant qu'il n'a rien
    // allumé ; et rien ne disait lequel des quatre les touches 1/2/3 repeignent.
    for (let i = 0; i < 4; i++) {
      const e = input.feedback ? this.emitters[i] : undefined;
      const o = 60 + i * 4;
      d[o] = (e?.pos[0] ?? 0) / GRID3 - 0.5;
      d[o + 1] = (e?.pos[1] ?? 0) / GRID3 - 0.5;
      d[o + 2] = (e?.pos[2] ?? 0) / GRID3 - 0.5;
      d[o + 3] = e === undefined ? -1 : e.ink + (i === this.selected ? 4 : 0);
    }
    // Options des gizmos : rayon d'émission (le monde EST le cube unité, donc la
    // fraction de grille est déjà le rayon monde) et boîte visible.
    d[76] = SIM3_DEFAULTS.emitterRadius;
    d[77] = input.feedback ? 1 : 0;
    d[78] = this.dragAxis;
    d[79] = SIM3_DEFAULTS.handleInner;
    // POIGNÉES de l'objet sélectionné : (centre monde, w longueur monde — ≤ 0 =
    // pas de sélection). La longueur est calculée ICI et non dans le shader pour
    // que le dessin et la SAISIE partagent exactement le même nombre.
    const sel = input.feedback ? this.selectedPos() : undefined;
    d[80] = (sel?.[0] ?? 0) / GRID3 - 0.5;
    d[81] = (sel?.[1] ?? 0) / GRID3 - 0.5;
    d[82] = (sel?.[2] ?? 0) / GRID3 - 0.5;
    const handle = sel === undefined ? 0 : this.handleLength(sel) / GRID3;
    d[83] = handle;
    // BOUTON D'ORIENTATION : axe unitaire + sa distance monde (0 = l'objet
    // sélectionné n'a pas d'orientation). Un seul nombre sert à la fois de
    // drapeau et de position.
    const axis = sel === undefined ? undefined : this.selectedAxis();
    d[84] = axis?.[0] ?? 0;
    d[85] = axis?.[1] ?? 1;
    d[86] = axis?.[2] ?? 0;
    d[87] = axis === undefined ? 0 : handle * SIM3_DEFAULTS.handleAim;
    // Densité de suie AU RENDU (slot z du vec4 « soot » côté raymarch).
    // PRISE DE VUE, trois valeurs dans un seul slot (le shader teste un
    // INTERVALLE pour l'extérieur) : 0 atelier · 1 extérieur · 2 nuit.
    d[88] = input.params.outdoor ? 1 : input.params.night ? 2 : 0;
    d[89] = input.params.sunHeight;
    d[90] = input.params.sootDensity;
    // Hauteur monde du domaine : le rendu en a besoin pour borner la boîte et
    // pour normaliser la coordonnée verticale de texture.
    d[91] = HEIGHT3;
    // OBSTACLE : la même forme que la simulation, en slot #23 côté rendu aussi.
    const shapeR = OBSTACLE_SHAPES[input.obstacleShape] ?? OBSTACLE_SHAPES[0]!;
    d[92] = input.obstacleShape;
    d[93] = shapeR.params[0];
    d[94] = shapeR.params[1];
    d[95] = shapeR.params[2];
    let dirty = false;
    for (let i = 0; i < d.length; i++) {
      if (d[i] !== this.lastRender[i]) {
        dirty = true;
        break;
      }
    }
    if (dirty) {
      this.lastRender.set(d);
      this.device.queue.writeBuffer(this.renderUniforms, 0, d);
    }
  }
}
