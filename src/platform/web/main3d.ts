/**
 * Point d'entrée du prototype 3D volumétrique (page 3d.html).
 * Couche plateforme minimale : device, canvas/resize, caméra orbitale à la souris
 * (glisser = orbiter, molette = zoom), espace = pause, R = reset, HUD FPS.
 * Mode ?selftest : rapport JSON dans #selftest + titre SELFTEST-OK/FAIL après 240 frames.
 */

import {
  defaultTuning3,
  estimateVram3,
  GRID3,
  GRID3_CHOICES,
  INK_COLORS,
  INK_NAMES,
  setGrid3,
  vcyclesFor,
  GRID3Y,
  HEIGHT3,
  FIELD_NAMES,
  SIM3_DEFAULTS,
  type Sim3Tuning,
} from '../../core3d/config3d';
import { FluidSim3D, type Frame3DInput } from '../../core3d/sim3d';
import { encodeVdb } from '../../core3d/vdb';
import { Panel3D, Toolbar3D } from './panel3d';
import { acquireDevice } from './gpu';
import { showFatalError } from './overlay';

const SELFTEST_FRAMES = 240;

/** Jeux de réglages nommés — partagés par les presets du panneau ET la démo. */
const TUNE_CANDLE: Partial<Sim3Tuning> = {
  buoyancy: 80, velocityDissipation: 0.05, emitHeat: 1.6, emitInkRate: 0.7,
  heatCooling: 1.5, inkDissipation: 0.3, burnRate: 2, heatYield: 0.5, expansion: 8,
};
const TUNE_FURNACE: Partial<Sim3Tuning> = {
  buoyancy: 280, velocityDissipation: 0.02, emitHeat: 6.5, emitInkRate: 3.5,
  heatCooling: 0.55, inkDissipation: 0.15, burnRate: 6, heatYield: 1.05,
  expansion: 48, oxygenRecover: 0.05, blowForce: 320,
};
const TUNE_SMOKE: Partial<Sim3Tuning> = {
  buoyancy: 110, emitHeat: 1.2, emitInkRate: 5.5,
  heatCooling: 1.2, inkDissipation: 0.04,
};
// Le vent est COUPÉ partout ailleurs : ce preset est le seul qui l'allume, pour
// qu'on puisse le voir d'un clic sans qu'il change quoi que ce soit par défaut.
const TUNE_WIND: Partial<Sim3Tuning> = {
  buoyancy: 160, emitHeat: 2.6, emitInkRate: 3.0, heatCooling: 1.0,
  inkDissipation: 0.10, vorticityStrength: 16,
  // 35 CALIBRÉ par captures : à 55 le panache commence à se disperser, à 95 le
  // vent l'écrase au ras de l'émetteur. À 35 il monte haut, penche, et traîne.
  windStrength: 35, windSwing: 40, windPeriod: 8, windHeading: 20,
};

// CHAMPIGNON ATOMIQUE — le cas qui met le moteur à l'épreuve : il ne suffit pas
// de faire une boule de feu, il faut tenir un ANNEAU TOURBILLONNAIRE plusieurs
// secondes, et lui faire traîner une colonne derrière lui. D'où ces choix :
//  · temps RALENTI — une bombe monte lentement ; à vitesse normale la montée est
//    finie avant qu'on ait vu le tore se former ;
//  · dissipation de vitesse quasi nulle et vorticité forte — c'est ce qui garde
//    l'anneau vivant au lieu de le laisser s'étaler en champignon mou ;
//  · refroidissement lent — la boule de feu d'une bombe reste chaude longtemps,
//    et c'est cette chaleur qui alimente la poussée pendant toute la montée ;
//  · suie et poussière poussées — le chapeau sale et le pied sombre ;
//  · bande éponge RESSERRÉE : le chapeau doit pouvoir s'étaler près du plafond
//    au lieu d'y être mangé avant d'avoir pris sa forme.
const TUNE_MUSHROOM: Partial<Sim3Tuning> = {
  // FLAMME PILOTE COUPÉE. C'est le réglage le plus important du preset, et il
  // manquait : un champignon est un ÉVÉNEMENT, pas un régime. Laisser la flamme
  // brûler en continu avec cette dissipation quasi nulle, ce refroidissement
  // lent et cette poussée remplit la boîte de purée grise en quelques secondes —
  // et la détonation s'y perd. Mon harnais de test coupait la flamme avant
  // chaque tir, ce que l'utilisateur ne peut pas faire d'un clic : j'ai donc
  // validé longtemps un scénario irreproductible à la souris.
  emitHeat: 0, emitInkRate: 0,
  timeScale: 0.55, buoyancy: 430, velocityDissipation: 0.008, vorticityStrength: 22,
  heatCooling: 0.42, inkDissipation: 0.02, burnRate: 7, heatYield: 1.3,
  expansion: 24, oxygenRecover: 0.02,
  sootYield: 28, sootCooling: 3.2,
  // Éponge RESSERRÉE ET ADOUCIE — le réglage le plus contre-intuitif du preset.
  // Le chapeau d'un champignon ne TRAVERSE pas la bande, il s'y GARE : il monte
  // au plafond et y reste. Une éponge calibrée pour absorber un panache de
  // passage le lamine alors sur place, et on obtient une dalle diffuse au lieu
  // d'un chapeau à lobes. Mesuré en rejouant le tir entier (cdp-nuke-ab) : à
  // 0,07/26 le chapeau est une dalle ; boîte close il retrouve ses lobes ; à
  // 0,035/8 il les garde ET la boîte ne se remplit pas.
  // Parois FRANCHES pour évacuer la nappe qui s'étale au sol (c'est elle qui
  // empâtait le bas de la boîte et qu'il fallait attendre), plafond DOUX pour
  // que le chapeau garde ses lobes.
  openBand: 0.09, openStrength: 24,
  openCeilBand: 0.035, openCeilStrength: 6,
  // INVERSION : au-dessus du quart bas, l'air ambiant se réchauffe avec
  // l'altitude. Le nuage cesse d'être flottant à mi-hauteur et s'y ÉTALE de
  // lui-même, au lieu de monter jusqu'à s'écraser au plafond. C'est la vraie
  // mécanique du chapeau d'un champignon — dans un cube, le « chapeau » était en
  // réalité le plafond, et on ne s'en apercevait pas.
  // Inversion BASSE et FRANCHE : le chapeau s'arrête aux deux tiers et laisse du
  // ciel au-dessus. C'est ce qui distingue un nuage LIBRE d'un nuage étouffé —
  // tant qu'il touche le plafond, la boîte est encore ce qui l'arrête.
  stratStrength: 3.2, stratBase: 0.16,
  // Le plafond de chaleur ouvre le domaine des BOULES DE FEU. La poussée sature
  // à 2 de son côté, donc relever ce plafond n'ajoute pas de flottabilité : il
  // n'ajoute que de la LUMIÈRE.
  heatCeiling: 9,
};

/** Réglages de DÉTONATION d'un preset (ils vivent sur l'entrée, pas sur les
 *  paramètres de simulation : ce sont des réglages d'outil, pas de physique). */
interface BoomTune {
  explosionRadius: number;
  explosionFuel: number;
  explosionHeight: number;
  dustRate: number;
  dustTime: number;
  dustRadius: number;
  explosionSpark: number;
}
const BOOM_DEFAULT: BoomTune = {
  explosionRadius: SIM3_DEFAULTS.explosionRadius,
  explosionFuel: SIM3_DEFAULTS.explosionFuel,
  explosionHeight: SIM3_DEFAULTS.explosionHeight,
  dustRate: SIM3_DEFAULTS.dustRate,
  dustTime: SIM3_DEFAULTS.dustTime,
  dustRadius: SIM3_DEFAULTS.dustRadius,
  explosionSpark: SIM3_DEFAULTS.explosionSpark,
};
// Charge PETITE, posée au ras du sol. Contre-intuitif — « atomique » n'appelle
// pas une charge énorme : dans une boîte de taille fixe, une grosse charge
// remplit le volume AVANT d'avoir eu le temps de monter, et on obtient un
// couvercle de fumée au lieu d'un champignon. Mesuré : à 170 de charge et 30
// d'expansion, la boule occupe la moitié de la boîte dès 1,8 s.
// C'est la POUSSIÈRE qui est poussée à fond : froide, elle n'enfle rien, elle ne
// monte que parce que la boule l'aspire — c'est exactement ce qui fait le pied.
const BOOM_MUSHROOM: BoomTune = {
  // Charge PETITE : la silhouette d'un champignon atomique est HAUTE et étroite,
  // et ce rapport ne s'obtient qu'en laissant au nuage beaucoup de hauteur
  // devant lui par rapport à son propre diamètre. Une charge deux fois plus
  // grosse donne un chou-fleur trapu qui touche le plafond avant d'avoir formé
  // un pied — la forme d'une explosion ordinaire, pas d'une bombe.
  explosionRadius: 0.042,
  explosionFuel: 64,
  explosionHeight: 0.07,
  // Arrachement LONG et doux : c'est lui qui fait un pied CONTINU. En une seule
  // bouffée, la colonne est un paquet fini que la montée étire puis rompt.
  dustRate: 11,
  dustTime: 5.0,
  dustRadius: 0.22,
  // AMORCE ÉNORME : c'est elle qui fait la différence entre un gros feu et une
  // bombe. Avec le plafond de chaleur relevé (voir TUNE_MUSHROOM), elle porte le
  // cœur loin au-dessus du domaine des flammes, là où l'émission part en loi de
  // puissance et sature au blanc.
  explosionSpark: 190,
};

// PRESETS du panneau : appliqués À CHAUD par-dessus les défauts (déterministe
// quel que soit l'historique de clics). Exposition/pas de marche non touchés.
const PRESETS: readonly {
  label: string;
  values: Partial<Sim3Tuning>;
  boom?: BoomTune;
}[] = [
  { label: '↺ défaut', values: {} },
  { label: '🕯 bougie', values: TUNE_CANDLE },
  { label: '🔥 fournaise', values: TUNE_FURNACE },
  { label: '🌫 fumée épaisse', values: TUNE_SMOKE },
  { label: '🌬 vent', values: TUNE_WIND },
  { label: '🍄 champignon', values: TUNE_MUSHROOM, boom: BOOM_MUSHROOM },
];

/**
 * Mode DÉMO (touche D ou ?demo) : chorégraphie scriptée du moteur — le showcase.
 * SIX actes, ~107 s : bougie → trois matières → le souffle qui embrase la nappe
 * de carburant → fournaise (braises, boule en lemniscate) → BOMBARDEMENT monté
 * en CUTS, trois charges SUPERPOSÉES par plan (les charges multiples : une
 * boule fraîche vit à côté du nuage qui noircit ; le montage garde sa raison
 * d'être — dans une boîte close la fumée s'accumule, un cut offre à chaque
 * VOLÉE son air pur) → dehors : BARRAGE D'ARTILLERIE contre l'horizon, puis le
 * champignon en bouquet final. Le rendu est VERROUILLÉ à 100 % pendant le
 * spectacle (l'adaptation floutait précisément les détonations).
 * Les réglages de l'utilisateur sont SNAPSHOTÉS au lancement et RESTAURÉS à la
 * sortie (D) — détonation, boule et retours visuels compris.
 * Dev : `?demo=54` cale l'horloge au lancement — les actes antérieurs
 * s'appliquent en un tick (l'état converge), pour travailler un acte sans
 * attendre les précédents.
 */
class DemoDriver {
  private t = 0;
  private readonly fired = new Set<number>();
  /** Cible caméra de l'acte courant (vitesse d'orbite, élévation, distance). */
  private camT = { speed: 0.06, elevation: 0.1, radius: 1.55 };
  private saved: {
    params: Sim3Tuning;
    embersOn: boolean;
    glowStrength: number;
    exposure: number;
    boom: BoomTune;
    sphereActive: boolean;
    feedback: boolean;
  } | null = null;

  constructor(
    private readonly input: Frame3DInput,
    private readonly sim: FluidSim3D,
    private readonly toast: (text: string) => void,
  ) {}

  private at(mark: number, action: () => void): void {
    if (this.t >= mark && !this.fired.has(mark)) {
      this.fired.add(mark);
      action();
    }
  }

  /** Réglages d'acte : les écarts par-dessus les défauts (comme les presets). */
  private apply(values: Partial<Sim3Tuning>): void {
    Object.assign(this.input.params, defaultTuning3(), values);
  }

  private act(cam: { speed: number; elevation: number; radius: number }): void {
    this.camT = cam;
  }

  /** Tir SCRIPTÉ : la charge est posée en clair (fractions de boîte, hauteur
   *  comprise) via sim.explodeAt — le réalisateur choisit ses marques, il ne
   *  vise pas. L'ancienne visée au pointeur retombait au CENTRE dès que le plan
   *  de charge était bas et la caméra rasante (piège documenté) : chaque tir de
   *  la démo était en réalité un tir centré. */
  private boomAt(nx: number, ny: number, nz: number): void {
    this.sim.explodeAt(nx, ny, nz, this.input);
  }

  /** Lancement : snapshot des réglages du spectateur, scène vierge, acte I.
   *  `at` cale l'horloge (dev, `?demo=<s>`). */
  start(at = 0): void {
    this.saved = {
      params: { ...this.input.params },
      embersOn: this.input.embersOn,
      glowStrength: this.input.glowStrength,
      exposure: this.input.exposure,
      // L'acte V réécrit les réglages de DÉTONATION et retire la boule : ils
      // font partie de ce que la sortie doit rendre intact.
      boom: {
        explosionRadius: this.input.explosionRadius,
        explosionFuel: this.input.explosionFuel,
        explosionHeight: this.input.explosionHeight,
        dustRate: this.input.dustRate,
        dustTime: this.input.dustTime,
        dustRadius: this.input.dustRadius,
        explosionSpark: this.input.explosionSpark,
      },
      sphereActive: this.input.sphereActive,
      feedback: this.input.feedback,
    };
    this.t = at;
    this.fired.clear();
    this.input.reset = true;
  }

  /** Sortie (D) : la main revient au spectateur avec SES réglages. */
  stop(): void {
    if (this.saved !== null) {
      Object.assign(this.input.params, this.saved.params);
      this.input.embersOn = this.saved.embersOn;
      this.input.glowStrength = this.saved.glowStrength;
      this.input.exposure = this.saved.exposure;
      Object.assign(this.input, this.saved.boom);
      this.input.sphereActive = this.saved.sphereActive;
      this.input.feedback = this.saved.feedback;
      this.saved = null;
    }
    this.input.blow.active = false;
  }

  tick(dt: number): void {
    this.t += dt;
    const input = this.input;

    // Caméra : orbite continue, élévation/distance LISSÉES vers la cible de
    // l'acte + une respiration lente par-dessus.
    const k = 1 - Math.exp(-1.1 * dt);
    input.cam.azimuth += dt * this.camT.speed;
    input.cam.elevation +=
      (this.camT.elevation + 0.05 * Math.sin(this.t * 0.13) - input.cam.elevation) * k;
    input.cam.radius +=
      (this.camT.radius + 0.13 * Math.sin(this.t * 0.09) - input.cam.radius) * k;

    // ACTE I — une bougie dans le noir (caméra proche, flamme intime).
    this.at(0.02, () => {
      this.apply(TUNE_CANDLE);
      input.embersOn = false;
      input.glowStrength = 1.8;
      // La boule revient : les actes V et VI la retirent, et sans cette ligne
      // le DEUXIÈME tour de boucle jouait la lemniscate de l'acte IV avec une
      // boule invisible et sans effet.
      input.sphereActive = true;
      this.act({ speed: 0.06, elevation: 0.1, radius: 1.55 });
      this.toast('Acte I — une bougie dans le noir');
    });
    // ACTE II — les trois matières : la flamme prend ses aises, l'encre lourde
    // retombe, la vapeur de carburant nappe le sol.
    this.at(9, () => {
      this.apply({});
      this.act({ speed: 0.09, elevation: 0.22, radius: 1.95 });
      this.toast('Acte II — la flamme prend ses aises');
    });
    this.at(12, () => {
      this.sim.addEmitterAt(0.27, 0.08, 0.5, 1);
      this.toast("fontaine d'encre magenta — lourde, elle retombe en refroidissant");
    });
    this.at(15, () => {
      this.sim.addEmitterAt(0.62, 0.08, 0.35, 2);
      this.act({ speed: 0.08, elevation: 0.13, radius: 1.85 });
      this.toast('la vapeur de carburant coule et nappe le sol');
    });
    // ACTE III — le souffle pousse la nappe dans la flamme : embrasement.
    this.at(25, () => this.toast('Acte III — un souffle pousse la nappe vers la flamme…'));
    if (this.t > 25 && this.t < 29.5) {
      input.blow.active = true;
      input.blow.ndcX = 0.2;
      input.blow.ndcY = -0.4;
      input.blow.moveX += -dt * 1.4;
    }
    this.at(29.5, () => (input.blow.active = false));
    // ACTE IV — embrasement : fournaise, braises, lueur poussée, caméra ample.
    // Dissipation remontée : la brume des actes précédents se consume, le
    // brasier se détache au lieu de se noyer dans la boîte enfumée.
    this.at(33, () => {
      this.apply({ ...TUNE_FURNACE, inkDissipation: 0.4 });
      input.embersOn = true;
      input.glowStrength = 2.3;
      this.act({ speed: 0.12, elevation: 0.3, radius: 2.3 });
      this.toast('Acte IV — FOURNAISE : braises au vent, la fumée boit la lumière');
    });
    this.at(38, () => this.toast('la boule plonge dans le brasier'));
    if (this.t > 38 && this.t < 52) {
      const u = (this.t - 38) * 0.33;
      this.sim.driveSphere(
        0.5 + 0.3 * Math.sin(u),
        0.4 + 0.17 * Math.sin(u * 0.71),
        0.5 + 0.16 * Math.sin(2.0 * u),
        dt,
      );
    }
    // ACTE V — BOMBARDEMENT monté en CUTS : deux plans, TROIS charges
    // SUPERPOSÉES par plan. Les charges multiples changent la grammaire — une
    // boule fraîche vit à côté du nuage qui noircit, là où le slot unique
    // amputait la charge en cours à chaque tir. Le montage garde sa raison
    // d'être : dans une boîte close, une rafale en continu s'enterre dans sa
    // propre fumée (mesuré — la boule de feu est de la fumée qui rougeoie :
    // dissiper assez vite pour nettoyer la scène éteint la boule elle-même).
    // Un cut = reset + saut d'azimut, et chaque VOLÉE vit dans l'air pur.
    // Charges au réglage « spectacle » : calibre réduit (trois par boîte),
    // suie réduite de moitié, amorce incandescente poussée, dissipation
    // remontée à 0,3 — assez pour consumer les nuages entre deux charges,
    // pas assez pour éteindre les boules.
    const cut = (label?: string): void => {
      // Suie à 10 (contre 6 au montage à un tir) : le contraste de l'acte est
      // là — une boule d'or FRAÎCHE devant le nuage NOIRCI du tir précédent.
      this.apply({ emitHeat: 0, emitInkRate: 0, sootYield: 10, inkDissipation: 0.35 });
      // Calibre RÉDUIT : trois charges au calibre du tir isolé remplissaient la
      // boîte en mur de fumée avant la fin de la volée (capturé) — le volume
      // injecté par volée doit rester celui d'UN tir d'avant. Amorce 55 (pas
      // 90) : à trois par boîte, avec la lueur, l'amorce de spectacle brûlait
      // chaque boule jeune en ampoule blanche sans texture.
      Object.assign(input, BOOM_DEFAULT, {
        explosionRadius: 0.06,
        explosionFuel: 48,
        dustRate: 1.5,
        dustTime: 0.8,
        explosionSpark: 55,
      });
      input.reset = true;
      input.embersOn = true;
      input.glowStrength = 1.6;
      input.cam.azimuth += 2.1;
      // La boule se tiendrait en plein dans les tirs — retirée pour l'acte
      // (le snapshot la rend à la sortie, l'acte I au tour suivant).
      input.sphereActive = false;
      if (label !== undefined) {
        this.toast(label);
      }
      // Repères coupés APRÈS le toast (le narrateur passe par feedback) : le
      // gizmo de l'émetteur pilote trônait au milieu des boules de feu.
      // L'acte VI les laisse coupés ; la boucle et la sortie les rendent.
      input.feedback = false;
    };
    this.at(54, () => {
      cut('Acte V — BOMBARDEMENT : les charges se répondent');
      this.act({ speed: 0.05, elevation: 0.24, radius: 2.35 });
    });
    this.at(54.6, () => this.boomAt(0.3, input.explosionHeight, 0.62));
    this.at(56.1, () => this.boomAt(0.68, input.explosionHeight, 0.48));
    this.at(57.6, () => this.boomAt(0.46, input.explosionHeight, 0.28));
    this.at(62.5, () => cut());
    this.at(63.1, () => this.boomAt(0.62, input.explosionHeight, 0.66));
    this.at(64.6, () => this.boomAt(0.34, input.explosionHeight, 0.4));
    this.at(66.1, () => this.boomAt(0.55, input.explosionHeight, 0.24));
    // ACTE VI — DEHORS, en deux temps. D'abord le BARRAGE D'ARTILLERIE : des
    // impacts AU SOL contre l'horizon, chacun avec son flash, sa colonne de
    // fumée sale et sa poussière arrachée. (Un feu d'artifice aérien a été
    // essayé ici — verdict de Renaud : « pas top top, pas beaucoup de couleur,
    // un peu plat ». Juste : la lumière des explosions est du corps noir,
    // or/orange/blanc — pas de couleurs pyrotechniques sans charges d'encre.
    // Le barrage joue au contraire sur ce que le moteur fait le mieux.)
    // La CADENCE est IRRÉGULIÈRE — deux coups rapprochés, un temps mort : la
    // régularité d'un métronome tue l'impression d'artillerie. La poussière a
    // une fenêtre courte mais réelle (2,2 s) : chaque impact nourrit sa
    // colonne, c'est elle qui fait « obus » et pas « pétard ».
    this.at(72, () => {
      // Le MÊME MOTEUR DE RÉGLAGES QUE LE CHAMPIGNON — verdict de Renaud sur
      // la première version : « de grosses explosions grasses » à côté d'un
      // bouquet « fin ». La finesse du champignon ne vient pas de sa taille :
      // temps ralenti (la turbulence se développe sous l'œil), dissipations
      // quasi nulles (les filaments survivent), vorticité forte, suie sombre
      // qui dessine. Ma dissipation à 0,42 — mise là pour vider la boîte —
      // FONDAIT la structure : c'était ça, le gras. On repart de
      // TUNE_MUSHROOM et on ne change que ce que SIX impacts imposent :
      this.apply({
        ...TUNE_MUSHROOM,
        // … une fumée qui s'efface quand même (0,12 contre 0,02 — six nuages
        // coexistent, le cut fait le reste),
        inkDissipation: 0.12,
        sootYield: 20,
        // … un temps un peu moins ralenti (le barrage doit garder du nerf),
        timeScale: 0.6,
        // … et les éponges MUSCLÉES du barrage : sept impacts serrés avaient
        // dessiné un MONOLITHE aux arêtes de la boîte (capturé). Parois
        // franches, plafond large et doux, inversion un peu plus haute que
        // celle du champignon (les colonnes s'arrêtent aux deux tiers).
        openBand: 0.1,
        openStrength: 30,
        openCeilBand: 0.1,
        openCeilStrength: 12,
        stratBase: 0.24,
      });
      input.params.outdoor = true;
      // Charges au CALIBRE du champignon (petites — c'est aussi ça, la
      // finesse : des détails petits devant le domaine), amorce incandescente,
      // poussière réelle mais à fenêtre courte (2,2 s contre 5).
      Object.assign(input, BOOM_MUSHROOM, {
        explosionFuel: 48,
        explosionSpark: 130,
        dustRate: 8,
        dustTime: 2.2,
        dustRadius: 0.16,
      });
      input.reset = true;
      input.sphereActive = false;
      input.embersOn = true;
      input.glowStrength = 1.8;
      this.act({ speed: 0.05, elevation: 0.12, radius: 2.85 });
      // Le narrateur repasse un instant : l'acte V a coupé les retours.
      input.feedback = true;
      this.toast("Acte VI — dehors : barrage d'artillerie, puis le bouquet (D : reprendre la main)");
      // Comme le bouton « extérieur » : la boîte de verre et les gizmos n'ont
      // plus de sens face à l'horizon. Coupé APRÈS le toast (le narrateur passe
      // par feedback) ; rendu à la boucle et à la sortie.
      input.feedback = false;
    });
    // Impacts GROUPÉS au centre de l'empreinte (0,38-0,65) : dispersés sur
    // tout le sol, l'union des colonnes épousait la boîte — parois verticales
    // taillées par l'éponge (capturé). Un barrage crédible dans ce domaine est
    // un barrage sur UNE position, pas sur toute la plaine ; la profondeur
    // varie pour la parallaxe. Et DEUX SALVES coupées d'un cut (reset + saut
    // d'azimut — dehors, un autre plan du même champ de bataille) : même
    // groupés, cinq impacts d'affilée finissaient en dalle aux arêtes de la
    // boîte ; trois par ciel propre, jamais (capturé aussi). La leçon de
    // l'acte V vaut dehors : le montage est la seule façon d'offrir à chaque
    // salve son air pur.
    this.at(73.0, () => this.boomAt(0.42, input.explosionHeight, 0.6));
    this.at(74.1, () => this.boomAt(0.62, input.explosionHeight, 0.38));
    this.at(75.4, () => this.boomAt(0.5, input.explosionHeight, 0.55));
    this.at(77.2, () => {
      input.reset = true;
      input.cam.azimuth += 2.1;
    });
    this.at(77.8, () => this.boomAt(0.6, input.explosionHeight, 0.6));
    this.at(78.9, () => this.boomAt(0.4, input.explosionHeight, 0.42));
    this.at(80.2, () => this.boomAt(0.55, input.explosionHeight, 0.5));
    // LE BOUQUET FINAL — le champignon, même recette que le bouton « 🍄 »
    // (scène nettoyée par le reset : le ciel se vide des fumées du feu
    // d'artifice), en prise de vue extérieure : dans la boîte de verre, le
    // nuage à 256³ n'est qu'une bouffée boueuse ; contre le ciel et l'horizon,
    // la même simulation devient un champignon lointain — c'est l'échelle qui
    // fait le spectacle, pas la matière (vérifié A/B, captures EXT256-* contre
    // UNCLIC256-*). La caméra reste basse, presque au ras de l'horizon, puis
    // s'élève doucement avec le chapeau. (Pas de toast ici : le narrateur est
    // coupé avec les retours visuels depuis l'entrée de l'acte.)
    this.at(82, () => {
      this.apply(TUNE_MUSHROOM);
      input.params.outdoor = true;
      Object.assign(input, BOOM_MUSHROOM);
      input.reset = true;
      input.embersOn = false;
      input.glowStrength = 1.8;
      this.act({ speed: 0.045, elevation: 0.14, radius: 2.7 });
    });
    this.at(83.2, () => this.boomAt(0.5, input.explosionHeight, 0.5));
    this.at(93, () => this.act({ speed: 0.06, elevation: 0.22, radius: 2.65 }));
    // Boucle — l'acte I ramène l'atelier (apply() remet outdoor à faux). Le
    // chapeau se disperse vers 103 : ne pas laisser dix secondes de ciel vide.
    if (this.t > 107) {
      input.feedback = this.saved?.feedback ?? true;
      input.reset = true;
      this.t = 0;
      this.fired.clear();
    }
  }

}

async function boot(): Promise<void> {
  const urlParams = new URLSearchParams(location.search);
  const selftest = urlParams.has('selftest');
  let demoOn = urlParams.has('demo');
  // Résolution à la demande : ?grid=128|256|320 (défaut 256 ; 320 = le plafond
  // mesuré de la machine de référence, toujours à 60 FPS).
  // `?tall=2|3` : domaine plus HAUT que large, à cellules toujours cubiques. Le
  // défaut (1) rend exactement le cube. Une boîte deux fois plus haute coûte
  // deux fois plus de cellules — d'où le tableau de VRAM du sélecteur.
  setGrid3(Number(urlParams.get('grid') ?? 256), Number(urlParams.get('tall') ?? 1));
  const canvas = document.getElementById('canvas3d') as HTMLCanvasElement;
  const hud = document.getElementById('hud3d') as HTMLDivElement;

  // Retours visuels débrayables : toasts d'événements + fuseau du souffle (core).
  const toast = (text: string): void => {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = text;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 1700);
  };

  const { device, format } = await acquireDevice();
  // Garde-fou hautes résolutions : Dawn valide ses staging internes (zéro-init
  // des textures 3D) contre maxBufferSize — s'il est trop bas pour la plus
  // grosse texture (rgba16float pleine grille), chaque submit échouerait EN
  // SILENCE (écran noir, HUD vivant). Erreur claire plutôt que mystère.
  const biggestTexture = GRID3 * GRID3 * GRID3 * 8;
  if (device.limits.maxBufferSize < biggestTexture) {
    throw new Error(
      `La résolution ${GRID3}³ dépasse les capacités de ce GPU ` +
        `(maxBufferSize ${Math.round(device.limits.maxBufferSize / 2 ** 20)} Mio < ` +
        `${Math.round(biggestTexture / 2 ** 20)} Mio requis). Choisir une grille plus petite : ` +
        `?grid=${GRID3_CHOICES.filter((n) => n * n * n * 8 <= device.limits.maxBufferSize).join('|')}`,
    );
  }
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      showFatalError(`Le device GPU a été perdu : ${info.message}`);
    }
  });

  const context = canvas.getContext('webgpu');
  if (context === null) {
    throw new Error('Impossible de créer un contexte WebGPU sur le canvas.');
  }
  context.configure({ device, format, alphaMode: 'opaque' });

  // RÉSOLUTION DYNAMIQUE : le ray-marching est payé par pixel — l'échelle de rendu
  // s'ajuste au FPS (hystérésis), imperceptible en mouvement, gros gain en charge.
  let renderScale = 1;
  // Résolution de rendu VERROUILLÉE : l'adaptation automatique ci-dessous baisse
  // la résolution quand la scène s'alourdit — c'est le bon réflexe pour tenir
  // 60 FPS, et exactement le mauvais quand on juge une image. Elle se déclenche
  // précisément au moment le plus dense, donc au moment où l'on regarde, et le
  // flou qu'elle produit se lit à tort comme un défaut de SIMULATION (constaté :
  // un champignon « diffus » à 9 s l'était pour moitié parce que le rendu était
  // tombé à 60 %). Verrou explicite, état affiché au HUD.
  let lockScale = urlParams.has('lock');
  const resize = (): void => {
    const scale = Math.min(window.devicePixelRatio || 1, 2) * renderScale;
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
  };
  resize();
  window.addEventListener('resize', resize);

  // ?profile : timestamps GPU par passe, affichés au HUD — l'instrument du
  // chantier de perf (voir sim3d). Coût nul sans le flag.
  const sim = await FluidSim3D.create(device, format, urlParams.has('profile'));

  const input: Frame3DInput = {
    dt: 0,
    paused: false,
    reset: false,
    multigrid: SIM3_DEFAULTS.multigrid,
    // Le défaut suit la RÉSOLUTION : un seul cycle laisse la pression
    // sous-convergée au-delà de 256³, et le panache s'y couche au sol.
    vcycles: vcyclesFor(GRID3),
    jacobiIterations: SIM3_DEFAULTS.jacobiIterations,
    params: defaultTuning3(),
    emitInk: 0,
    cam: {
      azimuth: SIM3_DEFAULTS.camAzimuth,
      elevation: SIM3_DEFAULTS.camElevation,
      radius: SIM3_DEFAULTS.camRadius,
    },
    pointer: { ndcX: 0, ndcY: 0 },
    blow: { active: false, ndcX: 0, ndcY: 0, moveX: 0, moveY: 0 },
    grab: { active: false },
    addEmitter: false,
    removeEmitter: false,
    addField: false,
    removeField: false,
    fieldType: 0,
    fieldStrength: SIM3_DEFAULTS.fieldVortexStrength,
    fieldRadius: SIM3_DEFAULTS.fieldRadius,
    explode: false,
    explosionRadius: SIM3_DEFAULTS.explosionRadius,
    explosionFuel: SIM3_DEFAULTS.explosionFuel,
    explosionHeight: SIM3_DEFAULTS.explosionHeight,
    dustRate: SIM3_DEFAULTS.dustRate,
    dustTime: SIM3_DEFAULTS.dustTime,
    dustRadius: SIM3_DEFAULTS.dustRadius,
    explosionSpark: SIM3_DEFAULTS.explosionSpark,
    sphereActive: true,
    feedback: true,
    exposure: SIM3_DEFAULTS.exposure,
    raymarchSteps: SIM3_DEFAULTS.raymarchSteps,
    glowStrength: SIM3_DEFAULTS.glowStrength,
    bloomStrength: SIM3_DEFAULTS.bloomStrength,
    embersOn: SIM3_DEFAULTS.embersOn,
    emberStrength: SIM3_DEFAULTS.emberStrength,
  };
  if (selftest) {
    (window as unknown as Record<string, unknown>)['__frame3d'] = input;
    (window as unknown as Record<string, unknown>)['__sim3d'] = sim;
    // Les presets pour le harnais de test : sans ce hook, un script qui veut
    // vérifier « le champignon » recopierait leurs valeurs et vérifierait donc
    // sa propre copie plutôt que ce que voit l'utilisateur.
    (window as unknown as Record<string, unknown>)['__presets3d'] = PRESETS;
    // Les DÉFAUTS eux-mêmes : un preset s'applique par-dessus une table remise à
    // neuf (`Object.assign(p, defaultTuning3(), preset.values)`), et sans ce hook
    // un harnais qui veut comparer « preset » et « défaut » recopierait sa propre
    // idée des défauts — exactement le piège ci-dessus, un cran plus bas.
    (window as unknown as Record<string, unknown>)['__defaults3d'] = defaultTuning3;
    // Forme de la grille, pour le harnais : sans ça, un problème de domaine non
    // cubique ne se diagnostique qu'à l'œil.
    (window as unknown as Record<string, unknown>)['__grid3'] = {
      x: GRID3,
      y: GRID3Y,
      height: HEIGHT3,
    };
  }

  // Caméra orbitale : glisser (gauche) pour tourner — SAUF si le clic attrape un
  // objet (flamme ou sphère : hitTest), auquel cas l'objet suit le pointeur.
  // Souffle : glisser au clic DROIT — le geste pousse le fluide le long du rayon.
  let dragging = false;
  let blowing = false;
  let lastX = 0;
  let lastY = 0;
  const updatePointer = (e: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect();
    input.pointer.ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    input.pointer.ndcY = 1 - ((e.clientY - rect.top) / rect.height) * 2;
  };
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    updatePointer(e);
    if (e.button === 2) {
      blowing = true;
    } else if (sim.hitTest(input.pointer.ndcX, input.pointer.ndcY)) {
      input.grab.active = true;
    } else {
      dragging = true;
    }
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.classList.add('dragging');
    try {
      canvas.setPointerCapture(e.pointerId);
    } catch {
      // Certains environnements tactiles refusent la capture : sans gravité.
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    updatePointer(e);
    if (blowing) {
      const rect = canvas.getBoundingClientRect();
      const b = input.blow;
      b.active = true;
      b.ndcX = input.pointer.ndcX;
      b.ndcY = input.pointer.ndcY;
      b.moveX += ((e.clientX - lastX) / rect.width) * 2;
      b.moveY += -((e.clientY - lastY) / rect.height) * 2;
      lastX = e.clientX;
      lastY = e.clientY;
      return;
    }
    if (!dragging) {
      return;
    }
    input.cam.azimuth += (e.clientX - lastX) * 0.008;
    input.cam.elevation = Math.min(
      Math.max(input.cam.elevation + (e.clientY - lastY) * 0.006, -1.25),
      1.45,
    );
    lastX = e.clientX;
    lastY = e.clientY;
  });
  const endDrag = (): void => {
    dragging = false;
    blowing = false;
    input.blow.active = false;
    input.grab.active = false;
    canvas.classList.remove('dragging');
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      input.cam.radius = Math.min(Math.max(input.cam.radius * Math.exp(e.deltaY * 0.001), 0.9), 4.5);
    },
    { passive: false },
  );
  // Export OpenVDB : readback ponctuel + encodage TS pur + téléchargement.
  // Grilles « density » (fumée) et « temperature » (chaleur), boîte de taille 1
  // centrée sur l'origine, Y simulation mappé sur +Z Blender (la fumée monte).
  let exporting = false;
  const exportVdbFile = async (): Promise<void> => {
    if (exporting) {
      return;
    }
    exporting = true;
    try {
      // Readback plafonné par maxBufferSize (demandé au max de l'adapter dans
      // gpu.ts — 2 Gio sur la machine de référence : 512³ passe tout juste).
      const { density, heat } = await sim.exportVolume();
      const data = encodeVdb(
        [
          { name: 'density', values: density },
          { name: 'temperature', values: heat },
        ],
        GRID3,
        1 / GRID3,
      );
      const url = URL.createObjectURL(new Blob([data], { type: 'application/octet-stream' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = `liquidvm-${Date.now()}.vdb`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      toast(`export impossible à ${GRID3}³ (limite mémoire GPU) — réduire la résolution`);
      console.warn('export VDB:', err);
    } finally {
      exporting = false;
    }
  };
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      input.paused = !input.paused;
    } else if (e.code === 'KeyR') {
      input.reset = true;
    } else if (e.code === 'KeyE') {
      void exportVdbFile();
    } else if (e.code === 'Digit1' || e.code === 'Numpad1') {
      input.emitInk = 0;
    } else if (e.code === 'Digit2' || e.code === 'Numpad2') {
      input.emitInk = 1;
    } else if (e.code === 'Digit3' || e.code === 'Numpad3') {
      input.emitInk = 2;
    }
    // Lettres via e.key : indépendant de la disposition clavier (AZERTY…).
    const k = e.key.toLowerCase();
    const say = (text: string): void => {
      if (input.feedback) {
        toast(text);
      }
    };
    if (k === 'a') {
      input.addEmitter = true;
      say(`➕ émetteur : ${INK_NAMES[input.emitInk] ?? '?'}`);
    } else if (k === 'x') {
      // Une seule touche « supprimer », routée vers la famille de l'objet
      // désigné : c'est ce qu'on VOIT en surbrillance qui disparaît.
      if (sim.selectedIsField) {
        input.removeField = true;
        say('➖ champ de force');
      } else if (sim.emitterCount > 1) {
        input.removeEmitter = true;
        say('➖ émetteur');
      } else {
        say('la flamme pilote ne se retire pas');
      }
    } else if (e.code === 'Escape') {
      sim.deselect();
      say('rien de sélectionné');
    } else if (k === 'm') {
      runMushroom();
    } else if (k === 'l') {
      lockScale = !lockScale;
      say(lockScale ? 'rendu verrouillé à 100 %' : 'rendu adaptatif (60-100 %)');
    } else if (k === 'g') {
      input.explode = true;
      say('💥 détonation');
    } else if (k === 'o') {
      input.sphereActive = !input.sphereActive;
      say(input.sphereActive ? 'boule : présente' : 'boule : retirée');
    } else if (k === 'b') {
      input.embersOn = !input.embersOn;
      say(input.embersOn ? '✨ braises : actives' : 'braises : coupées');
    } else if (k === 'd') {
      toggleDemo();
      say(demoOn ? 'DÉMO — D pour reprendre la main' : 'à toi de jouer');
    } else if (k === 'f') {
      input.feedback = !input.feedback;
      toast(input.feedback ? 'retours visuels : actifs' : 'retours visuels : coupés');
    }
    if (e.code === 'Digit1' || e.code === 'Digit2' || e.code === 'Digit3') {
      say(`encre : ${INK_NAMES[input.emitInk] ?? '?'}`);
    }
  });

  // SCÉNARIO EN UN CLIC. Un preset seul ne suffit pas à faire découvrir un
  // scénario : l'utilisateur clique le bouton qui ressemble à ce qu'il veut —
  // ici « boum » — et il obtient la détonation ordinaire, pas le champignon. Le
  // preset était donc caché derrière une étape que personne ne devine.
  // La détonation est DIFFÉRÉE de quelques frames : le nettoyage de la scène
  // occupe la frame courante, et tirer dedans donnerait une boule amputée.
  let boomIn = 0;
  const runMushroom = (): void => {
    const preset = PRESETS.find((x) => x.boom !== undefined);
    if (preset === undefined) {
      return;
    }
    Object.assign(p, defaultTuning3(), preset.values);
    Object.assign(input, preset.boom ?? BOOM_DEFAULT);
    input.reset = true;
    input.sphereActive = false; // elle se tiendrait en plein dans le souffle
    panel.refresh();
    boomIn = 4;
    if (input.feedback) {
      toast('🍄 champignon : scène nettoyée, détonation au sol…');
    }
  };

  const demoDriver = new DemoDriver(input, sim, (text) => {
    if (input.feedback) {
      toast(text);
    }
  });
  if (selftest) {
    // L'horloge de la démo, pour le harnais : capturer « à t mural » dérive de
    // plusieurs secondes (boot variable) — viser un instant du SCÉNARIO exige
    // de lire son horloge à elle.
    (window as unknown as Record<string, unknown>)['__demo3d'] = demoDriver;
  }

  // INTERFACE déclarative : panneau à sections (Tab) + barre d'outils tactile.
  // Ajouter un réglage futur = une entrée ici, rien d'autre.
  const p = input.params;
  const inkCss = INK_COLORS.map((c) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`);
  // Le mode démo scénarise TOUT (presets, braises, lueur, caméra) : snapshot
  // des réglages au lancement, restauration à la sortie.
  let lockBeforeDemo = false;
  const toggleDemo = (): void => {
    demoOn = !demoOn;
    if (demoOn) {
      // RENDU VERROUILLÉ À 100 % pendant le spectacle : l'adaptation baisse la
      // résolution exactement quand la scène est la plus lourde — c'est-à-dire
      // pendant les détonations, le moment qu'on est venu regarder. Les FPS
      // peuvent plonger un peu ; le flou, lui, ruinait le plan.
      lockBeforeDemo = lockScale;
      lockScale = true;
      demoDriver.start();
    } else {
      lockScale = lockBeforeDemo;
      demoDriver.stop();
    }
  };
  if (demoOn) {
    // ?demo au chargement : même verrou de rendu que par la touche D.
    // `?demo=54` cale l'horloge du scénario (dev — travailler un acte sans
    // attendre les précédents ; les actes antérieurs s'appliquent en un tick).
    lockBeforeDemo = lockScale;
    lockScale = true;
    demoDriver.start(Number(urlParams.get('demo')) || 0);
  }
  const panel = new Panel3D(document.body, [
    {
      title: 'presets',
      buttons: PRESETS.map((preset) => ({
        label: preset.label,
        action: (): void => {
          Object.assign(p, defaultTuning3(), preset.values);
          Object.assign(input, preset.boom ?? BOOM_DEFAULT);
          // Un preset de DÉTONATION est un scénario, pas une ambiance : il part
          // d'un air PROPRE. Sans ce nettoyage, on tire dans ce que la scène
          // précédente a laissé et on ne voit que le mélange des deux.
          const scenario = preset.boom !== undefined;
          if (scenario) {
            input.reset = true;
          }
          panel.refresh();
          if (input.feedback) {
            toast(
              scenario
                ? `${preset.label} — scène nettoyée, flamme coupée. G pour détoner${
                    input.sphereActive ? ' · O retire la boule, elle est dans le souffle' : ''
                  }`
                : `preset : ${preset.label}`,
            );
          }
        },
      })),
    },
    {
      title: 'simulation',
      sliders: [
        { label: 'vitesse du temps', min: 0, max: 1.5, step: 0.05, get: () => p.timeScale, set: (x) => (p.timeScale = x), format: (x) => `×${x.toFixed(2)}` },
        { label: 'poussée thermique', min: 0, max: 400, step: 5, get: () => p.buoyancy, set: (x) => (p.buoyancy = x), format: (x) => x.toFixed(0) },
        { label: 'vorticité', min: 0, max: 30, step: 0.5, get: () => p.vorticityStrength, set: (x) => (p.vorticityStrength = x), format: (x) => x.toFixed(1) },
        { label: 'viscosité', min: 0, max: 0.2, step: 0.005, get: () => p.velocityDissipation, set: (x) => (p.velocityDissipation = x), format: (x) => x.toFixed(3) },
        { label: 'vent', min: 0, max: 250, step: 5, get: () => p.windStrength, set: (x) => (p.windStrength = x), format: (x) => (x === 0 ? 'aucun' : x.toFixed(0)) },
        { label: 'vent : cap', min: 0, max: 360, step: 5, get: () => p.windHeading, set: (x) => (p.windHeading = x), format: (x) => `${x.toFixed(0)}°` },
        { label: 'vent : oscillation', min: 0, max: 90, step: 5, get: () => p.windSwing, set: (x) => (p.windSwing = x), format: (x) => (x === 0 ? 'constant' : `±${x.toFixed(0)}°`) },
        { label: 'vent : période', min: 1, max: 30, step: 1, get: () => p.windPeriod, set: (x) => (p.windPeriod = x), format: (x) => `${x.toFixed(0)} s` },
      ],
      checks: [
        { label: 'pression multigrid (sinon Jacobi)', get: () => input.multigrid, set: (v) => (input.multigrid = v) },
      ],
    },
    {
      title: 'matières',
      sliders: [
        { label: 'débit chaleur', min: 0, max: 8, step: 0.1, get: () => p.emitHeat, set: (x) => (p.emitHeat = x) },
        { label: 'débit matière', min: 0, max: 6, step: 0.1, get: () => p.emitInkRate, set: (x) => (p.emitInkRate = x) },
        { label: 'refroidissement', min: 0.2, max: 2, step: 0.05, get: () => p.heatCooling, set: (x) => (p.heatCooling = x) },
        { label: 'dissipation', min: 0, max: 0.6, step: 0.01, get: () => p.inkDissipation, set: (x) => (p.inkDissipation = x) },
      ],
    },
    {
      title: 'combustion',
      sliders: [
        { label: 'taux de réaction', min: 0, max: 8, step: 0.1, get: () => p.burnRate, set: (x) => (p.burnRate = x) },
        { label: 'chaleur dégagée', min: 0, max: 1.5, step: 0.05, get: () => p.heatYield, set: (x) => (p.heatYield = x) },
        { label: 'expansion', min: 0, max: 60, step: 1, get: () => p.expansion, set: (x) => (p.expansion = x), format: (x) => x.toFixed(0) },
        { label: 'retour d’oxygène', min: 0, max: 0.08, step: 0.002, get: () => p.oxygenRecover, set: (x) => (p.oxygenRecover = x), format: (x) => x.toFixed(3) },
      ],
    },
    {
      title: 'interaction & rendu',
      sliders: [
        { label: 'force du souffle', min: 0, max: 800, step: 10, get: () => p.blowForce, set: (x) => (p.blowForce = x), format: (x) => x.toFixed(0) },
        { label: 'exposition', min: 0.4, max: 3, step: 0.05, get: () => input.exposure, set: (x) => (input.exposure = x) },
        { label: 'lueur du feu', min: 0, max: 3, step: 0.05, get: () => input.glowStrength, set: (x) => (input.glowStrength = x) },
        { label: 'bloom', min: 0, max: 1, step: 0.05, get: () => input.bloomStrength, set: (x) => (input.bloomStrength = x) },
        { label: 'braises : débit', min: 0, max: 1, step: 0.05, get: () => input.emberStrength, set: (x) => (input.emberStrength = x) },
        // Le maximum doit rester ≤ au plafond de boucle de raymarch.wgsl (1024).
        { label: 'pas de marche', min: 64, max: 1024, step: 16, get: () => input.raymarchSteps, set: (x) => (input.raymarchSteps = x), format: (x) => x.toFixed(0) },
      ],
      checks: [
        { label: 'boule-obstacle (O)', get: () => input.sphereActive, set: (v) => (input.sphereActive = v) },
        { label: 'braises (B)', get: () => input.embersOn, set: (v) => (input.embersOn = v) },
        { label: 'retours visuels (F)', get: () => input.feedback, set: (v) => (input.feedback = v) },
      ],
      buttons: [
        { label: '⏯ pause (espace)', action: () => (input.paused = !input.paused) },
        { label: '↺ reset (R)', action: () => (input.reset = true) },
        { label: '🎬 démo (D)', action: toggleDemo },
        { label: '⬇ .vdb (E)', action: () => void exportVdbFile() },
      ],
    },
    {
      // EXPLOSION : une bouffée de carburant + son amorce, sous le pointeur. La
      // boule de feu, la suie et le souffle sortent de la combustion existante —
      // il n'y a rien ici qu'un calibre.
      title: 'explosion',
      sliders: [
        { label: 'rayon', min: 0.03, max: 0.2, step: 0.005, get: () => input.explosionRadius, set: (x) => (input.explosionRadius = x), format: (x) => x.toFixed(3) },
        { label: 'charge', min: 4, max: 200, step: 2, get: () => input.explosionFuel, set: (x) => (input.explosionFuel = x), format: (x) => x.toFixed(0) },
        { label: 'hauteur', min: 0.05, max: 0.7, step: 0.01, get: () => input.explosionHeight, set: (x) => (input.explosionHeight = x), format: (x) => (x <= 0.13 ? 'au sol' : x.toFixed(2)) },
        { label: 'poussière', min: 0, max: 90, step: 2, get: () => input.dustRate, set: (x) => (input.dustRate = x), format: (x) => (x <= 0 ? 'aucune' : x.toFixed(0)) },
        // La suie ne dépend pas de l'explosion : elle naît partout où le
        // carburant chaud manque d'air. Mais c'est là qu'on la règle, parce que
        // c'est une explosion qui en fabrique.
        { label: 'suie : rendement', min: 0, max: 40, step: 0.5, get: () => p.sootYield, set: (x) => (p.sootYield = x) },
        { label: 'suie : opacité', min: 0, max: 3, step: 0.05, get: () => p.sootDensity, set: (x) => (p.sootDensity = x) },
        { label: 'suie : évanouis.', min: 0, max: 0.6, step: 0.01, get: () => p.sootFade, set: (x) => (p.sootFade = x) },
        { label: 'suie : rayonnement', min: 0, max: 8, step: 0.1, get: () => p.sootCooling, set: (x) => (p.sootCooling = x) },
        { label: 'ciel ouvert', min: 0, max: 0.25, step: 0.005, get: () => p.openBand, set: (x) => (p.openBand = x), format: (x) => (x <= 0 ? 'boîte close' : x.toFixed(3)) },
        { label: 'ciel : absorption', min: 2, max: 90, step: 1, get: () => p.openStrength, set: (x) => (p.openStrength = x), format: (x) => x.toFixed(0) },
        { label: 'plafond : bande', min: 0, max: 0.25, step: 0.005, get: () => p.openCeilBand, set: (x) => (p.openCeilBand = x), format: (x) => (x <= 0 ? 'fermé' : x.toFixed(3)) },
        { label: 'plafond : absorption', min: 1, max: 90, step: 1, get: () => p.openCeilStrength, set: (x) => (p.openCeilStrength = x), format: (x) => x.toFixed(0) },
        { label: 'inversion', min: 0, max: 5, step: 0.05, get: () => p.stratStrength, set: (x) => (p.stratStrength = x), format: (x) => (x <= 0 ? 'air neutre' : x.toFixed(2)) },
        { label: 'inversion : base', min: 0.05, max: 0.9, step: 0.01, get: () => p.stratBase, set: (x) => (p.stratBase = x), format: (x) => x.toFixed(2) },
        { label: 'soleil : hauteur', min: 0.02, max: 0.9, step: 0.01, get: () => p.sunHeight, set: (x) => (p.sunHeight = x), format: (x) => x.toFixed(2) },
      ],
      buttons: [{ label: '💥 détoner (G)', action: () => (input.explode = true) }],
    },
    {
      // MODIFICATEURS : objets posés dans la scène, saisissables comme les
      // émetteurs et la boule. Les réglages ci-dessous s'appliquent au PROCHAIN
      // champ posé et à celui qu'on tient — jamais à distance.
      title: 'champs de force (modificateurs)',
      sliders: [
        {
          label: 'type',
          min: 0,
          max: FIELD_NAMES.length - 1,
          step: 1,
          get: () => input.fieldType,
          set: (x) => {
            input.fieldType = x;
            input.fieldStrength =
              x < 0.5 ? SIM3_DEFAULTS.fieldVortexStrength : SIM3_DEFAULTS.fieldWindStrength;
          },
          format: (x) => FIELD_NAMES[Math.round(x)] ?? '',
        },
        { label: 'force', min: 0, max: 600, step: 10, get: () => input.fieldStrength, set: (x) => (input.fieldStrength = x), format: (x) => x.toFixed(0) },
        { label: 'rayon', min: 0.05, max: 0.4, step: 0.01, get: () => input.fieldRadius, set: (x) => (input.fieldRadius = x), format: (x) => x.toFixed(2) },
      ],
      buttons: [
        { label: '＋ poser un champ', action: () => (input.addField = true) },
        { label: '－ retirer', action: () => (input.removeField = true) },
      ],
    },
    {
      title: 'résolution (recharge la page)',
      buttons: GRID3_CHOICES.map((n) => ({
        // Estimation VRAM affichée : au-delà du budget du GPU, l'échec est un
        // écran noir silencieux — le chiffre permet de choisir en connaissance.
        label: `${n}³ · ${(estimateVram3(n) / 2 ** 30).toFixed(1)} Go${n === GRID3 ? ' ✓' : ''}`,
        action: (): void => {
          const url = new URL(location.href);
          url.searchParams.set('grid', String(n));
          location.href = url.toString();
        },
      })),
    },
  ]);
  const toolbar = new Toolbar3D(document.body, [
    INK_NAMES.map((name, i) => ({
      label: name,
      color: inkCss[i]!,
      isActive: () => input.emitInk === i,
      action: (): void => {
        input.emitInk = i;
      },
    })),
    [
      { label: '➕ émetteur', action: () => (input.addEmitter = true) },
      {
        label: '➖',
        action: () => {
          if (sim.selectedIsField) {
            input.removeField = true;
          } else {
            input.removeEmitter = true;
          }
        },
      },
      { label: '⚪ boule', isActive: () => input.sphereActive, action: () => (input.sphereActive = !input.sphereActive) },
    ],
    [
      { label: '💥 boum', action: () => (input.explode = true) },
      { label: '🍄 champignon', action: runMushroom },
      {
        label: '🏞 extérieur',
        isActive: () => p.outdoor,
        action: () => {
          p.outdoor = !p.outdoor;
          // Les repères et la boîte de verre n'ont plus de sens en extérieur.
          input.feedback = !p.outdoor;
        },
      },
      { label: '🎬 démo', isActive: () => demoOn, action: toggleDemo },
      { label: '⚙ réglages', action: () => panel.toggle() },
      // Autres pages (URL relatives : valides en dev et sous /LiquidVM/ sur Pages).
      { label: '🌊 2D', action: () => (location.href = './') },
      { label: '💧 eau', action: () => (location.href = './eau.html') },
    ],
  ]);
  let last = performance.now();
  let fps = 60;
  let frames = 0;
  let hudTimer = 0;
  const tick = (now: number): void => {
    const dt = Math.min(Math.max((now - last) / 1000, 0), 0.05);
    last = now;
    if (dt > 0) {
      fps = fps * 0.95 + (1 / dt) * 0.05;
    }
    if (demoOn) {
      demoDriver.tick(dt);
    }
    input.dt = dt;
    // Une exception ici tuerait la boucle SANS RIEN DIRE : le canvas WebGPU
    // cesse d'être présenté et l'écran devient noir, symptôme parfaitement
    // muet (vécu deux fois). On la montre au lieu de la subir.
    try {
      sim.frame(input, context.getCurrentTexture().createView(), canvas.width, canvas.height);
    } catch (err: unknown) {
      showFatalError(
        `La simulation s'est arrêtée : ${err instanceof Error ? err.message : String(err)}`,
      );
      document.title = 'SELFTEST-FAIL';
      return;
    }
    input.reset = false;
    input.blow.moveX = 0;
    input.blow.moveY = 0;
    input.addEmitter = false;
    input.addField = false;
    input.removeField = false;
    input.explode = false;
    if (boomIn > 0 && --boomIn === 0) {
      input.explode = true;
    }
    input.removeEmitter = false;
    frames++;

    hudTimer += dt;
    if (hudTimer > 0.5) {
      hudTimer = 0;
      panel.refresh();
      toolbar.refresh();
      if (lockScale) {
        if (renderScale !== 1) {
          renderScale = 1;
          resize();
        }
      } else if (fps < 50 && renderScale > 0.6) {
        renderScale = Math.max(0.6, renderScale - 0.1);
        resize();
      } else if (fps > 58.5 && renderScale < 1) {
        renderScale = Math.min(1, renderScale + 0.05);
        resize();
      }
      const solver = input.multigrid ? `MG ×${input.vcycles}` : `jacobi ${input.jacobiIterations} it.`;
      const profil = Object.entries(sim.profileMs)
        .map(([k, v]) => `${k} ${v.toFixed(1)}`)
        .join(' · ');
      hud.innerHTML =
        `<b>LiquidVM 3D</b> · ${GRID3}³ · encre : ${INK_NAMES[input.emitInk] ?? '?'} · ${solver} · ${Math.round(fps)} FPS` +
        `${lockScale ? ' · rendu 100 % 🔒' : renderScale < 1 ? ` · rendu ${Math.round(renderScale * 100)} %` : ''}` +
        `${input.paused ? ' · ⏸ pause' : ''}${demoOn ? ' · <b>DÉMO</b> (D : reprendre la main)' : ' · D : démo'}<br>` +
        `${profil !== '' ? `<b>GPU (ms)</b> · ${profil}<br>` : ''}` +
        '1/2/3 : encre · glisser un objet : déplacer · ses poignées : un seul axe · son bouton violet : orienter · A : + émetteur · X : supprimer · G : 💥 · M : 🍄 champignon · L : rendu 100 % · O : boule · B : braises · F : repères<br>' +
        'glisser : orbiter · clic droit : souffler · molette : zoom · Échap : désélectionner · espace : pause · R : reset · E : export .vdb';
    }
    if (selftest && frames === SELFTEST_FRAMES) {
      const report = document.createElement('div');
      report.id = 'selftest';
      report.setAttribute('data-stage', `frames-${frames}`);
      report.textContent = JSON.stringify({ ok: true, frames, fps: Math.round(fps) });
      report.style.display = 'none';
      document.body.appendChild(report);
      document.title = 'SELFTEST-OK';
    }
    if (selftest) {
      document.getElementById('selftest')?.setAttribute('data-stage', `frames-${frames}`);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

boot().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  showFatalError(message);
  document.title = 'SELFTEST-FAIL';
});
