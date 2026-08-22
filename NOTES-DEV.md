# Notes de reprise (dev)

Mémo pour reprendre le projet après une pause. Le README couvre l'usage ; ici : où
toucher quoi, comment vérifier, et les pièges déjà payés.

## Démarrage

`npm i && npm run dev` → http://localhost:5173 (Chrome/Edge). `npm run build` =
typecheck strict + bundle. Aucune dépendance runtime.

## Carte du code (2 minutes)

- `src/core/` — tout le GPU, zéro DOM. `simulation.ts` orchestre la frame (schéma
  complet commenté en tête de fichier). `passes/*` créent pipelines + bind groups.
  `shaders/*.wgsl` sont autonomes : le struct `SimParams` y est DUPLIQUÉ — toute
  modification doit être synchronisée avec `uniforms.ts` (et tous les .wgsl).
- `src/platform/web/` — device, canvas, rAF, input (`FrameInput` muté en place, zéro
  alloc), panneau de réglages (Tab), barre tactile, overlay.
- Résolutions : `GRID_SIZE` / `DYE_SIZE` / `SCENE_SIZE` dans `core/config.ts` — tout
  suit automatiquement (dispatchs, niveaux multigrid, rayon de splat, éponge,
  `FORCE_SCALE` pour la buoyancy).

## Modifier / étendre

- **Réglage runtime** : `types.ts` (SimTuning) → `uniforms.ts` (fillSlot — slots de
  256 o, 8 vec4 utilisés) → `panel.ts` (slider). Les réglages de rendu passent par
  `renderUniforms`, réécrits seulement au changement.
- **Nouvelle passe compute** : shader WGSL + layout (`pipelines.ts`) + `createXxxPass`
  (`passes/`) + encodage dans `simulation.ts`. Règles non négociables : bind groups
  pré-créés à l'init (toutes les variantes ping-pong), zéro allocation par frame, un
  seul CommandEncoder, aucune lecture GPU→CPU.
- **Formats** : vélocité/densité/curl `rgba16float` (filtrable ET storage — `rg16float`
  n'est pas storage-capable) ; pression/divergence/obstacles `r32float` (storage mais
  NON filtrable → `textureLoad` uniquement).

## Vérifier (indispensable après tout changement de shader)

- `?selftest` : pilote synthétique (dessine un mur → orbite en injectant les 3 fluides
  → bascule les 3 modes de frontières), rapport JSON dans `#selftest`, titre
  SELFTEST-OK/FAIL. Options : `&hold=0|1|2` (fige le mode de frontières), `&nowall`.
  Hook de debug CDP : `window.__frame` (muter les réglages en plein vol).
- `node .selftest/cdp-check.mjs 9333` : attend le rapport, capture les 5 vues.
  `cdp-timeline.mjs 9333 [action] [index] [ms] [n] [tag] [urlFilter]` : captures
  périodiques — l'outil pour localiser une mort dans le temps.
- Chrome de test : `--remote-debugging-port=9333 --user-data-dir=<tmp>
  --enable-unsafe-webgpu --disable-backgrounding-occluded-windows` (sans ce dernier,
  rAF est gelé pour une fenêtre hors écran). Le headless pur (`--headless=new`) ne
  fournit PAS d'adapter WebGPU sur cette machine — toujours headed hors écran + CDP.
- **Vue 6 « résidu MG »** : mosaïque de la pyramide multigrid (4 colonnes × 2 rangées,
  du niveau fin au grossier ; chaque cellule = résidu du dernier V-cycle en haut,
  pression du niveau en bas — mg_debug.wgsl, composée seulement si la vue est active).
  Lecture : un résidu qui flambe désigne l'étage qui ne converge pas ; une pression qui
  explose désigne l'étage où la stabilité se perd en premier ; on y voit aussi le masque
  d'obstacles se dégrader en descendant la pyramide (restriction majoritaire). En mode
  Jacobi, les étages > 0 restent figés sur le dernier V-cycle (normal). L'instrument à
  ouvrir EN PREMIER si le multigrid re-déraille — construit pour le chantier 3D.

## Pièges durement payés — ne pas re-tomber dedans

- **Grille MAC** : texel (i,j) .x = face u gauche (position (i, j+½)), .y = face v
  haute ((i+½, j)). Chaque composante s'échantillonne avec SON décalage d'un demi-texel.
- **Le lisseur de pression doit être l'adjoint exact du couple divergence/gradient.**
  Jacobi pardonne un opérateur approximatif, le multigrid NON (il converge fort vers la
  solution du mauvais système → l'écart pompe de l'énergie → NaN).
- **Espace nul** : en Neumann/périodique la pression n'est définie qu'à une constante
  près, et la restriction MG amplifie sa dérive ×4 par niveau → le pin p(0,0)=0 au
  niveau 8² dans `multigrid.wgsl` est VITAL. Ne pas le retirer.
- **Mode ouvert = boîte fermée + bande éponge** (fn `sponge()` dans les deux
  `advect_*.wgsl`). Une vraie sortie Dirichlet avec face virtuelle extrapolée rend le
  système non symétrique (dernière colonne découplée) → MG diverge par construction.
  Ne pas « réparer » dans ce sens sans stocker les N+1 faces.
- **Float16** : les vélocités explosent en Inf → NaN, puis les clamps transforment les
  NaN en zéros silencieux. Symptôme « tout s'éteint d'un coup » = chercher une pression
  qui diverge, pas un bug de rendu.
- **Un champ de force purement radial est exactement annulé par la projection**
  (irrotationnel). D'où l'outil souffle = jet directionnel, pas radial.
- La vue debug divergence montre le résidu POST-projection (recalculé après le
  gradient) : elle doit rester ≈ noire.
- **Particules contre les parois** : au bord, la vitesse normale est nulle
  (non-pénétration) — les particules qui touchent un mur n'ont plus rien pour les
  décoller et surfent indéfiniment le courant tangentiel de la couche limite
  (observé : bande de particules glissant le long du bord haut en mode parois).
  Mitigation en place : timer « collée » dans misc.z de particle_advect.wgsl,
  respawn au-delà de 2,5 s. Si le phénomène réapparaît, vérifier ce timer d'abord.

## Perf mesurée (2026-08-20, desktop de référence)

512² et 2048² : 60 FPS (vsync) · 4096² : 32–60 FPS, ~1,6 Go VRAM.
Config actuelle : sim 1024², dye/scène 2048².

## Branches garées (fonctionnelles, non mergées — décision, pas dette)

- **`marbrure`** — atelier de marbrure mathématique (Lu/Jaffer) : outils goutte /
  stylet / peigne = déformations inverses préservant les aires sur la texture
  d'encres (`marble_warp.wgsl`), rendu « papier » pigmentaire, export PNG 2048²,
  murs respectés par raymarch même en pause. Le tout derrière UN interrupteur
  « mode marbrure » au panneau (pause + gravité 0 + papier + bain préparé ;
  R = bain neuf). État : complet et vérifié (selftest `&marble` +
  `.selftest/cdp-marblemode.mjs`, dernier commit 16758f1). Garée le 2026-08-20 :
  l'utilité artistique n'a pas convaincu à l'essai — à re-évaluer, pas à re-coder.
  Avant toute reprise : rebaser sur main (la branche date d'avant les évolutions
  suivantes de main).
- **Candidat à rapatrier seul depuis `marbrure`** : l'aperçu fantôme des outils
  (`platform/web/preview.ts` + son câblage main.ts/styles) — il sert TOUS les
  outils, pas que la marbrure.

## Chantier suivant : EAU liquide (branche `eau`)

Conception complète dans [PLAN-EAU.md](PLAN-EAU.md) — méthode (PIC/FLIP →
APIC), architecture (`eau.html` + `src/liquid3d/`, pont atomics→textures),
jalons J0–J6 avec critères de sortie mesurables ET LEUR JOURNAL (résultats,
batailles gagnées, leçons). À lire AVANT d'écrire du code eau.
LA leçon de J1 (2026-08-23) : les cellules de BORD doivent pouvoir contenir
des particules — une marge de 1 voxel aux parois laissait la rangée 0 vide,
donc classée AIR (p = 0) : le sol était une surface libre, zéro hydrostatique,
toute l'eau écrasée en une couche (c'était ça, pas « Jacobi ne tient pas
l'hydrostatique »). Instruments clés, à utiliser AVANT de spéculer :
le recensement (`census_pass` dans eau_g2p.wgsl — valides/perdues/rapides +
8 particules brutes) et le **recensement des cellules** (`cell_census` dans
eau_surface.wgsl — histogramme d'occupation 1-3/4-7/8-11/12-23/24+ au HUD :
phase dynamique saine = 8-11 dominant, 24+ < 2 k), plus les vues debug du
rendu (« coupe z=0 » du quart inférieur ×4, « densité max »). Une image de
points additifs ne dit RIEN de l'épaisseur ni de la densité : mesurer.
Rendu : `eau_surface.wgsl` = boîte (sol/parois/grille/arêtes) + iso-surface de
la densité floutée 3³ (calculée par sous-pas, lue aussi par le contrôle de
densité) marchée par rayon avec réfraction ; points = touche P.
Contrôle de densité (eau_grid.wgsl) : EXPANSION SEULE au-delà d'une zone morte
de 25 %, taux 10/s — la compression des cellules sous-denses fabrique des
grumeaux, l'absence totale de contrôle compacte l'eau de ~25 % (tableau des
mesures dans le journal J2 de PLAN-EAU). Transfert APIC par défaut (J2), FLIP
conservé (case / `?flip`) ; `?tall` = colonne haute 32×64.
Boule-obstacle (J3, touche O) : ANALYTIQUE comme celle du feu — face touchant la
boule = débit prescrit (divergence ET gradient), voisin solide = p_centre,
particules repoussées avec annulation de la vitesse relative entrante ; saisie
par `hitTest` au pointeur (glisser dessus = brasser, ailleurs = orbiter). Le
compteur « dans la boule » du HUD doit rester à 0.
Jalons J0–J3 verts ; la branche `eau` n'est PAS mergée dans `main` (le site
public ne montre rien de l'eau — demander avant de merger).

## Chantier 3D (branche `3d`)

- **Page séparée `3d.html`** (`src/platform/web/main3d.ts`) — l'appli 2D n'est pas
  touchée. `src/core3d/` : même doctrine que core/ (zéro DOM, zéro alloc/frame, un
  encoder, bind groups pré-créés), `config3d.ts` (GRID3 = 128) + `sim3d.ts` (tout
  l'orchestrateur) + `shaders3d/`.
- **État** : résolution 128³–512³ (GRID3 via setGrid3, défaut 256³ ; forces
  remises à l'échelle par SCALE3 = GRID3/128 pour garder le comportement
  physique), MAC 3D MacCormack RK2, TROIS ENCRES COLORÉES (canaux xyz de la
  densité, palette INK_COLORS bleu/magenta/ambre dupliquée dans raymarch.wgsl,
  touches 1/2/3 = encre de l'émetteur via blow_force.w, albédo mélangé par voxel
  au rendu), poussée thermique, projection MULTIGRID 3D (V-cycle GRID3→coarsest
  < 16, 1 cycle/frame warm-starté — pin p(0,0,0)=0 au niveau grossier appliqué
  d'office, le clamp des voisins EST l'opérateur Neumann exact, adjoint du
  couple divergence/gradient compact ; repli Jacobi comparatif via
  `__frame3d.multigrid=false`), émetteur de panache balancé, ray-marching
  (Beer-Lambert + corps noir + éclairage volumétrique, voir plus bas), souffle
  au clic droit, caméra orbitale, E = export VDB (encodeur généralisé
  multi-nœuds 16³ ; density = somme des encres, ≤ 512³ depuis le fix
  maxBufferSize). 60 FPS à 256³, ~26 à 384³, ~20 à 512³ (machine de référence),
  VDB re-validé dans Blender.
- **Interactions saisies** : clic gauche sur la flamme ou la boule = déplacer
  (hitTest par rayon décide saisie vs orbite ; déplacement sur le plan face
  caméra passant par l'objet), A = + émetteur sous le pointeur (max 4, chacun
  son encre — 1/2/3 recolore l'émetteur ACTIF, le dernier ajouté ou saisi),
  X = − émetteur, O = sphère-obstacle. La SPHÈRE est ANALYTIQUE : aucune texture
  d'obstacles — chaque shader (advection, forces, confinement, gradient, lisseur
  Jacobi ET multigrid à tous les niveaux via remontée en voxels fins) teste
  centre-de-cellule-dans-la-sphère depuis l'uniforme, avec la même règle partout
  (adjonction exacte préservée, aucun NaN). La fumée se fend autour de la boule
  et se referme. 60 FPS avec sphère + 3 émetteurs à 256³.
- **La boule brasse le fluide** : condition de bord MOBILE — les faces bloquées
  portent la vitesse de la sphère (suivie à la saisie, EMA 0.55/0.45, plafond
  420·SCALE3 voxels/s, amortie ×0.82 au relâcher, uniform sphere_vel). Le
  gradient PRESCRIT ces faces (ne les zéroe plus), la divergence les lit comme
  débit connu, le lisseur reste l'adjoint exact (faces fermées côté pression).
- **Rendu ancré** : sol sous la boîte (plan y=−0.502, tapis radial) recevant
  l'ombre volumétrique de la fumée (12 pas vers la lumière) + l'ombre analytique
  de la boule ; la boule s'ombre aussi DANS la fumée (test rayon-sphère par
  échantillon éclairé) et rougeoie au corps noir (chaleur échantillonnée à sa
  surface). Le tout tient les 60 FPS à 256³.
- **COMBUSTION (Feldman/Fedkiw simplifié)** : matière 3 = vapeur de CARBURANT,
  émise froide (pas de chaleur) et légèrement lourde → nappes au sol. Réaction
  dans advect_density (ignition smoothstep(0.28, 0.55) sur la chaleur, taux
  emit_meta.y, chaleur dégagée emit_meta.z plafonnée 1.9, suie 0.35·burn dans
  le canal fumée) ; EXPANSION = source de divergence au front de flamme
  (emit_meta.w, la passe divergence lit les densités [vel][den] et recalcule le
  même critère). Poids propres des matières dans forces (ink_weights slot 14 —
  Boussinesq deux voies, l'encre magenta retombe en fontaines). CALIBRATION
  DUREMENT PAYÉE : chaleur 2.0 + expansion 40 = runaway → boîte entière
  incandescente (boîte fermée : l'expansion nette ne peut pas sortir, tout
  s'accumule) ; valeurs sûres 0.7/10, dissipation matières 0.20/s pour
  l'auto-nettoyage. UX : 1/2/3 = encre des FUTURS émetteurs + celui EN MAIN
  uniquement (recolorer l'actif à distance éteignait la flamme pilote).
  Allumage jouable : amener/souffler la nappe vers une flamme. 53-60 FPS.
- **Numérique** : advection à traces RK2 point-milieu (backtrace/forwardtrace
  dans les deux advections MacCormack) ; refroidissement RADIATIF T⁴ sur la
  chaleur (0.30·T⁴/s — pointes de flammes nettes, fumée tiède persistante).
- **OXYGÈNE / système d'espèces** : texture rgba16float ping-pong (x = O₂ init 1
  via clear_one, yzw = espèces futures), passe species3d (advection RK2 + chimie)
  APRÈS l'advection des densités (elle lit les densités fraîches — cohérence du
  taux de réaction). La combustion et l'expansion sont modulées par
  clamp(O₂/0.25, 0, 1) ; consommation 0.55·burn ; récupération 0.015/s (boîte
  qui fuit) ; le souffle INJECTE de l'O₂ le long de son rayon (soufflet de
  forge). En vase clos une flamme soutenue s'étouffe — la raviver au souffle.
- **Perf (60 FPS partout à 256³)** : résolution de rendu dynamique pilotée FPS
  (0.6–1.0×, hystérésis 50/58.5, HUD « rendu N % ») ; pas adaptatif du raymarch
  (air vide = double enjambée, intégrales sur la longueur réelle) ; V-cycles = 1
  (warm start — 2 coûtaient ~6 balayages de plus sans gain visible). Si ça
  ralentit à nouveau : le budget sim est plein, chercher là (le rendu s'adapte
  seul).
- **Résolutions ?grid=128|256|320|384|512** — et le mystère de l'écran noir
  RÉSOLU : ce n'était PAS la VRAM. Dawn valide ses staging buffers INTERNES
  (zéro-init des textures 3D, `Dawn_DynamicUploaderStaging`) contre
  `maxBufferSize`, défaut 256 Mio : une rgba16float 384³ = 432 Mo → chaque
  submit échouait en silence (warning console UNIQUEMENT — écouter
  `Log.entryAdded` en CDP, c'est comme ça qu'on l'a trouvé). 320³ = 262 Mo
  passait à 2 % près, d'où l'ancien « plafond ». Fix : gpu.ts demande
  `requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize }` (2 Gio sur
  la RTX 5070 Ti). Garde-fou dans main3d : si l'adapter ne suffit pas pour la
  plus grosse texture, erreur fatale claire au lieu du noir. Mesures machine de
  référence (16 Go) : 384³ (~4,8 Go) ≈ 30 FPS, 512³ (~11,3 Go) ≈ 20-22 FPS,
  rendu adaptatif à 60 % dans les deux cas (budget sim plein, normal). Les
  boutons du panneau affichent l'estimation VRAM (estimateVram3, ~90 o/voxel).
  Piège de banc : après un run 512³, le processus GPU garde de la mémoire
  résidente — un test suivant peut tomber à 20 FPS à 256³ ; redémarrer le
  Chrome de test avant de mesurer. L'export VDB suit maxBufferSize : 384³ et
  512³ exportables désormais (multiples de 128 ; fichiers ~450 Mo / ~1,1 Go).
- **Interface (panel3d.ts)** : panneau à sections (Tab) + barre d'outils,
  ENTIÈREMENT déclaratifs — ajouter un réglage = une entrée de spec dans main3d.
  Toute la physique est réglable à chaud via Sim3Tuning (Frame3DInput.params,
  défauts de defaultTuning3()) : combustion, débits, dissipations, poussée,
  souffle, vitesse du temps, rendu. Barre : matières en pastilles, ± émetteur,
  boule, démo, réglages, retour 2D ; sélecteur de résolution (recharge avec ?grid=).
- **Presets (section en tête de panneau)** : défaut / bougie / fournaise /
  fumée épaisse — chaque preset = un Partial<Sim3Tuning> appliqué À CHAUD
  par-dessus defaultTuning3() (déterministe quel que soit l'historique de clics,
  pas de reset : la flamme change de caractère en direct). Ajouter un preset =
  une entrée dans PRESETS (main3d.ts). Exposition/pas de marche non touchés.
- **Mode DÉMO (showcase, REMASTER 2026-08-22)** : touche D ou `?demo` — boucle
  ~88 s en QUATRE ACTES qui traversent les presets : I bougie (caméra proche au
  ras, lueur intime) → la flamme prend (défaut) + fontaine magenta → II nappe
  de carburant + souffle scripté → III FOURNAISE (braises ON, lueur 2.3,
  dissipation remontée à 0.4 pour consumer la brume des actes précédents,
  caméra ample) + boule en lemniscate + grand souffle → IV fumée épaisse
  (braises OFF, caméra haute lente) → accalmie → reset. Caméra par CIBLES
  LISSÉES par acte (camT : vitesse d'orbite/élévation/distance, exp-smoothing
  + respiration). Les réglages du spectateur sont SNAPSHOTÉS à l'entrée et
  RESTAURÉS à la sortie (start()/stop() appelés par toggleDemo — la démo mute
  params/braises/lueur sans laisser de trace). TUNE_CANDLE/FURNACE/SMOKE au
  niveau module, partagés avec les presets du panneau. Piège de banc : le
  chronométrage mural d'une capture de démo doit se caler sur les ÉVÉNEMENTS
  (ex. bascule embersOn), pas sur l'horloge — le boot décale tout de ~10 s.
- **MacCormack 3D** : prédicteur/correcteur clampé par composante MAC (stencil du
  point rétro-advecté) — LE raffineur : le panache laminaire devient turbulent et
  structuré à lui seul.
- **Confinement de vorticité 3D — RÉACTIVÉ (défaut ε = 12)** : ω vectoriel aux
  centres, F = ε N̂×ω par face — et |ω| est FLOUTÉ (passe blur_curl, boîte 3³
  écrite dans velScratch libre à ce point, ω passe en xyz, |ω|flou en w) avant
  le gradient. C'était le fix pressenti : le gradient de la magnitude brute à
  ±1 voxel injectait du grain de grille dès ε≈3 (ancienne calibration EPS-*,
  défaut 0). Recalibré (EPS2-0/8/16/24 à 256³) : 8 subtil, 16 riche encore
  cohérent, 24 déchiqueté mais JAMAIS granuleux — l'échec a changé de nature.
  Slider « vorticité » (0–30).
- **Braises (2026-08-22)** : 32 768 particules-étincelles (embers3d.wgsl update,
  embers_draw.wgsl tracé ; buffer STORAGE fixe 1 Mo zéro-initialisé = toutes
  mortes). NAISSANCES AUTORÉGULÉES par rejet : chaque morte tire UNE position
  aléatoire par frame et ne naît que si chaleur > 0.55 (plus de feu = plus de
  braises, zéro chaud = zéro braise). Portées par le fluide (velocity_at MAC,
  relaxation exp(-6·dt)) + scintillement sinusoïdal propre + kick de naissance ;
  meurent hors boîte / dans la boule / à bout de vie (0.9–2.7 s). TRACÉ :
  billboards ADDITIFS dans la passe HDR (avant bloom — halo hérité), projection
  = INVERSE de la construction des rayons du raymarch (pas de matrice : clip =
  [d·right/(tanf·aspect), d·up/tanf, ·, d·fwd]) ; occlusion PAR PARTICULE au
  vertex (4 échantillons d'extinction vers la caméra + segment contre la
  boule) ; braise morte = quad dégénéré (zéro pixel). Corps noir 1.05–1.45
  refroidissant (doré→orange, jamais blanc). INTERRUPTEUR « braises (B) »
  (défaut COUPÉ — l'utilisateur évalue encore l'utilité) + slider de débit ;
  coupées = passes entièrement sautées. R.style.z = débit (lu par le compute
  via renderUniforms bindé en groupe 1), R.style.w = N. COÛT MESURÉ : 0,0 FPS
  de différence en A-B-A-B à 384³ à état de sim égal — le ralentissement
  ressenti pendant l'essai venait de la BOÎTE QUI SE REMPLIT de fumée (le
  raymarch paie chaque voxel non vide : le FPS à 384³ passe de ~34 panache
  jeune à ~22 en régime enfumé, braises ou pas). Leçon de banc : toujours
  mesurer en A-B-A-B alterné, jamais avant/après — l'état de la sim évolue.
  PIÈGE WGSL : « target » est un mot réservé (comme « from »).
- **Éclairage volumétrique (2026-08-22)** : la flamme éclaire sa propre fumée.
  (1) VOLUME DE LUEUR (glow3d.wgsl, grille GRID3/8, paire ping-pong) : inject =
  émission corps noir moyennée par pavé (8 prises trilinéaires), 3 blurs boîte
  3³ = diffusion ≈ scattering multiple ; le raymarch l'échantillonne en
  in-scattering (∝ densité locale), le SOL reçoit une flaque de lumière chaude,
  la SPHÈRE est éclairée par la lueur. Slider « lueur du feu » (0–3, défaut
  1.8, input.glowStrength). Tourne aussi en pause (passe compute dédiée).
  (2) OMBRE À DEUX OCTAVES (« powder ») : 0.62·exp(-τ·0.9) + 0.38·exp(-τ·0.22)
  — la lumière pénètre les cœurs épais. (3) PHASE Henyey-Greenstein g=0.45 sur
  la lumière directionnelle (constante par rayon) : liseré argenté à
  contre-jour. (4) CHAÎNE HDR + BLOOM (post3d.wgsl) : le raymarch écrit du
  rgba16float LINÉAIRE (tone-mapping déplacé en présentation), texture HDR
  taille canvas recréée au resize SEULEMENT (bind groups bloomDown/present
  avec elle — doctrine zéro-alloc tenue), bloom en taille FIXE 384×216 (seuil
  doux 0.9–1.9, gaussienne séparable espacement 1.5 texel), présentation =
  HDR + bloom·strength → tone-map expo + gamma. Slider « bloom » (0–1, défaut
  0.35). Coût mesuré : 60 FPS à 256³, ~26 FPS à 384³ (le rendu ajouté est payé
  par pixel, pas par voxel). frame() prend désormais (input, target, width,
  height) — la taille pilote la cible HDR.
  long du rayon caméra→scène, dirigée selon le geste (base caméra lue depuis
  renderData — écrite AVANT les uniforms sim). Réglages : blowRadius/blowForce.
- **Export OpenVDB (touche E)** : readback ponctuel hors boucle (seule lecture
  GPU→CPU du moteur 3D) + encodeur .vdb pur TS (core3d/vdb.ts) + téléchargement.
  Grilles float denses `density` et `temperature`, arbre 5-4-3 sans compression,
  Y sim → +Z monde (la fumée monte dans Blender), boîte 1³ centrée. VALIDÉ dans
  Blender 5.1.2 : import + volume-to-mesh (4720 sommets) + rendu Cycles du panache.
  Pièges du format durement payés : masque de feuille = 8 **u64** = 64 octets (pas
  8 octets !) ; masques du nœud 32³ = 4096 octets chacun ; UUID = 36 chars BRUTS
  sans préfixe de longueur. Référence : article JangaFX « VDB: a deep dive » +
  github.com/jangafx/simple-vdb-writer (et tinyvdbio pour le point de vue lecteur).
  Piège Blender headless : --factory-startup garde le cube par défaut (dans la
  collection enfant) — purger bpy.data.objects avant tout rendu de test.
- **Prochaines briques** : séquences VDB animées (File System Access API) ;
  encres colorées (canaux yz réservés dans la densité) ; qualité du confinement
  de vorticité (flouter |ω|).

## Pistes suivantes

Test smartphone (1024², vérifier les limites WebGPU mobiles) · suite du chantier 3D
ci-dessus.
