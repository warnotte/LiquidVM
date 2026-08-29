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
  `cdp-gizmos.mjs 9333` (page 3D) : repères + poignées, avec de VRAIS événements
  souris — vérifie qu'un geste diagonal sur une poignée ne bouge qu'un axe.
  `cdp-boom.mjs 9333 [tag] [rayon] [charge]` : une détonation, capturée à neuf
  instants — une explosion se juge sur sa CHRONOLOGIE, pas sur une image.
  `cdp-nuke-ab.mjs 9333` : rejoue le TIR ENTIER par variante (reset complet,
  même temps SIMULÉ) — pour tout ce qui est cumulatif, où un état figé ne dit
  rien. Lancer les mesures d'image avec `?lock` (résolution verrouillée).
  `cdp-suie-ab.mjs 9333` : détone, FIGE la scène, puis bascule un terme de
  couleur à la fois sur exactement le même état. À l'œil, émission, diffusion et
  lueur se ressemblent — c'est le seul moyen honnête de savoir ce qui teinte un
  nuage, et c'est ce qui a démasqué la suie qui ne se produisait pas.
  `cdp-timeline.mjs 9333 [action] [index] [ms] [n] [tag] [urlFilter]` : captures
  périodiques — l'outil pour localiser une mort dans le temps.
  `cdp-salve.mjs 9333 [tag] [ecartMs]` : trois détonations VISÉES à 0,5 s
  d'écart — le banc des charges multiples, jugé sur la COEXISTENCE (une boule
  fraîche à côté d'un nuage déjà noirci) et sur les slots (centres distincts).
  Il NAVIGUE À NEUF d'abord : les slots de charge survivent au reset, un banc
  précédent sur la même page fausserait l'attribution.
  `cdp-vise.mjs 9333` : sonde de visée — rejoue le calcul de rayon hors tir et
  dit quels points NDC tombent DANS la boîte sur le plan de charge. À passer
  AVANT d'écrire un banc qui vise : le plan est bas, une visée à peine trop
  haute ou large le coupe hors boîte et le tir est rabattu au centre.
  `cdp-profil384.mjs 9333` : recharge en ?grid=384&profile&lock et imprime la
  médiane de la table ?profile par passe — la jauge avant/après d'une retouche
  de shader.
  `cdp-artifice.mjs 9333 [tag]` : les actes V-VI de la démo (bombardement, feu
  d'artifice, bouquet) capturés AUX INSTANTS DU SCÉNARIO en lisant l'horloge
  `__demo3d.t` (le mural dérive du boot), plus le début du tour 2 de boucle —
  c'est là qu'un état oublié par un acte se voit. S'appuie sur `?demo=<s>` qui
  cale l'horloge au lancement (les actes antérieurs s'appliquent en un tick).
  `cdp-sparks.mjs 9333 [tag]` : le banc des ÉTINCELLES de feu d'artifice
  (chantier PLAN-ARTIFICES) — pivoine VERTE (la teinte-preuve : le corps noir
  ne la produit pas), duo bicolore rapproché, chronologies saule et éclat,
  le TIR COMPLET (montée traçante → flash à l'apogée → sphère → retombée →
  fumée), la SALVE de cinq teintes (coexistence, mélanges propres), le
  BI-COULEUR (coque + cœur), puis NON-RÉGRESSION des braises en fournaise.
  Tir scripté :
  `__sim3d.launchFirework(x, y, z, r, g, b, n?, vitesse?, patron?)` (l'éclat
  seul) et `__sim3d.launchRocket(x, z, apexY, r, g, b, n?, vitesse?, patron?,
  r2?, g2?, b2?)` (le tir complet ; r2 ≥ 0 = bi-couleur, le cœur prend la
  seconde teinte) ; `launchRocketAt(ndcX, ndcY, ...)` = le même tir VISÉ
  (rayon ∩ plan d'éclat, centre hors boîte). À la main : touche T sur la
  page 3D — le tir suivant d'une table déterministe, visé au pointeur.
  `cdp-obstacle-chaos.mjs 9333 [tag] [champignon|defaut] [posee[:x]|trainee|sans]
  [retouche-js]` : le banc de l'OBSTACLE. Une seule chose varie à la fois ; la
  détonation, la flamme coupée et la caméra sont tenues fixes. Le geste compte
  autant que le reste : « posee » ne donne à la boule qu'une PRÉSENCE, « trainee »
  lui donne une VITESSE — donc un bord mobile, un tout autre problème.
  `mesure-nuage.mjs <png…>` : mesure HORS LIGNE (zlib de Node, aucun navigateur,
  donc aucun GPU volé à la passe en cours). Compte la clarté d'une fenêtre prise
  DANS la boîte, à gauche de l'axe de la colonne, moins celle du ciel : un
  champignon y laisse ~2‰, une boîte envahie 50 à 100‰. Compter la matière ne
  sépare rien — c'est OÙ elle est qui sépare. À étalonner sur le témoin avant de
  s'en servir (cf. pièges).
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

- **PRESSION : JAMAIS UN SEUL V-CYCLE PAR FRAME, et le lisseur rouge-noir
  (2026-08-27)** — chantier « 384³ confortable ». Trois choses apprises, chacune
  payée plusieurs heures.
  1. **À ×1 exactement, le solve dégénère.** Un V-cycle par frame, warm start :
     l'erreur s'accumule frame à frame et la scène meurt (à 384³, boîte vide en
     quelques secondes — la flamme pilote écrasée au sol). Deux cycles par
     frame, STRICTEMENT identiques un à un : sain. Vrai pour tous les lisseurs
     essayés (Jacobi pondéré, rouge-noir en place, rouge-noir ping-pong, avec ou
     sans SOR). Le « panache couché » du journal du 26 était la forme DOUCE de
     cette maladie — je l'avais lue comme de la simple sous-convergence et
     conclu qu'il fallait ×4 ; en réalité ×2 suffit avec un bon lisseur, et ×1
     est un cas maudit qu'aucun compte de balayages ne rattrape. Non élucidé au
     fond ; contourné : `vcyclesFor()` rend 2 partout, le second cycle est quasi
     gratuit à 256³ (59,3 contre 60,3 FPS).
  2. **Le driver fait payer certaines FORMES sur les textures minuscules.** Sur
     les niveaux ≤ 24³ de la pyramide, trois formes font passer la frame 384³ de
     ~50 à ~230 ms, mesuré chacune isolément, sans explication propre : le
     storage read_write (lisseur en place OU prolongation en place), deux
     pipelines alternés entre dispatches, les offsets dynamiques d'uniforme. La
     forme qui passe : UN pipeline, ping-pong write-only, offsets STATIQUES par
     bind group. Les grands niveaux, eux, tolèrent le read_write sans broncher.
     Découpage en passes séparées : aucun effet.
  3. **Une config « mystérieusement lente » peut être une config numériquement
     MALADE.** Un champ qui explose coûte des FPS par le RENDU (les early-exit
     du raymarch ne s'amorcent plus) : 4 FPS qui ressemblent à une pathologie
     driver. Une partie de mes mesures « driver » de la journée mesuraient en
     fait la maladie du ×1. Règle : devant une lenteur inexpliquée, REGARDER LA
     SCÈNE (une capture) avant de profiler quoi que ce soit.
  Résultat net : lisseur rouge-noir V(1,1) (Gauss-Seidel, ~2× le facteur de
  lissage du Jacobi amorti), SOR (ω ≈ 1,7) au niveau le plus grossier qui RÉSOUT
  au lieu de lisser, pyramide arrêtée à 24³ (le 12³ était dans une des formes
  maudites), ×2 partout. À rendu verrouillé 100 % : 384³ passe de ~14 à
  20,0 FPS, 320³ à 36,8, 256³ reste à 60. Le plafond SANS pression est ~30 FPS à
  384³ : les 60 FPS demanderaient de s'attaquer au reste de la frame
  (advection, vorticité, rendu), pas au solveur.

- **LE CONFINEMENT DE VORTICITÉ RENFORCE LE BORD DE L'OBSTACLE (2026-08-27)** —
  fin de l'affaire « la sphère influence la simulation », signalée par Renaud et
  restée ouverte trois commits. Symptôme : à 384³, preset champignon, boule
  TRAÎNÉE à la souris puis lâchée, détonation — vers 6 s la boîte est envahie par
  un tourbillon, avec un jet visible tangent à la boule.
  **Ce que ce n'est pas**, mesuré et non raisonné : ce n'est pas la PRÉSENCE de
  l'obstacle (boule posée au même endroit, jamais animée → champignon propre à
  9 s) ; ce n'est pas la pyramide de pression (Jacobi montre un tout autre défaut,
  celui déjà connu de la sous-convergence) ; ce n'est pas la résolution seule
  (384³ sans boule → propre).
  **Ce que c'est** : `curl` ne connaît pas l'obstacle. Il lit la vitesse PRESCRITE
  sur les faces bloquées à côté de la vitesse du fluide une cellule plus loin ; la
  marche d'escalier entre les deux donne un |ω| qui n'est pas un tourbillon mais
  un artefact de discrétisation du bord. Le confinement, lui, RENFORCE ce qu'il
  trouve : il en fait un jet tangent à la boule, qui s'entretient. Et la marche
  vaut Δv sur UNE cellule, donc |ω| ≈ Δv/h **grandit quand la grille s'affine** —
  d'où un défaut invisible à 256³ et net à 384³, sans que ε y soit pour rien.
  Même famille que le tourbillon parasite du ×SCALE3 (ci-dessous) : le confinement
  injecte de l'énergie sans borne. Là c'était le facteur, ici c'est le bord.
  **Correctif** : le confinement s'éteint en FONDU sur une peau de trois cellules
  autour de la sphère (`fondu_obstacle`, vorticity3d.wgsl). En fondu et jamais net
  — une coupure franche serait elle-même une discontinuité de force, donc une
  nouvelle source. Sans sphère le rayon vaut −1 et le chemin est identique à
  l'ancien, au bit près.
  **Mesuré** (fenêtre « gauche » de `mesure-nuage.mjs`, 384³, boule traînée,
  champignon) : 77‰ à 6 s avant, 3‰ après — soit le niveau du témoin sans boule.
  Non-régression sur la scène où la boule est un SUJET (fournaise, boule dans les
  flammes, 256³, 24 s) : images indiscernables avant/après.
  **Leçon de diagnostic, la plus chère de la session** : une mesure doit être
  passée sur le TÉMOIN avant de conclure. Ma fenêtre a signalé « chaos » sur le
  preset par défaut ; le même preset SANS boule donne le même chiffre — elle
  mesurait la largeur de la boule de feu, pas le chaos. J'ai failli conclure que
  le défaut ne demandait pas les réglages extrêmes, sur la foi d'un instrument que
  je n'avais pas étalonné.
  **Correction d'une entrée précédente** : le commit b5e0dac écrit que le
  déplacement de la boule avait été éliminé comme cause (« même symptôme avec une
  boule posée »). Cela ne se reproduit pas : à HEAD, posée = propre, traînée =
  envahie. C'est bien le DÉPLACEMENT qui amorce, en lançant l'écoulement qui rend
  la nappe de bord assez forte pour s'emballer.

- **SOUS-CONVERGENCE DE LA PRESSION À HAUTE RÉSOLUTION (2026-08-26)** — signalé
  par Renaud comme « la sphère semble attirer la fumée », et « comme si le mode
  démo était activé ». Ce n'était NI la sphère NI le mode démo.
  Mesuré, page fraîche, scène par défaut à 384³ (`.selftest/cdp-sphere.mjs`) :
   - multigrid ×1 → le panache est couché au ras du sol et étalé ;
   - **boule RETIRÉE, même symptôme** — donc l'obstacle n'est pas en cause ;
   - **Jacobi 60 itérations → propre** ; c'est donc le solveur de pression ;
   - multigrid ×4 → le panache remonte.
  Un seul V-cycle suffit jusqu'à 256³ et plus au-delà : la pression
  sous-convergée laisse un écoulement latéral parasite. Le défaut suit désormais
  la résolution (`vcyclesFor()` : 1 jusqu'à 256, 2 au-delà, 4 à partir de 384).
  **Leçon de diagnostic** : un symptôme qui DÉSIGNE un objet (« c'est la
  sphère ») doit être vérifié en RETIRANT l'objet. Ici l'obstacle ne faisait que
  rendre visible un écoulement parasite qui existait sans lui.
  **REFERMÉ le 2026-08-27** : ce qui restait — la boîte envahie à 384³ avec la
  boule — n'était pas la pyramide de pression mais le CONFINEMENT DE VORTICITÉ au
  bord de l'obstacle (entrée suivante). La piste « masques d'obstacle dégradés en
  descendant la pyramide » a été traitée quand même (restriction conservatrice,
  commit b5e0dac) : elle est correcte, elle n'était pas la cause.
  Coût mesuré du passage à ×4 : 384³ tombe à 20 FPS.

- **TESTER CE QUE L'UTILISATEUR PEUT FAIRE (2026-08-26)** — le piège le plus
  coûteux de la session, et il n'était pas dans le moteur. Mes harnais CDP
  préparaient TOUJOURS la scène avant un tir : flamme pilote coupée, boule
  retirée, caméra placée, panneau replié. J'ai donc validé pendant des heures un
  scénario **irreproductible à la souris** : en cliquant « champignon » puis
  « boum », Renaud voyait une purée grise, parce que le preset laissait la flamme
  pilote brûler à plein débit avec une dissipation quasi nulle et une poussée de
  430 — réglages calibrés pour un ÉVÉNEMENT, appliqués à un régime continu.
  Deux règles qui en sortent :
   1. Un harnais qui PRÉPARE la scène mesure le moteur, pas le produit. En garder
      un qui ne fait QUE ce que fait l'utilisateur — `.selftest/cdp-user.mjs`
      clique le preset, clique la détonation, et rien d'autre.
   2. Un preset de SCÉNARIO (détonation) n'est pas un preset d'AMBIANCE : il doit
      couper ce qui alimente en continu et repartir d'un air propre. Le champignon
      coupe donc la flamme pilote et déclenche un reset.

- **UNITÉS ET RÉSOLUTION (3D, 2026-08-26)** — signalé par Renaud (« si j'augmente
  la résolution, ça devient encore pire »), et c'était un vrai bug d'unités.
  La règle, à appliquer à CHAQUE nouvelle grandeur :
   - une **ACCÉLÉRATION** (monde/s²) vaut N fois plus en voxels/s² → **×SCALE3**.
     Poussée, vent, poids des matières, force des champs : correctement traités.
   - une **VITESSE** (monde/s) vaut aussi N fois plus en voxels/s → **×SCALE3**.
   - le ε de **VORTICITÉ** est l'EXCEPTION, et elle mérite d'être comprise : la
     forme de Fedkiw est `f = ε·h·(N̂ × ω)`, et le facteur h (taille de cellule,
     1/N en monde) annule EXACTEMENT la conversion monde→voxels de
     l'accélération. Le ε discret est donc **déjà indépendant de la résolution**
     et ne se met PAS à l'échelle. J'ai cru le contraire en lisant la formule
     discrète sans son h : à 384³ et 512³ le confinement injectait alors assez
     d'énergie pour faire naître un TOURBILLON PARASITE qui envahissait la boîte
     après quelques secondes (invisible aux résolutions basses, où le facteur
     restait petit). **Règle de sûreté : le confinement de vorticité injecte de
     l'énergie sans borne — toute modification de son ε se vérifie sur PLUSIEURS
     SECONDES à la plus HAUTE résolution, jamais sur une image.**
   - une **DIVERGENCE** est en 1/s et ne se met **PAS** à l'échelle. La
     divergence discrète est la somme des écarts de vitesse d'une face à
     l'autre, en voxels/s par voxel : c'est déjà du 1/s. L'expansion de
     combustion était multipliée par SCALE3 à tort.
  **Symptôme** : à temps SIMULÉ égal, la même détonation donnait une colonnette
  à 128³ et un nuage massif à 384³ — la haute résolution paraissait « pire »
  parce qu'elle explosait plus fort, pas parce qu'elle simulait moins bien.
  **Méthode qui l'a montré** : `.selftest/cdp-res.mjs` — une passe par
  résolution, page fraîche, capture au même temps SIMULÉ. Comparer à la montre
  n'aurait rien donné : une grille fine tourne moins vite, on aurait comparé
  deux ÂGES différents du même nuage.
  Après correction, 128³ et 384³ donnent la même taille et la même forme, la
  finesse en plus. C'est l'EXPANSION qui portait tout le défaut ; les défauts et
  presets ont été rebasés en conséquence (expansion ×2 à 256³).
  `.selftest/cdp-vortex.mjs` rejoue la scène qui a démasqué le tourbillon
  parasite (boule poussée sur le côté, détonation, observation jusqu'à 8 s de
  temps simulé) — à garder pour toute retouche du confinement.


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
Jalons J0–J4 verts, MERGÉS dans `main` (les trois pages sont cross-linkées).
SOLVEUR DE PRESSION : multigrid masqué par défaut (`eau_mg.wgsl` = celui du feu +
un masque de type de cellule ; l'opérateur du feu est le cas « aucune cellule
d'air » de celui de l'eau). MG ×4 mesuré 2,5× plus exact et 48 % plus rapide que
Jacobi 100 à 192³ ; Jacobi reste le repli (case / `?jacobi`). Instruments au HUD :
résidu de divergence (moyen/max) et compteur de particules rapides — il FAUT les
deux, un champ uniforme parasite est à divergence nulle et le résidu ne le voit
pas. Le réglage « niveaux multigrid » (1 = lisseur seul) sert à bissecter.
Résolution : `?grid=96|128|160|192` ou boutons du panneau — bindings vivants via
`setGridEau()` appelé avant toute création, particules = n³, grandeurs en
voxels/s multipliées par `SCALE_EAU`. Mesuré : 60 FPS jusqu'à 128³, ~50 à 160³,
~27 à 192³.
TROIS PIÈGES d'échelle, tous invisibles à 128³ (détail dans le journal PLAN-EAU) :
`maxStorageBufferBindingSize` (128 Mio par défaut) était frôlé à l'octet près et
bloquait 160³ — relevé au max de l'adapter dans gpu.ts ;
`maxComputeWorkgroupsPerDimension` (65535) était dépassé à 192³ — les passes de
particules dispatchent en 2D (`PARTICLE_ROW`) ; et l'échantillonnage du
recensement, à pas figé, écrasait son propre histogramme au-delà de 2 M de
particules. Quand une limite n'apparaît qu'en WARNING console, écouter
`Log.entryAdded` et juger l'IMAGE, jamais le compteur de FPS.

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
  (1 bis, 2026-08-30) LUEUR TEINTÉE : les ÉTINCELLES de feu d'artifice
  splattent leur couleur dans un tampon à la résolution de la lueur (3 × u32
  par cellule, virgule fixe ×1024, atomicAdd dans sparks3d.wgsl — remis à
  zéro par clearBuffer, l'update des étincelles précède la passe de lueur) ;
  l'injection l'ajoute au corps noir (layout dédié `glowInject`, les blurs
  restent sur celui de curl). Poids : const SPARK_GLOW (glow3d.wgsl) — le
  seul chemin du moteur où une lumière verte/bleue existe.
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
- **Champs de force posés dans la scène — « modificateurs » (2026-08-24)** :
  objets placés au pointeur, SAISISSABLES et supprimables, qui agissent
  localement sur le fluide. Ils réutilisent tel quel le patron des émetteurs et
  de la boule : `pickTarget` les reconnaît (encodage `FIELD_TAG + i`, soit 100+i,
  à côté de −1 pour la boule et 0..n pour les émetteurs), la saisie les déplace
  sur le plan face caméra, et deux vec4 par champ vivent dans l'uniforme
  (slots 64-91 ; le tampon sim est passé à 512 o).
  DEUX TYPES, traités différemment pour une raison PHYSIQUE :
   · **tourbillon** — rotation solide autour d'un axe. Un champ rotationnel est
     à divergence nulle, donc la projection le PRÉSERVE : il agit sur l'air
     lui-même, matière ou pas.
   · **vent local** — direction constante, donc irrotationnel : uniforme il
     serait annulé par la projection (leçon du souffle radial 2D). Il doit être
     DIFFÉRENTIEL, pondéré par la matière, comme la poussée.
  Réglages (type / force / rayon) appliqués aux FUTURS champs et à celui qu'on
  TIENT, jamais à distance — le même piège que l'encre des émetteurs, qui
  éteignait la flamme pilote. GIZMOS : coque lumineuse à la frontière du champ
  dans le raymarch (bleu = tourbillon, ambre = vent), coupée avec les retours
  visuels (F). La zone de SAISIE suit le rayon visible du champ (même bonus que la
  boule) : sans ça on cliquait sur le gizmo sans rien attraper. Étendre à un
  nouveau type = un cas dans `field_force` (forces3d.wgsl) et une entrée dans
  FIELD_NAMES.
  ⚠️ **`grabbed` est un index TAGUÉ** (−2 rien · −1 boule · 0..n émetteur ·
  100+i champ). Un `grabbed >= 0` naïf attrape aussi les champs : c'est le bug
  de l'écran noir du 2026-08-24 — l'encre était appliquée à `emitters[100]`,
  l'exception tuait la boucle de rendu et le canvas WebGPU cessait d'être
  présenté, symptôme parfaitement muet. Passer par `heldEmitter` / `heldField`,
  jamais par un test brut. Et depuis, une exception dans la frame est ATTRAPÉE
  et affichée par l'overlay au lieu de noircir l'écran (les deux pages 3D).
- **Couche de GIZMOS (`gizmo3d.wgsl`, 2026-08-24)** — la base d'outillage : une
  passe de LIGNES 3D dessinée après la présentation, directement sur le canvas.
  Dès qu'on sait tracer des segments dans le monde, tout repère se dessine
  (champs de force, émetteurs, boîte et manipulateurs y vivent tous).
  Trois choix la rendent solide :
   1. AUCUN tampon de géométrie — tout vient de `vertex_index` et de l'uniforme
      de rendu, comme les braises et les points d'eau : la doctrine zéro-alloc
      tient sans effort. Le compte `GIZMO_VERTS` (sim3d.ts) doit rester d'accord
      avec les constantes du shader.
   2. APRÈS le tone-mapping, pas dedans : les lignes gardent exactement leur
      couleur, sans être délavées par l'exposition ni bavées par le bloom. C'est
      ce qui sépare un repère d'un halo — la première version, une coque
      lumineuse dans le raymarch, ne ressemblait à rien.
   3. Pas de tampon de profondeur : les gizmos passent devant le volume, comme
      les repères de Blender. Un repère à moitié caché ne repère plus rien.
  Géométrie par champ : trois cercles orthogonaux (la « sphère vide » de
  Blender — donne le rayon d'action et se lit sous tout angle) et un axe fléché
  qui donne l'ORIENTATION. Le champ ACTIF passe en blanc opaque : la sélection
  se voit. Uniforme : deux vec4 par champ à partir du slot 36 de `renderData`
  (centre monde + rayon monde ; axe unitaire + type, +2 si actif).
  Mots réservés WGSL rencontrés ici : **`ref`** et **`active`** — la famille
  compte maintenant `from`, `target`, `move`, `smooth`, `ref`, `active`.
  **Trois familles de repères (2026-08-25)**, dans cet ordre le long de
  `vertex_index` — l'ordre est la seule chose qui les sépare, et `GIZMO_VERTS`
  doit rester d'accord avec les constantes du shader :
   - CHAMPS DE FORCE : ci-dessus. Slots 36-59 de `renderData`.
   - ÉMETTEURS : un anneau posé à plat AU VRAI RAYON D'ÉMISSION et une tige
     fléchée vers le haut, à la couleur de l'encre ; l'actif passe en clair.
     Deux trous bouchés : un émetteur posé loin de la flamme n'avait aucune
     trace visible tant qu'il n'avait rien allumé, et rien ne disait lequel des
     quatre les touches 1/2/3 allaient repeindre. Slots 60-75.
   - BOÎTE : les 12 arêtes du cube unité, très pâles. Les parois n'étaient
     visibles qu'à leurs effets. Slot 77.
   - POIGNÉES (manipulateurs) : trois flèches d'axe portées par l'objet
     SÉLECTIONNÉ. Slots 78-83.
   - BOUTON D'ORIENTATION : anneau face caméra au bout de l'axe d'un champ de
     force (les émetteurs et la boule n'ont pas d'orientation). Slots 84-87.
  Têtes de flèche BILLBOARDÉES (perpendiculaire prise dans le plan de l'écran) :
  une flèche vue dans l'axe de son propre plan se réduisait à un trait,
  précisément quand on l'orientait vers la caméra.
  `renderData` : 60 → 84 floats, tampon de rendu 256 → 512 o. Les autres shaders
  déclarent un `RenderParams` PLUS COURT (ils s'arrêtent à `style`) — c'est
  légal, un préfixe suffit tant que le tampon lié est au moins aussi grand.
- **Manipulateurs (poignées d'axe, 2026-08-25)** — glisser un objet le déplaçait
  sur le plan face caméra : trois degrés de liberté d'un coup, impossible de le
  monter sans le décaler. Saisir une flèche contraint le geste à SON axe. Quatre
  points qui ne s'improvisent pas :
   1. **La droite de contrainte est FIGÉE à la saisie** (`dragOrigin`). La
      recalculer depuis l'objet qui bouge fait boucler la mesure sur elle-même
      et l'objet dérive tout seul. Le point de prise est mémorisé aussi
      (`dragGrip`) : rien ne saute sous le curseur au premier pas.
   2. **Le centre reste libre** (`handleInner`, 30 % de la longueur). Sans ce
      vide, cliquer au milieu d'un objet tombe dans la zone de saisie des TROIS
      poignées à la fois et donne un déplacement contraint là où on voulait le
      libre. Le vide est transmis au shader (`opts.w`), pas recopié.
   3. **Longueur constante à l'ÉCRAN** (elle grandit avec la distance) : une
      poignée de taille fixe dans le monde est invisible de loin et démesurée de
      près. Calculée côté CPU et transmise (`sel.w`), pour que le dessin et la
      saisie partagent exactement le même nombre.
   4. `selected` (le porteur des poignées, toutes familles confondues) n'est PAS
      `activeEmitter`/`activeField` (deux curseurs par famille, cibles des
      réglages). Trois notions voisines : ne pas les fusionner à la légère.
  Vérifié par vrais événements souris CDP (`.selftest/cdp-gizmos.mjs`, scène en
  pause) : geste diagonal sur la poignée Y → seul Y bouge (Δ = 0 · 0,2306 · 0) ;
  même geste sur le corps → les trois axes bougent.
- **Orienter un champ (bouton d'orientation, 2026-08-25)** — la flèche d'un champ
  AFFICHAIT son axe sans qu'on puisse le changer : il était imposé par le type
  (vertical pour un tourbillon, horizontal pour un vent). Un bouton tiré à la
  souris donne les deux degrés de liberté d'une direction, d'un seul geste.
   1. **Le bouton coulisse sur une SPHÈRE** centrée sur l'objet, de rayon figé à
      la saisie. Quand le rayon du pointeur manque la sphère, on prend le point
      le plus proche de sa surface : le geste reste continu au-delà de la
      silhouette au lieu de décrocher net au bord. Quand il la traverse, on
      prend l'intersection AVANT — sinon l'axe bascule vers l'arrière au moindre
      tremblé. L'axe reste unitaire par construction.
   2. **Il est posé au-delà des flèches de déplacement** (`handleAim` = 1,5 fois
      leur longueur, donc constant à l'écran lui aussi). Le placer au bout de la
      flèche du champ le mettait PILE sur la pointe de la poignée Y — le
      tourbillon a l'axe vertical par défaut — et l'une mangeait l'autre.
   3. **Piège trouvé en chemin** : les réglages du panneau s'appliquent au champ
      TENU, et remettaient `axis` à sa valeur canonique à chaque frame de
      saisie. L'orientation était donc effacée dès qu'on retouchait le champ.
      L'axe ne se réinitialise plus que si le TYPE change.
  Les deux types acceptent un axe quelconque sans rien changer au solveur
  (`cross(axe, rel)` pour le tourbillon, `axe · force` pour le vent local).
  Vérifié CDP : axe (1,0,0) → (0,822 · −0,273 · −0,500), norme 1,00000, position
  inchangée, et l'axe survit à une nouvelle saisie du champ.
- **La sélection devient le modèle d'interaction (2026-08-25)** — elle n'était
  qu'un détail de dessin ; elle est maintenant la seule notion d'« objet
  courant ». Trois conséquences :
   1. **Une seule surbrillance.** `activeEmitter` et `activeField` (deux
      curseurs séparés, un par famille) ne servaient QUE ça, et donnaient trois
      objets en évidence pour une seule idée. Supprimés — la mise en évidence
      vient de `selected`. Les accesseurs `fieldCount`/`activeFieldIndex`
      étaient morts, partis avec eux.
   2. **X supprime ce qu'on a DÉSIGNÉ**, pas le dernier posé : un émetteur ou un
      champ du milieu était jusqu'ici impossible à effacer sans effacer ses
      voisins. Repli sur le dernier quand rien n'est sélectionné, ce qui garde
      l'ancien enchaînement A/X. Après un `splice`, les index au-delà glissent
      d'un cran : la sélection ET `grabbed` doivent tomber, sinon on manipule le
      voisin sans le savoir. La touche est routée vers la bonne famille par
      `selectedIsField`, et refuse honnêtement de retirer la flamme pilote.
   3. **Échap désélectionne** (`deselect()`). Cliquer dans le vide ne le fait
      PAS : le clic gauche dans le vide sert à orbiter, on perdrait les poignées
      à chaque coup d'œil autour de l'objet.
  Vérifié CDP : trois émetteurs d'encres [0,1,2], le MILIEU désigné à la souris
  puis supprimé → [0,2] (un « pop » du dernier aurait laissé [0,1]).
- **Vent horizontal (2026-08-24)** : force DIFFÉRENTIELLE proportionnelle à la
  matière, ajoutée dans `forces3d.wgsl` à côté de la poussée, avec un cap qui
  OSCILLE (uniform `wind` = force / amplitude / période / cap, slots 60-63).
  Pourquoi différentielle : un vent UNIFORME serait vain dans une boîte close —
  un champ constant viole la non-pénétration aux parois, la projection lui
  oppose un gradient de pression et l'annule presque entièrement (même leçon
  que le souffle radial du 2D, annulé parce qu'irrotationnel). Pourquoi
  oscillant : un cap fixe couche le panache une fois pour toutes, un cap qui
  tourne le fait onduler. COUPÉ par défaut (`windStrength: 0`) — la scène est
  alors strictement identique à ce qu'elle était sans la fonction ; le preset
  « 🌬 vent » l'allume d'un clic. Force calibrée par captures : 35 = panache
  haut, penché, avec traîne ; 55 = il commence à se disperser ; 95 = il est
  écrasé au ras de l'émetteur.
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
- **EXPLOSIONS (2026-08-25)** — le moteur en contenait DÉJÀ une : la combustion
  allume le carburant au-dessus d'un seuil de température, ce qui brûle chauffe
  et fume, et l'expansion au front de flamme est injectée comme **source de
  divergence** (`project3d.wgsl`) — que la projection convertit en gradient de
  pression, donc en vitesse sortante. C'est le mécanisme correct du souffle, et
  la raison pour laquelle une force radiale ne marcherait pas (elle est
  irrotationnelle, la projection l'annule — leçon du souffle radial 2D).
  Ce qui manquait n'était pas de la physique mais **l'ÉVÉNEMENT** : le moteur ne
  savait qu'émettre en continu. Une explosion = une bouffée de carburant + son
  amorce en chaleur, et la combustion existante fait tout le reste.
   1. **Injection à débit constant pendant une DURÉE fixe** (`explosionTime`),
      jamais « pendant une frame » : à débit fixe, une frame donne un résultat
      qui dépend du framerate. La même détonation rend la même boule à 30 comme
      à 120 FPS.
   2. **La charge doit être GRUMELEUSE.** Une gaussienne analytique est
      parfaitement symétrique, et une boule de feu parfaitement symétrique
      ressemble à une AMPOULE — les instabilités qui font l'aspect d'une
      détonation n'ont rien à amplifier. Un bruit de valeur à deux octaves,
      ancré sur le centre de la bouffée et décalé par une graine, module
      fortement le carburant (il dessine les langues de flamme) et faiblement
      l'amorce (le cœur doit partir à coup sûr, sinon la charge couve).
      Le coût ne se paie que dans la petite boule, pendant les ~3 frames
      d'injection.
   3. **Piège de calibrage** : le terme d'expansion lit `min(carburant, 1)` — il
      SATURE. Rendre la charge grumeleuse à moyenne constante réduit donc le
      souffle, puisque les creux passent sous la saturation sans que les bosses
      compensent. Il a fallu doubler la charge (26 → 55) pour retrouver la
      taille de boule d'avant le bruit, ET la déchirure en plus.
  Graine avancée d'un pas FIXE à chaque détonation : deux explosions ne sont pas
  jumelles, mais la même suite de détonations rejoue à l'identique — ce qu'on
  attend d'un simulateur dont on rebaie les sorties.
  Chronologie mesurée (`.selftest/cdp-boom.mjs`, planche datée) : amorce à 80 ms,
  boule déchirée à 260 ms, chou-fleur à 600 ms, champignon qui monte à 900 ms,
  suie ensuite. 60 FPS à 256³.
- **SUIE (2026-08-25)** — le nuage d'après-explosion restait rose pâle là où une
  vraie détonation vire au noir. La suie est le chaînon manquant, et elle se
  déduit du modèle existant sans réglage séparé : **une flamme qui manque d'air
  craque son carburant et noircit**. Une bougie brûle du carburant très dilué
  dans l'air libre et reste propre ; une charge concentre plusieurs unités de
  carburant dans un volume qui n'a qu'une unité d'oxygène et noircit. Le même
  terme donne les deux.
   - **Logement** : canal y des ESPÈCES (`species3d.wgsl`), qui était libre ET
     déjà advecté par cette passe — la suie voyage avec le gaz sans coûter un
     transport de plus. Le rendu lie la texture d'espèces en binding 4 : un
     sondage de plus par pas de marche ET par pas d'ombre.
   - **PIÈGE, coûté une itération complète** : la première loi mesurait le
     manque d'air par `1 − oxy_factor`. Or `oxy_factor = clamp(o2/0,25, 0, 1)`
     **SATURE à 1** dès qu'il reste un quart d'oxygène — le facteur valait donc
     zéro presque partout et la suie était invisible. Diagnostic par A/B sur un
     état FIGÉ (`.selftest/cdp-suie-ab.mjs`) : basculer la densité de suie de 0
     à 3 ne changeait rien à l'image, ce qu'aucune inspection de code n'aurait
     dit aussi vite. La bonne mesure est le rapport de ce que le carburant
     présent RÉCLAME (`carburant × O2_PER_FUEL`) à ce qui est disponible.
   - **La boucle doit se refermer par le RAYONNEMENT.** Avec la seule extinction,
     le nuage devenait opaque mais gardait sa chaleur : il émettait en orange
     sous une peau désormais impénétrable — une braise géante, pas de la suie.
     Or les particules de suie rayonnent bien mieux que les gaz (c'est ce qui
     rend une flamme riche lumineuse ET la refroidit vite) : `sootCooling`
     accélère le refroidissement en proportion de la suie présente. C'est CE
     terme qui fait virer le nuage au noir.
   - Extinction 46 par unité (contre 22 pour la fumée) et albédo quasi noir
     (0,055) : la suie absorbe au lieu de diffuser. L'albédo du voxel est un
     mélange pondéré par l'extinction que chacun apporte — là où il n'y a pas de
     suie, l'image est strictement celle d'avant.
   - `sootDensity = 0` rend exactement l'image d'avant la fonction. 60 FPS tenus
     à 256³ malgré le sondage supplémentaire.
- **CIEL OUVERT (bande éponge 3D, 2026-08-25)** — la boîte 3D était close de
  partout (Neumann par clamp) : tout ce qu'on injecte finit par la remplir. Un
  panache plafonne et repeint le plafond, une explosion sature le volume en deux
  secondes — ce qui ruine n'importe quel rendu. C'est le défaut structurel déjà
  nommé dans la section « articulation des modes ».
  La solution n'est PAS une vraie sortie Dirichlet avec face virtuelle : elle
  rend le système non symétrique et le multigrid diverge par construction (leçon
  payée en 2D, cf. « Pièges durement payés »). On garde la boîte fermée — donc
  l'opérateur de pression symétrique et le MG convergent — et on AMORTIT dans
  une bande près des parois, exactement comme `sponge()` en 2D : matière et
  vitesse s'y éteignent avant d'avoir pu s'accumuler.
  Deux différences avec le 2D : le **SOL est exclu** (c'est la seule paroi qui
  soit un objet de la scène — le raymarch l'éclaire et y projette les ombres ;
  une explosion doit rebondir dessus, pas s'y dissoudre), et la vitesse est
  amortie **à sa position MAC** composante par composante. Amortir la vitesse
  introduit de la divergence, que la projection de la frame suivante reprend :
  c'est le prix, et il est faible devant le jet qui longeait le plafond.
  `openBand = 0` referme la boîte et rend exactement le comportement d'avant.
  Mesuré : à 2,2 s l'explosion donne un vrai champignon à cœur noir au lieu
  d'une dalle rose collée au plafond. 60 FPS tenus.
- **PLAFOND DE LA MARCHE (2026-08-25)** — la boucle du raymarch était bornée en
  dur à `256u`, et le curseur « pas de marche » s'arrêtait exactement à 256 :
  rien de cassé côté UI, mais un plafond de QUALITÉ, devenu contraignant depuis
  qu'on vise des rendus plutôt que la frame d'un jeu.
  **Le symptôme, si on dépasse ce plafond, ne ressemble pas à un plafond** : le
  rayon s'arrête avant la sortie de boîte, tout ce qui est derrière disparaît, et
  le nuage s'ÉCLAIRCIT. Poussé à 560 pas par un script, ça se lisait comme « la
  suie s'évapore quand j'augmente la qualité » — un faux problème de rendu qui a
  coûté une hypothèse avant que la borne ne saute aux yeux. Le plafond est passé
  à 1024 et le curseur avec ; les deux doivent rester d'accord.
  Le GRAIN visible dans les zones de forte extinction vient du décalage
  aléatoire du départ de rayon (`hash12`, l'anti-bandes) : il est proportionnel à
  la longueur de pas, donc il s'efface en montant les pas.
  **Mesuré** (scène chargée, 256³) : 160 pas → 60 FPS · 320 → 57 · 560 → 51.
  La qualité d'image s'achète donc pour presque rien — c'est le SOLVEUR qui tient
  le budget, pas la marche. Le sondage de suie, lui, ne se mesure pas (160 pas
  avec et sans : 60 FPS dans les deux cas).
- **CHAMPIGNON ATOMIQUE (preset 🍄, 2026-08-25)** — le cas qui met le moteur à
  l'épreuve : il ne suffit pas de faire une boule de feu, il faut tenir un
  ANNEAU TOURBILLONNAIRE plusieurs secondes et lui faire traîner une colonne.
  Le moteur y arrive, à trois conditions qui ne se devinent pas :
   1. **Le pied n'est pas fait de la charge, il est fait du SOL.** Une bouffée en
      l'air donne un chapeau qui flotte sans colonne. La détonation est donc
      posée au ras du sol (`explosionHeight`) et soulève une galette de matière
      FROIDE autour du point d'impact (`dustRate`) : froide, elle ne monte que
      parce que la boule l'aspire — c'est exactement ce qui donne au pied sa
      lenteur et son étranglement.
   2. **« Atomique » n'appelle PAS une grosse charge.** Contre-intuitif et mesuré :
      à 170 de charge et 30 d'expansion, la boule occupe la moitié de la boîte
      dès 1,8 s et on obtient un couvercle de fumée. Dans un domaine de taille
      fixe, il faut une charge PETITE, pour lui laisser de la place à monter.
      Séparer les deux rôles est la clé : **l'expansion fait ENFLER, la poussée
      fait MONTER**. Le preset baisse donc l'expansion (12) et pousse la
      poussée thermique (330).
   3. **Le temps est ralenti** (timeScale 0,55), la dissipation de vitesse presque
      nulle et la vorticité forte : c'est ce qui garde l'anneau vivant assez
      longtemps pour qu'on le voie se former, au lieu d'un champignon mou.
      La bande éponge est resserrée (0,07) pour que le chapeau puisse s'étaler
      près du plafond au lieu d'y être mangé avant d'avoir pris sa forme.
  Piège de VISÉE trouvé en chemin : avec un plan de charge BAS, une caméra
  presque de niveau coupe ce plan très loin de la boîte, et rabattre le point sur
  la paroi la plus proche posait la bombe dans un COIN. Hors de la boîte, on
  retombe désormais au centre.
  Chronologie mesurée : boule à 0,3 s, colonne formée à 1,8 s, chapeau étalé à
  4,5 s, silhouette mûre tenue jusqu'à ~10 s. 59-60 FPS à 256³.
  Les presets sont exposés au harnais (`window.__presets3d` en `?selftest`) :
  `cdp-boom.mjs 9333 NUKE champignon` applique CE QUE VOIT L'UTILISATEUR plutôt
  qu'une copie des réglages, et bascule seul sur la chronologie longue.
- **CHAPEAU DIFFUS : deux causes, aucune dans le solveur (2026-08-25)** — le
  chapeau du champignon perdait sa structure passé quelques secondes. Diagnostic
  en rejouant le TIR ENTIER avec une seule variable changée (`cdp-nuke-ab.mjs`,
  capture au même temps SIMULÉ) — le flou est cumulatif, il ne se lit pas sur un
  état figé. Quatre suspects écartés par la mesure : dissipation d'encre (nuage
  plus dense, toujours flou), vorticité ×2 (le PIED gagne du détail, pas le
  chapeau), temps normal (pire), et l'idée que ralentir le temps multiplierait la
  diffusion numérique (réfutée : c'est l'inverse ici).
   1. **L'ÉPONGE lamine ce qui se gare dedans.** Un panache TRAVERSE la bande ;
      le chapeau d'un champignon, lui, monte au plafond et y RESTE. Une éponge
      calibrée pour absorber un passage le rabote alors sur place. Mesuré : boîte
      close, le chapeau retrouve ses lobes ; à 0,07/26 c'est une dalle ; à
      0,035/8 il garde ses lobes ET la boîte ne se remplit pas. Le preset
      champignon prend donc une éponge resserrée et adoucie ; le défaut général
      (calibré pour un panache continu) ne bouge pas.
   2. **La RÉSOLUTION DE RENDU baissait toute seule.** L'adaptation automatique
      tombe à 60 % dès que la scène s'alourdit — donc précisément au moment le
      plus dense, donc au moment où l'on regarde. Le flou qu'elle produit se lit
      à tort comme un défaut de simulation : un champignon « diffus » à 9 s
      l'était pour moitié à cause de ça. Verrou à 100 % (touche **L** ou
      `?lock`), état affiché au HUD. **À utiliser pour toute mesure d'image** —
      sans lui, deux captures ne sont pas comparables.
- **DEUX DOMAINES D'INCANDESCENCE (2026-08-25)** — la détonation « ne ressemblait
  pas à une bombe A », et la cause était un PLAFOND, pas un réglage : le canal de
  chaleur était écrêté à 2 et `blackbody` saturait à 1,7. Une flamme et une bombe
  y étaient donc **aussi lumineuses l'une que l'autre**, et aucune charge, si
  grosse soit-elle, ne pouvait le contourner — la chaleur était coupée avant
  d'arriver au rendu. `blackbody` a maintenant deux domaines : celui des FLAMMES
  (inchangé au poil près jusqu'à 1,7) puis celui des BOULES DE FEU, où l'émission
  part en loi de puissance et sature au blanc. Le plafond du canal devient
  réglable (`heatCeiling`, 2 par défaut = comportement d'avant ; 9 pour le
  champignon).
  **La poussée, elle, sature à 2** : le modèle de flottabilité est une
  linéarisation valable sur une plage étroite, et sans cette saturation ouvrir
  le domaine d'incandescence aurait multiplié la poussée d'autant. Cette chaleur
  RAYONNE, elle ne soulève pas.
- **LA CHARGE DÉPENDAIT DU FRAMERATE — donc de la RÉSOLUTION (2026-08-25)** —
  signalé par Renaud (« un problème en fonction de la résolution quand je lance
  une détonation »), et vérifié : c'était bien un bug, avec TROIS régimes fautifs
  selon le pas de temps.
  La bouffée injecte un débit tant que sa fenêtre est ouverte, et la fenêtre se
  fermait AVANT que la tranche partielle soit injectée :
   - à 60 FPS, la dernière tranche était perdue → 0,033 s injectés sur 0,05
     (**−33 %**) ;
   - aux framerates intermédiaires, une tranche de trop → jusqu'à **+67 %** ;
   - dès que `dt` atteignait la durée de la bouffée (grosses grilles, ou le
     plafond `dt = 1/30` avec `timeScale`), la fenêtre se refermait sans AUCUNE
     injection : **la détonation ne faisait rien du tout.**
  Correction : la tranche vaut `min(reste, dt)`, transmise aux débits en fraction
  de frame — la charge délivrée vaut alors exactement débit × durée, quel que
  soit le pas de temps. La fenêtre ne s'écoule plus en PAUSE non plus (sinon la
  charge se consume pendant qu'on regarde). Vérifié en changeant `timeScale`
  (0,55 / 1 / 1,65), ce qui change `dt` à framerate égal exactement comme le fait
  un changement de résolution : `.selftest/cdp-dt-ab.mjs`, même temps SIMULÉ.
  **Règle à retenir** : toute injection à débit sur une fenêtre doit borner sa
  tranche par ce qui reste de la fenêtre. Le symptôme est perfide — la scène ne
  plante pas, elle rend simplement une explosion plus faible, plus forte, ou
  absente, selon la machine.
- **CE QUI SÉPARE UN GROS FEU D'UNE BOMBE (2026-08-25)** — trois manques, tous
  identifiés en regardant ce qu'une bombe fait et que le moteur ne faisait pas.
   1. **Le PIED doit être ALIMENTÉ, pas injecté d'un coup.** La poussière avait
      la durée de la charge (50 ms) : la colonne était donc un paquet FINI que la
      montée étirait puis rompait — chapeau et pied finissaient en deux nuages
      séparés. Le courant ascendant, lui, continue d'arracher le sol tant qu'il
      monte. Fenêtre d'arrachement SÉPARÉE (`dustTime`, 5 s pour le champignon
      contre 50 ms par défaut) : le pied devient continu.
   2. **Deux parois, deux rôles OPPOSÉS.** La nappe qui s'étale au sol TRAVERSE
      les parois latérales ; le chapeau, lui, SE GARE au plafond. Les traiter
      avec la même sévérité obligeait à choisir entre un chapeau laminé et une
      boîte qui s'empâte. Bandes séparées : parois franches (elles évacuent la
      nappe, plus rien d'intéressant ne s'y gare), plafond très doux.
   3. **La silhouette est HAUTE et étroite.** Le rapport ne s'obtient qu'en
      laissant au nuage beaucoup de hauteur devant lui par rapport à son propre
      diamètre : charge deux fois plus petite (rayon 0,042), poussée montée à
      430. Une charge plus grosse donne un chou-fleur trapu qui touche le
      plafond avant d'avoir formé un pied — la forme d'une explosion ordinaire.
  **Limite de fond, non contournée** : le domaine est un CUBE, alors qu'un vrai
  champignon est bien plus haut que large. Le rapport d'aspect est donc plafonné
  à ~1:1 ; on en tire une silhouette juste, pas la démesure d'une photo d'essai.
  Aller plus loin demanderait un domaine anisotrope (grille plus haute que
  large), donc un pas de grille différent selon l'axe — dans l'advection, le
  laplacien et le rendu. Chantier réel, non entrepris.
- **CHARGES MULTIPLES (2026-08-29)** — le slot d'explosion unique devient
  QUATRE charges en vol (`maxBursts`, config3d) : tirer n'ampute plus la charge
  précédente — une salve donne une boule incandescente fraîche À CÔTÉ du nuage
  déjà noirci du tir d'avant (banc `cdp-salve.mjs`). Ce qui ne se devine pas :
   1. **Chaque charge porte SES DEUX fenêtres** (injection ET poussière) : un
      champignon garde son pied alimenté 5 s même si on tire ailleurs pendant.
      Le débit de poussière est passé PAR CHARGE (`bursts[2k+1].z` — `dust.x`
      libéré) ; rayon/épaisseur de la galette restent des réglages communs.
   2. **Attribution de slot** : un slot ÉTEINT de préférence (les deux fenêtres
      consumées), sinon le plus ancien (rang de tir `stamp`) — un bombardement
      recycle ses charges mortes avant d'amputer une charge en vol. La graine
      reste GLOBALE, avancée d'un pas fixe par tir : la même suite de
      détonations rejoue à l'identique, slots ou pas.
   3. **Le RESET éteint les fenêtres.** Avant, le tir suivant écrasait l'unique
      slot ; avec plusieurs, la poussière d'un champignon re-cliqué aurait
      survécu au reset et arraché le sol de la scène NEUVE au point de l'ancien
      impact. Positions et graine restent (inertes).
   4. **Uniforme** : tampon sim 512 → 640 o, charges en `array<vec4f, 8>` aux
      floats 116-147 (2 vec4 par charge), les slots 92-99 de l'ancien slot
      unique sont LIBRES. Seul advect_density3d lit les charges — les autres
      shaders n'ont pas bougé (un struct préfixe suffit). Le drapeau binaire
      « injection en cours » a disparu : un débit nul gate le bloc.
   5. **Coût : nul, mesuré** (`cdp-profil384.mjs`) : aval 7,89 ms contre 7,9 de
      référence — la boucle des 4 charges est en contrôle UNIFORME, les groupes
      sautent les blocs d'un même pas quand rien n'est en vol.
  C'est la « capacité moteur nouvelle » que la note de la démo demandait — et
  Renaud a relancé la démo dessus le jour même (entrée suivante).
- **DÉMO RELANCÉE : BOMBARDEMENT SUPERPOSÉ + BARRAGE D'ARTILLERIE (2026-08-29,
  demande de Renaud)** — l'acte V passe de trois tirs isolés en cuts à DEUX
  PLANS DE TROIS CHARGES SUPERPOSÉES, et l'acte VI s'ouvre sur un barrage
  d'artillerie contre l'horizon avant le champignon. Ce qui a été payé en
  captures :
   1. **Chaque tir de l'ancienne démo était un tir CENTRÉ.** La visée au
      pointeur (`shoot(ndc)`) retombe au centre dès que le plan de charge est
      bas et la caméra rasante — le piège de visée documenté, et il valait pour
      TOUS les tirs scriptés. Remplacée par `sim.explodeAt(nx, ny, nz, input)`
      (pilotage scripté, pendant de driveSphere/addEmitterAt — fractions de N,
      même convention que `explosionHeight`) : le réalisateur pose ses marques,
      il ne vise pas. Les bancs y gagnent le même pistolet.
   2. **À trois charges par boîte, les réglages « spectacle » d'un tir unique
      saturent.** Calibre 0,06/48 (le volume injecté par VOLÉE ≈ un tir
      d'avant), amorce 55 (à 90 + lueur 2,2, chaque boule jeune brûlait en
      ampoule blanche sans texture), lueur 1,6, suie 10 — le contraste de
      l'acte est là : une boule d'or fraîche devant le nuage NOIRCI du tir
      précédent. Dissipation 0,35 entre les charges.
   3. **Le FEU D'ARTIFICE aérien a été essayé ici et REMPLACÉ** — verdict de
      Renaud : « pas top top, pas beaucoup de couleur, un peu plat ». Le
      diagnostic tient en une phrase : la lumière des explosions est du corps
      noir (or/orange/blanc) — pas de couleurs pyrotechniques sans charges
      d'ENCRE, une capacité qui n'existe pas. Leçon gardée au passage : un feu
      d'artifice se regarde d'en bas (à élévation 0,2 la caméra dominait le
      plafond de la boîte, tout éclat passait sous l'horizon). Remplacé par le
      BARRAGE D'ARTILLERIE — des impacts au sol, ce que le moteur fait le
      mieux — et le barrage a ses trois leçons, toutes capturées :
       · les impacts se GROUPENT au centre de l'empreinte (0,38-0,65) — étalés
         sur tout le sol, l'union des colonnes ÉPOUSE la boîte, parois
         verticales taillées par l'éponge, la pire image contre un horizon
         ouvert ;
       · DEUX SALVES de trois, coupées d'un cut (reset + saut d'azimut —
         dehors, un autre plan du même champ de bataille) : même groupés, cinq
         impacts d'affilée finissent en dalle ; la leçon de l'acte V vaut
         dehors ;
       · l'INVERSION (l'outil du champignon) arrête les colonnes aux deux
         tiers — c'est le plafond PLAT qui trahissait la boîte. Cadence
         irrégulière (la régularité d'un métronome tue l'artillerie),
         poussière à fenêtre courte mais réelle (c'est elle qui fait « obus »
         et pas « pétard »).
      Second verdict de Renaud sur cette version : « de grosses explosions
      GRASSES » à côté d'un bouquet « fin ». Le gras avait UNE cause : la
      dissipation à 0,42 — mise là pour vider la boîte — FONDAIT les
      filaments. La finesse du champignon n'est pas sa taille, c'est son
      moteur de réglages : temps ralenti (la turbulence se développe sous
      l'œil), dissipations quasi nulles, vorticité forte, suie sombre qui
      dessine, charge PETITE devant le domaine. Le barrage repart donc de
      TUNE_MUSHROOM tel quel et ne change que ce que six impacts imposent :
      dissipation 0,12 (contre 0,02 — six nuages coexistent, le cut fait le
      reste), suie 20, temps 0,6, éponges musclées, charges au calibre du
      champignon (BOOM_MUSHROOM, amorce 130, poussière 2,2 s). Règle à
      retenir : POUR QU'UNE EXPLOSION SOIT FINE, PARTIR DU MOTEUR DU
      CHAMPIGNON et n'en dévier que sous contrainte mesurée.
   4. **Deux états qui fuyaient d'un acte à l'autre** : le gizmo de l'émetteur
      pilote trônait au milieu des boules de feu (repères coupés dès l'acte V,
      rendus par la boucle et la sortie) ; et après le premier tour, la boule
      de l'acte IV restait désactivée — le lemniscate tournait à vide (l'acte I
      la rallume). La boucle passe à ~107 s (bouquet décalé à 82).
  Vérifié par `cdp-artifice.mjs` (chronologie des trois segments + tour 2 de
  boucle), 60 FPS constants à 256³. Le champignon extérieur du bouquet est à
  parité avec sa référence EXT256-*.
- **Prochaines briques** : séquences VDB animées (File System Access API) ;
  encres colorées (canaux yz réservés dans la densité) ; qualité du confinement
  de vorticité (flouter |ω|).

## Articulation des trois modes — ANALYSE, PLUS UNE FEUILLE DE ROUTE

> Le chantier de l'eau est CLOS depuis le 2026-08-25. Ce qui suit garde sa
> valeur d'analyse (ce qui se mutualise, ce qui ne doit surtout pas fusionner,
> pourquoi le couplage a échoué) mais **aucun de ses points numérotés n'est à
> entreprendre**. Écrit le 2026-08-23, après le merge de l'eau :

Le projet a maintenant TROIS pages : `index.html` (fluide 2D), `3d.html` (feu
volumétrique), `eau.html` (liquide APIC). Elles partagent déjà `gpu.ts`,
`pipelines.ts`, `overlay.ts`, `panel3d.ts` et la doctrine (zéro alloc/frame,
zéro readback en boucle, cœur sans DOM). Ce qu'elles ne partagent PAS encore,
et ce que ça coûte :

**Duplication réelle (mesurable).** L'eau a ré-écrit, en les adaptant : la
caméra orbitale et la construction de rayon, le hitTest/saisie d'objet, la
boule-obstacle analytique et son bord mobile, la boucle de frame + HUD +
`?selftest`, le sélecteur de résolution. À chaque fois c'était une transposition
fidèle du feu, donc sans surprise numérique — mais c'est du code en double qui
divergera.

**Ce qui n'est PAS à unifier.** Les solveurs. Le feu est un gaz compressible-
par-expansion en boîte fermée ; l'eau est un liquide à surface libre avec
marqueurs air/eau/solide. Fusionner les deux passes de projection produirait un
opérateur plein de branches, plus lent et plus fragile que les deux séparés. La
2D est encore plus à part (autre dimensionnalité, autre modèle d'outils).

**Ordre recommandé (du plus rentable au plus spéculatif) :**

1. ~~**J4 — généraliser le multigrid du feu à l'eau.**~~ **FAIT** (2026-08-23) :
   `eau_mg.wgsl`, MG ×4 par défaut, 2,5× plus exact et 48 % plus rapide que
   Jacobi 100 à 192³. La thèse « c'est le même opérateur » s'est vérifiée.
   L'idée d'ancrer le mode constant pour autoriser une restriction de masques
   permissive a été implémentée puis RÉFUTÉE par la mesure (journal J4) : le
   domaine fluide grossier déborderait le domaine fin, ce qu'aucune
   régularisation ne rattrape. L'ancrage est conservé (ε = 0,1) comme assurance
   contre les poches d'eau closes. La règle conservatrice est la bonne.
2. **Extraire une coquille 3D commune** (`platform/web/scene3d.ts`) : caméra
   orbitale + rayon + saisie, boucle de frame, HUD, selftest, sélecteur de
   résolution, navigation. Aucun changement visible, mais J6 (presets, démo,
   retours visuels de l'eau) devient presque gratuit puisque le feu a déjà
   `Panel3D`, `Toolbar3D`, `DemoDriver` et les toasts.

   ⚠️ **L'étage 2 du couplage a été ESSAYÉ (2026-08-24, branche `melange`) et
   SUPPRIMÉ — verdict utilisateur : « catastrophique ».** L'architecture était
   pourtant juste et l'a prouvé : la phase gazeuse n'a demandé aucun second
   solveur (même opérateur, masque inversé — l'eau devient un solide mobile —,
   V-cycle partagé tel quel, débit prescrit = la vitesse d'eau projetée), l'eau
   est restée intacte (résidu 0,005–0,02, 0 perdue) et le tout tournait à
   60 FPS à 128³. **C'est la QUALITÉ DE LA FUMÉE qui a échoué**, et la cause est
   claire rétrospectivement : la phase gazeuse était un solveur de fumée
   appauvri par rapport à celui du feu — advection de vitesse semi-lagrangienne
   (donc très diffusive, aucune structure de tourbillon), pas de vorticity
   confinement, 128³ au lieu de 256³, et une pression de gaz sous-convergée
   (4 V-cycles partagés avec l'eau). À quoi s'ajoute un fait structurel : dans
   une boîte quasi close, une émission continue FINIT TOUJOURS par remplir
   l'espace — le feu a le même défaut, mais là-bas la fumée est le sujet.
   **Leçon : le couplage ne se juge pas sur la plomberie mais sur la qualité de
   la phase la plus faible.** Refaire l'étage 2 n'a de sens qu'après avoir donné
   au gaz la VRAIE machinerie du feu (MacCormack sur la vitesse, vorticité,
   résolution) — c'est-à-dire après le point 2 ci-dessus, pas avant. Ne pas
   relancer sans décision explicite.
3. **Feu + eau sur UNE page à deux modes** — seulement APRÈS 2, et seulement si
   on vise le point 4. Tant qu'ils ne se parlent pas, deux pages coûtent moins
   cher qu'un sélecteur de mode.
4. **Le couplage (vapeur)** : l'eau qui s'évapore au contact de la flamme, la
   vapeur qui monte dans le système d'espèces du feu. C'est LE morceau
   spectaculaire — et un vrai chantier : une seule grille pour les deux, un
   solveur qui accepte deux phases de densités très différentes, un rendu qui
   compose liquide opaque et volume participatif. À ne PAS commencer sans
   décision explicite : c'est exactement le genre d'excursion séduisante qui a
   fait dériver le projet par le passé (marbrure, pipeline Blender).

Un couplage PARTIEL, beaucoup moins cher, existe si l'envie vient sans le
budget : la fumée du feu voyant la surface de l'eau comme un obstacle mobile
(sens unique, aucune modification du solveur d'eau).

## Pistes suivantes

La DÉMO (D) : arrêtée le 2026-08-28 (« pas super super » — l'état de la boîte
close en cuts, journal dans le commit c1adbd1), RELANCÉE par Renaud le
2026-08-29 sur la capacité charges multiples : acte V en volées superposées,
acte VI feu d'artifice + bouquet (entrée « DÉMO RELANCÉE » du chantier 3D).
En attente de son verdict sur cette version.

Le chantier de l'EAU est **CLOS** (décision du 2026-08-25) : `eau.html` reste en
ligne tel quel, J5 et J6 de PLAN-EAU ne seront pas faits, et l'articulation des
modes décrite plus haut tombe avec — sa coquille 3D commune n'avait de sens que
pour faire hériter l'eau de l'outillage du feu. La section qui la décrit reste
comme ANALYSE (ce qui se mutualise, ce qui ne doit surtout pas fusionner), pas
comme feuille de route.

Le chantier PERF 384³ est **CLOS** (décision du 2026-08-29) : ~25 FPS au défaut
(27 pendant une explosion de spectacle) est accepté comme régime de croisière —
les 60 FPS auraient demandé de diviser la frame par 2,3, hors de portée du
grattage. Son journal : le solveur est FAIT (rouge-noir ×2, journal du
2026-08-27) ; première passe sur le reste FAITE (2026-08-28 : vorticité en
mi-résolution −4 ms, occlusion des braises par particule −4 ms, ombre du
raymarch réutilisée un pas sur deux −4 ms sur scène DENSE — défaut ~21 FPS,
fournaise 19,5 → 21,3). Confinement à une évaluation et
espèces fusionnées dans le correcteur : FAITS (~23 FPS au défaut). PROFILEUR GPU
par passe : FAIT (?profile — timestamps matériels, table au HUD et dans
__sim3d.profileMs ; les toggles rAF sous-estimaient la pression de moitié). La
VRAIE table à 384³/défaut : pression 17,7 ms · amont (advection vitesse + forces
+ vorticité) 13,8 · aval (gradient + densités) 7,5 · divergence 2,0 · raymarch
2,6 · reste ~0,3 — total 44. Pression ATTAQUÉE
(2026-08-28 soir : restriction fusionnée au résidu — zéro texture intermédiaire,
260 Mo de VRAM rendus — et pré-lissage fin des cycles enchaînés sauté) :
17,7 → 13,5 ms, total 44,0 → 39,2, ~25 FPS au défaut, 27 FPS mesurés PENDANT une
explosion de spectacle (captures REVE-384-*). À 512³ : 11 FPS (pression 33,
amont 32 — la table ?profile y est encore plus parlante). Advection vitesse ATTAQUÉE
(même soir) : forces FUSIONNÉES dans le correcteur MacCormack (termes locaux —
la passe séparée payait un aller-retour vitesse de 450 Mo, forces3d.wgsl
supprimé) et les traces avant/arrière d'une même face partagent leur première
évaluation de vitesse (même valeur dans les deux formules RK2) : amont
13,8 → 12,5 ms, résultat identique à la réassociation flottante près, les trois
jauges re-passées. Le solde de la frame à 384³ : pression 13,7 · amont 12,5 ·
aval 7,9 · divergence 2,0 · raymarch 2,8 ≈ 39 ms. Rouvrir demanderait du
STRUCTUREL — advection à échantillons partagés entre faces (quality gate
serré), tuiles creuses, ou l'aval (7,9 ms, seul gros poste jamais fusionné) —
et une décision explicite. Se souvenir, si réouverture : mesurer au PROFILEUR,
plus aux toggles ; les gains de rendu se jugent sur une scène DENSE (le panache
par défaut a trop peu de pas occupés).

Pistes restantes, côté feu : qualité du confinement de vorticité · test
smartphone (1024², limites WebGPU mobiles). Les charges multiples : FAITES
(2026-08-29, entrée « CHARGES MULTIPLES » du chantier 3D).

**FEUX D'ARTIFICE COLORÉS — commande de Renaud (2026-08-29), les NÉCESSITÉS.**
Le feu d'artifice aérien a échoué d'abord par la COULEUR : la lumière d'une
explosion est du corps noir (or/orange/blanc) et sa fumée porte l'albédo des
trois matières — il n'existe AUCUN chemin pour du vert ou du bleu
pyrotechnique aujourd'hui. Ce qu'il faudrait, en deux étages :
 1. **CHARGES D'ENCRE (l'étage bon marché — 2-3 teintes fixes).** Une
    détonation qui injecte de l'ENCRE (canaux x/y de la densité, palette
    INK_COLORS) au lieu de fumée+carburant, avec son amorce en chaleur pour le
    souffle. Le RENDU est déjà prêt (albédo mélangé par voxel). Nécessite : une
    « encre de la charge » PAR SLOT — burst_b est plein (carburant/amorce/
    poussière/graine), donc passer à 3 vec4 par charge (tampon sim 640 → 768 o)
    — et un cas dans l'injection d'advect_density3d. Limite assumée : 2-3
    teintes, pas une palette.
 2. **TEINTES ARBITRAIRES (le vrai feu d'artifice).** La grille de densité n'a
    que 3 canaux à albédos FIXES : deux teintes différentes au même voxel sont
    structurellement impossibles. La voie grille (2ᵉ texture de densité, ou
    champ de teinte advecté) coûte cher et reste sale aux mélanges. La voie
    PROBABLEMENT juste : des ÉTINCELLES COLORÉES par charge, famille des
    braises — particules additives, occlusion par particule, la couleur vit
    SUR la particule et pas dans la grille, mélanges propres par construction.
    Un feu d'artifice est d'ailleurs PHYSIQUEMENT ça : des particules
    incandescentes, pas un gaz coloré. Nécessite : un tampon d'étincelles par
    teinte OU une teinte par particule (un float de plus dans le layout), des
    naissances liées à la charge (position + kick radial) plutôt qu'à la
    chaleur ambiante, et une variante de embers_draw sans corps noir (couleur
    prescrite). La LUEUR resterait or (le volume de lueur injecte l'émission
    corps noir) — une lueur teintée demanderait d'y injecter émission ×
    albédo local, à juger à l'image.
Et pour la FORME, la leçon du 2026-08-29 vaut d'avance : partir du moteur de
réglages du champignon (fin), pas des défauts (gras).
(« Encres colorées » a longtemps traîné ici : c'est FAIT depuis le 2026-08-21,
commit 7cefd12 — touches 1/2/3, fumée/encre/carburant, albédo mélangé par voxel.
La ligne datait d'avant ce commit et a survécu à une réécriture de la section.)
