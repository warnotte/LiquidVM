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

## Chantier 3D (branche `3d`)

- **Page séparée `3d.html`** (`src/platform/web/main3d.ts`) — l'appli 2D n'est pas
  touchée. `src/core3d/` : même doctrine que core/ (zéro DOM, zéro alloc/frame, un
  encoder, bind groups pré-créés), `config3d.ts` (GRID3 = 128) + `sim3d.ts` (tout
  l'orchestrateur) + `shaders3d/`.
- **État** : 256³ (GRID3, forces remises à l'échelle par SCALE3 = GRID3/128 pour
  garder le comportement physique), MAC 3D MacCormack, TROIS ENCRES COLORÉES
  (canaux xyz de la densité, palette INK_COLORS bleu/magenta/ambre dupliquée dans
  raymarch.wgsl, touches 1/2/3 = encre de l'émetteur via blow_force.w, albédo
  mélangé par voxel au rendu), poussée thermique, projection MULTIGRID 3D
  (V-cycle GRID3→8³, 2 cycles/frame — pin p(0,0,0)=0 au 8³ appliqué d'office,
  le clamp des voisins EST l'opérateur Neumann exact, adjoint du couple
  divergence/gradient compact ; repli Jacobi comparatif via
  `__frame3d.multigrid=false`), émetteur de panache balancé, ray-marching
  (Beer-Lambert + corps noir + ombre interne 6 pas + liseré de boîte), souffle au
  clic droit, caméra orbitale, E = export VDB (encodeur généralisé multi-nœuds 16³,
  jusqu'à 4096³ ; density = somme des encres). 60 FPS à 256³ desktop, VDB 256³
  re-validé dans Blender.
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
- **MacCormack 3D** : prédicteur/correcteur clampé par composante MAC (stencil du
  point rétro-advecté) — LE raffineur : le panache laminaire devient turbulent et
  structuré à lui seul.
- **Confinement de vorticité 3D** : implémenté (ω vectoriel aux centres, F = ε N̂×ω
  par face) mais DÉFAUT ε = 0 — à 128³, le gradient de |ω| à ±1 voxel injecte du
  grain à l'échelle de la grille dès ε≈3 et détruit le panache vers ε≈10 (calibré
  par captures EPS-0/3/6/10). Réglable à chaud : `__frame3d.vorticityStrength`.
  Avant de le réactiver : lisser |ω| (flou 3D) avant d'en prendre le gradient.
- **Souffle au pointeur** : clic droit + glisser = force gaussienne dans un TUBE le
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
