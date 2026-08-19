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

## Perf mesurée (2026-08-20, desktop de référence)

512² et 2048² : 60 FPS (vsync) · 4096² : 32–60 FPS, ~1,6 Go VRAM.
Config actuelle : sim 1024², dye/scène 2048².

## Pistes suivantes

Vue debug du résidu multigrid par niveau · test smartphone (1024², vérifier les limites
WebGPU mobiles) · sim 3D ~128³ + rendu volumétrique (la vraie « vue de côté »).
