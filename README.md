# LiquidVM

Simulation de fluides 2D temps réel en WebGPU pur — solveur eulérien incompressible sur
**grille décalée MAC** (pression aux centres de cellules, composantes de vélocité sur les
faces : les opérateurs divergence/gradient sont compacts et adjoints, aucun mode en
damier ne survit à la projection) : advection MacCormack d'ordre 2, vorticity confinement
réglable, résolutions configurables en une constante (`GRID_SIZE`/`DYE_SIZE`, défaut
1024² pour la vélocité et 2048² pour les densités), trois fluides indépendants
(eau, encre, fumée), murs dessinables à la souris, frontières fermées / périodiques
(tore) / ouvertes, éclairage par gradient de densité + bloom HDR, vues de debug des
champs internes, tout l'état résidant sur GPU. TypeScript strict + Vite, zéro dépendance
runtime, shaders WGSL en fichiers séparés.

**Démo en ligne** : [warnotte.github.io/LiquidVM](https://warnotte.github.io/LiquidVM/) —
chaque push sur `main` redéploie automatiquement (GitHub Pages).

## Simulation 3D volumétrique (`3d.html`)

Deuxième page de l'appli (bouton « 🧊 3D » de la barre d'outils, ou
[/3d.html](https://warnotte.github.io/LiquidVM/3d.html)) : solveur MAC 3D 256³
(128³–320³ via `?grid=`), advection MacCormack à traces RK2, projection multigrid
V-cycle, combustion à trois réactifs (carburant / chaleur / oxygène) avec expansion
volumique au front de flamme et refroidissement radiatif T⁴, trois encres colorées,
sphère-obstacle déplaçable à condition de bord mobile (elle brasse le fluide),
multi-émetteurs, souffle au pointeur, rendu ray-marching (Beer-Lambert, corps noir,
ombres volumétriques, sol), export OpenVDB (touche `E`, validé dans Blender), mode
démo chorégraphié (`D`), panneau de réglages déclaratif (`Tab`, toute la physique à
chaud). Même contrat que le 2D : état 100 % GPU, zéro readback en boucle de frame,
`src/core3d/` portable sans DOM.

## Commandes

```sh
npm i          # installe les outils (vite, typescript, @webgpu/types — dev uniquement)
npm run dev    # serveur de dev → http://localhost:5173
npm run build  # typecheck strict + bundle de production dans dist/
```

Navigateur requis : Chrome/Edge récents (WebGPU actif par défaut), ou Firefox/Safari
avec WebGPU activé. Sans WebGPU, la page affiche un message d'erreur explicite.

Pour reprendre le développement (carte du code, workflow de vérification, pièges
connus) : voir [NOTES-DEV.md](NOTES-DEV.md).

## Contrôles

| Entrée / Geste | Effet                                                        |
| -------------- | ------------------------------------------------------------ |
| glisser (souris / doigt) | applique l'outil actif (injection, souffle, tourbillon, gomme, mur) |
| clic droit / outil « mur » | construit un mur sous le pointeur (fonctionne aussi en pause) |
| maj + clic droit / outil « gomme mur » | gomme les murs                                       |
| `1` / `2` / `3` (ou barre d'outils) | choisit le fluide : eau (coule), encre (neutre), fumée (monte) |
| `T` (ou barre d'outils) | cycle les outils : injecter → gommer → tourbillon → souffle → mur → gomme mur |
| `B` (ou barre d'outils) | frontières : parois → périodique (tore) → ouvert (ce qui sort disparaît) |
| `V` (ou barre d'outils) | vue : fluides → vélocité → pression → divergence résiduelle → vorticité |
| `Tab` / bouton ⚙️ | panneau de réglages : viscosité, vorticité, pression, splat, temps, MacCormack, exposition, bloom |
| `N`            | avancer d'une frame exactement (pendant la pause)             |
| `Espace` / bouton ⏸ | pause (le rendu continue)                                 |
| `R` / bouton ↺ | reset des champs de fluide (les murs sont conservés)          |
| `X`            | efface tous les murs                                          |
| `+` / `-`      | itérations de pression ±5 (défaut 30, bornes 5–100)           |

**Barre d'outils tactile (bas de l'écran)** : déployable / rétractable avec ▲/▼, elle donne accès à tous les fluides, outils, modes de frontière, vues et réglages sans clavier physique.

**Webcam (panneau Tab)** : « caméra » — le flux optique convertit tout mouvement devant
la webcam en forces sur le fluide (encart de contrôle en haut à gauche : image + flux,
sliders force/seuil) ; « mains » — MediaPipe suit votre index (curseur-anneau à l'écran)
et le pincement pouce-index presse l'outil actif : toute l'interface se pilote au geste.
Le modèle (~8 Mo) n'est téléchargé qu'à l'activation ; sans caméra, tout le reste
fonctionne normalement.

Overlay (bas gauche) : taille de grille, fluide actif, itérations, FPS — mis à jour 2×/s.

## Schéma d'une frame

Un seul `CommandEncoder`, une seule compute pass (un dispatch = un usage scope, la
synchronisation entre étapes est implicite), une soumission :

1. *(si reset / X)* clear des champs de fluide / du champ de murs
2. *(si clic droit)* **pinceau à murs** : storage `r32float` read-write, pas de ping-pong
3. **Advection MacCormack de la vélocité** (Selle et al. 2008), composante par
   composante à sa position de face MAC (le vecteur y est reconstruit par
   échantillonnage demi-décalé de l'autre composante) : prédicteur semi-lagrangien →
   scratch, puis correcteur d'ordre 2 φ' = φ̂ + ½(φ − φ̃), clampé aux extrema du stencil
   bilinéaire d'origine (aucun overshoot ; désactivable dans le panneau pour comparer)
4. **Forces** : buoyancy par fluide + impulsion souris
5. **Vorticity confinement** : rotationnel puis force ε·ω·(N.y, −N.x) qui réinjecte
   les petits tourbillons dissipés par la diffusion numérique (Fedkiw 2001)
6. **Divergence** MAC compacte : div = u(i+1)−u(i) + v(j+1)−v(j) par cellule
   (`r32float`) — l'opérateur voit toute la divergence représentable
7. **Jacobi ×N** : pression en ping-pong, warm start depuis la frame précédente
8. **Soustraction du gradient** compact (adjoint exact de la divergence), faces
   bloquées forcées à zéro, puis recalcul de la **divergence résiduelle** (vue de
   debug 3) — les hautes fréquences y sont ≈ 0 ; le résidu basse fréquence restant est
   la limite de convergence de Jacobi (montez les itérations, ou un jour : multigrid)
9. **Advection MacCormack des densités** sur la grille dye 1024² + injection souris —
   la résolution visible est celle du dye, le coût de la projection celui de la grille
   de vélocité ; la vélocité est échantillonnée en normalisé, les grilles sont découplées
10. **Rendu en trois étages** : (a) scène — triangle plein écran vers une texture HDR
    fixe 1024², composition linéaire des fluides avec éclairage par gradient de densité
    (pseudo-normale → diffus + spéculaire), ou vue de debug ; (b) bloom (vue fluides
    uniquement) — seuil + downsample ½ et ¼, flou gaussien séparable en compute ;
    (c) présentation vers le canvas — scène + bloom, tone-mapping expo, fond sombre,
    murs en ardoise, gamma. Vues de debug (touche V) : vélocité en roue chromatique,
    pression/divergence/vorticité en colormap divergente — sans bloom ni tone-mapping.

Frontières (touche B) : **parois** — voisins clampés + réflexion de la composante
normale aux bords ; **périodique** — coordonnées wrappées (`%`) et sampler `repeat`
(simple sélection de variante du bind group 0 à l'encodage) ; **ouvert** — boîte fermée
+ **bande éponge** de 16 texels qui amortit exponentiellement vélocité et densité aux
bords : ce qui les atteint disparaît. L'opérateur de pression du mode ouvert est ainsi
identique à celui des parois (Neumann symétrique) — une vraie sortie Dirichlet avec
faces virtuelles extrapolées rendrait le système non symétrique et ferait diverger le
multigrid. Les murs dessinés sont des obstacles statiques : vélocité nulle, Neumann
pour la pression, free-slip tangentiel, aucune densité ne s'y advecte.

Sous-pas : si `dt > 1/45 s`, la frame est découpée en 2 sous-pas (slots d'uniforms
sélectionnés par offset dynamique — l'advection est stable quel que soit dt, les sous-pas
préservent seulement la qualité).

Contraintes tenues : aucune lecture GPU→CPU ; seul flux CPU→GPU par frame = un uniform
buffer ≤ 512 o ; bind groups/pipelines/vues tous pré-créés à l'init (zéro allocation en
boucle de frame) ; workgroups 16×16 (= 256, la limite d'invocations par défaut de WebGPU).

## Architecture

```
src/
├─ core/                 # portable : ne touche ni DOM, ni window, ni events
│  ├─ config.ts          # GRID_SIZE, formats (choix justifiés), propriétés des fluides
│  ├─ types.ts           # FrameInput abstrait (pointeur normalisé, fluide, actions)
│  ├─ uniforms.ts        # remplissage CPU de l'uniform buffer (seul trafic par frame)
│  ├─ resources.ts       # textures ping-pong, sampler, buffers — tout labellisé
│  ├─ (grille MAC : texel (i,j) = u face gauche + v face haute, packés en .xy)
│  ├─ pipelines.ts       # bind group layouts explicites, error scopes, compil WGSL vérifiée
│  ├─ passes/            # advect (MacCormack), forces, vorticity, project, walls, clear
│  ├─ render.ts          # scène HDR + chaîne de bloom + présentation
│  ├─ simulation.ts      # FluidSim : orchestration d'une frame
│  └─ shaders/*.wgsl     # features de base uniquement, portables Dawn/wgpu tels quels
└─ platform/web/         # jetable/remplaçable : adapter/device, canvas, rAF, souris,
   ├─ gpu.ts             #   clavier, overlay, resize, gestion device.lost
   ├─ input.ts
   ├─ overlay.ts
   └─ main.ts            # + mode ?selftest (input synthétique, rapport dans le DOM)
```

## Portage natif (Dawn / wgpu-native)

1. Réimplémenter `platform/` : fenêtre + surface (GLFW/SDL), boucle, events → `FrameInput`.
2. Fournir à `FluidSim.create` un device et le format de la swapchain, à `frame()` un dt et la vue cible.
3. Les WGSL se compilent tels quels (aucune extension, aucun override, formats core).
4. Porter les appels JS de `core/` vers l'API C (`webgpu.h`) — structure 1:1, labels inclus.
5. Rien d'autre : `core/` ne référence aucun symbole navigateur.

## Vérification headless (dev)

`.selftest/cdp-check.mjs` pilote un Chrome hors écran via le protocole DevTools :
`?selftest` remplace la souris par une orbite synthétique, publie un rapport JSON dans le
DOM après 200 frames et le script capture une frame dans `.selftest/shot.png`.
Lancer Chrome avec `--disable-backgrounding-occluded-windows` (sinon rAF est suspendu
hors écran) ; le WebGPU headless pur (`--headless=new`) ne fournit pas d'adapter sur
toutes les machines.
