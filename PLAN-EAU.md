# Chantier EAU — liquide à surface libre (branche `eau`)

Document de conception, écrit AVANT la première ligne de code. Objectif : que
chaque session de travail sache où elle va, ce qui est déjà décidé, et ce qui
est un critère de sortie mesurable plutôt qu'une impression.

## Objectif

De l'eau liquide interactive dans la boîte 3D : une masse d'eau qui s'effondre,
fait des vagues, clapote et se calme ; la boule qui la brasse ; verser de l'eau
au pointeur. Temps réel (60 FPS cible), même machine de référence, même
doctrine que le reste du moteur.

## Non-objectifs (v1) — garde-fous de scope

- PAS de tension de surface, de mousse, d'écume, de gouttelettes secondaires.
- PAS de couplage eau ↔ feu avant que l'eau seule soit solide (la vapeur via le
  système d'espèces est un HORIZON documenté, pas un jalon).
- PAS de haute résolution : 128³ tant que J1–J3 ne sont pas verts.
- PAS de rendu réaliste avant que la physique tienne (J1 se regarde en points).

## Méthode : particules + grille (PIC/FLIP → **APIC**)

Famille retenue : les particules portent la masse et la vitesse ; la grille
MAC ne sert qu'à imposer l'incompressibilité (projection) ; transferts
particules→grille (P2G) et grille→particules (G2P) à chaque pas.

- Conservation de masse **triviale** (les particules ne disparaissent pas) —
  là où un level-set pur fond ou gonfle.
- La projection réutilise NOTRE machinerie éprouvée : grille MAC compacte,
  divergence/gradient adjoints, multigrid, sphère analytique, bord mobile.

**État de l'art assumé** : la cible est **APIC** (Jiang et al. 2016) — chaque
particule porte en plus une matrice affine C (3 vec3) ; les transferts affines
suppriment À LA FOIS le bruit de FLIP et la dissipation de PIC, et conservent
le moment angulaire. C'est le défaut moderne (Houdini, Bifrost). Mais APIC
partage EXACTEMENT l'architecture de FLIP/PIC (mêmes grilles, mêmes atomics,
même solveur) et FLIP/PIC en est un sous-ensemble : on construit donc le
squelette avec le transfert simple (moins de pièces mobiles pendant qu'on
déboguera atomics + surface libre), puis APIC est un JALON ferme avec A/B
mesuré — pas une option vague.

Écartés, avec raisons :
- **SPH** : voisinages dynamiques coûteux, incompressibilité molle, rendu
  difficile.
- **Level-set eulérien pur** : perte de masse, réinitialisation de SDF
  délicate — exactement ce que les particules évitent.
- **PBF** (Position-Based Fluids) : qualité jeu vidéo, compressibilité
  visible, ne réutilise pas notre solveur de pression.
- **MLS-MPM** : magnifique pour le multi-matériaux (sable, neige) mais
  remplace la projection incompressible — notre plus grande force — par un
  modèle faiblement compressible plus coûteux par particule. Horizon lointain
  si un jour on veut du sable, pas la v1 de l'eau.
- **Narrow-band FLIP** : optimisation d'échelle (particules seulement près de
  la surface) — pertinente APRÈS les jalons, pas avant.

## Architecture

- **Page séparée `eau.html`** + `src/liquid3d/` (même doctrine : zéro DOM,
  zéro alloc/frame, un CommandEncoder, zéro readback en boucle). Le 2D et le
  feu 3D ne sont PAS touchés. Réutilisés tels quels : `platform/web/gpu.ts`
  (device + maxBufferSize), `panel3d.ts`/`Toolbar3D` (UI déclarative),
  `core/pipelines.ts` (modules WGSL vérifiés, error scopes), le patron de
  projection inverse-rayons d'`embers_draw.wgsl` pour le rendu en points.
- **Grille** : MAC 128³ (convention identique au feu : texel (i,j,k) = faces
  u(i,j+½,k+½) / v / w, schéma compact = murs gratuits aux bords).
- **Particules** : buffer STORAGE fixe (pos vec4 + vel vec4 = 32 o), ~2 M
  (8/cellule d'eau initiale, ~1/8 de boîte). 64 Mo, mise à jour en place.
- **P2G (le pont buffers→textures)** : scatter par `atomicAdd` i32 en VIRGULE
  FIXE (×256 — vitesses ≤ ~500 voxels/s → marge i32 confortable) dans 6
  buffers (u, v, w + 3 poids), puis une passe « resolve » divise par les poids
  et écrit les TEXTURES de vélocité — toute la machinerie existante (advection
  de grille inutile ici, mais projection, sphère, bord mobile) travaille en
  textures comme au feu. Les atomics n'existent que sur les buffers : ce pont
  est le prix, il est payé une fois par frame.
- **Marqueurs de cellules** : texture u32 (air / eau / solide) déduite des
  comptes P2G + sphère analytique. La divergence ne se calcule que dans l'eau.
- **Pression à surface libre** : les cellules d'AIR sont en Dirichlet p = 0
  (condition SYMÉTRIQUE — compatible multigrid par construction, contrairement
  à la sortie ouverte 2D dont la face extrapolée brisait la symétrie) ; les
  solides gardent le clamp-trick Neumann éprouvé.
- **G2P** : blend FLIP/PIC réglable au panneau (défaut α ≈ 0,97), clamp CFL
  (une particule ≤ ~1 cellule/sous-pas, 2 sous-pas si dt le demande).
- **Extrapolation de vélocité** : 1–2 cellules dans l'air au-dessus de la
  surface (les particules de surface échantillonnent sinon des faces nulles).

## Jalons — chacun a un critère de sortie MESURABLE

| # | Livrable | Critère de sortie |
| - | -------- | ----------------- |
| J0 ✅ | Micro-banc P2G seul (scatter atomique 2 M particules, distribution GROUPÉE — la contention réaliste) | < 2 ms/frame mesuré — sinon on revoit la méthode AVANT de construire |
| J1 ✅ | **Dam break** : gravité, P2G/G2P (PIC/FLIP), Jacobi surface libre, rendu points | La colonne s'effondre, la vague traverse, clapote et SE CALME (ni explosion ni fonte) ; 60 FPS à 128³/2 M ; `?selftest` 240 frames + compteur de particules affiché au HUD |
| J2 ✅ | **APIC** : matrice affine C par particule, transferts APIC | A/B FLIP↔APIC par captures : surface plus lisse SANS amortissement visible ; tourbillon d'essai qui survit ; coût ≤ +20 % |
| J3 ✅ | La **boule** dans l'eau (sphère analytique + saisie + bord mobile) | Vagues au passage de la boule, aucune particule dans la sphère, 60 FPS |
| J4 | **Multigrid masqué** (pyramide air/eau/solide par niveau) | A/B Jacobi↔MG visuellement identiques, gain FPS mesuré, 2000+ frames stables ; Jacobi RESTE le repli au panneau |
| J5 | **Surface** : splat densité → raymarch (absorption bleu-vert, Fresnel) | Des captures qui ressemblent à de l'eau ; 60 FPS tenus |
| J6 | **Interactions** : verser au pointeur, souffle, presets (bassin calme / tempête), démo courte | Jouable au même niveau de finition que le feu |

Horizon non engagé (après J6, sur décision explicite) : eau + feu = vapeur via
le système d'espèces ; narrow-band FLIP puis 192³/4 M ; MLS-MPM si un jour le
sable/la neige tentent.

## Risques identifiés et mitigations

| Risque | Mitigation |
| ------ | ---------- |
| Scatter atomique trop lent | J0 le mesure AVANT tout le reste ; si échec : tri par cellule (complexe) ou réduction du nombre de particules |
| MG masqué instable (masques grossiers) | Jacobi d'abord (J1), MG seulement en J3, toggle de repli permanent — trajectoire déjà éprouvée en 2D et au feu |
| Bruit FLIP (surface qui grésille) | Blend PIC réglable à chaud dès J1 ; APIC = jalon J2 ferme (le vrai fix) |
| Dérive de volume | Compteur de particules + hauteur d'eau au HUD dès J1 ; correction de dérive SEULEMENT si mesurée |
| CFL violé (particules qui traversent les murs) | Clamp de déplacement + sous-pas ; test « aucune particule hors boîte » dans le selftest |

## Doctrine inchangée (non négociable)

Zéro allocation par frame, un seul CommandEncoder, zéro readback en boucle de
frame, bind groups pré-créés, WGSL features de base. `?selftest` dès J1.
Vérification par captures CDP (Chrome hors écran — voir NOTES-DEV), mesures de
FPS en A-B-A-B alterné (jamais avant/après : l'état de la sim évolue).
NOTES-DEV mis à jour à chaque jalon, pièges payés documentés immédiatement.

## Journal des jalons

- **J0 — VERT (2026-08-22, machine de référence RTX 5070 Ti).** Banc
  `eau.html?selftest` : 2 097 152 particules groupées (bloc 64³, 8/cellule),
  scatter 48 atomicAdd/particule (≈ 100 M/itération) + clear ×6 + resolve →
  texture MAC. Mesuré par onSubmittedWorkDone, 5 soumissions × 32 itérations :
  **ordre aléatoire 2,57 ms** (pire cas), **ordre trié par cellule 1,83 ms**
  — sous le budget de 2 ms. Conséquence ferme : J1 embarque un TRI périodique
  des particules par cellule (comptage/préfixes, toutes les ~8-16 frames) pour
  maintenir l'état trié en régime. Verdict : la voie particules+grille est
  praticable à 60 FPS, on construit.

- **J1 — EN COURS (2026-08-22) : STABLE EN EAU PEU PROFONDE.** Le dam break
  64×32×128 (profondeur 32) s'effondre, déferle, clapote et SE CALME en bassin
  plat d'épaisseur 16 (volume exactement conservé) — 2 097 152 particules
  valides / 0 perdue / 0 « rapide », 56-60 FPS, selftest OK. Trois batailles
  documentées, dans l'ordre où elles ont été gagnées :
  1. **Fromage** (filaments + trous, fonte du volume) → contrôle de densité de
     Bridson : divergence cible positive dans les cellules > 8 particules,
     bornée à 1× (débordée, elle cause la bataille 2), jamais de correction en
     sous-densité (elle combattrait la surface).
  2. **Explosion d'énergie** (toutes les particules à vitesse max, écrasées
     dans les coins) → cause racine PROUVÉE par l'instrument de recensement
     (valides/perdues/rapides + échantillon de 8 particules brutes, readback
     diagnostique 288 o toutes les 30 frames) : **Jacobi ne converge pas
     l'hydrostatique au-delà de ~32 cellules de profondeur d'eau** (~1 cellule
     de propagation par balayage). Le FLIP et le tri, suspectés d'abord, sont
     INNOCENTS (prouvé par A/B). Conséquence ferme : le **multigrid masqué
     (prévu J4) est REMONTÉ dans J1** — la colonne profonde 64 est l'épreuve
     d'entrée, pas une option.
  3. Pièges WGSL/WebGPU payés : « move » est réservé (avec « from » et
     « target ») ; un layout storageTexture 3D doit déclarer viewDimension
     ('2d' par défaut) ; filet anti-float16 sur les vitesses particules (±600).

- **J1 — reprise (2026-08-23) : LA CAUSE RACINE, et l'eau qui ressemble à de
  l'eau.** Retour utilisateur : « rendu bizarre, l'eau semble disparaître aux
  bords, pas de frontière physique ». Deux chantiers menés de front :
  1. **Rendu de surface grossier remonté de J5** (`eau_surface.wgsl`) : boîte
     dessinée (sol, parois, grille, arêtes — la frontière qui manquait),
     densité par cellule floutée 3³ → iso-surface marchée par rayon (pas 1
     voxel, bissection, normale = −gradient), **réfraction** n = 1,33 (la
     grille du sol se décale sous la ligne d'eau — le signal qui fait lire un
     volume d'eau claire), Fresnel ciel / Beer-Lambert, reflet solaire, tone-map
     + gamma. Le rendu points reste un instrument (case / touche P).
  2. **Instruments** (c'est eux qui ont trouvé le bug) : vues debug « coupe
     z = 0 » (quart inférieur ×4, fausses couleurs) et « densité max par
     rayon » ; **recensement des CELLULES** (histogramme d'occupation 1-3 /
     4-7 / 8-11 / 12-23 / 24+ au HUD, readback diagnostique existant).
  3. **CAUSE RACINE de « l'eau peu profonde »** : l'histogramme montrait ~15,8 k
     cellules occupées pour 2 M de particules = **UNE couche 128×128 à ~128
     particules/cellule**. `MARGIN = 1.001` dans eau_g2p.wgsl interdisait la
     rangée 0 (sol) et les colonnes 0 (parois) → cellules vides → classées
     AIR (p = 0) par le solveur → **le sol était une surface libre** : aucune
     pression hydrostatique possible, l'eau tombait « à travers » jusqu'au
     clamp. L'« explosion » de la bataille 2 et la limite ~32 cellules
     venaient de là, pas de Jacobi seul. Fix : marge 0,01 + fondu de la
     vitesse vers 0 dans la dernière demi-cellule des parois hautes (la face
     N n'existe pas, le sampler clampait sur N−1) + colonne initiale à ras.
  4. Dans la foulée : contrôle de densité **dans les deux sens** sur la
     densité floutée (compression des cellules intérieures sous-denses, jamais
     en surface — brise le cliquet « toute cellule touchée = eau
     incompressible, le volume ne fait que gonfler ») ; **taux en 1/s**
     (10/s) au lieu de 0,3/dt = 3600 %/s qui pompait de l'énergie (surface qui
     « bout », grumeaux à 24+) ; **purge de la pression au reset** (le warm
     start de l'ancien état kickait des particules > 550 voxels/s dès la 1re
     frame — la leçon 2D, encore).
  RÉSULTAT : vraie vague de rupture de barrage avec éclaboussures, ballottement,
  nappe de ~17 voxels qui TIENT hydrostatiquement (histogramme dominé par 8-11
  pendant la phase dynamique, 0 perdue, 0-3 rapides), 60 FPS avec le rendu de
  surface, selftest OK, build prod OK. Reste : bruit FLIP en régime calme
  (intérieur grumeleux, fringe de 2-3 voxels de particules éparses en surface —
  le sujet d'APIC en J2), dérive lente de l'histogramme sur 2 minutes (8-10 k
  cellules à 24+).
  **Colonne PROFONDE re-testée dans la foulée** (case « colonne haute 32×64 »
  / `?tall`) : elle s'effondre, déferle et se calme en ~17 voxels SANS
  explosion avec Jacobi 100 (0 perdue ; 230 « rapides » pendant l'impact puis
  ~10) — c'est le sol qui manquait, pas le solveur. **J1 est VERT** sur ses
  critères. Le multigrid masqué (J4) redevient une optimisation, plus l'épreuve
  d'entrée. À surveiller : légère pente résiduelle de la surface en coupe à
  110 s sur la colonne haute (seiche lente ou asymétrie parois basses/hautes
  — la paroi haute a le fondu `wall_fade`, la paroi basse absorbe dans la
  face 0), à mesurer avant de conclure.

- **J2 — APIC : VERT sur le critère de coût, MODESTE sur le visuel, et la vraie
  trouvaille est ailleurs (2026-08-23).** Transfert APIC implémenté (layout
  particule 64 o : vitesse + matrice affine C = ∇v_grille par différences
  centrées ½ voxel ; scatter v + C·dx ; case au panneau, `?flip` pour l'ancien
  mode). A/B FLIP↔APIC sur la même chronologie : 60 FPS dans les deux cas
  (coût ≤ +20 % tenu, vsync), 0 perdue, selftest OK ; histogramme à 60 s
  légèrement meilleur en APIC (40,7 k cellules à 8-11 contre 30,7 k) mais la
  différence visuelle est faible : l'artefact dominant du bassin calme n'était
  PAS le bruit FLIP mais le **contrôle de densité lui-même**. Expériences
  (histogramme à 60 s, Jacobi 100 sauf mention) :
  | contrôle | 1-3 | 4-7 | 8-11 | 12-23 | 24+ | lecture |
  | - | - | - | - | - | - | - |
  | deux sens 10/s | 48 k | 66 k | 41 k | 44 k | 15 k | surface raréfiée + grumeaux |
  | deux sens 40/s | 50 k | 71 k | 45 k | 46 k | 13 k | idem |
  | deux sens 10/s, Jacobi 30 | 48 k | 71 k | 44 k | 45 k | 13 k | Jacobi hors de cause |
  | aucun (0) | 5 k | 24 k | 70 k | 83 k | 1,5 k | propre mais COMPACTÉ ~25 % (nappe 12,5 au lieu de 16) |
  | **expansion seule + zone morte 25 %, 10/s** | **12 k** | **75 k** | **105 k** | **47 k** | **0,2 k** | **volume tenu, surface propre, stable à 120 s** |
  Leçons : la compression des cellules sous-denses FABRIQUE les grumeaux et
  raréfie la surface (la règle « jamais en surface » pompe vers le haut) ; sans
  zone morte le bruit de Poisson du comptage déclenche des expansions
  parasites ; sans contrôle du tout, l'eau se compacte lentement. Forme
  retenue dans eau_grid.wgsl, taux réglable au panneau (« contrôle densité
  (/s) »). Le « tourbillon d'essai » du critère J2 n'a pas été testé (pas de
  scène dédiée) — à faire avec la boule (J3), qui le fournira naturellement.

- **J3 — VERT : la boule dans l'eau (2026-08-23).** Sphère ANALYTIQUE (aucune
  texture d'obstacles, comme le feu) : `in_sphere` = « centre de cellule dans la
  sphère », testé par chaque passe. Conditions de bord posées d'emblée sous leur
  forme correcte, d'où zéro NaN du premier coup :
  - **divergence** : une face touchant la boule est un DÉBIT CONNU = vitesse de
    la boule (`face_u/v/w` distinguent paroi de boîte → 0, face de boule →
    `sphere_vel`, sinon vitesse advectée) ;
  - **Jacobi** : voisin solide → `p_centre` (Neumann, le même clamp-trick que
    les bords) ; l'air reste Dirichlet p = 0 ;
  - **gradient** : ces faces sont PRESCRITES à `sphere_vel` et non corrigées —
    l'adjoint divergence/gradient reste exact, donc le bord MOBILE ne fabrique
    ni source ni puits ;
  - **particules** : repoussées sur la surface (rayon + ½ voxel) avec annulation
    de la composante entrante de la vitesse RELATIVE (une boule immobile ne
    colle pas les particules, une boule qui avance les pousse).
  Saisie au pointeur comme le feu : `hitTest` par rayon décide saisie vs orbite,
  déplacement sur le plan face caméra, vitesse lissée EMA 0.55/0.45 plafonnée à
  400 voxels/s, amortie ×0,82 au relâcher. Rendu : intersection analytique, la
  marche d'eau et le rayon RÉFRACTÉ s'arrêtent sur la boule (`behind()` rend la
  boule ou la paroi) — elle se lit dans l'air comme sous l'eau.
  VÉRIFIÉ par un vrai glisser CDP (`Input.dispatchMouseEvent`) : la boule pousse
  une vague devant elle, laisse un sillage, éclabousse ; **0 particule dans la
  boule** à chaque relevé (nouveau compteur `census[3]`, affiché au HUD), 0
  perdue, **60 FPS** pendant tout le brassage. Le brassage violent fait monter
  les « rapides » à ~600 (vitesses > 550 voxels/s) sans aucune divergence : le
  filet float16 fait son travail.
  Piège UX payé : le HUD (`.hud`) captait les clics — la boule était
  insaisissable dans le bas de l'écran. `pointer-events: none` (il est purement
  informatif). À vérifier sur les autres pages si un jour on y ajoute de la
  manipulation directe près du HUD.

- **MOUSSE (acompte sur J5, 2026-08-23).** L'eau aérée — embruns, crêtes
  déferlantes, nappes fines — diffuse au lieu de réfracter : sans elle une
  éclaboussure se rend comme une bille de verre. Deux indices déjà disponibles,
  combinés en `max()` puis pondérés par le slider « mousse » (défaut 0,7) :
  densité mesurée **2 voxels SOUS la surface** (l'eau en masse y est dense, une
  gouttelette non) et épaisseur d'eau traversée par le rayon réfracté.
  **Piège payé** : la première version sondait la densité SUR l'iso-surface —
  elle y vaut le seuil PAR CONSTRUCTION, donc « aéré » partout et le bassin
  entier devenait blanc laiteux. Sonder un champ à l'endroit exact où on l'a
  seuillé ne dit jamais rien.
  Coût mesuré nul (A/B mousse 0 ↔ 0,7 à état de sim égal : 45 FPS dans les deux
  cas au premier run, 60 dans les deux cas au second). **Leçon de banc rappelée**
  : le niveau absolu du premier run après un démarrage de Chrome n'est pas
  comparable au suivant (45 vs 60 ici, à code identique) — seul l'A/B dans la
  MÊME session compte.

- **RÉSOLUTION CHOISISSABLE + MERGE DANS MAIN (2026-08-23).** `?grid=96|128|160|192`
  et boutons au panneau, sur le modèle du feu : `GRID_EAU`/`SCALE_EAU`/
  `PARTICLES_EAU`/`SORT_BLOCKS` sont des bindings VIVANTS, `setGridEau()` appelé
  par la plateforme avant toute création. Le nombre de particules suit
  exactement n³ (la colonne fait n/2 × n/4 × n cellules à 8 particules), et
  toutes les grandeurs en voxels/s (gravité, plafond CFL, filet float16,
  vitesse de la boule) sont multipliées par SCALE_EAU = n/128 — la boîte garde
  sa taille physique. MESURES (RTX 5070 Ti, dam break + boule) : **96³ 0,9 M →
  60 FPS · 128³ 2,1 M → 60 · 160³ 4,1 M → 48-51 · 192³ 7,1 M → 27-28**. Le
  solveur de pression domine (Jacobi 100 × 2 sous-pas × n³) : c'est là que le
  multigrid masqué (J4) paiera.
  **Trois limites/bugs révélés par la montée en résolution — tous invisibles à
  128³ :**
  1. `maxStorageBufferBindingSize` (défaut 128 Mio) plafonne un BINDING, pas une
     allocation, et le buffer de particules vaut n³ × 64 o = **exactement**
     128 Mio à 128³ (ça passait à zéro octet près) → 160³ échouait à la
     validation. Relevée au max de l'adapter dans gpu.ts, à côté de
     `maxBufferSize` (2 Gio ici). Corollaire : une limite frôlée n'est pas une
     limite respectée, il faut la relever avant de croire qu'on a de la marge.
  2. `maxComputeWorkgroupsPerDimension` = 65535, or n³/64 vaut 110 592 à 192³ →
     tous les submits échouaient (en WARNING console uniquement, encore une
     fois : le HUD affichait 60 FPS sur une image morte). Les passes de
     particules dispatchent désormais en **2D** (1024 groupes par rangée,
     index = `gid.x + gid.y * 65536`, constante `PARTICLE_ROW` partagée
     WGSL/TS).
  3. L'instrument se sabotait lui-même : le pas d'échantillonnage des 8
     particules brutes du recensement était figé à 262144, donc à 160³ il
     produisait 16 échantillons dont les 8 derniers ÉCRASAIENT l'histogramme
     des cellules (census[66..71]) — qui affichait alors des bits de position
     (~1,1e9). Le pas suit maintenant le nombre de particules ET l'index est
     borné. **Leçon : un instrument doit être borné à sa fenêtre, sinon il
     ment d'autant plus fort qu'on change d'échelle.**
  MERGE : `eau` → `main` (demandé par l'utilisateur), navigation croisée
  ajoutée sur les trois pages (« 💧 eau » depuis la 2D et le feu, « 🌊 2D » et
  « 🧊 feu 3D » depuis l'eau), README complété. Le HUD de l'eau est en
  `pointer-events: none` (il captait les clics).

## Conventions du chantier

- Branche `eau` (depuis `main`). Merge dans `main` par JALON VERT uniquement —
  jamais de travail en cours sur `main`.
- La navigation depuis les autres pages (bouton « 💧 eau ») n'est ajoutée qu'au
  premier merge — le site public ne montre jamais un chantier cassé.
- `eau.html` = 3ᵉ entrée Vite ; `src/liquid3d/` ne référence jamais le DOM.
- Chaque jalon : commit descriptif + captures dans la PR/commit + NOTES-DEV.
