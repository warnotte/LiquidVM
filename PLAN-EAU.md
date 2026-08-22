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
| J0 | Micro-banc P2G seul (scatter atomique 2 M particules, distribution GROUPÉE — la contention réaliste) | < 2 ms/frame mesuré — sinon on revoit la méthode AVANT de construire |
| J1 | **Dam break** : gravité, P2G/G2P (PIC/FLIP), Jacobi surface libre, rendu points | La colonne s'effondre, la vague traverse, clapote et SE CALME (ni explosion ni fonte) ; 60 FPS à 128³/2 M ; `?selftest` 240 frames + compteur de particules affiché au HUD |
| J2 | **APIC** : matrice affine C par particule, transferts APIC | A/B FLIP↔APIC par captures : surface plus lisse SANS amortissement visible ; tourbillon d'essai qui survit ; coût ≤ +20 % |
| J3 | La **boule** dans l'eau (sphère analytique + saisie + bord mobile) | Vagues au passage de la boule, aucune particule dans la sphère, 60 FPS |
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

## Conventions du chantier

- Branche `eau` (depuis `main`). Merge dans `main` par JALON VERT uniquement —
  jamais de travail en cours sur `main`.
- La navigation depuis les autres pages (bouton « 💧 eau ») n'est ajoutée qu'au
  premier merge — le site public ne montre jamais un chantier cassé.
- `eau.html` = 3ᵉ entrée Vite ; `src/liquid3d/` ne référence jamais le DOM.
- Chaque jalon : commit descriptif + captures dans la PR/commit + NOTES-DEV.
