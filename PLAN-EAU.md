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

## Méthode : FLIP/PIC hybride — et pourquoi

**FLIP/PIC** : les particules portent la masse et la vitesse ; la grille MAC ne
sert qu'à imposer l'incompressibilité (projection) ; transferts
particules→grille (P2G) et grille→particules (G2P) à chaque pas.

- Conservation de masse **triviale** (les particules ne disparaissent pas) —
  là où un level-set pur fond ou gonfle.
- La projection réutilise NOTRE machinerie éprouvée : grille MAC compacte,
  divergence/gradient adjoints, multigrid, sphère analytique, bord mobile.
- Standard de l'industrie (Zhu & Bridson 2005) : comportement connu, pièges
  documentés, extensions balisées (APIC si le bruit FLIP gêne).
- SPH rejeté : voisinages dynamiques coûteux, incompressibilité molle, rendu
  difficile. Level-set eulérien pur rejeté : perte de masse, réinitialisation
  de SDF délicate — exactement ce que FLIP évite.

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
| J0 | Micro-banc P2G seul (scatter atomique 2 M particules) | < 2 ms/frame mesuré — sinon on revoit la méthode AVANT de construire |
| J1 | **Dam break** : gravité, P2G/G2P, Jacobi surface libre, rendu points | La colonne s'effondre, la vague traverse, clapote et SE CALME (ni explosion ni fonte) ; 60 FPS à 128³/2 M ; `?selftest` 240 frames + compteur de particules affiché au HUD |
| J2 | La **boule** dans l'eau (sphère analytique + saisie + bord mobile) | Vagues au passage de la boule, aucune particule dans la sphère, 60 FPS |
| J3 | **Multigrid masqué** (pyramide air/eau/solide par niveau) | A/B Jacobi↔MG visuellement identiques, gain FPS mesuré, 2000+ frames stables ; Jacobi RESTE le repli au panneau |
| J4 | **Surface** : splat densité → raymarch (absorption bleu-vert, Fresnel) | Des captures qui ressemblent à de l'eau ; 60 FPS tenus |
| J5 | **Interactions** : verser au pointeur, souffle, presets (bassin calme / tempête), démo courte | Jouable au même niveau de finition que le feu |

Horizon non engagé (après J5, sur décision explicite) : eau + feu = vapeur via
le système d'espèces ; APIC ; 192³/4 M.

## Risques identifiés et mitigations

| Risque | Mitigation |
| ------ | ---------- |
| Scatter atomique trop lent | J0 le mesure AVANT tout le reste ; si échec : tri par cellule (complexe) ou réduction du nombre de particules |
| MG masqué instable (masques grossiers) | Jacobi d'abord (J1), MG seulement en J3, toggle de repli permanent — trajectoire déjà éprouvée en 2D et au feu |
| Bruit FLIP (surface qui grésille) | Blend PIC réglable à chaud ; APIC en piste documentée |
| Dérive de volume | Compteur de particules + hauteur d'eau au HUD dès J1 ; correction de dérive SEULEMENT si mesurée |
| CFL violé (particules qui traversent les murs) | Clamp de déplacement + sous-pas ; test « aucune particule hors boîte » dans le selftest |

## Doctrine inchangée (non négociable)

Zéro allocation par frame, un seul CommandEncoder, zéro readback en boucle de
frame, bind groups pré-créés, WGSL features de base. `?selftest` dès J1.
Vérification par captures CDP (Chrome hors écran — voir NOTES-DEV), mesures de
FPS en A-B-A-B alterné (jamais avant/après : l'état de la sim évolue).
NOTES-DEV mis à jour à chaque jalon, pièges payés documentés immédiatement.

## Conventions du chantier

- Branche `eau` (depuis `main`). Merge dans `main` par JALON VERT uniquement —
  jamais de travail en cours sur `main`.
- La navigation depuis les autres pages (bouton « 💧 eau ») n'est ajoutée qu'au
  premier merge — le site public ne montre jamais un chantier cassé.
- `eau.html` = 3ᵉ entrée Vite ; `src/liquid3d/` ne référence jamais le DOM.
- Chaque jalon : commit descriptif + captures dans la PR/commit + NOTES-DEV.
