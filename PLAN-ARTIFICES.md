# PLAN — FEUX D'ARTIFICE COLORÉS (chantier ouvert le 2026-08-29)

**CHANTIER RANGÉ (2026-08-30) — verdict de Renaud : « ce n'est pas que je
n'aime pas, mais ce n'est pas ce que j'avais vraiment espéré, même s'il y a
de l'idée. »** J0-J3 verts et mergés, outil de tir visé (touche T) et lueur
teintée livrés — tout est DANS main et INERTE par défaut (système
événementiel, coût au repos strictement nul, passes sautées). NE PAS
RE-PROPOSER ; à re-évaluer seulement si Renaud le relance — le journal
ci-dessous dit où tout se trouve, rien n'est à re-coder.

Commande de Renaud, après l'échec du feu d'artifice « gaz » de la démo (« pas
top top, pas beaucoup de couleur, un peu plat ») : de VRAIS feux d'artifice,
colorés. Ce document est la référence du chantier — à relire avant d'écrire du
code artifices. Jalons à critères de sortie mesurables, journal en fin de
chaque jalon.

## La méthode, et pourquoi

**La couleur vit sur des PARTICULES, pas dans la grille.** Deux faits
l'imposent, tous deux payés :

- La grille de densité n'a que trois canaux à albédos FIXES (fumée / encre /
  carburant) : deux teintes différentes au même voxel sont structurellement
  impossibles, et la lumière d'une explosion est du corps noir — il n'existe
  aucun chemin vers un vert ou un bleu pyrotechnique par le gaz. (Le feu
  d'artifice « gaz » de la démo l'a montré à l'image.)
- Les BRAISES ont déjà prouvé la voie : particules additives dans la chaîne
  HDR, occlusion par particule contre la fumée, coût mesuré 0,0 FPS en
  A-B-A-B à 384³. Un feu d'artifice est d'ailleurs physiquement CELA : des
  étoiles incandescentes qui retombent, pas un gaz coloré.

Le FLUIDE reste le décor et le moteur : le vent et la turbulence portent les
étoiles (traînée), la bouffée de l'éclat pousse l'air et laisse une fumée
grise qui dérive — celle-là, la grille la fait très bien.

**Écarté, et pourquoi :**
- *Charges d'encre* (bouffées de gaz coloré) : 2-3 teintes fixes maximum,
  mélanges sales au même voxel — gardé en bonus éventuel pour de la fumée
  colorée de jour, PAS comme voie principale.
- *Deuxième texture de densité / champ de teinte advecté* : cher (mémoire +
  advection), et les mélanges de teintes dans un champ advecté bavent par
  construction.

## Architecture

- **Tampon d'ÉTINCELLES dédié**, distinct des braises (les braises sont une
  fonctionnalité validée, défaut coupé — on ne les touche pas). Ordre de
  grandeur 65 k particules, layout enrichi d'une COULEUR par particule.
  Storage fixe, zéro-initialisé = toutes mortes, zéro allocation par frame.
- **Naissance par ÉVÉNEMENT DE TIR** (pas par rejet sur la chaleur comme les
  braises) : un tir = position d'éclat + calibre + vitesse radiale + palette +
  graine, consommé en une frame (le patron des charges). Attribution des
  particules par CURSEUR EN ANNEAU côté CPU — pas d'atomics, déterministe.
- **Physique d'une étoile** : vitesse radiale initiale (l'éclat), puis
  gravité + traînée exponentielle vers la vitesse du fluide (les étoiles
  dérivent dans le vent), vie bornée avec extinction et scintillement.
- **Rendu** : billboards additifs dans la passe HDR avant le bloom (halo
  hérité), occlusion par particule contre la fumée, couleur PRESCRITE par
  particule — aucun corps noir.
- **Pilotage** : `launchFirework(...)` scripté (famille driveSphere /
  addEmitterAt / explodeAt), exposé au harnais via `__sim3d`.
- **Couplage gaz** : l'éclat tire aussi une petite charge de CHALEUR sans
  carburant (flash + poussée + fumée grise) — le système de charges multiples
  existant fait ce travail tel quel.

Doctrine inchangée et non négociable : zéro alloc/frame, zéro readback, UN
CommandEncoder, bind groups pré-créés, mots réservés WGSL (from, target,
move, smooth, ref, active).

## Jalons

**J0 — Le squelette étincelles.** Tampon dédié + passes update/draw clonées
des braises + naissance par événement (`launchFirework`) + couleur par
particule. Une teinte franche IMPOSSIBLE au corps noir (un VERT) comme preuve.
*Sortie : un tir donne une sphère d'étincelles vertes qui s'éteignent, 60 FPS
à 256³, braises non régressées (A/B), selftest OK.*

**J1 — La balistique d'une étoile.** Gravité + traînée vers le fluide, vie
avec extinction et scintillement, deux-trois patrons (pivoine sphérique,
saule à longue retombée, éclat bref). Traînées : version simple d'abord
(billboard étiré par la vitesse, comme les particules 2D) — jugée à l'image,
abandonnée sans regret si elle ne paie pas.
*Sortie : chronologie capturée d'une étoile — expansion freinée, suspension,
chute — reconnaissable comme du feu d'artifice sur planche datée.*

**J2 — Le tir complet.** La fusée qui monte (traçante), l'éclat à l'apogée
(charge de chaleur sans carburant : flash + fumée grise qui dérive), les
étoiles.
*Sortie : montée → flash → sphère → retombée → fumée, en une chronologie
continue capturée.*

**J3 — Palette et variété.** Teintes par tir (rouge, vert, bleu, or, violet),
bi-couleurs, calibres. En option : la LUEUR TEINTÉE (le volume de lueur
n'injecte que du corps noir — à juger à l'image, somptueux ou criard).
*Sortie : une salve de cinq tirs de cinq teintes, chaque teinte lisible,
mélanges propres.*

**J4 — Jouable, puis mis en scène.** Touche/outil « fusée » (clic = tir visé,
la plomberie explodeAt existe), et SEULEMENT ensuite l'acte de démo refait en
couleurs — c'est Renaud qui calibre le spectacle en tirant.
*Sortie : verdict de Renaud.*

Convention de merge : chaque jalon vert est commité ; la page publique ne
change de comportement par défaut qu'à partir de J4 (d'ici là, tout passe par
le pilotage scripté et le harnais).

## Journal des jalons

- **J0 — VERT (2026-08-29, machine de référence).** Le squelette tient tout
  entier du premier coup : `sparks3d.wgsl` (update, naissance par anneau) +
  `sparks_draw.wgsl` (billboards + occlusion par particule) + plomberie
  sim3d.ts, 65 536 étincelles × 48 o = 3 Mo. Mesuré (`cdp-sparks.mjs`) :
  pivoine VERTE nette (la teinte-preuve — le corps noir ne sait pas la faire),
  duo bleu/rouge à 0,5 s d'écart séparé proprement par l'anneau, braises de la
  fournaise INTACTES, 60 FPS à 256³, fenêtre de vie refermée à 3 s (les trois
  passes re-sautées — coût au repos strictement nul).
  Choix qui ont payé :
   · l'événement de tir vit dans la QUEUE du tampon d'uniformes (floats
     148-159, trois vec4 déjà alloués dans les 640 o mais jamais écrits) — le
     shader y accède par un struct à BLOCS DE PADDING `array<vec4f, N>`, aucun
     slot étranger, aucun agrandissement de tampon ; l'arithmétique d'offsets
     a été vérifiée par relecteur adversarial dédié (rien trouvé) ;
   · curseur d'anneau CÔTÉ CPU : pas d'atomics, déterministe (curseur, graine
     à pas fixe et époque avancent même pour un tir écrasé — le rejeu est
     identique) ;
   · l'ÉPOQUE (spark_c.z ↔ tint.w, modulo 2²⁴ — le dernier entier exact en
     f32) : le reset tue les étoiles au premier update qui suit, sans passe de
     clear dédiée.
  Relecture adversariale (3 lentilles) : zéro bloquant ; corrigé dans la
  foulée — le reset DÉSARME un tir en attente (sinon pivoine fantôme dans la
  scène neuve), modulo d'époque élargi, contrat « un événement par pas
  SIMULÉ » documenté.
  **Dette connue, à solder si le coût devient visible** : les passes
  étincelles sont invisibles au profileur `?profile` — l'update se fond dans
  la section « lueur », `sparks-occ` n'a pas de timestamps (le QuerySet est
  plein à 8 sections × 2). Sans gravité tant que le système est événementiel
  et bref ; à traiter si un acte de démo les fait vivre en continu.

- **J1 — VERT (2026-08-29, machine de référence).** La balistique d'une
  étoile, par PATRON : pivoine (traînée 1,8, gravité 0,12·N), saule (vie
  2,6-4 s, traînée lâche 0,9, gravité 0,20·N), éclat (vif, 0,45-0,9 s,
  presque sans chute). Le patron voyage AVEC l'époque dans `tint.w` = époque
  × 8 + patron (époque modulo 2²¹ pour rester sous 2²⁴, l'exact f32) — deux
  patrons coexistent en vol. Scintillement par l'ÂGE (pas d'horloge dans R :
  l'âge avance par frame et suffit), fort dès la naissance pour l'éclat.
  TRAÎNÉES par étirement écran : l'axe uv.y du billboard suit la vitesse
  projetée (×10 pour le saule), le dégradé radial du fragment fait le reste —
  la version simple a payé du premier coup, pas besoin de mieux pour l'instant.
  Touche **T** sur la page 3D : chaque appui tire la fusée suivante d'une
  table déterministe (le test à la main demandé par Renaud — l'outil visé
  reste J4). Chronologie capturée (`cdp-sparks.mjs`, planches J1B-*) :
  expansion freinée → suspension → chute → extinction, reconnaissable.
  LEÇON du jalon : le fondu de fin de vie est un réglage DE PATRON — en
  quadratique, le saule s'éteignait précisément pendant sa retombée, qui est
  sa raison d'être ; il fond en linéaire et brûle jusqu'au sol.

- **J2 — VERT (2026-08-30, machine de référence).** Le tir complet, SANS
  READBACK : la fusée est intégrée CÔTÉ CPU (même Euler semi-implicite que le
  shader — vitesse d'abord, position ensuite —, même gravité ROCKET_G = 0,55·N,
  traînée NULLE et vent ignoré, sinon les trajectoires divergent) pendant que
  la TRAÇANTE GPU (patron 3 : comète d'or de 24 étoiles, étirement ×10 plafonné
  à 7, scintillante dès la naissance) suit la même balistique. Les deux
  culminent au MÊME critère (vy ≤ 0) : le CPU y tire l'éclat exactement là où
  la comète s'éteint. L'éclat = les étoiles au patron demandé PLUS une charge
  de chaleur presque sans carburant via le système de charges existant, doté
  pour ça de MULTIPLICATEURS PAR CHARGE (calibre 0,045·N, carburant ×0,12,
  amorce ×0,8, POUSSIÈRE NULLE — un éclat en l'air n'arrache pas le sol ; la
  charge au pointeur reste ×1, inchangée). Calibration à l'image en trois
  passes : amorce ×3 donnait un petit CHAMPIGNON salmon qui montait et
  OCCULTAIT l'hémisphère haut de la sphère d'étoiles (l'occlusion par
  particule fait son travail — contre nous) ; ×0,8 laisse un flash bref et
  une bouffée tiède qui dérive. Plomberie neuve : FILE DES TIRS (anneau
  pré-alloué de 8 — un tir complet = deux événements, launch et apogée, et
  deux fusées peuvent culminer la même frame ; la file les étale d'un pas au
  lieu de les écraser, la graine voyage avec l'événement), 4 fusées en vol
  max (slots pré-alloués, patron des charges). API `launchRocket(x, z, apexY,
  r, g, b, n?, vitesse?, patron?)` ; touche T = tirs complets. Mesuré
  (`cdp-sparks.mjs`, planches J2F-*) : montée 400-1300 ms → flash 1600 →
  sphère verte → retombée 2900 → fumée qui dérive 4200-6000 ; 60 FPS à 256³,
  fenêtre refermée, braises fournaise INTACTES, saule/éclat/duo inchangés
  (les constantes par patron refactorées en tableaux — valeurs identiques).

- **J3 — VERT (2026-08-30, machine de référence).** Palette et variété — le
  gros du jalon était DÉJÀ LÀ (la teinte, le calibre — vitesse et compte — et
  le patron sont des paramètres depuis J0/J1) ; ce qui manquait : le
  BI-COULEUR et la preuve de la salve. Bi-couleur par la FILE DES TIRS, aucun
  slot d'uniforme de plus : à l'apogée, deux éclats au même point — la coque
  à la teinte principale (65 % des étoiles, pleine vitesse), le CŒUR à la
  seconde (35 %, vitesse ×0,55) — étalés d'un pas simulé par la file,
  invisible à l'œil. `launchRocket(..., r2, g2, b2)`, r2 ≥ 0 arme le
  bi-couleur ; deux entrées bi-colores dans la table de la touche T (cœur
  d'or, cœur bleu). Mesuré (`cdp-sparks.mjs`, planches J3A-salve-*/
  J3A-bicolore-*) : SALVE de cinq tirs complets étagés de 350 ms — rouge,
  vert, bleu, or, violet — les cinq sphères COEXISTENT, chaque teinte
  lisible, mélanges propres (l'anneau sépare, comme au duo de J0) ; le
  bi-couleur se lit nettement à 2,1 s (coque verte, cœur d'or). 60 FPS à
  256³, braises intactes. La LUEUR TEINTÉE (le volume de lueur n'injecte que
  du corps noir) reste EN OPTION, non prise : à juger à l'image le jour où
  un acte de démo la réclame — somptueux ou criard.

- **J4 — EN COURS (2026-08-30) : l'outil est livré, le spectacle attend
  Renaud.** La touche T tire désormais VISÉ AU POINTEUR : l'apogée est posée
  sous la souris (`launchRocketAt` — rayon ∩ plan horizontal à la hauteur
  d'éclat de l'entrée de table, même géométrie que la visée des charges,
  même rabattement au centre hors boîte), la fusée part du sol à la
  verticale du point visé ; la table (8 tirs : cinq teintes, trois patrons,
  deux bi-couleurs) fournit la rotation. PIÈGE de visée retrouvé à
  l'identique, version plan HAUT : vu de la caméra basse du banc, un NDC à
  +0,25 en Y passe AU-DESSUS du plan d'éclat dans la boîte et le coupe
  derrière elle — rabattu au centre ; viser bas (+0,05) place les tirs
  exactement où l'on pointe (banc `cdp-vise-fusee.mjs` du scratchpad :
  deux tirs NDC ±0,35 → deux éclats gauche/droite ; vrai appui T par
  Input.dispatchKeyEvent → pivoine au repos du pointeur, centre). RESTE de
  J4, qui n'est pas à moi : Renaud tire, calibre, et l'acte de démo se
  refait en couleurs SEULEMENT ensuite — sortie : son verdict.

- **LUEUR TEINTÉE — PRISE À L'ESSAI (2026-08-30), après le doute de Renaud
  (« pas super content […] on ne joue ici que sur des particules »).**
  Diagnostic : les étoiles brillaient dans le VIDE — le spectacle n'utilisait
  pas la force du moteur, la lumière volumétrique. L'option gardée au plan
  est donc branchée : chaque étoile vivante SPLATTE sa couleur dans un tampon
  à la résolution du volume de lueur (3 × u32 par cellule, virgule fixe
  ×1024, atomicAdd dans la passe update — le premier atomics du feu, cantonné
  à un tampon dédié), remis à zéro par `clearBuffer` au niveau encodeur ;
  l'INJECTION l'ajoute à l'émission corps noir (layout dédié — les blurs
  restent sur celui de curl) et toute la chaîne existante suit : diffusion
  ×3, in-scattering ∝ densité, flaque au sol, sphère éclairée. L'update des
  étincelles est passé AVANT la passe de lueur (l'injection lit le splat de
  la frame même) ; pause = lumière gelée cohérente ; un clear de plus solde
  le résidu à la mort du dernier tir. Poids face au corps noir :
  `SPARK_GLOW` (glow3d.wgsl) — 0,35 trop discret, **1,2 retenu à l'essai** :
  aura franche autour de la sphère, volutes voisines teintées, masse
  lointaine neutre (physique, pas criard — l'in-scattering veut un MILIEU,
  donc l'effet culmine en scène enfumée). Mesuré (`cdp-lueur.mjs` du
  scratchpad, planches LU-B-* ; non-régression LU-C-*) : pivoine à aura
  verte en boîte claire, brume illuminée en vert autour de l'éclat, SALVE
  transfigurée (chaque tir teinte sa propre fumée), braises INTACTES,
  60 FPS à 256³. Verdict final : Renaud, à la touche T — le réglage
  « somptueux ou criard » est cette seule constante.

- **RANGEMENT (2026-08-30).** Verdict de Renaud, après l'essai lueur
  teintée : « ce n'est pas que je n'aime pas, mais ce n'est pas ce que
  j'avais vraiment espéré, même s'il y a de l'idée. » Le chantier s'arrête
  là — J4 reste non validé (pas d'acte de démo en couleurs). Rien n'est
  retiré : étincelles, tirs complets, salves, bi-couleurs et lueur teintée
  restent dans main, inertes par défaut, jouables à la touche T et pilotables
  par `launchFirework` / `launchRocket` / `launchRocketAt`. Reprise
  éventuelle : relire ce journal — l'architecture (couleur sur particules,
  file des tirs, splat de lueur) est complète et vérifiée, seule la mise en
  scène manque.
