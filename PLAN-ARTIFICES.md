# PLAN — FEUX D'ARTIFICE COLORÉS (chantier ouvert le 2026-08-29)

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
