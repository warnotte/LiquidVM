# PLAN — OBSTACLES DE FORME QUELCONQUE (chantier ouvert le 2026-08-30)

Commande de Renaud, après l'état des lieux face à **EmberGen** (JangaFX) : le
seul écart de MOTEUR qui change ce qu'on peut *fabriquer* est l'obstacle. Eux
encaissent des colliders arbitraires ; nous avons une sphère analytique. Et le
mot d'ordre du jour : « si on peut montrer un truc qu'ils ne font pas, ça
serait comique » — d'où le jalon J1, qui n'est pas une forme de plus mais une
PHYSIQUE qu'ils n'ont pas.

## La méthode, et pourquoi

**Une seule fonction : la distance signée.** L'obstacle est aujourd'hui testé
dans dix passes, et TOUJOURS sous la même forme — « ce centre de cellule est-il
dans la sphère ? ». Chaque occurrence est déjà isolée dans une petite fonction
(`solid_cell`, `fondu_obstacle`, `sphere_hit`). Il suffit donc de remplacer le
prédicat par `solid_sd(p) < 0`, où `solid_sd` rend une distance SIGNÉE en
voxels, négative dedans.

Ce que cela préserve, et c'est tout l'intérêt :
- **l'adjonction exacte** du couple divergence/gradient — l'opérateur ne
  change pas, seul le masque binaire de cellules change ;
- **la règle du domaine qui RÉTRÉCIT** en descendant la pyramide multigrille :
  « cellule entièrement dedans » s'écrit `sd < −demi-diagonale`, ce qui est la
  même inégalité qu'avant avec le rayon ;
- **la peau sans confinement** (`fondu_obstacle`) : c'était déjà
  `distance − rayon`, c'est-à-dire la distance signée de la sphère. Elle
  devient `smoothstep(0, 3, solid_sd(p))` sans autre changement.

**Écarté pour l'instant :** une TEXTURE de distance signée (qui autoriserait
des maillages importés). Elle coûte une passe de re-voxélisation à chaque
déplacement et 64 Mo à 256³, et elle n'apporte rien tant que les primitives ne
sont pas en place. La fonction `solid_sd` est le point d'insertion : le jour où
elle échantillonne une texture au lieu de calculer une formule, tout le reste
du moteur suit sans être retouché.

**Convention de taille :** `sphere.w` reste LE rayon (≤ 0 = pas d'obstacle) et
les paramètres de forme sont des RATIOS de ce rayon. Le curseur « taille » de
l'utilisateur continue donc de régler l'obstacle, quelle que soit sa forme.

## Jalons

**J0 — Le prédicat généralisé.** `solid_sd` dupliquée dans les dix passes
(sphère · boîte · tore), rendu par sphere tracing borné à la sphère
englobante, saisie au pointeur laissée sur l'englobante.
*Sortie : la SPHÈRE est indiscernable de l'existant sur les bancs (obstacle,
fournaise, champignon) ; une boîte et un tore fendent la fumée et se
referment ; 60 FPS à 256³.*

**J1 — LA CLOCHE, et l'ÉTOUFFEMENT.** Une forme CREUSE (coquille ouverte en
bas), posée sur le sol au-dessus de la flamme. Le moteur a déjà tout : l'O₂ se
consomme, la combustion est modulée par lui, la récupération est lente. Sous
la cloche, la flamme doit MOURIR faute d'air — et repartir quand on soulève.
C'est le « truc qu'ils ne font pas » : chez eux la suie et l'extinction sont
des réglages d'artiste, ici ce sont des conséquences.
*Sortie : chronologie capturée — flamme vive, déclin, mort ; cloche levée,
reprise. Le compteur d'O₂ doit raconter la même histoire que l'image.*

**J2 — Jouable.** Choix de la forme au panneau, et l'acte de démo s'il paie.
*Sortie : verdict de Renaud.*

Doctrine inchangée : zéro alloc/frame, zéro readback, un CommandEncoder, bind
groups pré-créés, mots réservés WGSL (from, target, move, smooth, ref, active).

## Journal des jalons

- **J0 — VERT (2026-08-30, machine de référence).** Le prédicat est généralisé
  d'un bout à l'autre : dix passes lisent la même `solid_sd` (sphère · boîte ·
  tore), et la sphère reste le cas EXACT — elle ne paie rien à la
  généralisation. Ce qui a rendu la bascule mécanique : le prédicat était
  partout la MÊME petite fonction isolée (`solid_cell`, `fondu_obstacle`,
  `sphere_hit`), et deux d'entre elles étaient déjà des distances signées
  déguisées — la peau du confinement était `distance − rayon`, le test
  conservateur du multigrid était « rayon − demi-diagonale ». Les deux se
  réécrivent sans changer une inégalité.
  Choix qui ont payé :
   · **taille = ratios.** Les paramètres de forme sont des ratios du rayon, donc
     le curseur de taille existant vaut pour toute forme, sans nouveau réglage ;
   · **le rendu marche, mais borné.** L'analytique devient du sphere tracing,
     encadré par la sphère ENGLOBANTE : hors d'elle, aucun pas — le coût reste
     nul pour l'immense majorité des rayons. Et la sphère court-circuite la
     marche (une évaluation) ;
   · **l'occlusion des particules** (braises, étincelles) se contente de 16 pas :
     c'est une occultation binaire, pas une image.
  **PIÈGE PAYÉ, et il était muet** : `renderData` faisait EXACTEMENT 92 floats,
  et une écriture en 92-95 sur un Float32Array est SILENCIEUSEMENT IGNORÉE. La
  simulation voyait la boîte et le tore (le panache s'étalait bien sous une face
  plate), mais le rendu dessinait toujours une sphère — deux images qui se
  contredisent et aucune erreur nulle part. `lastRender` devait suivre : `set()`
  LÈVE si la source est plus longue. Règle : quand on ajoute un slot d'uniforme,
  vérifier la LONGUEUR du tableau de staging, pas seulement la taille du tampon
  GPU.
  Mesuré (`cdp-formes.mjs` du scratchpad) : 60 FPS pour les trois formes à
  256³ ; le tore est TRAVERSÉ par le panache et son ombre au sol est un anneau ;
  la boîte étale la fumée sous sa face plate et projette une ombre rectangulaire ;
  la sphère est indiscernable de l'existant (et l'est par construction — même
  inégalité, même branche analytique au rendu).
  **Dette connue** : la SAISIE au pointeur teste encore la sphère englobante —
  on attrape un tore par son trou. Sans gravité tant que les formes sont
  convexes-ish ; à traiter avec la cloche si elle gêne.
