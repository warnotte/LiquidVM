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

- **J1 — EN COURS (2026-08-30). La cloche EST là ; l'étouffement ne l'est pas
  encore.** Ce qui marche et est commité :
   · **la forme** — coquille sphérique `abs(|q| − R) − e`, sans plan de coupe :
     c'est le PLANCHER de la boîte (déjà un mur de non-pénétration) qui ferme
     le bas. Poser le centre assez bas pour que la coquille traverse le sol
     suffit à sceller. Élégant, et zéro paramètre de plus ;
   · **le VERRE** — un obstacle qui arrête aussi les rayons cacherait
     précisément ce qu'on veut montrer. La cloche ne tronque donc pas la marche
     du raymarch : elle n'ajoute qu'un liseré (Fresnel simplifié) et un peu de
     la lueur qu'elle capte, ne porte pas d'ombre au sol et n'occulte pas les
     particules. On voit la flamme brûler DANS le verre — l'image est là.
  Ce qui ne marche pas encore, et les deux leçons qui l'expliquent :
   1. **PIÈGE DE BANC : poser l'obstacle AVANT le reset ne sert à rien.** Le
      reset restaure la position par défaut (`processInteraction`), donc la
      cloche flottait au-dessus de la flamme et la « mort par étouffement »
      observée n'était qu'une flamme à l'air libre qui faiblissait. Poser
      APRÈS, une frame plus tard. Même famille que la leçon des harnais qui
      préparent la scène : c'est l'ORDRE des gestes qui mentait.
   2. **Sans COMBUSTION, il n'y a rien à étouffer.** Le preset bougie émet de
      la fumée CHAUDE, pas du carburant : l'oxygène n'est jamais consommé, et
      la flamme brûle sous cloche indéfiniment (24 s vérifiées). Mais émettre
      du carburant ne suffit pas non plus : il est émis FROID par conception,
      et un émetteur seul ne peut pas à la fois le cracher et l'allumer — la
      boîte se remplit alors de vapeur imbrûlée (capturé). L'étouffement
      demande donc le montage de l'acte III de la démo : une FLAMME PILOTE
      (encre 0, chaude) + un émetteur de CARBURANT, la nappe portée vers la
      flamme. C'est un travail de SCÉNARIO, pas de moteur.
  À reprendre par là. Les deux témoins du banc (`cdp-cloche.mjs` du
  scratchpad : sans cloche · cloche qui fuit) sont écrits et prêts — sans eux
  une flamme qui meurt ne prouverait rien.
  **À surveiller** : 25-26 FPS mesurés dans ces scènes contre 60 ailleurs.
  Probablement la BOÎTE PLEINE de vapeur (le raymarch paie chaque voxel non
  vide — leçon documentée des braises), pas la cloche ; à vérifier au
  profileur avant de conclure quoi que ce soit.

- **J1 (suite, 2026-08-30) — LA SCÈNE EST LÀ, L'ÉTOUFFEMENT N'EST PAS PROUVÉ.**
  Livré, cliquable, vérifié : un preset **🔔 cloche** qui monte la scène en un
  clic — flamme PILOTE (encre 0, chaude) + émetteur de CARBURANT collé à elle,
  cloche de verre posée, air non renouvelé. L'allumage marche : une vraie
  combustion brûle dans le verre, et l'image est superbe. Trois choses apprises
  en chemin, toutes payées au banc :
   1. **La pose se fait APRÈS le reset** (compteur différé `bellIn`), sinon la
      position par défaut de l'obstacle écrase la nôtre.
   2. **Le joint au sol a un critère exact** : la coquille INTÉRIEURE (R − e)
      doit passer sous le plancher. À 0,2 voxel près d'affleurer, la cavité
      fuit et la boîte se remplit — ce n'est pas « à peu près posé », c'est
      une inégalité.
   3. **Une source de masse dans un vase clos est intenable.** L'EXPANSION est
      une source de divergence : sans sortie, pression et vitesses montent
      jusqu'à ce que la matière TUNNELE à travers la coquille (une trace qui
      parcourt plus que l'épaisseur en un pas la traverse). Expansion à 0 sous
      cloche, paroi épaissie à 0,17 R.
  **Ce qui N'EST PAS démontré, et pourquoi je ne le déclare pas fait** : les
  deux TÉMOINS refusent de séparer. « Cloche + air non renouvelé » et « cloche
  + air renouvelé » donnent la même image à 14 et 20 s, sur sept réglages
  successifs. Tant que le témoin ne sépare pas, l'oxygène n'est pas la cause de
  ce qu'on voit — et une flamme qui faiblit ne prouve rien. Deux facteurs
  confondants ont été identifiés et retirés, sans que ça suffise :
   · réduire la flamme pilote après l'allumage éteint la flamme AVEC OU SANS
     oxygène — c'était une mise en scène, pas une conséquence ;
   · la chaleur du pilote était injectée SANS CONDITION : un émetteur chaud est
     pourtant une combustion, il doit demander de l'air. Corrigé (`heat_rate ×
     oxy_factor`) — physiquement juste, sans effet ailleurs (`oxy_factor`
     sature à 1 dès un quart d'oxygène) et la scène par défaut est
     indiscernable.
  Deux pistes restent, dans cet ordre : (a) l'échelle de temps — avec la
  stœchiométrie par défaut (0,55) un volume d'air tient près d'une MINUTE ;
  d'où le nouveau réglage `oxygenBurn` (curseur « O₂ consommé »), mais même à 9
  la fenêtre de 20 s ne suffit visiblement pas ; (b) **l'oxygène traverse-t-il
  la paroi ?** Son advection est fusionnée dans `advect_density3d` et sa
  rétro-trace, elle, N'A AUCUNE notion de solide — c'est la seule grandeur du
  moteur dans ce cas. À instrumenter AVANT de re-régler quoi que ce soit : il
  faut mesurer l'O₂ dans la cavité, pas le déduire de l'image.
  Livré au passage : `oxygenBurn` (défaut 0,55 = l'historique au bit près) et
  son curseur ; le banc `cdp-etouffe.mjs` (scratchpad) qui clique le PRESET
  comme l'utilisateur et porte ses deux témoins.

- **J1 — VERT (2026-08-30). L'étouffement est DÉMONTRÉ, et c'est l'instrument
  qui l'a permis.** Après sept réglages jugés à l'image, tous non concluants,
  j'ai cessé de régler et j'ai MESURÉ : `__sim3d.sampleOxygen()` lit l'oxygène
  (readback ponctuel, même exception assumée que l'export VDB) et rend sa
  moyenne DANS la cavité et AUTOUR. Le premier relevé a tout dit :
  ```
  A (cloche, air non renouvelé)   O2 dedans 0,81 → 0,53 puis PLATEAU, dehors 1,000
  ```
  Un plateau, avec zéro renouvellement et un réservoir extérieur intact : ce
  n'est pas une combustion qui s'arrête, c'est une FUITE à l'équilibre. Et
  rétrécir la cavité de 3,6× n'a pas bougé le plateau — la source ne suivait
  donc pas le VOLUME mais la SURFACE. Ce qui désignait la paroi.
  **LA CAUSE** : les cellules SOLIDES gardaient leur oxygène initial (1,0) pour
  toujours — rien ne les consomme — et l'échantillonnage trilinéaire des
  cellules voisines les mélangeait dans la cavité. La paroi était un
  RÉSERVOIR D'AIR INFINI. Correctif d'une ligne, et c'est la convention déjà
  suivie par les densités : **un solide ne contient pas de gaz**.
  Après correctif, les trois cas SE SÉPARENT enfin :
  ```
  A  cloche, air non renouvelé   O2 0,17 → 0,043 → 0,009 → 0,000   (mort)
  B  sans cloche                 O2 0,90 → 0,94 → 0,96 → 0,95      (vit)
  C  cloche, air renouvelé       O2 0,17 → 0,11 → 0,13 → 0,12      (bridée)
  ```
  et l'IMAGE raconte la même histoire : flamme orange vive à 3 s sous le verre,
  braise sombre à 20 s ; sans cloche, panache haut et vif au même instant.
  **Un second correctif de physique, trouvé en chemin** : la chaleur d'un
  émetteur est désormais modulée par l'oxygène — un émetteur chaud EST une
  combustion, et tant que la sienne était inconditionnelle, il rougeoyait dans
  le vide et aucune flamme ne pouvait mourir. Sans effet ailleurs
  (`oxy_factor` sature à 1 dès un quart d'oxygène) ; scènes standard
  re-capturées, indiscernables, 60 FPS.
  **Ce qui n'a PAS lieu, et ne doit pas** : la flamme ne se rallume pas quand
  on lève la cloche. C'est juste — une mèche étouffée ne se rallume pas seule,
  il lui faut une source. Ne pas le vendre comme un bug.
  **LA LEÇON, et elle est la plus chère de la journée** : sept réglages jugés à
  l'image n'ont rien produit ; une mesure a donné la cause en un relevé. Un
  témoin qui ne SÉPARE PAS ne dit pas « l'effet est faible », il dit « tu ne
  mesures pas ce que tu crois ». Instrumenter AVANT de re-régler.
  Livré : preset **🔔 cloche** (un clic monte la scène), réglage `oxygenBurn`
  (stœchiométrie, défaut 0,55 = l'historique), instrument `sampleOxygen`, banc
  `cdp-o2.mjs` (les nombres) et `cdp-etouffe.mjs` (les images), tous deux avec
  leurs deux témoins.

- **LA CLOCHE QUI BOUGE (2026-08-30, signalé par Renaud : « si je bouge la
  cloche tout explose »).** Je n'ai PAS reproduit son explosion — ni au
  pilotage scripté (glissement doux, puis secousse violente), ni par de vrais
  événements souris (qui ont orbité la caméra au lieu d'attraper). Ce que je
  n'ai pas reproduit, je ne prétends pas l'avoir corrigé.
  En revanche le diagnostic STRUCTUREL, lui, ne dépend pas du geste, et il est
  dur : **une coquille fermée crée une seconde région de fluide isolée**, et
  deux choses y sont intenables.
   1. **Un débit prescrit sur son bord.** Le solveur n'ancre qu'UNE constante
      de pression (le pin p(0,0,0)=0, documenté comme vital). La cavité a son
      propre espace nul, et la somme des débits imposés sur son bord doit
      valoir EXACTEMENT zéro pour que le système reste soluble. En escalier
      discret elle ne le vaut pas : le résidu n'a nulle part où aller. Donc
      **une coquille creuse ne prescrit plus sa vitesse au bord** — on soulève
      une cloche de verre, l'air qu'elle enferme ne se fait pas traîner en
      bloc. La sphère, elle, continue de brasser le fluide comme avant.
   2. **Une source de masse.** L'expansion de combustion est une source de
      divergence ; dans un volume clos elle n'a aucune sortie. Elle est donc
      COUPÉE tant qu'une coquille est présente, quel que soit le preset —
      c'était déjà 0 dans le preset « cloche », mais un preset appliqué
      par-dessus la rendait, et le runaway revenait par cette porte.
  Vérifié après correctif : secousse violente (aller-retour large + vertical à
  la vitesse maximale acceptée), aucune divergence, O₂ et FPS sains, sphère et
  autres formes indiscernables (60 FPS).
  **Reste à faire** : obtenir de Renaud le geste exact. Deux suspects que je
  n'ai pas pu écarter — la SAISIE au pointeur teste encore l'englobante (dette
  connue de J0 : on attrape la cloche par son trou), et rien n'empêche de la
  traîner hors de la boîte.
