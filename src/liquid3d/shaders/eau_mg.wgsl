// J4 — MULTIGRID MASQUÉ pour la pression de l'eau. C'est le multigrid du feu
// (multigrid3d.wgsl) auquel on ajoute UNE chose : le TYPE de cellule. Le reste
// — V-cycle, restriction, prolongation, lisseur — est identique, parce que
// l'opérateur l'est déjà :
//
//   A p = Σ_voisins contrib − 6·p   avec contrib = p_voisin (fluide)
//                                                 p_centre (solide, Neumann)
//                                                 0        (air, Dirichlet)
//
// Le feu est le cas « aucune cellule d'air » de cet opérateur, l'eau le cas
// général. Écrire un second multigrid aurait été la faute : c'est le même.
//
// PYRAMIDE DE MASQUES, reconstruite à chaque frame (la surface libre bouge).
// Règle retenue : une cellule grossière est AIR dès qu'UN enfant est air, sinon
// FLUIDE si un enfant est fluide, sinon SOLIDE — c'est la seule des trois
// règles essayées qui soit STABLE. Mesures à l'appui :
//
//   « fluide dès qu'UN enfant est fluide » → EXPLOSION dès deux niveaux. Les
//   embruns isolés propagent « fluide » vers le haut, le niveau grossier perd
//   ses cellules d'air donc ses conditions de Dirichlet, son système devient
//   singulier (Neumann pur) et sa solution dérive d'une constante énorme.
//   Prolongée dans le fluide, cette constante rencontre les cellules d'air
//   remises à zéro au niveau FIN : le saut crée un gradient de pression
//   gigantesque à la surface et toutes les particules partent au plafond.
//   PIÈGE MAJEUR : le résidu de divergence NE VOIT RIEN de tout ça — il tombait
//   à 3e-5, « mieux » que Jacobi. Un champ de vitesse uniforme parasite est à
//   divergence nulle. Un solveur peut donc être irréprochable au sens du résidu
//   et complètement faux ; il faut regarder AUSSI l'énergie (compteur de
//   particules rapides) et l'image.
//
//   « fluide à la MAJORITÉ des enfants » → explose aussi. Le seuil n'est donc
//   pas le bon levier.
//
//   ANCRAGE DES NIVEAUX GROSSIERS (essayé le 2026-08-24, sur l'hypothèse que la
//   dérive du mode constant était la cause) : il RÉGLE le cas à deux niveaux —
//   permissif + ε devient stable là où il explosait — mais NE SAUVE PAS la
//   pyramide complète, quel que soit ε (0,5 · 1 · 2 · 4 essayés). Le mécanisme
//   dominant n'est donc pas la singularité : avec le fluide dominant, le
//   domaine fluide grossier finit par DÉBORDER largement le domaine fin (au 4ᵉ
//   niveau, toute cellule contenant la moindre goutte est « fluide »), si bien
//   que le niveau grossier modélise un autre problème que le vrai. Aucune
//   régularisation ne rattrape une inconsistance d'opérateur. Le domaine
//   grossier doit RÉTRÉCIR, jamais grandir — c'est la règle conservatrice.
//
//   « air dès qu'UN enfant est air » (retenue) → STABLE. Le domaine fluide
//   grossier est plus petit que le vrai — une nappe de 16 voxels n'a presque
//   plus rien à corriger au 3ᵉ niveau — donc la correction s'essouffle en
//   profondeur, mais la condition de Dirichlet ne disparaît jamais.
//
// L'ancrage est CONSERVÉ malgré tout (ε petit, coût mesuré nul) : il ne sert
// pas la règle permissive, mais il couvre le seul cas réellement singulier —
// une poche d'eau entièrement close par du solide, sans aucune cellule d'air
// pour fixer sa constante. Voir `relax_anchored`.

struct UEau {
  sim: vec4f,   // x: N (grille FINE), y: nb particules, z: dt, w: gravité
  sim2: vec4f,
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
  render: vec4f,
  sphere: vec4f, // xyz: centre (voxels FINS), w: rayon (≤ 0 = absente)
  sphere_vel: vec4f,
  mg: vec4f,     // x: règle de masques (0 = air dominant, 1 = permissive), y: ε d'ancrage
}

@group(0) @binding(0) var<uniform> U: UEau;
// Rôle par entry point (comme multigrid3d.wgsl du feu) :
//   src_tex : restrict = résidu FIN · prolong = correction GROSSIÈRE ·
//             mask_restrict = masque FIN
//   p_src   : smooth/residual = pression du niveau · prolong = pression FINE
//   rhs     : smooth/residual = second membre du niveau
//   mask    : type de cellule du niveau COURANT pour smooth/residual, du niveau
//             FIN pour restrict_rhs, du niveau GROSSIER pour prolong_add
@group(1) @binding(0) var src_tex: texture_3d<f32>;
@group(1) @binding(1) var p_src: texture_3d<f32>;
@group(1) @binding(2) var rhs: texture_3d<f32>;
@group(1) @binding(3) var mask: texture_3d<f32>;
@group(1) @binding(4) var scalar_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(5) var<storage, read> cell_count: array<u32>;

const AIR = 0.0;
const FLUID = 1.0;
const SOLID = 2.0;
const OMEGA = 0.8;

fn n_of(t: texture_3d<f32>) -> i32 {
  return i32(textureDimensions(t).x);
}

// Même chose pour la cible : une texture de stockage n'est pas du même type,
// WGSL n'a pas de surcharge — d'où deux helpers.
fn n_dst() -> i32 {
  return i32(textureDimensions(scalar_dst).x);
}

// Hors grille = solide (paroi de boîte) : même convention qu'au niveau 0.
fn type_at(c: vec3i, n: i32) -> f32 {
  if (any(c < vec3i(0)) || any(c >= vec3i(n))) {
    return SOLID;
  }
  return textureLoad(mask, c, 0).x;
}

fn p_at(c: vec3i, n: i32) -> f32 {
  return textureLoad(p_src, clamp(c, vec3i(0), vec3i(n - 1)), 0).x;
}

fn offset_of(a: i32) -> vec3i {
  let s = select(-1, 1, (a & 1) == 0);
  switch (a >> 1) {
    case 0: { return vec3i(s, 0, 0); }
    case 1: { return vec3i(0, s, 0); }
    default: { return vec3i(0, 0, s); }
  }
}

// Voisinage d'une cellule d'eau, sous la forme qui rend l'opérateur EXPLICITE :
//
//   A p = sum_fluid − (6 − solides)·p
//
// où sum_fluid ne compte QUE les voisins d'eau (un voisin d'air contribue 0 :
// c'est la surface libre ; un voisin solide n'a pas de face, il sort de la
// diagonale). Écrire l'opérateur ainsi évite le piège payé ici : la première
// version sommait aussi la contribution s·p des solides PUIS divisait par
// (6 − s) — somme complète et diagonale réduite mélangées, donc sur-relaxation
// et explosion. Le Jacobi de production (eau_grid.wgsl) fait l'autre choix
// cohérent : somme complète divisée par 6, plus amorti. Les deux sont valides,
// leur mélange ne l'est pas.
struct Stencil {
  sum_fluid: f32,
  solids: f32,
}

fn stencil(c: vec3i, n: i32) -> Stencil {
  var out: Stencil;
  out.sum_fluid = 0.0;
  out.solids = 0.0;
  for (var a = 0; a < 6; a++) {
    let nb = c + offset_of(a);
    let t = type_at(nb, n);
    if (t == SOLID) {
      out.solids += 1.0;
    } else if (t == FLUID) {
      out.sum_fluid += p_at(nb, n);
    }
    // air : contribution nulle (Dirichlet p = 0)
  }
  return out;
}

// Lisseur — nommé `relax` parce que « smooth » est un MOT RÉSERVÉ WGSL (même
// famille que « from », « target » et « move »). Jacobi pondéré avec la VRAIE
// diagonale −(6 − solides) : le Jacobi de production (eau_grid.wgsl) divise
// toujours par 6, ce qui l'amortit près des parois ; ici on veut lisser vite.
@compute @workgroup_size(4, 4, 4)
fn relax(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_of(p_src);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  if (type_at(c, n) != FLUID) {
    textureStore(scalar_dst, gid, vec4f(0.0));
    return;
  }
  let st = stencil(c, n);
  let diag = max(6.0 - st.solids, 1.0);
  let upd = (st.sum_fluid - textureLoad(rhs, c, 0).x) / diag;
  textureStore(scalar_dst, gid, vec4f(mix(p_at(c, n), upd, OMEGA), 0.0, 0.0, 0.0));
}

// LISSEUR ANCRÉ — réservé au niveau le PLUS GROSSIER, le seul où l'on enchaîne
// assez de balayages (16) pour qu'un mode constant s'installe ; aux niveaux
// intermédiaires, 2 balayages ne laissent pas le temps de dériver. C'est la même
// logique que le pin p(0,0,0)=0 du multigrid du feu, mais sous une forme qui ne
// suppose RIEN sur la géométrie du fluide : un sous-domaine d'eau sans aucune
// cellule d'air voisine donne un système de Neumann pur, singulier, dont la
// solution dérive d'une constante arbitraire — et un pin sur une cellule fixe
// n'aiderait pas si cette cellule n'appartient pas au sous-domaine fautif.
//
// Régularisation : A' p = sum_fluid − (6 − solides + ε)·p. Le mode constant y
// est borné (c ≈ −moyenne(résidu)/ε) au lieu de dériver, quel que soit le
// nombre de composantes connexes. Le niveau grossier n'a pas besoin d'être
// exact — il n'est qu'un préconditionneur —, seulement BORNÉ.
@compute @workgroup_size(4, 4, 4)
fn relax_anchored(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_of(p_src);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  if (type_at(c, n) != FLUID) {
    textureStore(scalar_dst, gid, vec4f(0.0));
    return;
  }
  let st = stencil(c, n);
  let diag = max(6.0 - st.solids, 1.0) + U.mg.y;
  let upd = (st.sum_fluid - textureLoad(rhs, c, 0).x) / diag;
  textureStore(scalar_dst, gid, vec4f(mix(p_at(c, n), upd, OMEGA), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn residual(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_of(p_src);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  if (type_at(c, n) != FLUID) {
    textureStore(scalar_dst, gid, vec4f(0.0));
    return;
  }
  let st = stencil(c, n);
  let ap = st.sum_fluid - (6.0 - st.solids) * p_at(c, n);
  textureStore(scalar_dst, gid, vec4f(textureLoad(rhs, c, 0).x - ap, 0.0, 0.0, 0.0));
}

// Masque du niveau 0 : eau si la cellule contient au moins une particule,
// solide si elle est dans la boule (les parois sont hors grille, donc solides
// par type_at), air sinon.
@compute @workgroup_size(4, 4, 4)
fn mask_fine(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(U.sim.x);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  if (U.sphere.w > 0.0 && distance(vec3f(c) + vec3f(0.5), U.sphere.xyz) < U.sphere.w) {
    textureStore(scalar_dst, gid, vec4f(SOLID, 0.0, 0.0, 0.0));
    return;
  }
  let occupied = cell_count[u32(c.x + n * (c.y + n * c.z))] > 0u;
  textureStore(scalar_dst, gid, vec4f(select(AIR, FLUID, occupied), 0.0, 0.0, 0.0));
}

// Masque grossier : AIR si ≥ 1 enfant air, sinon FLUIDE si ≥ 1 enfant fluide,
// sinon SOLIDE (src_tex = masque FIN). Voir l'en-tête : cet ordre de priorité
// est ce qui rend le V-cycle stable.
@compute @workgroup_size(4, 4, 4)
fn mask_restrict(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_dst();
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  let f = c * 2;
  var any_fluid = false;
  var any_air = false;
  for (var i = 0; i < 8; i++) {
    let t = textureLoad(src_tex, f + vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1), 0).x;
    any_fluid = any_fluid || t == FLUID;
    any_air = any_air || t == AIR;
  }
  // Deux règles, commutables à chaud pour pouvoir les MESURER (voir l'en-tête) :
  //   conservative — l'air domine : sûre même sans ancrage, mais le domaine
  //     fluide grossier perd une couche par niveau ;
  //   permissive — le fluide domine : garde tout le domaine, mais crée des
  //     sous-systèmes sans Dirichlet → EXIGE l'ancrage du niveau grossier.
  var out = SOLID;
  if (U.mg.x > 0.5) {
    if (any_fluid) {
      out = FLUID;
    } else if (any_air) {
      out = AIR;
    }
  } else if (any_air) {
    out = AIR;
  } else if (any_fluid) {
    out = FLUID;
  }
  textureStore(scalar_dst, gid, vec4f(out, 0.0, 0.0, 0.0));
}

// Résidu fin → second membre grossier : moyenne des enfants FLUIDES × 4
// (facteur h²). Les enfants non fluides portent un résidu nul par construction,
// mais les compter dans la moyenne diluerait le second membre.
@compute @workgroup_size(4, 4, 4)
fn restrict_rhs(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_dst();
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  let f = c * 2;
  var s = 0.0;
  var k = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = f + vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    if (textureLoad(mask, o, 0).x == FLUID) {
      s += textureLoad(src_tex, o, 0).x;
      k += 1.0;
    }
  }
  let value = select(0.0, s / k * 4.0, k > 0.0);
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

// Correction grossière → niveau fin : trilinéaire manuelle (r32float non
// filtrable), pondérée par le masque GROSSIER et renormalisée — une cellule
// grossière d'air ou de solide ne porte pas de correction valide.
// Les cellules fines non fluides reçoivent une valeur sans signification ; le
// post-lissage qui SUIT TOUJOURS cette passe les remet à zéro. (Ne jamais
// régler MG_POST à 0 sans ajouter ici un test sur le masque fin — qui
// demanderait un binding de plus.)
@compute @workgroup_size(4, 4, 4)
fn prolong_add(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_of(p_src);
  let c = vec3i(gid);
  if (any(c >= vec3i(n))) {
    return;
  }
  let fine_p = textureLoad(p_src, c, 0).x;
  let nc = n_of(src_tex);
  let q = (vec3f(c) + vec3f(0.5)) * 0.5 - vec3f(0.5);
  let base = vec3i(floor(q));
  let t = q - vec3f(base);
  var corr = 0.0;
  var wsum = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let cc = clamp(base + o, vec3i(0), vec3i(nc - 1));
    let w = mix(1.0 - t.x, t.x, f32(o.x)) * mix(1.0 - t.y, t.y, f32(o.y)) *
      mix(1.0 - t.z, t.z, f32(o.z));
    if (textureLoad(mask, cc, 0).x == FLUID) {
      corr += w * textureLoad(src_tex, cc, 0).x;
      wsum += w;
    }
  }
  let value = fine_p + select(0.0, corr / wsum, wsum > 1e-6);
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn clear_level(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_dst();
  if (any(vec3i(gid) >= vec3i(n))) {
    return;
  }
  textureStore(scalar_dst, gid, vec4f(0.0));
}
