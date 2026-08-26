// Multigrid 3D pour la pression : V-cycle géométrique 128³ → 8³.
// Les tailles se déduisent de textureDimensions — les mêmes entry points servent
// tous les niveaux. Leçons du 2D, payées cher, appliquées d'office :
// 1) le couple divergence/gradient compact et ce lisseur partagent EXACTEMENT le
//    même opérateur A p = Σ(p_voisin) − 6p (le clamp des voisins hors boîte ajoute
//    k·p_centre qui reproduit précisément le Neumann des faces manquantes) ;
// 2) en Neumann pur la pression n'est définie qu'à une constante près et la
//    restriction amplifie sa dérive ×4 par niveau → ancrage p(0,0,0) = 0 au
//    niveau le plus grossier (8³), directement dans le lisseur. NE PAS RETIRER.

struct Params {
  misc: vec4f,      // z: Nx fin (remonte aux voxels fins), w: hauteur du domaine
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,    // xyz: centre (voxels fins), w: rayon (voxels fins, ≤0 = absente)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var src_tex: texture_3d<f32>;  // restrict : résidu fin ; prolong : correction grossière
@group(1) @binding(1) var p_src: texture_3d<f32>;
@group(1) @binding(2) var rhs: texture_3d<f32>;
@group(1) @binding(3) var scalar_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var fine_src: texture_3d<f32>;  // prolong : pression fine courante

// Le niveau le plus grossier est le SEUL de taille < 16 quelle que soit la
// pyramide (8 pour les grilles 2^k, 12 pour 384) — l'ancrage se détecte ainsi.
const COARSEST_LIMIT = 16;
const OMEGA = 0.8;

fn p_at(c: vec3i, n: vec3i) -> f32 {
  return textureLoad(p_src, clamp(c, vec3i(0), n - vec3i(1)), 0).x;
}

// Sphère-obstacle vue depuis un niveau grossier. RESTRICTION CONSERVATRICE : la
// cellule n'est solide que si elle est ENTIÈREMENT dans la sphère, pas
// seulement son centre.
//
// C'est la règle du journal J4 — « le domaine grossier doit RÉTRÉCIR, jamais
// grandir » — et elle vaut pour l'obstacle comme elle valait pour le fluide.
// Avec le critère du centre, une cellule à moitié dedans devient solide, donc
// l'obstacle GROSSIT à chaque niveau : le niveau grossier corrige alors un
// problème que le niveau fin ne pose pas, et cette incohérence pompe de
// l'énergie tout autour de la boule. Le défaut n'apparaît qu'en HAUTE
// résolution parce que la pyramide y a plus de niveaux, et il met plusieurs
// secondes à devenir visible — d'où l'impression que la sphère « attire » la
// fumée au bout d'un moment.
fn solid_cell(c: vec3i, n: vec3i) -> bool {
  if (P.sphere.w <= 0.0) {
    return false;
  }
  // Les CELLULES sont cubiques et tous les axes se divisent par deux à chaque
  // niveau : un seul facteur d'échelle suffit, quelle que soit la forme du
  // domaine.
  let scale = P.misc.z / f32(n.x);
  // Demi-diagonale de la cellule grossière, en voxels fins : ce qu'il faut
  // retrancher au rayon pour exiger la cellule ENTIÈRE à l'intérieur.
  let half_diag = scale * 0.866025;
  return distance((vec3f(c) + vec3f(0.5)) * scale, P.sphere.xyz) < P.sphere.w - half_diag;
}

fn neighbor_sum(c: vec3i, n: vec3i) -> f32 {
  let pc = p_at(c, n);
  var sum = 0.0;
  for (var a = 0; a < 6; a++) {
    var off = vec3i(0);
    let axis = a >> 1;
    let sign = select(-1, 1, (a & 1) == 0);
    if (axis == 0) {
      off.x = sign;
    } else if (axis == 1) {
      off.y = sign;
    } else {
      off.z = sign;
    }
    let nb = c + off;
    sum += select(p_at(nb, n), pc, solid_cell(nb, n));
  }
  return sum;
}

@compute @workgroup_size(4, 4, 4)
fn smooth_jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(p_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let b = textureLoad(rhs, c, 0).x;
  let upd = (neighbor_sum(c, n) - b) / 6.0;
  var value = mix(p_at(c, n), upd, OMEGA);
  // Ancrage de l'espace nul au niveau le plus grossier.
  if (n.x < COARSEST_LIMIT && all(c == vec3i(0))) {
    value = 0.0;
  }
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn residual(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(p_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let ap = neighbor_sum(c, n) - 6.0 * p_at(c, n);
  let r = textureLoad(rhs, c, 0).x - ap;
  textureStore(scalar_dst, gid, vec4f(r, 0.0, 0.0, 0.0));
}

// Résidu fin → second membre grossier : moyenne des 8 enfants × 4 (facteur h²).
@compute @workgroup_size(4, 4, 4)
fn restrict_rhs(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(scalar_dst));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let f = c * 2;
  var s = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    s += textureLoad(src_tex, f + o, 0).x;
  }
  textureStore(scalar_dst, gid, vec4f(s * 0.5, 0.0, 0.0, 0.0)); // (s/8)·4
}

// Correction grossière → niveau fin : interpolation trilinéaire manuelle
// (r32float non filtrable) ajoutée à la pression fine courante.
@compute @workgroup_size(4, 4, 4)
fn prolong_add(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(fine_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let nc = vec3i(textureDimensions(src_tex));
  let q = (vec3f(c) + vec3f(0.5)) * 0.5 - vec3f(0.5);
  let base = vec3i(floor(q));
  let t = q - vec3f(base);
  var corr = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let w = mix(1.0 - t.x, t.x, f32(o.x)) * mix(1.0 - t.y, t.y, f32(o.y)) *
      mix(1.0 - t.z, t.z, f32(o.z));
    corr += w * textureLoad(src_tex, clamp(base + o, vec3i(0), nc - vec3i(1)), 0).x;
  }
  let value = textureLoad(fine_src, c, 0).x + corr;
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}
