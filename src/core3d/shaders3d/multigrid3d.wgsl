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
  misc: vec4f,
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var src_tex: texture_3d<f32>;  // restrict : résidu fin ; prolong : correction grossière
@group(1) @binding(1) var p_src: texture_3d<f32>;
@group(1) @binding(2) var rhs: texture_3d<f32>;
@group(1) @binding(3) var scalar_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var fine_src: texture_3d<f32>;  // prolong : pression fine courante

const COARSEST = 8;
const OMEGA = 0.8;

fn p_at(c: vec3i, n: i32) -> f32 {
  return textureLoad(p_src, clamp(c, vec3i(0), vec3i(n - 1)), 0).x;
}

fn neighbor_sum(c: vec3i, n: i32) -> f32 {
  return p_at(c + vec3i(1, 0, 0), n) + p_at(c - vec3i(1, 0, 0), n) +
    p_at(c + vec3i(0, 1, 0), n) + p_at(c - vec3i(0, 1, 0), n) +
    p_at(c + vec3i(0, 0, 1), n) + p_at(c - vec3i(0, 0, 1), n);
}

@compute @workgroup_size(4, 4, 4)
fn smooth_jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(textureDimensions(p_src).x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let b = textureLoad(rhs, c, 0).x;
  let upd = (neighbor_sum(c, n) - b) / 6.0;
  var value = mix(p_at(c, n), upd, OMEGA);
  // Ancrage de l'espace nul au niveau le plus grossier.
  if (n == COARSEST && all(c == vec3i(0))) {
    value = 0.0;
  }
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn residual(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(textureDimensions(p_src).x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let ap = neighbor_sum(c, n) - 6.0 * p_at(c, n);
  let r = textureLoad(rhs, c, 0).x - ap;
  textureStore(scalar_dst, gid, vec4f(r, 0.0, 0.0, 0.0));
}

// Résidu fin → second membre grossier : moyenne des 8 enfants × 4 (facteur h²).
@compute @workgroup_size(4, 4, 4)
fn restrict_rhs(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(textureDimensions(scalar_dst).x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
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
  let n = i32(textureDimensions(fine_src).x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let nc = i32(textureDimensions(src_tex).x);
  let q = (vec3f(c) + vec3f(0.5)) * 0.5 - vec3f(0.5);
  let base = vec3i(floor(q));
  let t = q - vec3f(base);
  var corr = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let w = mix(1.0 - t.x, t.x, f32(o.x)) * mix(1.0 - t.y, t.y, f32(o.y)) *
      mix(1.0 - t.z, t.z, f32(o.z));
    corr += w * textureLoad(src_tex, clamp(base + o, vec3i(0), vec3i(nc - 1)), 0).x;
  }
  let value = textureLoad(fine_src, c, 0).x + corr;
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}
