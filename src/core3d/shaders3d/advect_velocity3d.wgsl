// Advection MacCormack de la vélocité sur grille MAC 3D — deux entry points.
// predict : semi-lagrangien φ̂ = A(φ) vers la texture scratch (aux).
// correct : φ̃ = A⁻¹(φ̂) (re-advection en remontant le temps), φ' = φ̂ + ½(φ − φ̃),
// clampé au min/max du stencil trilinéaire du point rétro-advecté (Selle 2008) —
// l'ordre 2 sans les oscillations. Dissipation et murs appliqués au correcteur.
// Convention MAC : texel (i,j,k) → u à (i, j+½, k+½), v à (i+½, j, k+½), w à (i+½, j+½, k).

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,    // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
}

// Sphère-obstacle analytique : la cellule est solide si son CENTRE est dans la
// sphère — la même règle que le gradient et le lisseur (adjonction exacte).
fn solid_cell(c: vec3i) -> bool {
  if (P.sphere.w <= 0.0) {
    return false;
  }
  return distance(vec3f(c) + vec3f(0.5), P.sphere.xyz) < P.sphere.w;
}

fn face_blocked(c: vec3i, axis: vec3i) -> bool {
  return solid_cell(c) || solid_cell(c - axis);
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var aux: texture_3d<f32>;
@group(1) @binding(2) var vel_dst: texture_storage_3d<rgba16float, write>;

const OFF_U = vec3f(0.5, 0.0, 0.0);
const OFF_V = vec3f(0.0, 0.5, 0.0);
const OFF_W = vec3f(0.0, 0.0, 0.5);

fn n_size() -> i32 {
  return i32(P.misc.z);
}
fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(P.misc.z);
}

fn src_u(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_U) * inv_n(), 0.0).x;
}
fn src_v(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_V) * inv_n(), 0.0).y;
}
fn src_w(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_W) * inv_n(), 0.0).z;
}
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(src_u(p), src_v(p), src_w(p));
}

fn aux_u(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_U) * inv_n(), 0.0).x;
}
fn aux_v(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_V) * inv_n(), 0.0).y;
}
fn aux_w(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_W) * inv_n(), 0.0).z;
}

// Min/max d'une composante sur les 8 coins du stencil trilinéaire au point `pos`
// (grille des faces de cette composante : coordonnée texel continue = pos + off).
fn stencil_minmax(pos: vec3f, off: vec3f, comp: i32) -> vec2f {
  let base = vec3i(floor(pos + off - vec3f(0.5)));
  var lo = 1e30;
  var hi = -1e30;
  for (var i = 0; i < 8; i++) {
    let corner = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let c = clamp(base + corner, vec3i(0), vec3i(n_size() - 1));
    let v4 = textureLoad(vel_src, c, 0);
    let v = select(select(v4.z, v4.y, comp == 1), v4.x, comp == 0);
    lo = min(lo, v);
    hi = max(hi, v);
  }
  return vec2f(lo, hi);
}

@compute @workgroup_size(4, 4, 4)
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let fc = vec3f(c);
  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);
  let u = src_u(pu - dt * velocity_at(pu));
  let v = src_v(pv - dt * velocity_at(pv));
  let w = src_w(pw - dt * velocity_at(pw));
  textureStore(vel_dst, gid, vec4f(u, v, w, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let decay = 1.0 / (1.0 + P.diss.x * dt);
  let fc = vec3f(c);
  let orig = textureLoad(vel_src, c, 0);
  let hat = textureLoad(aux, c, 0);

  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);
  let vel_u = velocity_at(pu);
  let vel_v = velocity_at(pv);
  let vel_w = velocity_at(pw);

  // φ' = φ̂ + ½(φ − φ̃), clampé au stencil du point rétro-advecté.
  var u = hat.x + 0.5 * (orig.x - aux_u(pu + dt * vel_u));
  var v = hat.y + 0.5 * (orig.y - aux_v(pv + dt * vel_v));
  var w = hat.z + 0.5 * (orig.z - aux_w(pw + dt * vel_w));
  let mu = stencil_minmax(pu - dt * vel_u, OFF_U, 0);
  let mv = stencil_minmax(pv - dt * vel_v, OFF_V, 1);
  let mw = stencil_minmax(pw - dt * vel_w, OFF_W, 2);
  u = clamp(u, mu.x, mu.y) * decay;
  v = clamp(v, mv.x, mv.y) * decay;
  w = clamp(w, mw.x, mw.y) * decay;

  // Boîte fermée + sphère-obstacle : non-pénétration.
  u = select(u, 0.0, c.x == 0 || face_blocked(c, vec3i(1, 0, 0)));
  v = select(v, 0.0, c.y == 0 || face_blocked(c, vec3i(0, 1, 0)));
  w = select(w, 0.0, c.z == 0 || face_blocked(c, vec3i(0, 0, 1)));

  textureStore(vel_dst, gid, vec4f(u, v, w, 0.0));
}
