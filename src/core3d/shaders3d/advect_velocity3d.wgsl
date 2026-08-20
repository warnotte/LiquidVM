// Advection semi-lagrangienne de la vélocité sur grille MAC 3D.
// Convention : le texel (i,j,k) porte u à la face gauche (i, j+½, k+½),
// v à la face basse (i+½, j, k+½), w à la face arrière (i+½, j+½, k).
// Chaque composante est advectée depuis SA position de face, échantillonnée avec
// SON demi-décalage. Boîte fermée : les faces frontières (i=0 / j=0 / k=0) sont
// zéroées à l'écriture ; les faces opposées n'existent pas (schéma compact) et
// valent 0 dans la divergence — murs gratuits sur les six côtés.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: libre
  emitter: vec4f,   // xyz: centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit fumée, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: dissipation fumée, z: refroidissement, w: buoyancy
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var vel_dst: texture_storage_3d<rgba16float, write>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(f32(P.misc.z));
}

fn sample_u(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x;
}
fn sample_v(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y;
}
fn sample_w(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z;
}
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(sample_u(p), sample_v(p), sample_w(p));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.misc.z);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let decay = 1.0 / (1.0 + P.diss.x * dt);
  let fc = vec3f(c);

  // Position de chaque face portée par ce texel.
  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);

  var u = sample_u(pu - dt * velocity_at(pu)) * decay;
  var v = sample_v(pv - dt * velocity_at(pv)) * decay;
  var w = sample_w(pw - dt * velocity_at(pw)) * decay;

  // Boîte fermée : non-pénétration aux faces frontières.
  u = select(u, 0.0, c.x == 0);
  v = select(v, 0.0, c.y == 0);
  w = select(w, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(u, v, w, 0.0));
}
