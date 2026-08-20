// Advection semi-lagrangienne des densités (x: fumée, yz: réservés aux encres
// futures, w: chaleur) + injection de l'émetteur, aux centres des cellules.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: libre
  emitter: vec4f,   // xyz: centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit fumée, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: dissipation fumée, z: refroidissement, w: buoyancy
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var den_src: texture_3d<f32>;
@group(1) @binding(1) var vel_src: texture_3d<f32>;
@group(1) @binding(2) var den_dst: texture_storage_3d<rgba16float, write>;

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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.misc.z);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let center = vec3f(c) + vec3f(0.5);
  let vel = vec3f(sample_u(center), sample_v(center), sample_w(center));
  let back = center - dt * vel;
  var val = textureSampleLevel(den_src, lin, back * inv_n(), 0.0);

  // Dissipations : la fumée s'estompe lentement, la chaleur se refroidit vite.
  val = vec4f(val.xyz / (1.0 + P.diss.y * dt), val.w / (1.0 + P.diss.z * dt));

  // Émetteur gaussien : chaleur + fumée.
  let d = distance(center, P.emitter.xyz) / max(P.emitter.w, 1e-3);
  let g = exp(-d * d * 3.0);
  val.w = min(val.w + P.emit_vals.x * dt * g, 2.0);
  val.x = min(val.x + P.emit_vals.y * dt * g, 3.0);

  textureStore(den_dst, gid, max(val, vec4f(0.0)));
}
