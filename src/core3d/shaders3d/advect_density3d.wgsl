// Advection MacCormack des densités (x: fumée, yz: encres futures, w: chaleur),
// aux centres des cellules. predict : φ̂ vers scratch ; correct : correction clampée
// + dissipations + injection de l'émetteur (uniquement à l'écriture finale).

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,   // xyz: centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit d'encre, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f, // w: index de l'encre émise (0/1/2)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var den_src: texture_3d<f32>;
@group(1) @binding(1) var vel_src: texture_3d<f32>;
@group(1) @binding(2) var aux: texture_3d<f32>;
@group(1) @binding(3) var den_dst: texture_storage_3d<rgba16float, write>;

fn n_size() -> i32 {
  return i32(P.misc.z);
}
fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(P.misc.z);
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
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let center = vec3f(c) + vec3f(0.5);
  let back = center - P.misc.x * velocity_at(center);
  textureStore(den_dst, gid, textureSampleLevel(den_src, lin, back * inv_n(), 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let center = vec3f(c) + vec3f(0.5);
  let vel = velocity_at(center);
  let orig = textureLoad(den_src, c, 0);
  let hat = textureLoad(aux, c, 0);
  let tilde = textureSampleLevel(aux, lin, (center + dt * vel) * inv_n(), 0.0);
  var val = hat + 0.5 * (orig - tilde);

  // Clamp au stencil trilinéaire du point rétro-advecté (par canal).
  let back = center - dt * vel;
  let base = vec3i(floor(back - vec3f(0.5)));
  var lo = vec4f(1e30);
  var hi = vec4f(-1e30);
  for (var i = 0; i < 8; i++) {
    let corner = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let s = textureLoad(den_src, clamp(base + corner, vec3i(0), vec3i(n - 1)), 0);
    lo = min(lo, s);
    hi = max(hi, s);
  }
  val = clamp(val, lo, hi);

  // Dissipations : la fumée s'estompe lentement, la chaleur se refroidit vite.
  val = vec4f(val.xyz / (1.0 + P.diss.y * dt), val.w / (1.0 + P.diss.z * dt));

  // Émetteur gaussien : chaleur + l'encre sélectionnée (canal xyz selon l'index).
  let d = distance(center, P.emitter.xyz) / max(P.emitter.w, 1e-3);
  let g = exp(-d * d * 3.0);
  let ink = u32(P.blow_force.w + 0.5);
  var inject = vec3f(0.0);
  if (ink == 0u) {
    inject.x = 1.0;
  } else if (ink == 1u) {
    inject.y = 1.0;
  } else {
    inject.z = 1.0;
  }
  val = vec4f(
    min(val.xyz + inject * (P.emit_vals.y * dt * g), vec3f(3.0)),
    min(val.w + P.emit_vals.x * dt * g, 2.0),
  );

  textureStore(den_dst, gid, max(val, vec4f(0.0)));
}
