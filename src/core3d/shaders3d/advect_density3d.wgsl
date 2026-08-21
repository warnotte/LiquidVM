// Advection MacCormack des densités (x: fumée, yz: encres futures, w: chaleur),
// aux centres des cellules. predict : φ̂ vers scratch ; correct : correction clampée
// + dissipations + injection de l'émetteur (uniquement à l'écriture finale).

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,   // émetteur 0 : xyz centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit d'encre, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,    // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f, // x: nombre d'émetteurs actifs (1..4)
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f, // encre (0/1/2) de chaque émetteur
}

fn emitter_pos(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.emitter; }
    case 1u: { return P.emitter1; }
    case 2u: { return P.emitter2; }
    default: { return P.emitter3; }
  }
}

fn emitter_ink(i: u32) -> u32 {
  switch i {
    case 0u: { return u32(P.emit_inks.x + 0.5); }
    case 1u: { return u32(P.emit_inks.y + 0.5); }
    case 2u: { return u32(P.emit_inks.z + 0.5); }
    default: { return u32(P.emit_inks.w + 0.5); }
  }
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

  // Émetteurs gaussiens : chaleur + l'encre propre à chacun (canal xyz selon l'index).
  let count = u32(P.emit_meta.x + 0.5);
  for (var i = 0u; i < count; i++) {
    let e = emitter_pos(i);
    let d = distance(center, e.xyz) / max(e.w, 1e-3);
    let g = exp(-d * d * 3.0);
    let ink = emitter_ink(i);
    var inject = vec3f(0.0);
    if (ink == 0u) {
      inject.x = 1.0;
    } else if (ink == 1u) {
      inject.y = 1.0;
    } else {
      inject.z = 1.0;
    }
    val = vec4f(
      val.xyz + inject * (P.emit_vals.y * dt * g),
      val.w + P.emit_vals.x * dt * g,
    );
  }
  val = min(val, vec4f(3.0, 3.0, 3.0, 2.0));

  // Jamais d'encre ni de chaleur dans la sphère-obstacle.
  if (P.sphere.w > 0.0 && distance(center, P.sphere.xyz) < P.sphere.w) {
    val = vec4f(0.0);
  }

  textureStore(den_dst, gid, max(val, vec4f(0.0)));
}
