// Forces : poussée thermique (la chaleur monte, +y) + impulsion de l'émetteur
// (jet vertical avec balancement latéral périodique — l'asymétrie qui déclenche
// les instabilités naturelles du panache). Appliquées aux faces MAC concernées.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: libre
  emitter: vec4f,   // xyz: centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit fumée, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: dissipation fumée, z: refroidissement, w: buoyancy
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var den_src: texture_3d<f32>;
@group(1) @binding(2) var vel_dst: texture_storage_3d<rgba16float, write>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(f32(P.misc.z));
}

fn heat_at(p: vec3f) -> f32 {
  // Les densités sont aux centres des cellules : échantillonnage direct.
  return textureSampleLevel(den_src, lin, p * inv_n(), 0.0).w;
}

fn emitter_gauss(p: vec3f) -> f32 {
  let d = distance(p, P.emitter.xyz) / max(P.emitter.w, 1e-3);
  return exp(-d * d * 3.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.misc.z);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let t = P.misc.y;
  let fc = vec3f(c);
  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);

  var vel = textureLoad(vel_src, c, 0).xyz;

  // Poussée thermique sur les faces verticales.
  vel.y += dt * P.diss.w * heat_at(pv);

  // Impulsion de l'émetteur : jet montant + balancement latéral (fréquences
  // incommensurables pour ne jamais boucler visiblement).
  let lat = P.emit_vals.w * vec2f(sin(t * 2.9 + 1.3), cos(t * 2.3));
  vel.x += dt * emitter_gauss(pu) * lat.x;
  vel.y += dt * emitter_gauss(pv) * P.emit_vals.z;
  vel.z += dt * emitter_gauss(pw) * lat.y;

  // Boîte fermée : les faces frontières restent des murs.
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
