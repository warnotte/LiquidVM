// Projection 3D : divergence MAC compacte, lisseur Jacobi (Neumann par clamp des
// voisins — la boîte fermée), soustraction du gradient. Trois entry points, un
// layout commun (les bindings inutilisés par un entry point sont ignorés).
// Schéma compact : la face N n'est pas stockée et vaut 0 (mur) dans la divergence,
// exactement comme en 2D — le couple divergence/gradient reste l'adjoint du lisseur.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: libre
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var pressure_src: texture_3d<f32>;
@group(1) @binding(2) var divergence_src: texture_3d<f32>;
@group(1) @binding(3) var scalar_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var vel_dst: texture_storage_3d<rgba16float, write>;

fn n_size() -> i32 {
  return i32(P.misc.z);
}

fn in_bounds(c: vec3i) -> bool {
  let n = n_size();
  return c.x < n && c.y < n && c.z < n;
}

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let n = n_size();
  let v0 = textureLoad(vel_src, c, 0).xyz;
  // Faces opposées : hors grille = mur (0), schéma compact.
  let ux = select(0.0, textureLoad(vel_src, c + vec3i(1, 0, 0), 0).x, c.x + 1 < n);
  let vy = select(0.0, textureLoad(vel_src, c + vec3i(0, 1, 0), 0).y, c.y + 1 < n);
  let wz = select(0.0, textureLoad(vel_src, c + vec3i(0, 0, 1), 0).z, c.z + 1 < n);
  textureStore(scalar_dst, gid, vec4f(ux - v0.x + vy - v0.y + wz - v0.z, 0.0, 0.0, 0.0));
}

fn p_at(c: vec3i) -> f32 {
  // Clamp = condition de Neumann (le voisin hors boîte vaut la valeur du bord).
  let n = n_size();
  return textureLoad(pressure_src, clamp(c, vec3i(0), vec3i(n - 1)), 0).x;
}

@compute @workgroup_size(4, 4, 4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let sum = p_at(c + vec3i(1, 0, 0)) + p_at(c - vec3i(1, 0, 0)) +
    p_at(c + vec3i(0, 1, 0)) + p_at(c - vec3i(0, 1, 0)) +
    p_at(c + vec3i(0, 0, 1)) + p_at(c - vec3i(0, 0, 1));
  let div = textureLoad(divergence_src, c, 0).x;
  textureStore(scalar_dst, gid, vec4f((sum - div) / 6.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn gradient(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let pc = p_at(c);
  var vel = textureLoad(vel_src, c, 0).xyz;
  vel.x = select(0.0, vel.x - (pc - p_at(c - vec3i(1, 0, 0))), c.x > 0);
  vel.y = select(0.0, vel.y - (pc - p_at(c - vec3i(0, 1, 0))), c.y > 0);
  vel.z = select(0.0, vel.z - (pc - p_at(c - vec3i(0, 0, 1))), c.z > 0);
  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
