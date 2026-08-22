// Vorticity confinement 3D. En 3D la vorticité est un VECTEUR : ω = ∇×v.
// curl : ω aux centres des cellules (différences centrées des vitesses centrées),
//        |ω| dans le canal w.
// blur_curl : |ω| flouté (boîte 3³) — LE fix du grain : le gradient de |ω| brut
//        à ±1 voxel injectait du bruit à l'échelle de la grille (calibré EPS-*,
//        d'où l'ancien défaut ε=0) ; flouter la magnitude AVANT d'en prendre le
//        gradient donne des normales N̂ cohérentes sur ~3 voxels. ω passe tel quel.
// confine : force F = ε (N̂ × ω) avec N̂ = ∇|ω|flou normalisé — pousse vers les
//        cœurs de tourbillons, restitue l'énergie que l'advection dissipe.
// Chaque composante de force est évaluée à SA position de face MAC.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité ε
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,     // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,
  sphere_vel: vec4f, // xyz: vitesse de la sphère (voxels/s)
}

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
@group(1) @binding(1) var curl_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var curl_src: texture_3d<f32>;
@group(1) @binding(3) var vel_dst: texture_storage_3d<rgba16float, write>;

fn n_size() -> i32 {
  return i32(P.misc.z);
}
fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(P.misc.z);
}

// Vitesse au centre de la cellule c (moyenne des deux faces de chaque axe ;
// coordonnées clampées : au bord, la face manquante est un mur → biais négligeable).
fn center_vel(c: vec3i) -> vec3f {
  let n = n_size();
  let cc = clamp(c, vec3i(0), vec3i(n - 1));
  let v0 = textureLoad(vel_src, cc, 0).xyz;
  let ux = select(0.0, textureLoad(vel_src, clamp(cc + vec3i(1, 0, 0), vec3i(0), vec3i(n - 1)), 0).x, cc.x + 1 < n);
  let vy = select(0.0, textureLoad(vel_src, clamp(cc + vec3i(0, 1, 0), vec3i(0), vec3i(n - 1)), 0).y, cc.y + 1 < n);
  let wz = select(0.0, textureLoad(vel_src, clamp(cc + vec3i(0, 0, 1), vec3i(0), vec3i(n - 1)), 0).z, cc.z + 1 < n);
  return 0.5 * vec3f(v0.x + ux, v0.y + vy, v0.z + wz);
}

@compute @workgroup_size(4, 4, 4)
fn curl(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  // Différences centrées des vitesses centrées voisines.
  let xp = center_vel(c + vec3i(1, 0, 0));
  let xm = center_vel(c - vec3i(1, 0, 0));
  let yp = center_vel(c + vec3i(0, 1, 0));
  let ym = center_vel(c - vec3i(0, 1, 0));
  let zp = center_vel(c + vec3i(0, 0, 1));
  let zm = center_vel(c - vec3i(0, 0, 1));
  let omega = 0.5 * vec3f(
    (yp.z - ym.z) - (zp.y - zm.y),
    (zp.x - zm.x) - (xp.z - xm.z),
    (xp.y - xm.y) - (yp.x - ym.x),
  );
  textureStore(curl_dst, gid, vec4f(omega, length(omega)));
}

// |ω| flouté (boîte 3³, 27 loads), ω du centre passé tel quel : le confinement
// lit UNE texture. Entrée : la texture de rotationnel via le binding 0 (même
// layout que curl), sortie : velScratch (libre à ce point de la frame).
@compute @workgroup_size(4, 4, 4)
fn blur_curl(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(c + vec3i(dx, dy, dz), vec3i(0), vec3i(n - 1));
        sum += textureLoad(vel_src, q, 0).w;
      }
    }
  }
  let omega = textureLoad(vel_src, c, 0).xyz;
  textureStore(curl_dst, gid, vec4f(omega, sum / 27.0));
}

fn omega_at(p: vec3f) -> vec3f {
  return textureSampleLevel(curl_src, lin, p * inv_n(), 0.0).xyz;
}

// |ω| FLOUTÉ (canal w écrit par blur_curl) — jamais la magnitude brute.
fn omega_len(p: vec3f) -> f32 {
  return textureSampleLevel(curl_src, lin, p * inv_n(), 0.0).w;
}

// Force de confinement à la position p : ε (N̂ × ω), N̂ = ∇|ω|flou normalisé.
fn confine_force(p: vec3f) -> vec3f {
  let grad = 0.5 * vec3f(
    omega_len(p + vec3f(1.0, 0.0, 0.0)) - omega_len(p - vec3f(1.0, 0.0, 0.0)),
    omega_len(p + vec3f(0.0, 1.0, 0.0)) - omega_len(p - vec3f(0.0, 1.0, 0.0)),
    omega_len(p + vec3f(0.0, 0.0, 1.0)) - omega_len(p - vec3f(0.0, 0.0, 1.0)),
  );
  let nrm = grad / max(length(grad), 1e-5);
  return P.misc.w * cross(nrm, omega_at(p));
}

@compute @workgroup_size(4, 4, 4)
fn confine(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let fc = vec3f(c);
  var vel = textureLoad(vel_src, c, 0).xyz;
  vel.x += dt * confine_force(fc + vec3f(0.0, 0.5, 0.5)).x;
  vel.y += dt * confine_force(fc + vec3f(0.5, 0.0, 0.5)).y;
  vel.z += dt * confine_force(fc + vec3f(0.5, 0.5, 0.0)).z;

  vel.x = select(vel.x, P.sphere_vel.x, face_blocked(c, vec3i(1, 0, 0)));
  vel.y = select(vel.y, P.sphere_vel.y, face_blocked(c, vec3i(0, 1, 0)));
  vel.z = select(vel.z, P.sphere_vel.z, face_blocked(c, vec3i(0, 0, 1)));
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
