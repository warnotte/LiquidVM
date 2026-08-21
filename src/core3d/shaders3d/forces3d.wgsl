// Forces : poussée thermique (la chaleur monte, +y) + impulsion de l'émetteur
// (jet vertical avec balancement latéral périodique — l'asymétrie qui déclenche
// les instabilités naturelles du panache). Appliquées aux faces MAC concernées.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,   // émetteur 0 : xyz centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit d'encre, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f, // xyz: origine du rayon du pointeur (voxels), w: rayon du pinceau (voxels)
  blow_dir: vec4f,    // xyz: direction du rayon (normalisée), w: actif (0/1)
  blow_force: vec4f,  // xyz: force du souffle (voxels/s²), w: libre
  sphere: vec4f,      // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,   // x: nombre d'émetteurs actifs (1..4)
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,   // encre (0/1/2) de chaque émetteur
  sphere_vel: vec4f,  // xyz: vitesse de la sphère (voxels/s) — condition de bord mobile
  ink_weights: vec4f, // xyz: poids propre des matières (voxels/s² par unité)
}

fn emitter_pos(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.emitter; }
    case 1u: { return P.emitter1; }
    case 2u: { return P.emitter2; }
    default: { return P.emitter3; }
  }
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
@group(1) @binding(1) var den_src: texture_3d<f32>;
@group(1) @binding(2) var vel_dst: texture_storage_3d<rgba16float, write>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(f32(P.misc.z));
}

fn density_at(p: vec3f) -> vec4f {
  // Les densités sont aux centres des cellules : échantillonnage direct.
  return textureSampleLevel(den_src, lin, p * inv_n(), 0.0);
}

// Poids gaussien cumulé de TOUS les émetteurs actifs à la position p.
fn emitter_gauss(p: vec3f) -> f32 {
  var g = 0.0;
  let count = u32(P.emit_meta.x + 0.5);
  for (var i = 0u; i < count; i++) {
    let e = emitter_pos(i);
    let d = distance(p, e.xyz) / max(e.w, 1e-3);
    g += exp(-d * d * 3.0);
  }
  return min(g, 1.5);
}

// Souffle du pointeur : poids gaussien autour du RAYON caméra→scène (un tube),
// nul derrière la caméra. Le geste pousse le fluide partout où son rayon passe.
fn blow_gauss(p: vec3f) -> f32 {
  if (P.blow_dir.w < 0.5) {
    return 0.0;
  }
  let v = p - P.blow_origin.xyz;
  let t = dot(v, P.blow_dir.xyz);
  if (t < 0.0) {
    return 0.0;
  }
  let d = distance(p, P.blow_origin.xyz + P.blow_dir.xyz * t) / max(P.blow_origin.w, 1e-3);
  return exp(-d * d * 2.0);
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

  // Boussinesq à deux voies : la chaleur pousse vers le haut, le poids propre des
  // matières tire vers le bas (l'encre lourde retombe, le carburant froid coule).
  let dn = density_at(pv);
  vel.y += dt * (P.diss.w * dn.w - dot(P.ink_weights.xyz, dn.xyz));

  // Impulsion de l'émetteur : jet montant + balancement latéral (fréquences
  // incommensurables pour ne jamais boucler visiblement).
  let lat = P.emit_vals.w * vec2f(sin(t * 2.9 + 1.3), cos(t * 2.3));
  vel.x += dt * emitter_gauss(pu) * lat.x;
  vel.y += dt * emitter_gauss(pv) * P.emit_vals.z;
  vel.z += dt * emitter_gauss(pw) * lat.y;

  // Souffle du pointeur (clic droit + glisser), par face.
  vel.x += dt * blow_gauss(pu) * P.blow_force.x;
  vel.y += dt * blow_gauss(pv) * P.blow_force.y;
  vel.z += dt * blow_gauss(pw) * P.blow_force.z;

  // Sphère mobile : les faces bloquées portent sa vitesse ; parois de boîte à 0.
  vel.x = select(vel.x, P.sphere_vel.x, face_blocked(c, vec3i(1, 0, 0)));
  vel.y = select(vel.y, P.sphere_vel.y, face_blocked(c, vec3i(0, 1, 0)));
  vel.z = select(vel.z, P.sphere_vel.z, face_blocked(c, vec3i(0, 0, 1)));
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
