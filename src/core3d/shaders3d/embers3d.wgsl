// BRAISES : particules-étincelles nées dans les zones chaudes, portées par le
// fluide avec un scintillement propre, mortes en refroidissant. Le nombre de
// naissances s'autorégule : chaque particule morte tire UNE position aléatoire
// par frame et ne naît que si elle tombe dans du chaud — plus de feu = plus de
// braises, feu éteint = plus aucune. Positions en VOXELS (l'espace des vitesses).
// Buffer zéro-initialisé = toutes mortes (life 0). R.style.z = débit (slider).

struct Params {
  misc: vec4f, // x: dt, y: temps, z: N, w: vorticité (inutilisé ici)
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f, // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,
}

struct RenderParams {
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
  light: vec4f,
  sphere: vec4f,
  blow_a: vec4f,
  blow_b: vec4f,
  style: vec4f, // z: débit des braises
}

struct Ember {
  pos: vec4f, // xyz: voxels, w: âge (s)
  vel: vec4f, // xyz: voxels/s, w: durée de vie (s, 0 = jamais née)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var den_src: texture_3d<f32>;
@group(1) @binding(2) var<storage, read_write> embers: array<Ember>;
@group(1) @binding(3) var<uniform> R: RenderParams;

fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(P.misc.z);
}

// Convention MAC identique à advect_density3d.wgsl.
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(
    textureSampleLevel(vel_src, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x,
    textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y,
    textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z,
  );
}

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&embers)) {
    return;
  }
  var e = embers[i];
  let dt = P.misc.x;
  let n = P.misc.z;
  let seed = i * 1664525u + bitcast<u32>(P.misc.y);

  let alive = e.vel.w > 0.0 && e.pos.w < e.vel.w;
  if (!alive) {
    // Renaissance par rejet : une position tirée, il faut du CHAUD dessous.
    let r = vec3f(rand01(seed), rand01(seed ^ 0x9e3779b9u), rand01(seed ^ 0x85ebca6bu));
    let heat = textureSampleLevel(den_src, lin, r, 0.0).w;
    if (heat > 0.55 && rand01(seed ^ 0xc2b2ae35u) < R.style.z * 0.30) {
      let p = r * n;
      let kick = vec3f(
        (rand01(seed ^ 0x27d4eb2fu) - 0.5) * 0.08 * n,
        rand01(seed ^ 0x165667b1u) * 0.10 * n,
        (rand01(seed ^ 0xd3a2646cu) - 0.5) * 0.08 * n,
      );
      e.pos = vec4f(p, 0.0);
      e.vel = vec4f(velocity_at(p) + kick, 0.9 + 1.8 * rand01(seed ^ 0xfd7046c5u));
    }
    embers[i] = e;
    return;
  }

  // Vivante : relaxation vers le fluide (traînée) + scintillement propre.
  let h1 = rand01(i * 2654435761u);
  let h2 = rand01(i * 2246822519u);
  let t = P.misc.y;
  let wob = vec3f(
    sin(t * (6.0 + 8.0 * h1) + h2 * 6.283),
    0.5 * sin(t * (9.0 + 6.0 * h2) + h1 * 6.283),
    cos(t * (7.0 + 7.0 * h1) + (h1 + h2) * 3.14),
  ) * 0.02 * n;
  // NB : « target » est un mot réservé WGSL (comme « from » — piège documenté).
  let k = 1.0 - exp(-6.0 * dt);
  let goal = velocity_at(e.pos.xyz) * 1.03 + wob + vec3f(0.0, 0.015 * n, 0.0);
  let v = mix(e.vel.xyz, goal, k);
  var pos = e.pos.xyz + v * dt;
  var age = e.pos.w + dt;
  // Hors boîte ou dans la boule : morte (renaîtra ailleurs).
  let margin = 1.0;
  if (any(pos < vec3f(margin)) || any(pos > vec3f(n - margin)) ||
      (P.sphere.w > 0.0 && distance(pos, P.sphere.xyz) < P.sphere.w)) {
    age = e.vel.w + 1.0;
  }
  embers[i] = Ember(vec4f(pos, age), vec4f(v, e.vel.w));
}
