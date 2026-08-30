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
  // Traversée jusqu'à `shape` (#23 = floats 92-95, l'ancien slot
  // d'explosion unique, libre depuis les charges multiples).
  pad_shape: array<vec4f, 10>,
  shape: vec4f,     // x: type d'obstacle, yzw: paramètres
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

// OBSTACLE — DISTANCE SIGNÉE en voxels, négative dedans (voir PLAN-OBSTACLES).
// Le prédicat n'est plus « dans la sphère » mais « sd < 0 » : l'opérateur ne
// change pas, seul le masque binaire de cellules change. `sphere.w` reste LE
// rayon (≤ 0 = pas d'obstacle), les paramètres de forme sont des RATIOS de ce
// rayon. Dupliquée dans chaque passe — les shaders sont autonomes.
fn solid_sd(p: vec3f) -> f32 {
  let r = P.sphere.w;
  if (r <= 0.0) {
    return 1e9;
  }
  let q = p - P.sphere.xyz;
  let kind = i32(P.shape.x + 0.5);
  if (kind == 1) { // BOÎTE : demi-côtés = rayon × ratios
    let d = abs(q) - r * P.shape.yzw;
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  if (kind == 2) { // TORE d'axe vertical : grand rayon r, petit rayon r × y
    return length(vec2f(length(q.xz) - r, q.y)) - r * P.shape.y;
  }
  return length(q) - r; // SPHÈRE
}


// TAILLE DE LA GRILLE, par axe. Le domaine n'est plus forcément cubique :
// Nx = Nz = misc.z, Ny = misc.z × misc.w. Les CELLULES, elles, restent cubiques
// — c'est ce qui permet de ne rien changer à l'opérateur ni à l'advection.
fn n_size() -> vec3i {
  return vec3i(i32(P.misc.z), i32(P.misc.z * P.misc.w), i32(P.misc.z));
}

fn n_sizef() -> vec3f {
  return vec3f(P.misc.z, P.misc.z * P.misc.w, P.misc.z);
}
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var den_src: texture_3d<f32>;
@group(1) @binding(2) var<storage, read_write> embers: array<Ember>;
@group(1) @binding(3) var<uniform> R: RenderParams;

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
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
  // RÈGLE de la grille non cubique : les POSITIONS et les BORNES sont per-axe,
  // mais les VITESSES et les longueurs gardent le N horizontal — les cellules
  // sont cubiques, donc « un voxel » vaut la même distance dans les trois axes.
  // Utiliser Ny pour une vitesse verticale la multiplierait par la hauteur du
  // domaine, ce qui n'aurait aucun sens.
  let n = P.misc.z;
  let seed = i * 1664525u + bitcast<u32>(P.misc.y);

  let alive = e.vel.w > 0.0 && e.pos.w < e.vel.w;
  if (!alive) {
    // Renaissance par rejet : une position tirée, il faut du CHAUD dessous.
    let r = vec3f(rand01(seed), rand01(seed ^ 0x9e3779b9u), rand01(seed ^ 0x85ebca6bu));
    let heat = textureSampleLevel(den_src, lin, r, 0.0).w;
    if (heat > 0.55 && rand01(seed ^ 0xc2b2ae35u) < R.style.z * 0.30) {
      let p = r * n_sizef();
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
      (solid_sd(pos) < 0.0)) {
    age = e.vel.w + 1.0;
  }
  embers[i] = Ember(vec4f(pos, age), vec4f(v, e.vel.w));
}
