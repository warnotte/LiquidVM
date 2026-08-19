// Advection du système de particules : chaque particule échantillonne le champ de
// vélocité MAC à sa position (composantes u/v avec leur décalage d'un demi-texel) et
// s'intègre en RK2 (point milieu) — bien plus fidèle aux tourbillons que l'Euler simple.
// Les positions et vitesses sont stockées NORMALISÉES [0,1] : le rendu est indépendant
// des résolutions de grilles. Respawn (position pseudo-aléatoire par hash PCG) quand la
// particule meurt de vieillesse, entre dans un mur, ou atteint un bord en mode ouvert.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s), w: outil
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s²), w: temps simulé (s, graine de hash)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

// posvel: xy = position normalisée, zw = vitesse normalisée/s ; misc: x = âge (s), y = durée de vie (s).
struct Particle {
  posvel: vec4f,
  misc: vec4f,
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var lin: sampler;

@group(1) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(1) var src_vel: texture_2d<f32>;
@group(1) @binding(2) var obstacle: texture_2d<f32>;

fn sample_u(p: vec2f) -> f32 {
  return textureSampleLevel(src_vel, lin, (p + vec2f(0.5, 0.0)) * P.grid.zw, 0.0).x;
}
fn sample_v(p: vec2f) -> f32 {
  return textureSampleLevel(src_vel, lin, (p + vec2f(0.0, 0.5)) * P.grid.zw, 0.0).y;
}
fn velocity_at(p: vec2f) -> vec2f {
  return vec2f(sample_u(p), sample_v(p));
}

fn solid_at(p: vec2f) -> bool {
  let size = vec2i(P.grid.xy);
  let c = clamp(vec2i(p), vec2i(0), size - vec2i(1));
  return textureLoad(obstacle, c, 0).x > 0.5;
}

// Hash PCG : générateur déterministe par index — pas de Math.random côté GPU.
fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) / 4294967295.0;
}

// Synchronisé avec PARTICLE_WORKGROUP (core/config.ts).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&particles)) {
    return;
  }
  let dt = P.impulse.x;
  var p = particles[i];
  var pos = p.posvel.xy * P.grid.xy; // → texels sim
  var age = p.misc.x + dt;
  let life = p.misc.y;

  // Intégration RK2 (point milieu).
  let v1 = velocity_at(pos);
  let v2 = velocity_at(pos + v1 * (0.5 * dt));
  pos += v2 * dt;

  // Frontières : wrap en périodique, clamp sinon.
  let periodic = P.extra.x > 0.5 && P.extra.x < 1.5;
  if (periodic) {
    pos = fract(pos * P.grid.zw) * P.grid.xy;
  } else {
    pos = clamp(pos, vec2f(0.5), P.grid.xy - vec2f(0.5));
  }

  // Respawn : vie épuisée (ou jamais initialisée), mur, ou bord atteint en mode ouvert
  // (la bande éponge y absorbe le fluide — les particules « sortent » avec lui).
  let margin = P.grid.x / 64.0;
  let at_border = P.extra.x > 1.5 &&
    (min(pos.x, pos.y) < margin || max(pos.x - P.grid.x, pos.y - P.grid.y) > -margin);
  if (life <= 0.0 || age > life || solid_at(pos) || at_border) {
    let seed = i * 4u + u32(P.buoyancy.w * 977.0);
    pos = vec2f(rand01(seed), rand01(seed + 1u)) * P.grid.xy;
    age = rand01(seed + 2u) * 2.0; // désynchronise les morts groupées
    p.misc.y = 6.0 + 8.0 * rand01(seed + 3u);
  }

  p.posvel = vec4f(pos * P.grid.zw, velocity_at(pos) * P.grid.z);
  p.misc.x = age;
  particles[i] = p;
}
