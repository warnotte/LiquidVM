// Pinceau à murs : écrit dans le champ d'obstacles (1 = solide, 0 = fluide) sous le
// pointeur. Pinceau dur (disque net) plutôt que gaussien — des murs crisp. r32float
// est le seul format garanti en accès storage read-write, ce qui permet de préserver
// les murs existants sans ping-pong.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s)
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;

@group(1) @binding(0) var obstacle: texture_storage_2d<r32float, read_write>;

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(P.grid.xy);
  if (gid.x >= size.x || gid.y >= size.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let off = (vec2f(gid.xy) + vec2f(0.5)) - P.pointer.xy;
  // Pinceau un peu plus fin que le splat de fluide : des structures précises.
  let radius = P.impulse.w * 0.8;
  if (dot(off, off) > radius * radius) {
    return;
  }
  var value = textureLoad(obstacle, c).x;
  if (P.extra.y > 0.5) {
    value = 1.0;
  } else if (P.extra.y < -0.5) {
    value = 0.0;
  }
  textureStore(obstacle, c, vec4f(value, 0.0, 0.0, 0.0));
}
