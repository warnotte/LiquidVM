// Rotationnel sur grille MAC, évalué aux NŒUDS (coins de cellules, position (i,j)) —
// c'est là qu'il est naturellement défini avec un stencil compact exact :
//   ω(i,j) = (v(i,j) − v(i−1,j)) − (u(i,j) − u(i,j−1))
// (les quatre faces adjacentes au nœud, h = 1). Stocké en rgba16float (.x) pour être
// échantillonnable bilinéairement par la passe de confinement, qui en a besoin aux
// positions de faces.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s)
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;

@group(1) @binding(0) var src_vel: texture_2d<f32>;
@group(1) @binding(1) var dst_curl: texture_storage_2d<rgba16float, write>;

// Coordonnée d'un voisin : wrap en périodique, clamp sinon (parois et ouvert).
fn neighbor(q: vec2i, size: vec2i) -> vec2i {
  if (P.extra.x > 0.5 && P.extra.x < 1.5) {
    return ((q % size) + size) % size;
  }
  return clamp(q, vec2i(0), size - vec2i(1));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(P.grid.xy);
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  let here = textureLoad(src_vel, c, 0).xy;
  let v_left = textureLoad(src_vel, neighbor(vec2i(c.x - 1, c.y), size), 0).y;
  let u_up = textureLoad(src_vel, neighbor(vec2i(c.x, c.y - 1), size), 0).x;

  let curl = (here.y - v_left) - (here.x - u_up);
  textureStore(dst_curl, c, vec4f(curl, 0.0, 0.0, 0.0));
}
