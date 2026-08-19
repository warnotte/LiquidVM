// Divergence sur grille MAC : stencil COMPACT par cellule (h = 1) —
//   div(i,j) = u(i+1,j) − u(i,j) + v(i,j+1) − v(i,j)
// où les u/v sont les vitesses de faces adjacentes à la cellule. Contrairement aux
// différences centrées sur grille colocalisée, aucun mode en damier n'échappe à cet
// opérateur : la projection voit (et annule) toute la divergence représentable.
// Les faces bloquées ont déjà été forcées à zéro par les passes d'écriture ; il ne reste
// qu'à gérer la face manquante du bord droit/bas selon le mode de frontières.

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
@group(1) @binding(1) var dst_div: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var obstacle: texture_2d<f32>;

// Face u du texel q — q.x peut déborder de 1 (face droite de la dernière colonne).
// Parois ET ouvert : face de bord fermée (le mode ouvert est une boîte fermée + bande
// éponge dans l'advection — une face extrapolée rendrait le système de pression non
// symétrique et ferait diverger le multigrid).
fn u_at(q: vec2i, size: vec2i) -> f32 {
  if (q.x >= size.x) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return textureLoad(src_vel, vec2i(0, q.y), 0).x; // périodique : wrap
    }
    return 0.0;
  }
  return textureLoad(src_vel, q, 0).x;
}

fn v_at(q: vec2i, size: vec2i) -> f32 {
  if (q.y >= size.y) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return textureLoad(src_vel, vec2i(q.x, 0), 0).y;
    }
    return 0.0;
  }
  return textureLoad(src_vel, q, 0).y;
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(P.grid.xy);
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  if (textureLoad(obstacle, c, 0).x > 0.5) {
    textureStore(dst_div, c, vec4f(0.0));
    return;
  }
  let here = textureLoad(src_vel, c, 0).xy;
  let div = (u_at(vec2i(c.x + 1, c.y), size) - here.x) +
    (v_at(vec2i(c.x, c.y + 1), size) - here.y);
  textureStore(dst_div, c, vec4f(div, 0.0, 0.0, 0.0));
}
