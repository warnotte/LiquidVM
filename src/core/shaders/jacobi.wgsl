// Une itération de Jacobi pour l'équation de Poisson de la pression :
// ∇²p = div  ⇒  p' = (p_gauche + p_droite + p_haut + p_bas − div) / 4  (h = 1).
// Répétée N fois en ping-pong, elle converge vers le champ de pression dont le gradient
// annulera la divergence. Conditions aux limites selon le mode :
//  - parois / murs dessinés : Neumann (∂p/∂n = 0) — le voisin solide ou hors-bord
//    prend la pression de la cellule courante (clamp des coordonnées) ;
//  - périodique : wrap ;
//  - ouvert : Dirichlet p = 0 hors du domaine (sortie libre, rien ne retient le fluide).

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s)
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;

@group(1) @binding(0) var src_pressure: texture_2d<f32>;
@group(1) @binding(1) var src_div: texture_2d<f32>;
@group(1) @binding(2) var dst_pressure: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var obstacle: texture_2d<f32>;

// Pression du voisin en (c + off), selon le mode de frontières et les murs.
// Le mode ouvert utilise le MÊME opérateur Neumann que les parois (boîte fermée +
// bande éponge dans l'advection) : clamp des coordonnées = Neumann naturel.
fn neighbor_pressure(c: vec2i, off: vec2i, size: vec2i, center: f32) -> f32 {
  var q = c + off;
  if (P.extra.x > 0.5 && P.extra.x < 1.5) {
    q = ((q % size) + size) % size;
  } else {
    q = clamp(q, vec2i(0), size - vec2i(1));
  }
  if (textureLoad(obstacle, q, 0).x > 0.5) {
    return center; // Neumann sur les murs dessinés
  }
  return textureLoad(src_pressure, q, 0).x;
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(P.grid.xy);
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  let pc = textureLoad(src_pressure, c, 0).x;
  let pl = neighbor_pressure(c, vec2i(-1, 0), size, pc);
  let pr = neighbor_pressure(c, vec2i(1, 0), size, pc);
  let pt = neighbor_pressure(c, vec2i(0, -1), size, pc);
  let pb = neighbor_pressure(c, vec2i(0, 1), size, pc);
  let div = textureLoad(src_div, c, 0).x;

  let p = (pl + pr + pt + pb - div) * 0.25;
  textureStore(dst_pressure, c, vec4f(p, 0.0, 0.0, 0.0));
}
