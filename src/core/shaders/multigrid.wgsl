// Solveur multigrid géométrique pour l'équation de Poisson de la pression, A·p = div,
// avec A·p = Σp_voisins − 4p (Laplacien 5 points à pas unitaire, mêmes conditions aux
// limites que jacobi.wgsl : Neumann parois/murs, wrap périodique, Dirichlet 0 en ouvert).
// Cinq entry points, tous dimensionnés par textureDimensions (aucun uniform de taille —
// le même pipeline sert à tous les niveaux de la hiérarchie) :
//   smooth            — Jacobi pondéré ω = 0.8 : écrase les hautes fréquences du résidu
//   residual          — r = rhs − A·p
//   restrict_rhs      — moyenne 2×2 du résidu fin ×4 (mise à l'échelle h² de l'opérateur
//                       unitaire : le niveau grossier résout A·e = 4·R(r))
//   prolong_add       — p_fin += interpolation bilinéaire de la correction grossière
//   restrict_obstacle — masque grossier : solide si ≥ la moitié des 4 cellules fines l'est

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

@group(1) @binding(0) var mg_p: texture_2d<f32>;    // pression | résidu fin | correction grossière
@group(1) @binding(1) var mg_rhs: texture_2d<f32>;  // second membre | pression fine (prolong)
@group(1) @binding(2) var mg_write: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var mg_obstacle: texture_2d<f32>;

const OMEGA: f32 = 0.8; // poids du Jacobi amorti — le lisseur multigrid standard

fn mode_periodic() -> bool {
  return P.extra.x > 0.5 && P.extra.x < 1.5;
}
fn mode_open() -> bool {
  return P.extra.x > 1.5;
}

// Pression du voisin avec conditions aux limites (taille du niveau en paramètre).
// Le mode OUVERT utilise le MÊME opérateur Neumann que les parois : le domaine est une
// boîte fermée, la sortie du fluide est assurée par la bande éponge des passes
// d'advection. (Une vraie sortie Dirichlet avec faces virtuelles extrapolées rend le
// système non symétrique — la dernière colonne se découple — et le V-cycle diverge.)
fn p_neighbor(c: vec2i, off: vec2i, size: vec2i, center: f32) -> f32 {
  var q = c + off;
  if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) {
    if (mode_periodic()) {
      q = (q + size) % size;
    } else {
      return center; // parois et ouvert : Neumann
    }
  }
  if (textureLoad(mg_obstacle, q, 0).x > 0.5) {
    return center; // mur : Neumann
  }
  return textureLoad(mg_p, q, 0).x;
}

fn neighbor_sum(c: vec2i, size: vec2i, center: f32) -> f32 {
  return p_neighbor(c, vec2i(-1, 0), size, center) +
    p_neighbor(c, vec2i(1, 0), size, center) +
    p_neighbor(c, vec2i(0, -1), size, center) +
    p_neighbor(c, vec2i(0, 1), size, center);
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts). (`smooth` est un mot réservé WGSL.)
@compute @workgroup_size(16, 16)
fn smooth_jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(mg_p));
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  // Ancrage de l'espace nul (l'opérateur est Neumann ou périodique dans TOUS les
  // modes : la pression n'est définie qu'à une constante près, et la restriction ×4
  // du multigrid fait dériver cette constante très vite). Au niveau le plus grossier
  // (taille MG_COARSEST_SIZE, cf. config.ts), la cellule (0,0) est épinglée à 0 :
  // la constante de toute la hiérarchie est fixée, la pression reste d'amplitude
  // physique — sinon elle finit par dépasser le float16 de la vélocité, et NaN.
  if (size.x <= 8 && all(c == vec2i(0))) {
    textureStore(mg_write, c, vec4f(0.0));
    return;
  }
  let pc = textureLoad(mg_p, c, 0).x;
  if (textureLoad(mg_obstacle, c, 0).x > 0.5) {
    textureStore(mg_write, c, vec4f(pc, 0.0, 0.0, 0.0));
    return;
  }
  let rhs = textureLoad(mg_rhs, c, 0).x;
  let jacobi = (neighbor_sum(c, size, pc) - rhs) * 0.25;
  textureStore(mg_write, c, vec4f(mix(pc, jacobi, OMEGA), 0.0, 0.0, 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn residual(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(mg_p));
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  if (textureLoad(mg_obstacle, c, 0).x > 0.5) {
    textureStore(mg_write, c, vec4f(0.0));
    return;
  }
  let pc = textureLoad(mg_p, c, 0).x;
  let rhs = textureLoad(mg_rhs, c, 0).x;
  let r = rhs - (neighbor_sum(c, size, pc) - 4.0 * pc);
  textureStore(mg_write, c, vec4f(r, 0.0, 0.0, 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn restrict_rhs(@builtin(global_invocation_id) gid: vec3u) {
  let csize = vec2i(textureDimensions(mg_write));
  let c = vec2i(gid.xy);
  if (c.x >= csize.x || c.y >= csize.y) {
    return;
  }
  let f = c * 2;
  let sum = textureLoad(mg_p, f, 0).x +
    textureLoad(mg_p, f + vec2i(1, 0), 0).x +
    textureLoad(mg_p, f + vec2i(0, 1), 0).x +
    textureLoad(mg_p, f + vec2i(1, 1), 0).x;
  // moyenne (×0.25) puis ×4 pour l'échelle h² de l'opérateur unitaire → ×1.
  textureStore(mg_write, c, vec4f(sum, 0.0, 0.0, 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn prolong_add(@builtin(global_invocation_id) gid: vec3u) {
  let fsize = vec2i(textureDimensions(mg_write));
  let c = vec2i(gid.xy);
  if (c.x >= fsize.x || c.y >= fsize.y) {
    return;
  }
  let csize = vec2i(textureDimensions(mg_p));
  // Coordonnée texel continue dans la grille grossière (cell-centered, facteur 2).
  let g = (vec2f(c.xy) + vec2f(0.5)) * 0.5 - vec2f(0.5);
  let g0 = vec2i(floor(g));
  let f = g - floor(g);
  var corner: array<vec2i, 4> = array<vec2i, 4>(
    vec2i(0, 0), vec2i(1, 0), vec2i(0, 1), vec2i(1, 1));
  var e: array<f32, 4>;
  for (var k = 0; k < 4; k++) {
    var q = g0 + corner[k];
    if (mode_periodic()) {
      q = ((q % csize) + csize) % csize;
    } else {
      // Neumann (parois et ouvert) : clamp = extension à dérivée nulle, cohérente
      // avec l'opérateur des lisseurs.
      q = clamp(q, vec2i(0), csize - vec2i(1));
    }
    e[k] = textureLoad(mg_p, q, 0).x;
  }
  let interp = mix(mix(e[0], e[1], f.x), mix(e[2], e[3], f.x), f.y);
  let fine = textureLoad(mg_rhs, c, 0).x;
  textureStore(mg_write, c, vec4f(fine + interp, 0.0, 0.0, 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn restrict_obstacle(@builtin(global_invocation_id) gid: vec3u) {
  let csize = vec2i(textureDimensions(mg_write));
  let c = vec2i(gid.xy);
  if (c.x >= csize.x || c.y >= csize.y) {
    return;
  }
  let f = c * 2;
  let sum = textureLoad(mg_p, f, 0).x +
    textureLoad(mg_p, f + vec2i(1, 0), 0).x +
    textureLoad(mg_p, f + vec2i(0, 1), 0).x +
    textureLoad(mg_p, f + vec2i(1, 1), 0).x;
  textureStore(mg_write, c, vec4f(select(0.0, 1.0, sum >= 2.0), 0.0, 0.0, 0.0));
}
