// Soustraction du gradient de pression sur grille MAC : stencil COMPACT par face —
//   u(i,j) −= p(i,j) − p(i−1,j)   (face entre les cellules i−1 et i)
//   v(i,j) −= p(i,j) − p(i,j−1)
// L'adjoint exact de l'opérateur de divergence compact : le couple divergence/gradient
// est consistant, la projection retire toute la divergence visible (aux itérations de
// Jacobi près) et aucun mode en damier ne survit. Conditions aux limites : Neumann
// contre les murs et parois (différence nulle), wrap en périodique, p = 0 dehors en
// ouvert. Les faces bloquées sont forcées à zéro.

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

@group(1) @binding(0) var src_pressure: texture_2d<f32>;
@group(1) @binding(1) var src_vel: texture_2d<f32>;
@group(1) @binding(2) var dst_vel: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var obstacle: texture_2d<f32>;

fn bound_coord(q: vec2i, size: vec2i) -> vec2i {
  if (P.extra.x > 0.5 && P.extra.x < 1.5) {
    return ((q % size) + size) % size;
  }
  return clamp(q, vec2i(0), size - vec2i(1));
}

fn solid_at(q: vec2i, size: vec2i) -> bool {
  return textureLoad(obstacle, bound_coord(q, size), 0).x > 0.5;
}

fn u_face_blocked(c: vec2i, size: vec2i) -> bool {
  if (solid_at(c, size)) { return true; }
  if (c.x == 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return solid_at(vec2i(size.x - 1, c.y), size);
    }
    return true; // parois et ouvert : bord fermé
  }
  return solid_at(vec2i(c.x - 1, c.y), size);
}

fn v_face_blocked(c: vec2i, size: vec2i) -> bool {
  if (solid_at(c, size)) { return true; }
  if (c.y == 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return solid_at(vec2i(c.x, size.y - 1), size);
    }
    return true;
  }
  return solid_at(vec2i(c.x, c.y - 1), size);
}

// Pression de la cellule voisine en amont de la face (q peut déborder de −1).
// Le mode ouvert partage l'opérateur Neumann des parois (boîte fermée + bande éponge
// dans l'advection) — la face de bord est de toute façon bloquée, la valeur ne compte pas.
fn pressure_before(q: vec2i, size: vec2i, center: f32) -> f32 {
  if (q.x < 0 || q.y < 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return textureLoad(src_pressure, bound_coord(q, size), 0).x; // wrap
    }
    return center; // parois et ouvert : Neumann
  }
  if (solid_at(q, size)) {
    return center; // mur : Neumann
  }
  return textureLoad(src_pressure, q, 0).x;
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let usize = vec2u(P.grid.xy);
  if (gid.x >= usize.x || gid.y >= usize.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let size = vec2i(P.grid.xy);
  var vel = textureLoad(src_vel, c, 0).xy;
  let pc = textureLoad(src_pressure, c, 0).x;

  vel.x -= pc - pressure_before(vec2i(c.x - 1, c.y), size, pc);
  vel.y -= pc - pressure_before(vec2i(c.x, c.y - 1), size, pc);

  if (u_face_blocked(c, size)) { vel.x = 0.0; }
  if (v_face_blocked(c, size)) { vel.y = 0.0; }
  textureStore(dst_vel, c, vec4f(vel, 0.0, 0.0));
}
