// J1 — passes de GRILLE de l'eau : forces (gravité + parois), divergence sur
// les cellules d'EAU seulement, Jacobi à SURFACE LIBRE (air = Dirichlet p = 0,
// condition symétrique — voir PLAN-EAU.md ; hors-boîte = Neumann par le
// clamp-trick éprouvé du feu : lire le voisin clampé REND p_centre), gradient.
// Schéma MAC compact identique au feu : les faces N inexistantes sont des murs
// gratuits, les faces 0 sont forcées à zéro.

struct UEau {
  sim: vec4f,  // x: N, y: nb particules, z: dt sous-pas, w: gravité (voxels/s²)
  sim2: vec4f, // x: mélange FLIP
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var vel_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var<storage, read> cell_count: array<u32>;
@group(1) @binding(3) var div_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var p_src: texture_3d<f32>;
@group(1) @binding(5) var p_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(6) var div_src: texture_3d<f32>;

fn n_size() -> i32 {
  return i32(U.sim.x);
}

fn fluid(c: vec3i, n: i32) -> bool {
  if (any(c < vec3i(0)) || any(c >= vec3i(n))) {
    return false; // hors boîte = solide, pas eau
  }
  return cell_count[u32(c.x + n * (c.y + n * c.z))] > 0u;
}

fn solid(c: vec3i, n: i32) -> bool {
  return any(c < vec3i(0)) || any(c >= vec3i(n));
}

// Gravité sur la grille + parois : les faces 0 sont des murs (vitesse nulle).
@compute @workgroup_size(4, 4, 4)
fn forces(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  var vel = textureLoad(vel_src, c, 0).xyz;
  vel.y -= U.sim.w * U.sim.z;
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);
  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}

// Face haute inexistante (i = N) = mur : contribution nulle.
fn face_u(c: vec3i, n: i32) -> f32 {
  if (c.x >= n) {
    return 0.0;
  }
  return textureLoad(vel_src, c, 0).x;
}
fn face_v(c: vec3i, n: i32) -> f32 {
  if (c.y >= n) {
    return 0.0;
  }
  return textureLoad(vel_src, c, 0).y;
}
fn face_w(c: vec3i, n: i32) -> f32 {
  if (c.z >= n) {
    return 0.0;
  }
  return textureLoad(vel_src, c, 0).z;
}

// Divergence compacte, cellules d'EAU seulement (l'air n'est pas résolu) —
// avec CONTRÔLE DE DENSITÉ (Bridson) : la projection rend le champ de vitesse
// incompressible mais rien n'empêche les particules de s'agglutiner par dérive
// d'interpolation (fonte du volume, trous en « fromage »). Les cellules
// SURPEUPLÉES (> 8 particules = densité de repos) reçoivent une divergence
// cible positive : le solveur les fait s'étendre vers la densité de repos.
// Jamais de correction en sous-densité (elle combattrait la surface libre).
const REST_DENSITY = 8.0;
const DENSITY_RELAX = 0.3;

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  if (!fluid(c, n)) {
    textureStore(div_dst, gid, vec4f(0.0));
    return;
  }
  let d = face_u(c + vec3i(1, 0, 0), n) - textureLoad(vel_src, c, 0).x +
    face_v(c + vec3i(0, 1, 0), n) - textureLoad(vel_src, c, 0).y +
    face_w(c + vec3i(0, 0, 1), n) - textureLoad(vel_src, c, 0).z;
  let count = f32(cell_count[u32(c.x + n * (c.y + n * c.z))]);
  // Bornée à 1 : sans le clamp, une cellule très tassée (30+ particules)
  // demandait ~180 s⁻¹ d'expansion → pressions explosives → vitesses au-delà
  // du max float16 → NaN → particules invisibles (la saga float16 du 2D).
  let overdense = min(max(count - REST_DENSITY, 0.0) / REST_DENSITY, 1.0);
  let d_target = DENSITY_RELAX * overdense / max(U.sim.z, 1e-4);
  textureStore(div_dst, gid, vec4f(d - d_target, 0.0, 0.0, 0.0));
}

// Contribution d'un voisin au balayage Jacobi :
//  - solide (hors boîte) → p_centre (Neumann, clamp-trick),
//  - eau → sa pression,
//  - air → 0 (Dirichlet p = 0 : LA surface libre).
fn neighbor_p(c: vec3i, p_center: f32, n: i32) -> f32 {
  if (solid(c, n)) {
    return p_center;
  }
  if (!fluid(c, n)) {
    return 0.0;
  }
  return textureLoad(p_src, c, 0).x;
}

@compute @workgroup_size(4, 4, 4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  if (!fluid(c, n)) {
    textureStore(p_dst, gid, vec4f(0.0));
    return;
  }
  let pc = textureLoad(p_src, c, 0).x;
  let sum = neighbor_p(c + vec3i(1, 0, 0), pc, n) + neighbor_p(c - vec3i(1, 0, 0), pc, n) +
    neighbor_p(c + vec3i(0, 1, 0), pc, n) + neighbor_p(c - vec3i(0, 1, 0), pc, n) +
    neighbor_p(c + vec3i(0, 0, 1), pc, n) + neighbor_p(c - vec3i(0, 0, 1), pc, n);
  textureStore(p_dst, gid, vec4f((sum - textureLoad(div_src, c, 0).x) / 6.0, 0.0, 0.0, 0.0));
}

// Pression du point de vue du GRADIENT : hors boîte → Neumann (grad nul via
// p_centre) ; l'air a p = 0 dans la texture (écrit par jacobi) — lecture directe.
fn grad_p(c: vec3i, p_center: f32, n: i32) -> f32 {
  if (solid(c, n)) {
    return p_center;
  }
  return textureLoad(p_src, c, 0).x;
}

// Soustraction du gradient : sur toute face touchant au moins une cellule
// d'eau ; les faces 0 restent des murs.
@compute @workgroup_size(4, 4, 4)
fn gradient(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  var vel = textureLoad(vel_src, c, 0).xyz;
  let pc = textureLoad(p_src, c, 0).x;
  let here_fluid = fluid(c, n);
  if (here_fluid || fluid(c - vec3i(1, 0, 0), n)) {
    vel.x -= pc - grad_p(c - vec3i(1, 0, 0), pc, n);
  }
  if (here_fluid || fluid(c - vec3i(0, 1, 0), n)) {
    vel.y -= pc - grad_p(c - vec3i(0, 1, 0), pc, n);
  }
  if (here_fluid || fluid(c - vec3i(0, 0, 1), n)) {
    vel.z -= pc - grad_p(c - vec3i(0, 0, 1), pc, n);
  }
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);
  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
