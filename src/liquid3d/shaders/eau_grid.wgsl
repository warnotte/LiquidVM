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
@group(1) @binding(7) var dens_src: texture_3d<f32>; // densité floutée (1 = repos)

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
// avec CONTRÔLE DE DENSITÉ (Bridson) DANS LES DEUX SENS, sur la densité
// FLOUTÉE (boîte 3³, 1 = repos, calculée par sous-pas dans eau_surface.wgsl) :
//  - surpeuplée → divergence cible positive (expansion) : contre les trous en
//    « fromage » et la fonte du volume ;
//  - SOUS-peuplée ET sans voisin d'air → cible négative (compression) : sans
//    elle, le volume n'a qu'un cliquet — toute cellule touchée par une
//    particule devient « eau » incompressible, le volume gonfle à chaque
//    éclaboussure et ne se recompacte jamais (mesuré : nappe calme à ~4
//    particules/cellule, volume ×2). Jamais de compression en surface : elle
//    aspirerait la surface libre.
// Taux en 1/s (indépendant du sous-pas) : à rel = 1, la cellule demande 10 %
// de volume par 10 ms. L'ancien 0.3/dt (= 3600 %/s) sur-réagissait au bruit
// de Poisson du comptage et POMPAIT de l'énergie dans le bassin calme (surface
// qui « bout », grumeaux à 24+ particules/cellule).
const DENSITY_RATE = 10.0;
// Pas de pression : écrit 0 (purge du warm start au reset — leçon 2D).
@compute @workgroup_size(4, 4, 4)
fn clear_pressure(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  if (i32(gid.x) >= n || i32(gid.y) >= n || i32(gid.z) >= n) {
    return;
  }
  textureStore(p_dst, gid, vec4f(0.0));
}

fn air_adjacent(c: vec3i, n: i32) -> bool {
  let dirs = array<vec3i, 6>(
    vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1),
  );
  for (var k = 0; k < 6; k++) {
    let q = c + dirs[k];
    if (!solid(q, n) && !fluid(q, n)) {
      return true;
    }
  }
  return false;
}

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
  // Bornée à [−0.5, 1] : sans le clamp, une cellule très tassée (30+
  // particules) demandait ~180 s⁻¹ d'expansion → pressions explosives →
  // vitesses au-delà du max float16 → NaN (la saga float16 du 2D).
  var rel = clamp(textureLoad(dens_src, c, 0).x - 1.0, -0.5, 1.0);
  if (rel < 0.0 && air_adjacent(c, n)) {
    rel = 0.0;
  }
  let d_target = DENSITY_RATE * rel;
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
