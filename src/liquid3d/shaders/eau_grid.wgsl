// J1 — passes de GRILLE de l'eau : forces (gravité + parois), divergence sur
// les cellules d'EAU seulement, Jacobi à SURFACE LIBRE (air = Dirichlet p = 0,
// condition symétrique — voir PLAN-EAU.md ; hors-boîte = Neumann par le
// clamp-trick éprouvé du feu : lire le voisin clampé REND p_centre), gradient.
// Schéma MAC compact identique au feu : les faces N inexistantes sont des murs
// gratuits, les faces 0 sont forcées à zéro.

struct UEau {
  sim: vec4f,  // x: N, y: nb particules, z: dt sous-pas, w: gravité (voxels/s²)
  sim2: vec4f, // x: mélange FLIP, y: largeur colonne, z: APIC, w: taux densité (1/s)
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
  render: vec4f,
  sphere: vec4f,     // xyz: centre (voxels), w: rayon (voxels, ≤ 0 = absente)
  sphere_vel: vec4f, // xyz: vitesse de la boule (voxels/s) — bord MOBILE
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
@group(1) @binding(8) var<storage, read_write> census: array<atomic<u32>>;

fn n_size() -> i32 {
  return i32(U.sim.x);
}

// BOULE-OBSTACLE analytique (J3) : aucune texture d'obstacles — chaque passe
// teste « centre de cellule dans la sphère », comme le feu. La règle « centre »
// garde le lisseur adjoint exact du couple divergence/gradient.
fn in_sphere(c: vec3i) -> bool {
  if (U.sphere.w <= 0.0) {
    return false;
  }
  return distance(vec3f(c) + vec3f(0.5), U.sphere.xyz) < U.sphere.w;
}

fn out_of_box(c: vec3i, n: i32) -> bool {
  return any(c < vec3i(0)) || any(c >= vec3i(n));
}

fn fluid(c: vec3i, n: i32) -> bool {
  if (out_of_box(c, n) || in_sphere(c)) {
    return false; // hors boîte ou dans la boule = solide, pas eau
  }
  return cell_count[u32(c.x + n * (c.y + n * c.z))] > 0u;
}

fn solid(c: vec3i, n: i32) -> bool {
  return out_of_box(c, n) || in_sphere(c);
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

// Vitesse d'une FACE, du point de vue de la divergence. La face d'indice f est
// celle entre les cellules f−1 et f. Trois cas : paroi de boîte (0, faces 0 et
// N — le schéma compact rend les faces N inexistantes), face de la BOULE (débit
// PRESCRIT = vitesse de la boule, le même flux que le gradient y écrira), sinon
// la vitesse advectée.
fn face_u(f: vec3i, n: i32) -> f32 {
  if (f.x <= 0 || f.x >= n) {
    return 0.0;
  }
  if (in_sphere(f) || in_sphere(f - vec3i(1, 0, 0))) {
    return U.sphere_vel.x;
  }
  return textureLoad(vel_src, f, 0).x;
}
fn face_v(f: vec3i, n: i32) -> f32 {
  if (f.y <= 0 || f.y >= n) {
    return 0.0;
  }
  if (in_sphere(f) || in_sphere(f - vec3i(0, 1, 0))) {
    return U.sphere_vel.y;
  }
  return textureLoad(vel_src, f, 0).y;
}
fn face_w(f: vec3i, n: i32) -> f32 {
  if (f.z <= 0 || f.z >= n) {
    return 0.0;
  }
  if (in_sphere(f) || in_sphere(f - vec3i(0, 0, 1))) {
    return U.sphere_vel.z;
  }
  return textureLoad(vel_src, f, 0).z;
}

// Divergence compacte, cellules d'EAU seulement (l'air n'est pas résolu) —
// avec CONTRÔLE DE DENSITÉ (Bridson, EXPANSION SEULE) sur la densité FLOUTÉE
// (boîte 3³, 1 = repos, calculée par sous-pas dans eau_surface.wgsl) : les
// cellules tassées au-delà d'une ZONE MORTE de 25 % reçoivent une divergence
// cible positive. Mesures (histogramme d'occupation, 2026-08-23) qui fixent
// cette forme :
//  - sans contrôle, l'eau se COMPACTE de ~25 % (82 k cellules à 12-23
//    particules, nappe de 12,5 voxels au lieu de 16) — d'où l'expansion ;
//  - une compression des cellules sous-denses (essayée) FABRIQUE des grumeaux
//    (15 k cellules à 24+) et raréfie la surface (48 k cellules à 1-3) ;
//  - sans zone morte, le bruit de Poisson du comptage (σ ≈ 35 % par cellule,
//    ~7 % flouté) déclenche des expansions parasites qui soufflent la surface.
// Taux en 1/s (uniform sim2.w, indépendant du sous-pas) : à rel = 1 et 10/s,
// la cellule demande 10 % de volume par 10 ms. L'ancien 0.3/dt (= 3600 %/s)
// POMPAIT de l'énergie dans le bassin calme (surface qui « bout »).
const DENSITY_DEADBAND = 0.25;

// Divergence CIBLE d'une cellule (0 sauf si elle est tassée). Bornée à 1× :
// sans le clamp, une cellule très tassée (30+ particules) demandait ~180 s⁻¹
// d'expansion → pressions explosives → au-delà du max float16 → NaN (la saga
// float16 du 2D). Partagée par la passe de divergence ET par l'instrument de
// résidu : le solveur doit atteindre CETTE valeur, pas zéro.
fn density_target(c: vec3i) -> f32 {
  let over = textureLoad(dens_src, c, 0).x - 1.0 - DENSITY_DEADBAND;
  return U.sim2.w * clamp(over / (1.0 - DENSITY_DEADBAND), 0.0, 1.0);
}
// Pas de pression : écrit 0 (purge du warm start au reset — leçon 2D).
@compute @workgroup_size(4, 4, 4)
fn clear_pressure(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  if (i32(gid.x) >= n || i32(gid.y) >= n || i32(gid.z) >= n) {
    return;
  }
  textureStore(p_dst, gid, vec4f(0.0));
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
  let d = face_u(c + vec3i(1, 0, 0), n) - face_u(c, n) +
    face_v(c + vec3i(0, 1, 0), n) - face_v(c, n) +
    face_w(c + vec3i(0, 0, 1), n) - face_w(c, n);
  textureStore(div_dst, gid, vec4f(d - density_target(c), 0.0, 0.0, 0.0));
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

// INSTRUMENT D'EXACTITUDE : divergence RÉSIDUELLE après projection, sur les
// seules cellules d'eau. C'est la mesure qui départage deux solveurs — « ça a
// l'air pareil » n'est pas une preuve. Accumulée en virgule fixe (×256) dans
// census[72] (somme) / census[73] (max) / census[74] (cellules d'eau) ; la
// contribution par cellule est bornée pour que la somme ne déborde pas l'u32.
// À lire APRÈS le gradient : vel_src doit être la vitesse projetée.
const DIV_FIXED = 256.0;

@compute @workgroup_size(4, 4, 4)
fn divergence_census(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n || !fluid(c, n)) {
    return;
  }
  // RÉSIDU : la projection doit rendre div(u) ÉGAL à la cible du contrôle de
  // densité, pas nul. Mesurer |div| seul confond l'erreur du solveur avec la
  // divergence VOULUE (c'est ce que faisait la première version : son maximum
  // de ~9,7 était exactement le taux d'expansion demandé, pas une erreur).
  let d = abs(
    face_u(c + vec3i(1, 0, 0), n) - face_u(c, n) +
    face_v(c + vec3i(0, 1, 0), n) - face_v(c, n) +
    face_w(c + vec3i(0, 0, 1), n) - face_w(c, n) - density_target(c)
  );
  let fixed = u32(clamp(d * DIV_FIXED, 0.0, 1048576.0));
  atomicAdd(&census[72], fixed);
  atomicMax(&census[73], fixed);
  atomicAdd(&census[74], 1u);
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
  let here_solid = in_sphere(c);
  // Face de la boule : vitesse PRESCRITE (bord mobile — le gradient n'y touche
  // pas, la divergence l'a lue comme un débit connu : l'adjoint reste exact).
  if (here_solid || in_sphere(c - vec3i(1, 0, 0))) {
    vel.x = U.sphere_vel.x;
  } else if (here_fluid || fluid(c - vec3i(1, 0, 0), n)) {
    vel.x -= pc - grad_p(c - vec3i(1, 0, 0), pc, n);
  }
  if (here_solid || in_sphere(c - vec3i(0, 1, 0))) {
    vel.y = U.sphere_vel.y;
  } else if (here_fluid || fluid(c - vec3i(0, 1, 0), n)) {
    vel.y -= pc - grad_p(c - vec3i(0, 1, 0), pc, n);
  }
  if (here_solid || in_sphere(c - vec3i(0, 0, 1))) {
    vel.z = U.sphere_vel.z;
  } else if (here_fluid || fluid(c - vec3i(0, 0, 1), n)) {
    vel.z -= pc - grad_p(c - vec3i(0, 0, 1), pc, n);
  }
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);
  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
