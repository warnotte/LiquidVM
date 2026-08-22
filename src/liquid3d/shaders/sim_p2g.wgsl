// J1 — transferts particules → grille de la SIMULATION (le banc J0 vit dans
// p2g_bench.wgsl). init_dam : colonne d'eau 64×64×64 cellules contre une paroi,
// 8 particules/cellule, TRIÉE par construction (l'état que le tri périodique
// entretient). scatter : dispersion trilinéaire MAC + comptage de cellule
// (marqueur eau/air ET histogramme du tri). resolve : vitesses de grille
// (velOld — la référence du delta FLIP) depuis les accumulateurs atomiques.
// Le rayon de dispersion (±1 face) EST l'extrapolation de vélocité dans l'air
// au voisinage de la surface : le G2P trilinéaire retombe toujours dessus.

struct UEau {
  sim: vec4f,      // x: N, y: nb particules, z: dt sous-pas, w: gravité
  sim2: vec4f,     // x: mélange FLIP, y-w: libres
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(0) @binding(1) var<storage, read_write> particles: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> acc_u: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> acc_v: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> acc_w: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> wgt_u: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> wgt_v: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> wgt_w: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var vel_old_dst: texture_storage_3d<rgba16float, write>;

const SCALE = 256.0;

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

// Colonne d'eau : 64×32×128 cellules (basse et large — la profondeur 32 reste
// à portée du Jacobi tant que le multigrid masqué n'est pas là), 8/cellule,
// à RAS du sol et de la paroi (les cellules de bord sont de l'eau, leurs faces
// extérieures sont les murs).
@compute @workgroup_size(64)
fn init_dam(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let s = i * 1664525u + 77u;
  let cell = i / 8u;
  // Largeur de la colonne (uniform sim2.y : 64 = basse, 32 = haute) ; la
  // section largeur×hauteur vaut toujours 2048 cellules × 128 de profondeur.
  let w = max(u32(U.sim2.y), 1u);
  let h = 2048u / w;
  let cx = f32(cell % w);
  let cy = f32((cell / w) % h);
  let cz = f32(cell / 2048u);
  let pos = vec3f(cx, cy, cz) +
    vec3f(rand01(s), rand01(s ^ 0x9e3779b9u), rand01(s ^ 0x85ebca6bu));
  particles[2u * i] = vec4f(pos, 0.0);
  particles[2u * i + 1u] = vec4f(0.0);
}

fn cell_index(c: vec3i, n: i32) -> u32 {
  let q = clamp(c, vec3i(0), vec3i(n - 1));
  return u32(q.x + n * (q.y + n * q.z));
}

fn scatter_u(fp: vec3f, value: f32, n: i32) {
  let base = vec3i(floor(fp));
  let t = fp - floor(fp);
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        let w = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        let idx = cell_index(base + vec3i(dx, dy, dz), n);
        atomicAdd(&acc_u[idx], i32(round(value * w * SCALE)));
        atomicAdd(&wgt_u[idx], i32(round(w * SCALE)));
      }
    }
  }
}

fn scatter_v(fp: vec3f, value: f32, n: i32) {
  let base = vec3i(floor(fp));
  let t = fp - floor(fp);
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        let w = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        let idx = cell_index(base + vec3i(dx, dy, dz), n);
        atomicAdd(&acc_v[idx], i32(round(value * w * SCALE)));
        atomicAdd(&wgt_v[idx], i32(round(w * SCALE)));
      }
    }
  }
}

fn scatter_w(fp: vec3f, value: f32, n: i32) {
  let base = vec3i(floor(fp));
  let t = fp - floor(fp);
  for (var dz = 0; dz <= 1; dz++) {
    for (var dy = 0; dy <= 1; dy++) {
      for (var dx = 0; dx <= 1; dx++) {
        let w = mix(1.0 - t.x, t.x, f32(dx)) * mix(1.0 - t.y, t.y, f32(dy)) * mix(1.0 - t.z, t.z, f32(dz));
        let idx = cell_index(base + vec3i(dx, dy, dz), n);
        atomicAdd(&acc_w[idx], i32(round(value * w * SCALE)));
        atomicAdd(&wgt_w[idx], i32(round(w * SCALE)));
      }
    }
  }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let n = i32(U.sim.x);
  let pos = particles[2u * i].xyz;
  let vel = particles[2u * i + 1u].xyz;
  // Marqueur eau/air : la cellule qui CONTIENT la particule.
  atomicAdd(&cell_count[cell_index(vec3i(floor(pos)), n)], 1u);
  // Convention MAC du moteur : u en (i, j+½, k+½), v en (i+½, j, k+½), w en (i+½, j+½, k).
  scatter_u(pos - vec3f(0.0, 0.5, 0.5), vel.x, n);
  scatter_v(pos - vec3f(0.5, 0.0, 0.5), vel.y, n);
  scatter_w(pos - vec3f(0.5, 0.5, 0.0), vel.z, n);
}

@compute @workgroup_size(4, 4, 4)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(U.sim.x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let idx = cell_index(c, n);
  let u = f32(atomicLoad(&acc_u[idx])) / max(f32(atomicLoad(&wgt_u[idx])), 1.0);
  let v = f32(atomicLoad(&acc_v[idx])) / max(f32(atomicLoad(&wgt_v[idx])), 1.0);
  let w = f32(atomicLoad(&acc_w[idx])) / max(f32(atomicLoad(&wgt_w[idx])), 1.0);
  textureStore(vel_old_dst, gid, vec4f(u, v, w, 0.0));
}
