// J0 — banc du transfert particules→grille (P2G), LE risque technique du
// chantier eau (PLAN-EAU.md) : 2 M particules dispersent leur vitesse sur les
// faces MAC voisines par atomicAdd i32 en virgule fixe (×256), puis une passe
// resolve divise par les poids et écrit la texture de vélocité — le pont vers
// la machinerie de projection existante. 48 atomicAdd par particule
// (3 composantes × 8 coins × [valeur + poids]) ≈ 100 M atomics/itération.
//
// init : distribution GROUPÉE (bloc 64³ = 8 particules/cellule, façon dam
// break) — c'est la contention réaliste, pas un uniforme complaisant.

struct Params {
  misc: vec4f, // x: N (grille), y: nombre de particules, z: graine, w: libre
}

@group(0) @binding(0) var<uniform> P: Params;
// Particules : [2i] = pos.xyz (voxels), [2i+1] = vel.xyz (voxels/s).
@group(0) @binding(1) var<storage, read_write> particles: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> acc_u: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> acc_v: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> acc_w: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> wgt_u: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> wgt_v: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> wgt_w: array<atomic<i32>>;
@group(0) @binding(8) var vel_tex: texture_storage_3d<rgba16float, write>;

const SCALE = 256.0;

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

// Dispatch 2D des passes de particules (voir PARTICLE_DISPATCH_X dans
// config_eau.ts) : 1024 groupes de 64 par rangée. La limite
// maxComputeWorkgroupsPerDimension vaut 65535, or n³/64 la dépasse dès 192³.
const PARTICLE_ROW = 65536u;

// Ordre ALÉATOIRE : threads voisins → cellules quelconques. C'est l'état d'un
// nuage de particules jamais trié — le pire cas de cache pour les atomics.
@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * PARTICLE_ROW;
  if (f32(i) >= P.misc.y) {
    return;
  }
  let s = i * 1664525u + bitcast<u32>(P.misc.z);
  // Bloc 64³ posé dans un coin (marge 4) : 2 M particules / 262 144 cellules.
  let pos = vec3f(4.0) + vec3f(rand01(s), rand01(s ^ 0x9e3779b9u), rand01(s ^ 0x85ebca6bu)) * 64.0;
  let vel = (vec3f(rand01(s ^ 0x27d4eb2fu), rand01(s ^ 0x165667b1u), rand01(s ^ 0xd3a2646cu)) - vec3f(0.5)) * 240.0;
  particles[2u * i] = vec4f(pos, 0.0);
  particles[2u * i + 1u] = vec4f(vel, 0.0);
}

// Ordre TRIÉ PAR CELLULE : particule i vit dans la cellule i/8 du bloc — les
// threads d'un même warp frappent la même poignée de faces (agrégation
// d'atomics + lignes de cache partagées). C'est l'état qu'un tri périodique
// (pratique standard FLIP, prévu en J1) maintient en régime.
@compute @workgroup_size(64)
fn init_sorted(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * PARTICLE_ROW;
  if (f32(i) >= P.misc.y) {
    return;
  }
  let s = i * 1664525u + bitcast<u32>(P.misc.z);
  let cell = i / 8u;
  let cx = f32(cell % 64u);
  let cy = f32((cell / 64u) % 64u);
  let cz = f32(cell / 4096u);
  let pos = vec3f(4.0) + vec3f(cx, cy, cz) +
    vec3f(rand01(s), rand01(s ^ 0x9e3779b9u), rand01(s ^ 0x85ebca6bu));
  let vel = (vec3f(rand01(s ^ 0x27d4eb2fu), rand01(s ^ 0x165667b1u), rand01(s ^ 0xd3a2646cu)) - vec3f(0.5)) * 240.0;
  particles[2u * i] = vec4f(pos, 0.0);
  particles[2u * i + 1u] = vec4f(vel, 0.0);
}

fn cell_index(c: vec3i, n: i32) -> u32 {
  let q = clamp(c, vec3i(0), vec3i(n - 1));
  return u32(q.x + n * (q.y + n * q.z));
}

// Scatter trilinéaire d'UNE composante sur SA grille de faces MAC.
// fp = position en repère de faces (pos − offset de la composante).
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
  let i = gid.x + gid.y * PARTICLE_ROW;
  if (f32(i) >= P.misc.y) {
    return;
  }
  let n = i32(P.misc.x);
  let pos = particles[2u * i].xyz;
  let vel = particles[2u * i + 1u].xyz;
  // Convention MAC du moteur : u en (i, j+½, k+½), v en (i+½, j, k+½),
  // w en (i+½, j+½, k) — le repère de faces de chaque composante.
  scatter_u(pos - vec3f(0.0, 0.5, 0.5), vel.x, n);
  scatter_v(pos - vec3f(0.5, 0.0, 0.5), vel.y, n);
  scatter_w(pos - vec3f(0.5, 0.5, 0.0), vel.z, n);
}

// Résolution : vitesse = somme pondérée / somme des poids (les ×256 s'annulent),
// écrite dans la texture MAC — le format que toute la projection sait lire.
@compute @workgroup_size(4, 4, 4)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.misc.x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let idx = cell_index(c, n);
  let u = f32(atomicLoad(&acc_u[idx])) / max(f32(atomicLoad(&wgt_u[idx])), 1.0);
  let v = f32(atomicLoad(&acc_v[idx])) / max(f32(atomicLoad(&wgt_v[idx])), 1.0);
  let w = f32(atomicLoad(&acc_w[idx])) / max(f32(atomicLoad(&wgt_w[idx])), 1.0);
  textureStore(vel_tex, gid, vec4f(u, v, w, 0.0));
}
