// J1 — TRI PÉRIODIQUE des particules par bloc de 8³ voxels (16³ = 4096 blocs).
// La conséquence ferme de J0 : le P2G trié coûte 1,83 ms contre 2,57 en
// désordre. Granularité bloc (pas cellule) : l'essentiel du gain de cache pour
// un tri à trois petites passes — histogramme atomique, scan exclusif SÉRIEL
// (4096 additions sur un thread, une fois toutes les ~12 frames : négligeable
// et trivialement correct), réordonnancement par curseurs atomiques.

struct UEau {
  sim: vec4f, // x: N, y: nb particules
  sim2: vec4f,
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(1) @binding(0) var<storage, read> src: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> block_count: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> block_cursor: array<atomic<u32>>;

const BLOCK = 8.0;

// Blocs par axe = N / 8, déduit de la résolution (uniform) : rien à répliquer
// côté CPU quand la grille change.
fn blocks_per_axis() -> u32 {
  return max(u32(U.sim.x / BLOCK), 1u);
}

fn block_of(pos: vec3f) -> u32 {
  let bpa = blocks_per_axis();
  let b = vec3u(clamp(pos / BLOCK, vec3f(0.0), vec3f(f32(bpa) - 1.0)));
  return b.x + bpa * (b.y + bpa * b.z);
}

// Dispatch 2D des passes de particules (voir PARTICLE_DISPATCH_X dans
// config_eau.ts) : 1024 groupes de 64 par rangée. La limite
// maxComputeWorkgroupsPerDimension vaut 65535, or n³/64 la dépasse dès 192³.
const PARTICLE_ROW = 65536u;


@compute @workgroup_size(64)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * PARTICLE_ROW;
  if (f32(i) >= U.sim.y) {
    return;
  }
  atomicAdd(&block_count[block_of(src[4u * i].xyz)], 1u);
}

// Scan exclusif sériel : block_cursor[b] = départ du bloc b dans le buffer trié.
@compute @workgroup_size(1)
fn scan(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) {
    return;
  }
  var offset = 0u;
  let bpa = blocks_per_axis();
  let total = bpa * bpa * bpa;
  for (var b = 0u; b < total; b++) {
    atomicStore(&block_cursor[b], offset);
    offset += atomicLoad(&block_count[b]);
  }
}

@compute @workgroup_size(64)
fn reorder(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * PARTICLE_ROW;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let d = atomicAdd(&block_cursor[block_of(src[4u * i].xyz)], 1u);
  dst[4u * d] = src[4u * i];
  dst[4u * d + 1u] = src[4u * i + 1u];
  dst[4u * d + 2u] = src[4u * i + 2u];
  dst[4u * d + 3u] = src[4u * i + 3u];
}
