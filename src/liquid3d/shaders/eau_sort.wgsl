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
const BLOCKS_PER_AXIS = 16u;

fn block_of(pos: vec3f) -> u32 {
  let b = vec3u(clamp(pos / BLOCK, vec3f(0.0), vec3f(f32(BLOCKS_PER_AXIS) - 1.0)));
  return b.x + BLOCKS_PER_AXIS * (b.y + BLOCKS_PER_AXIS * b.z);
}

@compute @workgroup_size(64)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
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
  for (var b = 0u; b < 4096u; b++) {
    atomicStore(&block_cursor[b], offset);
    offset += atomicLoad(&block_count[b]);
  }
}

@compute @workgroup_size(64)
fn reorder(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let d = atomicAdd(&block_cursor[block_of(src[4u * i].xyz)], 1u);
  dst[4u * d] = src[4u * i];
  dst[4u * d + 1u] = src[4u * i + 1u];
  dst[4u * d + 2u] = src[4u * i + 2u];
  dst[4u * d + 3u] = src[4u * i + 3u];
}
