// Post-traitement 3D : le raymarch écrit désormais une scène HDR linéaire —
// ici : seuil doux + réduction vers une texture bloom de taille FIXE (la
// lueur est basse fréquence, la résolution du canvas ne lui apporte rien),
// gaussienne séparable 9 prises (espacement 1,5 texel : le bilinéaire élargit
// le noyau gratuitement), puis présentation = scène + bloom, tone-mapping
// exponentiel et gamma (déplacés depuis raymarch.wgsl).

struct RenderParams {
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,    // w: exposition
  cam_fwd: vec4f,
  light: vec4f,
  sphere: vec4f,
  blow_a: vec4f,
  blow_b: vec4f,
  style: vec4f,     // x: lueur du feu, y: intensité du bloom
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
// Présentation (layout dédié, mêmes bindings 0/1 + les deux scènes) :
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;
// Chaîne bloom (compute) :
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

// Seuil doux : seules les luminances HDR franches contribuent au halo.
const BLOOM_T0 = 0.9;
const BLOOM_T1 = 1.9;

@compute @workgroup_size(8, 8)
fn bloom_down(@builtin(global_invocation_id) gid: vec3u) {
  let m = textureDimensions(dst);
  if (gid.x >= m.x || gid.y >= m.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(m);
  let c = textureSampleLevel(src, lin, uv, 0.0).rgb;
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  textureStore(dst, gid.xy, vec4f(c * smoothstep(BLOOM_T0, BLOOM_T1, lum), 0.0));
}

const W0 = 0.227027;
const W = array<f32, 4>(0.1945946, 0.1216216, 0.0540541, 0.0162162);

fn blur_axis(gid: vec3u, axis: vec2f) {
  let m = textureDimensions(dst);
  if (gid.x >= m.x || gid.y >= m.y) {
    return;
  }
  let inv = vec2f(1.0) / vec2f(m);
  let uv = (vec2f(gid.xy) + vec2f(0.5)) * inv;
  var acc = textureSampleLevel(src, lin, uv, 0.0).rgb * W0;
  for (var i = 0; i < 4; i++) {
    let o = axis * (1.5 * f32(i + 1)) * inv;
    acc += textureSampleLevel(src, lin, uv + o, 0.0).rgb * W[i];
    acc += textureSampleLevel(src, lin, uv - o, 0.0).rgb * W[i];
  }
  textureStore(dst, gid.xy, vec4f(acc, 0.0));
}

@compute @workgroup_size(8, 8)
fn bloom_h(@builtin(global_invocation_id) gid: vec3u) {
  blur_axis(gid, vec2f(1.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn bloom_v(@builtin(global_invocation_id) gid: vec3u) {
  blur_axis(gid, vec2f(0.0, 1.0));
}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_present(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = corners[vi];
  var out: VSOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv = vec2f(p.x, -p.y) * 0.5 + vec2f(0.5);
  return out;
}

@fragment
fn fs_present(frag: VSOut) -> @location(0) vec4f {
  let hdr = textureSampleLevel(hdr_tex, lin, frag.uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloom_tex, lin, frag.uv, 0.0).rgb;
  let col = hdr + bloom * R.style.y;
  let mapped = vec3f(1.0) - exp(-col * R.cam_up.w);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
