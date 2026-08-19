// Chaîne de bloom en compute, sur la scène HDR :
//   bright_down : scène → ½ résolution, seuil doux (seules les zones > THRESHOLD glowent) ;
//   down        : ½ → ¼ résolution (élargit le halo pour trois fois rien) ;
//   blur_h/v    : gaussienne séparable 9 taps, appliquée à chaque niveau.
// La passe de présentation additionne les deux niveaux flous à la scène avant tone-mapping.

@group(0) @binding(0) var lin: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

// Seuil HDR : sous cette luminance, pas de contribution au bloom. Les couleurs de
// fluide saturées et le spéculaire de l'éclairage dépassent 1.0 et glowent.
const THRESHOLD: f32 = 0.85;

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn bright_down(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst);
  if (any(gid.xy >= dims)) {
    return;
  }
  // Un tap bilinéaire au centre du texel destination = moyenne 2×2 de la source.
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(dims);
  let c = textureSampleLevel(src, lin, uv, 0.0).rgb;
  textureStore(dst, vec2i(gid.xy), vec4f(max(c - vec3f(THRESHOLD), vec3f(0.0)), 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst);
  if (any(gid.xy >= dims)) {
    return;
  }
  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(dims);
  textureStore(dst, vec2i(gid.xy), vec4f(textureSampleLevel(src, lin, uv, 0.0).rgb, 0.0));
}

fn tap(c: vec2i, off: vec2i, dims: vec2i) -> vec3f {
  return textureLoad(src, clamp(c + off, vec2i(0), dims - 1), 0).rgb;
}

fn blur(c: vec2i, dir: vec2i) -> vec3f {
  let dims = vec2i(textureDimensions(src));
  var acc = textureLoad(src, c, 0).rgb * 0.227027;
  acc += (tap(c, dir, dims) + tap(c, -dir, dims)) * 0.1945946;
  acc += (tap(c, dir * 2, dims) + tap(c, dir * -2, dims)) * 0.1216216;
  acc += (tap(c, dir * 3, dims) + tap(c, dir * -3, dims)) * 0.054054;
  acc += (tap(c, dir * 4, dims) + tap(c, dir * -4, dims)) * 0.016216;
  return acc;
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= textureDimensions(dst))) {
    return;
  }
  textureStore(dst, vec2i(gid.xy), vec4f(blur(vec2i(gid.xy), vec2i(1, 0)), 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= textureDimensions(dst))) {
    return;
  }
  textureStore(dst, vec2i(gid.xy), vec4f(blur(vec2i(gid.xy), vec2i(0, 1)), 0.0));
}
