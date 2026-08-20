// Flux optique « normal » entre deux images de la caméra : pour chaque texel,
// f = −(∂I/∂t) · ∇I / (|∇I|² + ε) — la composante du mouvement dans la direction du
// gradient d'intensité (une seule paire d'images, pas d'itération : bruité mais très
// bon marché, et largement suffisant comme champ de FORCES une fois lissé par le
// rééchantillonnage bilinéaire et l'intégration du fluide).
// L'image est MIROIRÉE horizontalement (une webcam se vit comme un miroir).
// La passe écrit aussi la luminance courante pour la frame suivante (ping-pong).

@group(0) @binding(0) var camera_tex: texture_2d<f32>;
@group(0) @binding(1) var prev_lum: texture_2d<f32>;
@group(0) @binding(2) var dst_flow: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var dst_lum: texture_storage_2d<r32float, write>;

// Luminance de la caméra en (c), miroir horizontal + clamp aux bords.
fn cam_lum(c: vec2i, size: vec2i) -> f32 {
  let q = clamp(vec2i(size.x - 1 - c.x, c.y), vec2i(0), size - vec2i(1));
  let rgb = textureLoad(camera_tex, q, 0).rgb;
  return dot(rgb, vec3f(0.299, 0.587, 0.114));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(textureDimensions(dst_flow));
  let c = vec2i(gid.xy);
  if (c.x >= size.x || c.y >= size.y) {
    return;
  }
  let l1 = cam_lum(c, size);
  let l0 = textureLoad(prev_lum, c, 0).x;
  let dt_l = l1 - l0;

  var flow = vec2f(0.0);
  // Porte de bruit : les capteurs webcam scintillent (~1-2 % d'intensité).
  if (abs(dt_l) > 0.02) {
    let gx = 0.5 * (cam_lum(c + vec2i(1, 0), size) - cam_lum(c + vec2i(-1, 0), size));
    let gy = 0.5 * (cam_lum(c + vec2i(0, 1), size) - cam_lum(c + vec2i(0, -1), size));
    let g = vec2f(gx, gy);
    flow = -dt_l * g / (dot(g, g) + 0.02);
    let mag = length(flow);
    if (mag > 1.5) {
      flow = flow * (1.5 / mag);
    }
  }
  textureStore(dst_flow, c, vec4f(flow, 0.0, 0.0));
  textureStore(dst_lum, c, vec4f(l1, 0.0, 0.0, 0.0));
}
