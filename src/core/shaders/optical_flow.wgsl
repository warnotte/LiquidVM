// Flux optique « normal » entre deux images de la caméra : pour chaque texel,
// f = −(∂I/∂t) · ∇I / (|∇I|² + ε) — la composante du mouvement dans la direction du
// gradient d'intensité (une seule paire d'images, pas d'itération : bruité mais très
// bon marché, et largement suffisant comme champ de FORCES une fois lissé par le
// rééchantillonnage bilinéaire et l'intégration du fluide).
// L'image est MIROIRÉE horizontalement (une webcam se vit comme un miroir).
// La passe écrit aussi la luminance courante pour la frame suivante (ping-pong).
// PERSISTANCE TEMPORELLE : la caméra tourne à ~30 fps, la sim à 60 — une frame sur
// deux n'a aucun changement d'image. Le flux précédent (copié dans prev_flow) décroît
// au lieu d'être effacé, et les nouvelles mesures s'y accumulent : les impulsions
// intermittentes deviennent un champ de force continu et lisse.

// Réglages à chaud (panneau) : gain appliqué aux nouvelles mesures, porte de bruit,
// décroissance de la persistance. Écrits par le CPU uniquement au changement.
struct FlowParams {
  gain: f32,
  gate: f32,
  decay: f32,
  pad: f32,
}

@group(0) @binding(0) var camera_tex: texture_2d<f32>;
@group(0) @binding(1) var prev_lum: texture_2d<f32>;
@group(0) @binding(2) var dst_flow: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var dst_lum: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var prev_flow: texture_2d<f32>;
@group(0) @binding(5) var<uniform> FP: FlowParams;

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

  // Persistance : le flux d'hier décroît doucement (≈ 0 en ~0,5 s à 60 fps).
  var flow = textureLoad(prev_flow, c, 0).xy * FP.decay;
  // Porte de bruit : les capteurs webcam scintillent (~1-2 % d'intensité).
  if (abs(dt_l) > FP.gate) {
    let gx = 0.5 * (cam_lum(c + vec2i(1, 0), size) - cam_lum(c + vec2i(-1, 0), size));
    let gy = 0.5 * (cam_lum(c + vec2i(0, 1), size) - cam_lum(c + vec2i(0, -1), size));
    let g = vec2f(gx, gy);
    flow += -dt_l * g / (dot(g, g) + 0.02) * FP.gain;
  }
  let cap = 1.5 * max(FP.gain, 1.0);
  let mag = length(flow);
  if (mag > cap) {
    flow = flow * (cap / mag);
  }
  textureStore(dst_flow, c, vec4f(flow, 0.0, 0.0));
  textureStore(dst_lum, c, vec4f(l1, 0.0, 0.0, 0.0));
}
