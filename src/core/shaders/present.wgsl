// Passe finale vers le canvas : scène HDR + bloom (vue fluides uniquement), tone-mapping
// exponentiel, fond sombre dégradé, murs en ardoise (dans toutes les vues), gamma.
// Les vues de debug arrivent déjà en couleurs d'affichage : elles ne reçoivent ni bloom
// ni tone-mapping, seulement les murs et le gamma.

struct RenderParams {
  color0: vec4f, // rgb: couleur fluide 0 (eau)
  color1: vec4f, // rgb: couleur fluide 1 (encre)
  color2: vec4f, // rgb: couleur fluide 2 (fumée)
  tone: vec4f,   // x: exposition, y: vue (0–4), z: échelle debug vélocité, w: force du bloom
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var scene_tex: texture_2d<f32>;
@group(0) @binding(3) var bloom_mid: texture_2d<f32>;
@group(0) @binding(4) var bloom_wide: texture_2d<f32>;
@group(0) @binding(5) var obstacle_tex: texture_2d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = corners[vi];
  var out: VSOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv = p * vec2f(0.5, -0.5) + vec2f(0.5);
  return out;
}

// Échantillonnage bilinéaire manuel du masque d'obstacles (format non filtrable).
fn obstacle_mask(uv: vec2f) -> f32 {
  let dims = vec2i(textureDimensions(obstacle_tex));
  let g = uv * vec2f(dims) - vec2f(0.5);
  let g0 = vec2i(floor(g));
  let f = g - floor(g);
  let o00 = textureLoad(obstacle_tex, clamp(g0, vec2i(0), dims - 1), 0).x;
  let o10 = textureLoad(obstacle_tex, clamp(g0 + vec2i(1, 0), vec2i(0), dims - 1), 0).x;
  let o01 = textureLoad(obstacle_tex, clamp(g0 + vec2i(0, 1), vec2i(0), dims - 1), 0).x;
  let o11 = textureLoad(obstacle_tex, clamp(g0 + vec2i(1, 1), vec2i(0), dims - 1), 0).x;
  return mix(mix(o00, o10, f.x), mix(o01, o11, f.x), f.y);
}

@fragment
fn fs_main(frag: VSOut) -> @location(0) vec4f {
  var col = textureSample(scene_tex, lin, frag.uv).rgb;
  let view = u32(R.tone.y + 0.5);
  if (view == 0u) {
    let glow = textureSample(bloom_mid, lin, frag.uv).rgb * 0.7 +
      textureSample(bloom_wide, lin, frag.uv).rgb * 0.55;
    col += glow * R.tone.w;
    // Tone-mapping : compresse doucement les hautes densités au lieu d'écrêter.
    col = vec3f(1.0) - exp(-col * R.tone.x);
    // Fond sombre, à peine dégradé pour éviter l'aplat pur.
    let bg = mix(vec3f(0.012, 0.014, 0.020), vec3f(0.045, 0.052, 0.080), frag.uv.y);
    col = bg + col * (vec3f(1.0) - bg);
  }
  // Murs : ardoise mate, visibles dans toutes les vues.
  let mask = obstacle_mask(frag.uv);
  let wall = mix(vec3f(0.16, 0.17, 0.21), vec3f(0.24, 0.25, 0.30), frag.uv.y);
  col = mix(col, wall, smoothstep(0.2, 0.8, mask));
  // Le format de canvas préféré n'est pas une vue sRGB : correction gamma manuelle.
  col = pow(col, vec3f(1.0 / 2.2));
  return vec4f(col, 1.0);
}
