// Rendu de la scène vers la texture HDR intermédiaire (rgba16float, taille fixe).
// Vue 0 (fluides) : composition LINÉAIRE pré-tone-mapping avec éclairage par gradient
// de densité — le champ de densité totale sert de carte de hauteur, son gradient de
// pseudo-normale ; un diffus directionnel modèle le volume, un spéculaire étroit donne
// l'aspect mouillé. Les hautes lumières (spéculaire compris) dépassent 1.0 et
// alimenteront le bloom. Vues 1–4 : champs de debug en couleurs d'affichage, transmises
// telles quelles. Tone-mapping, fond, murs et gamma vivent dans present.wgsl.

struct RenderParams {
  color0: vec4f, // rgb: couleur fluide 0 (eau)
  color1: vec4f, // rgb: couleur fluide 1 (encre)
  color2: vec4f, // rgb: couleur fluide 2 (fumée)
  tone: vec4f,   // x: exposition, y: vue (0–4), z: échelle debug vélocité, w: force du bloom
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_2d<f32>;
@group(0) @binding(3) var velocity_tex: texture_2d<f32>;
@group(0) @binding(4) var pressure_tex: texture_2d<f32>;
@group(0) @binding(5) var divergence_tex: texture_2d<f32>;
@group(0) @binding(6) var curl_tex: texture_2d<f32>;

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

// Texel de la grille de simulation correspondant à un uv écran.
fn sim_texel(uv: vec2f) -> vec2i {
  let dims = vec2i(textureDimensions(pressure_tex));
  return clamp(vec2i(uv * vec2f(dims)), vec2i(0), dims - 1);
}

fn total_density(uv: vec2f) -> f32 {
  let d = textureSampleLevel(density_tex, lin, uv, 0.0).xyz;
  return d.x + d.y + d.z;
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3f {
  let k = (vec3f(5.0, 3.0, 1.0) + vec3f(h * 6.0)) % vec3f(6.0);
  return v - v * s * clamp(min(k, vec3f(4.0) - k), vec3f(0.0), vec3f(1.0));
}

// Colormap divergente sur fond sombre : bleu (négatif) ↔ noir (zéro) ↔ orange (positif).
fn diverging(t: f32) -> vec3f {
  let a = clamp(abs(t), 0.0, 1.0);
  let warm = vec3f(1.0, 0.45, 0.12);
  let cold = vec3f(0.15, 0.5, 1.0);
  return select(cold, warm, t >= 0.0) * a + vec3f(0.02, 0.022, 0.03);
}

@fragment
fn fs_main(frag: VSOut) -> @location(0) vec4f {
  let view = u32(R.tone.y + 0.5);
  var col: vec3f;
  switch view {
    case 0u: {
      let den4 = textureSampleLevel(density_tex, lin, frag.uv, 0.0);
      let den = den4.xyz;
      col = R.color0.rgb * den.x + R.color1.rgb * den.y + R.color2.rgb * den.z;
      // Éclairage : gradient central de la densité totale → pseudo-normale.
      let e = 1.0 / vec2f(textureDimensions(density_tex));
      let gx = total_density(frag.uv + vec2f(e.x, 0.0)) - total_density(frag.uv - vec2f(e.x, 0.0));
      let gy = total_density(frag.uv + vec2f(0.0, e.y)) - total_density(frag.uv - vec2f(0.0, e.y));
      let n = normalize(vec3f(-gx * 3.0, -gy * 3.0, 1.0));
      let light_dir = normalize(vec3f(-0.45, -0.65, 0.6)); // lumière haut-gauche
      let diffuse = clamp(dot(n, light_dir), 0.0, 1.0);
      let spec = pow(clamp(dot(reflect(-light_dir, n), vec3f(0.0, 0.0, 1.0)), 0.0, 1.0), 24.0);
      let presence = clamp(den.x + den.y + den.z, 0.0, 1.0);
      // Valeurs artistiques : 55 % d'ambiant pour ne pas éteindre les zones à l'ombre.
      col = col * (0.55 + 0.45 * diffuse) + vec3f(1.0, 0.98, 0.92) * (spec * 0.4 * presence);
      // Feu : rampe de corps noir sur la température (.a) — rouge sombre → orange →
      // blanc chaud. Les valeurs HDR > 1 nourrissent le bloom : les flammes rayonnent.
      let heat = clamp(den4.w, 0.0, 1.7);
      col += vec3f(heat * 1.6, heat * heat * 0.9, heat * heat * heat * 0.42);
    }
    case 1u: {
      // Grille MAC : reconstruit le vecteur au centre de l'écran-texel en échantillonnant
      // chaque composante de face avec son décalage d'un demi-texel.
      let dims = vec2f(textureDimensions(velocity_tex));
      let p = frag.uv * dims;
      let u = textureSampleLevel(velocity_tex, lin, (p + vec2f(0.5, 0.0)) / dims, 0.0).x;
      let vv = textureSampleLevel(velocity_tex, lin, (p + vec2f(0.0, 0.5)) / dims, 0.0).y;
      let v = vec2f(u, vv);
      let hue = atan2(v.y, v.x) / 6.2831853 + 0.5;
      let mag = clamp(length(v) / R.tone.z, 0.0, 1.0);
      col = hsv2rgb(hue, 0.85, mag) + vec3f(0.02, 0.022, 0.03);
    }
    case 2u: {
      col = diverging(textureLoad(pressure_tex, sim_texel(frag.uv), 0).x * 0.05);
    }
    case 3u: {
      // Échelle modérée : le résidu basse fréquence (que Jacobi ne tue que très
      // lentement) reste lisible sans saturer toute la vue.
      col = diverging(textureLoad(divergence_tex, sim_texel(frag.uv), 0).x * 0.2);
    }
    default: {
      col = diverging(textureLoad(curl_tex, sim_texel(frag.uv), 0).x * 0.02);
    }
  }
  return vec4f(col, 1.0);
}
