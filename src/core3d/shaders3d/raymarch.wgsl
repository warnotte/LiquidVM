// Rendu volumétrique : ray-marching à travers la boîte [-½, ½]³, absorption
// Beer-Lambert + émission corps noir (chaleur) + diffusion éclairée par une
// lumière directionnelle avec ombre portée interne (marche secondaire courte).
// La boîte est suggérée par un liseré discret sur ses arêtes à l'entrée du rayon.

struct RenderParams {
  cam_pos: vec4f,   // xyz: position caméra, w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect (largeur/hauteur)
  cam_up: vec4f,    // xyz, w: exposition
  cam_fwd: vec4f,   // xyz, w: pas de marche (nombre)
  light: vec4f,     // xyz: direction VERS la lumière, w: intensité
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_3d<f32>;

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
  out.uv = p;
  return out;
}

// Intersection rayon / boîte centrée (slab). Retourne (t_entrée, t_sortie).
fn box_hit(ro: vec3f, rd: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let t0 = (vec3f(-0.5) - ro) * inv;
  let t1 = (vec3f(0.5) - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

fn extinction(s: vec4f) -> f32 {
  return s.x * 22.0 + s.w * 5.0;
}

fn blackbody(heat: f32) -> vec3f {
  let h = clamp(heat, 0.0, 1.7);
  return vec3f(h * 1.6, h * h * 0.9, h * h * h * 0.42);
}

// Hachage rapide pour décaler le départ de chaque rayon (casse le banding).
fn hash12(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 345.45));
  q += dot(q, q + 34.345);
  return fract(q.x * q.y);
}

@fragment
fn fs_main(frag: VSOut) -> @location(0) vec4f {
  let tanf = R.cam_pos.w;
  let ndc = frag.uv * vec2f(1.0, 1.0);
  let rd = normalize(
    R.cam_fwd.xyz + R.cam_right.xyz * ndc.x * tanf * R.cam_right.w +
      R.cam_up.xyz * ndc.y * tanf,
  );
  let ro = R.cam_pos.xyz;

  // Fond : dégradé sombre vertical, cohérent avec l'appli 2D.
  let bg = mix(vec3f(0.045, 0.05, 0.065), vec3f(0.10, 0.105, 0.13), frag.uv.y * 0.5 + 0.5);

  let hit = box_hit(ro, rd);
  let t0 = max(hit.x, 0.0);
  var col: vec3f;
  if (hit.y <= t0) {
    col = bg;
  } else {
    let steps = max(R.cam_fwd.w, 16.0);
    let step_len = (hit.y - t0) / steps;
    var t = t0 + step_len * hash12(frag.uv * 917.0);
    var transmit = 1.0;
    var acc = vec3f(0.0);

    // Liseré des arêtes de la boîte au point d'entrée : ancre la « boîte de verre ».
    let entry = ro + rd * hit.x;
    let e = abs(entry);
    let edge = smoothstep(0.485, 0.499, max(min(e.x, e.y), min(max(e.x, e.y), e.z)));
    acc += vec3f(0.10, 0.11, 0.14) * edge * 0.5;

    let ldir = R.light.xyz;
    let shadow_step = 0.5 / 6.0;
    for (var i = 0u; i < 256u; i++) {
      if (f32(i) >= steps || transmit < 0.01) {
        break;
      }
      let pos = ro + rd * t;
      let s = textureSampleLevel(density_tex, lin, pos + vec3f(0.5), 0.0);
      let ext = extinction(s);
      if (ext > 0.005) {
        // Ombre interne : marche courte vers la lumière.
        var occ = 0.0;
        for (var j = 1u; j <= 6u; j++) {
          let sp = pos + ldir * (f32(j) * shadow_step);
          let ss = textureSampleLevel(density_tex, lin, sp + vec3f(0.5), 0.0);
          occ += extinction(ss);
        }
        let shade = exp(-occ * shadow_step * 0.9);
        let albedo = vec3f(0.60, 0.64, 0.72);
        let lo = blackbody(s.w) * 2.2 +
          albedo * R.light.w * (shade * 0.92 + 0.08) * min(ext, 3.0) * 0.28;
        let a = 1.0 - exp(-ext * step_len);
        acc += transmit * lo * a;
        transmit *= exp(-ext * step_len);
      }
      t += step_len;
    }
    col = acc + transmit * bg;
  }

  // Tone-mapping exponentiel + gamma (le format canvas n'est pas une vue sRGB).
  let exposure = R.cam_up.w;
  let mapped = vec3f(1.0) - exp(-col * exposure);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
