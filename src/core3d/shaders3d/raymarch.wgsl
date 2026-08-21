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
  sphere: vec4f,    // xyz: centre (monde), w: rayon (monde, ≤0 = absente)
  blow_a: vec4f,    // retour visuel du souffle : xyz origine (monde), w rayon
  blow_b: vec4f,    // xyz direction, w actif (0/1)
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

// Palette des trois matières (canaux xyz) — dupliquée depuis config3d.ts :
// fumée grise, encre magenta, vapeur de carburant ambrée.
const INK0 = vec3f(0.55, 0.60, 0.68);
const INK1 = vec3f(1.00, 0.30, 0.80);
const INK2 = vec3f(1.00, 0.80, 0.45);

fn extinction(s: vec4f) -> f32 {
  return (s.x + s.y + s.z) * 22.0 + s.w * 5.0;
}

// Albédo du voxel : mélange des couleurs d'encres pondéré par leurs concentrations.
fn ink_albedo(s: vec4f) -> vec3f {
  let total = s.x + s.y + s.z;
  return (INK0 * s.x + INK1 * s.y + INK2 * s.z) / max(total, 1e-4);
}

fn blackbody(heat: f32) -> vec3f {
  let h = clamp(heat, 0.0, 1.7);
  return vec3f(h * 1.6, h * h * 0.9, h * h * h * 0.42);
}

// Intersection rayon / sphère-obstacle : t d'entrée, ou 1e9 si manquée/absente.
fn sphere_hit(ro: vec3f, rd: vec3f) -> f32 {
  if (R.sphere.w <= 0.0) {
    return 1e9;
  }
  let oc = ro - R.sphere.xyz;
  let b = dot(oc, rd);
  let disc = b * b - (dot(oc, oc) - R.sphere.w * R.sphere.w);
  if (disc < 0.0) {
    return 1e9;
  }
  let t = -b - sqrt(disc);
  return select(1e9, t, t > 0.0);
}

// Hachage rapide pour décaler le départ de chaque rayon (casse le banding).
fn hash12(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 345.45));
  q += dot(q, q + 34.345);
  return fract(q.x * q.y);
}

// Ombrage du sol sous la boîte : tapis sombre qui reçoit l'ombre volumétrique de
// la fumée (marche courte vers la lumière à travers la boîte) et celle de la boule.
fn floor_shade(fp: vec3f, bg: vec3f, ldir: vec3f) -> vec3f {
  let reach = smoothstep(1.15, 0.40, length(fp.xz));
  if (reach <= 0.0) {
    return bg;
  }
  var occ = 0.0;
  let bh = box_hit(fp, ldir);
  let s0 = max(bh.x, 0.0);
  if (bh.y > s0) {
    let step_len = (bh.y - s0) / 12.0;
    for (var j = 0u; j < 12u; j++) {
      let sp = fp + ldir * (s0 + (f32(j) + 0.5) * step_len);
      occ += extinction(textureSampleLevel(density_tex, lin, sp + vec3f(0.5), 0.0)) * step_len;
    }
  }
  if (sphere_hit(fp, ldir) < 1e8) {
    occ += 3.0;
  }
  let shadow = exp(-occ * 0.8);
  let base = vec3f(0.075, 0.080, 0.098);
  return mix(bg, base * (0.30 + 0.70 * shadow), reach);
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

  // Fond : dégradé sombre vertical, remplacé par le sol si le rayon le touche.
  var bg = mix(vec3f(0.045, 0.05, 0.065), vec3f(0.10, 0.105, 0.13), frag.uv.y * 0.5 + 0.5);
  if (rd.y < -1e-4) {
    let t_floor = (-0.502 - ro.y) / rd.y;
    if (t_floor > 0.0) {
      bg = floor_shade(ro + rd * t_floor, bg, R.light.xyz);
    }
  }

  let hit = box_hit(ro, rd);
  let t0 = max(hit.x, 0.0);
  // La marche s'arrête à la sphère-obstacle si le rayon la touche.
  let t_sphere = sphere_hit(ro, rd);
  let t_end = min(hit.y, t_sphere);
  var col: vec3f;
  if (hit.y <= t0) {
    col = bg;
  } else {
    let steps = max(R.cam_fwd.w, 16.0);
    let step_len = (max(t_end, t0 + 1e-4) - t0) / steps;
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
      // Retour visuel du souffle : fuseau lumineux le long du rayon du geste —
      // visible même dans l'air vide, pour voir OÙ le souffle agit.
      if (R.blow_b.w > 0.5) {
        let bv = pos - R.blow_a.xyz;
        let bt = dot(bv, R.blow_b.xyz);
        if (bt > 0.0) {
          let bd = distance(pos, R.blow_a.xyz + R.blow_b.xyz * bt) / max(R.blow_a.w, 1e-4);
          acc += transmit * exp(-bd * bd * 2.0) * vec3f(0.05, 0.08, 0.13) * step_len * 22.0;
        }
      }
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
        var shade = exp(-occ * shadow_step * 0.9);
        // La boule projette son ombre dans la fumée.
        if (sphere_hit(pos, ldir) < 1e8) {
          shade *= 0.25;
        }
        let albedo = ink_albedo(s);
        let lo = blackbody(s.w) * 2.2 +
          albedo * R.light.w * (shade * 0.92 + 0.08) * min(ext, 3.0) * 0.28;
        let a = 1.0 - exp(-ext * step_len);
        acc += transmit * lo * a;
        transmit *= exp(-ext * step_len);
      }
      t += step_len;
    }
    // Sphère-obstacle : surface mate ardoise, diffuse + lueur de contour —
    // et elle ROUGEOIE au corps noir quand la flamme la lèche.
    if (t_sphere < hit.y && transmit > 0.005) {
      let sp = ro + rd * t_sphere;
      let nrm = normalize(sp - R.sphere.xyz);
      let diff = max(dot(nrm, R.light.xyz), 0.0);
      let rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 3.0);
      let heat_here = textureSampleLevel(density_tex, lin, sp + nrm * 0.012 + vec3f(0.5), 0.0).w;
      var sphere_col = vec3f(0.30, 0.32, 0.38) * (0.30 + 0.70 * diff) + vec3f(0.10) * rim;
      sphere_col += blackbody(heat_here) * 0.9;
      acc += transmit * sphere_col;
      transmit = 0.0;
    }
    col = acc + transmit * bg;
  }

  // Tone-mapping exponentiel + gamma (le format canvas n'est pas une vue sRGB).
  let exposure = R.cam_up.w;
  let mapped = vec3f(1.0) - exp(-col * exposure);
  return vec4f(pow(mapped, vec3f(1.0 / 2.2)), 1.0);
}
