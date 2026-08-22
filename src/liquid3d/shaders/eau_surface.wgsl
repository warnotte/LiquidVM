// Rendu de SURFACE « grossier » + BOÎTE (remonté du jalon J5 suite au retour
// « l'eau semble disparaître aux bords » : rien ne dessinait la boîte) :
//  - density_blur : cell_count (particules/cellule, 8 = repos) → densité floutée
//    en boîte 3³ (texture rgba16float), calculée à CHAQUE sous-pas : c'est la
//    densité que lit le contrôle de densité du solveur (eau_grid.wgsl), et
//    l'iso-surface de ce champ EST la surface de l'eau au rendu.
//  - fs_surface : un rayon par pixel (même base caméra que les points), boîte
//    dessinée (sol ardoise, parois dégradées, grille 8×8, arêtes claires),
//    marche d'un voxel jusqu'à l'iso-surface, bissection, normale = −gradient,
//    Fresnel ciel / transmission Beer-Lambert vers la paroi derrière, reflet
//    solaire. Rayon droit (pas de réfraction) : stable et lisible, le vrai J5
//    raffinera. Mode points : la boîte seule, les points se dessinent par-dessus.

struct UEau {
  sim: vec4f,       // x: N
  sim2: vec4f,
  cam_pos: vec4f,   // xyz: caméra (monde), w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect
  cam_up: vec4f,    // xyz, w: exposition
  cam_fwd: vec4f,   // xyz
  render: vec4f,    // x: 0 surface / 1 points (boîte seule) / 2 coupe / 3 densité max, y: absorption, z: seuil iso
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<storage, read> cell_count: array<u32>;
@group(0) @binding(3) var dens_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var dens: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> census: array<atomic<u32>>;

const REST_DENSITY = 8.0;

// Recensement des CELLULES (instrument) : histogramme d'occupation dans
// census[66..71] = [0, 1-3, 4-7, 8-11, 12-23, 24+] particules/cellule.
@compute @workgroup_size(4, 4, 4)
fn cell_census(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(U.sim.x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let count = cell_count[u32(c.x + n * (c.y + n * c.z))];
  var b = 5u;
  if (count == 0u) {
    b = 0u;
  } else if (count < 4u) {
    b = 1u;
  } else if (count < 8u) {
    b = 2u;
  } else if (count < 12u) {
    b = 3u;
  } else if (count < 24u) {
    b = 4u;
  }
  atomicAdd(&census[66u + b], 1u);
}

// Seuil d'iso-surface (réglable : un peu sous 0.5 garde visibles les
// éclaboussures clairsemées de 2-4 particules/cellule).
fn iso() -> f32 {
  return U.render.z;
}

@compute @workgroup_size(4, 4, 4)
fn density_blur(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(U.sim.x);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        // Clamp aux parois : l'eau « continue » dans le mur, la normale y reste
        // portée par les autres axes (pas de faux bord au contact des parois).
        let q = clamp(c + vec3i(dx, dy, dz), vec3i(0), vec3i(n - 1));
        sum += min(f32(cell_count[u32(q.x + n * (q.y + n * q.z))]) / REST_DENSITY, 3.0);
      }
    }
  }
  textureStore(dens_dst, gid, vec4f(sum / 27.0, 0.0, 0.0, 0.0));
}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) ndc: vec2f,
}

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VSOut {
  var tri = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  out.pos = vec4f(tri[vi], 0.0, 1.0);
  out.ndc = tri[vi];
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

fn hash12(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 345.45));
  q += dot(q, q + 34.345);
  return fract(q.x * q.y);
}

fn density_at(p: vec3f) -> f32 {
  return textureSampleLevel(dens, lin, p + vec3f(0.5), 0.0).x;
}

fn surface_normal(p: vec3f) -> vec3f {
  let h = 1.0 / U.sim.x;
  let g = vec3f(
    density_at(p + vec3f(h, 0.0, 0.0)) - density_at(p - vec3f(h, 0.0, 0.0)),
    density_at(p + vec3f(0.0, h, 0.0)) - density_at(p - vec3f(0.0, h, 0.0)),
    density_at(p + vec3f(0.0, 0.0, h)) - density_at(p - vec3f(0.0, 0.0, h)),
  );
  let l = length(g);
  if (l < 1e-5) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return -g / l;
}

// Fausses couleurs de la densité (instrument de debug) : noir → bleu → cyan →
// blanc (= repos) → jaune (tassé) ; liseré rouge sur l'iso-surface.
fn heat(d: f32) -> vec3f {
  var c = mix(vec3f(0.0), vec3f(0.05, 0.15, 0.6), clamp(d / 0.25, 0.0, 1.0));
  c = mix(c, vec3f(0.1, 0.8, 0.9), clamp((d - 0.25) / 0.25, 0.0, 1.0));
  c = mix(c, vec3f(1.0), clamp((d - 0.5) / 0.5, 0.0, 1.0));
  c = mix(c, vec3f(1.0, 0.9, 0.2), clamp((d - 1.0) / 0.4, 0.0, 1.0));
  return mix(c, vec3f(1.0, 0.1, 0.1), 1.0 - smoothstep(0.0, 0.03, abs(d - iso())));
}

// Médiane des trois |composantes| proche de 0.5 ⇔ sur une arête de la boîte.
fn edge_factor(p: vec3f) -> f32 {
  let e = abs(p);
  return smoothstep(0.482, 0.497, max(min(e.x, e.y), min(max(e.x, e.y), e.z)));
}

// Parois INTÉRIEURES de la boîte au point de sortie du rayon : sol ardoise,
// parois plus sombres et dégradées vers le bas, grille discrète 8×8.
fn container(p: vec3f) -> vec3f {
  let a = abs(p);
  var base: vec3f;
  var uv: vec2f;
  if (a.y >= a.x && a.y >= a.z) {
    base = select(vec3f(0.06, 0.065, 0.08), vec3f(0.125, 0.135, 0.16), p.y < 0.0);
    uv = p.xz;
  } else {
    base = select(vec3f(0.085, 0.095, 0.12), vec3f(0.095, 0.105, 0.13), a.x >= a.z);
    base *= 0.55 + 0.6 * (p.y + 0.5);
    uv = select(p.xy, p.zy, a.x >= a.z);
  }
  let dl = (vec2f(0.5) - abs(fract(uv * 8.0) - vec2f(0.5))) / 8.0;
  let line = 1.0 - smoothstep(0.0015, 0.005, min(dl.x, dl.y));
  return base * (1.0 + 0.45 * line) + vec3f(0.22, 0.24, 0.30) * edge_factor(p);
}

@fragment
fn fs_surface(frag: VSOut) -> @location(0) vec4f {
  let tanf = U.cam_pos.w;
  let rd = normalize(
    U.cam_fwd.xyz + U.cam_right.xyz * frag.ndc.x * tanf * U.cam_right.w +
      U.cam_up.xyz * frag.ndc.y * tanf,
  );
  let ro = U.cam_pos.xyz;
  let ldir = normalize(vec3f(0.35, 0.85, 0.30));

  // Fond : dégradé sombre + tapis sous la boîte (même ancrage que le feu).
  var bg = mix(vec3f(0.045, 0.05, 0.065), vec3f(0.10, 0.105, 0.13), frag.ndc.y * 0.5 + 0.5);
  if (rd.y < -1e-4) {
    let t_floor = (-0.502 - ro.y) / rd.y;
    if (t_floor > 0.0) {
      let fp = ro + rd * t_floor;
      bg = mix(bg, vec3f(0.075, 0.08, 0.098), smoothstep(1.15, 0.40, length(fp.xz)));
    }
  }

  // Vue debug 2 : COUPE verticale z = 0 du champ de densité, plaquée à l'écran
  // (x = largeur de la boîte, y = le QUART inférieur agrandi ×4 : ~31 px/voxel).
  if (U.render.x > 1.5 && U.render.x < 2.5) {
    let p = vec3f(frag.ndc.x * 0.5, -0.5 + (frag.ndc.y + 1.0) * 0.125, 0.0);
    let vox = (p.y + 0.5) * U.sim.x;
    let line = 1.0 - smoothstep(0.0, 0.08, abs(fract(vox) - 0.5) * 2.0 - 0.92);
    var c = heat(density_at(p));
    c = mix(c, vec3f(0.5), 0.25 * line * f32(u32(floor(vox)) % 4u == 0u));
    return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);
  }

  let hit = box_hit(ro, rd);
  let t0 = max(hit.x, 0.0);
  var col = bg;
  if (hit.y > t0) {
    let entry = ro + rd * hit.x;
    let back = container(ro + rd * hit.y);
    let glass = vec3f(0.10, 0.11, 0.14) * edge_factor(entry);
    if (U.render.x > 2.5) {
      // Vue debug 3 : densité MAX le long du rayon (rayon droit).
      var dmax = 0.0;
      var t = t0;
      for (var i = 0u; i < 256u; i++) {
        if (t >= hit.y) {
          break;
        }
        dmax = max(dmax, density_at(ro + rd * t));
        t += 1.0 / U.sim.x;
      }
      col = select(back * 0.5, heat(dmax), dmax > 0.02) + glass;
    } else if (U.render.x > 0.5) {
      col = back + glass;
    } else {
      let step_len = 1.0 / U.sim.x;
      var hit_found = false;
      var t_hit = t0;
      var n_hit = vec3f(0.0, 1.0, 0.0);
      // Entrée directement dans l'eau (vue à travers la paroi de verre) : la
      // surface est la face de la boîte elle-même.
      if (density_at(ro + rd * (t0 + 1e-3)) > iso()) {
        hit_found = true;
        let ae = abs(entry);
        n_hit = select(
          select(vec3f(0.0, 0.0, sign(entry.z)), vec3f(0.0, sign(entry.y), 0.0), ae.y >= ae.z),
          vec3f(sign(entry.x), 0.0, 0.0),
          ae.x >= ae.y && ae.x >= ae.z,
        );
      } else {
        var t_prev = t0;
        var t = t0 + step_len * hash12(frag.ndc * 917.0);
        for (var i = 0u; i < 256u; i++) {
          if (t >= hit.y) {
            break;
          }
          if (density_at(ro + rd * t) > iso()) {
            // Bissection entre le dernier point d'air et celui-ci.
            var ta = t_prev;
            var tb = t;
            for (var k = 0u; k < 4u; k++) {
              let tm = 0.5 * (ta + tb);
              if (density_at(ro + rd * tm) > iso()) {
                tb = tm;
              } else {
                ta = tm;
              }
            }
            t_hit = tb;
            n_hit = surface_normal(ro + rd * tb);
            hit_found = true;
            break;
          }
          t_prev = t;
          t += step_len;
        }
      }
      if (!hit_found) {
        col = back;
      } else {
        let p_hit = ro + rd * t_hit;
        n_hit = select(n_hit, -n_hit, dot(n_hit, rd) > 0.0);
        // Réfraction (n = 1,33) : la grille du sol se décale sous la ligne
        // d'eau — le signal qui fait lire un volume d'eau claire dans un bac.
        // Le rayon réfracté cumule l'épaisseur d'eau traversée (pas 1,5 voxel)
        // et va chercher la paroi derrière.
        let rd2 = refract(rd, n_hit, 0.752);
        let exit2 = max(box_hit(p_hit, rd2).y, 0.0);
        let step2 = 1.5 * step_len;
        var water_len = 0.0;
        var t2 = 0.5 * step2;
        for (var i = 0u; i < 192u; i++) {
          if (t2 >= exit2) {
            break;
          }
          if (density_at(p_hit + rd2 * t2) > iso()) {
            water_len += step2;
          }
          t2 += step2;
        }
        let back2 = container(p_hit + rd2 * exit2);
        let absorb = vec3f(6.0, 2.2, 1.0) * U.render.y;
        let through = back2 * exp(-absorb * water_len) +
          vec3f(0.02, 0.13, 0.20) * (1.0 - exp(-3.0 * water_len));
        let f = 0.02 + 0.98 * pow(1.0 - max(dot(n_hit, -rd), 0.0), 5.0);
        let r = reflect(rd, n_hit);
        // Ciel réfléchi : horizon clair, zénith plus bleu, halo large autour du
        // soleil — une surface plane le reflète uniformément, les vagues le découpent.
        let sky = mix(vec3f(0.50, 0.54, 0.60), vec3f(0.22, 0.32, 0.48), smoothstep(0.0, 0.9, r.y)) *
          smoothstep(-0.25, 0.05, r.y) +
          vec3f(0.9, 0.85, 0.7) * pow(max(dot(r, ldir), 0.0), 8.0) * 0.35;
        let spec = pow(max(dot(r, ldir), 0.0), 200.0) * 3.0;
        col = mix(through, sky, f) + spec * (0.1 + 0.9 * f);
      }
      col += glass;
    }
  }
  col = vec3f(1.0) - exp(-col * U.cam_up.w);
  return vec4f(pow(col, vec3f(1.0 / 2.2)), 1.0);
}
