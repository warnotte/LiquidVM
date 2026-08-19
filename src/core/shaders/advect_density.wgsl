// Advection MacCormack des champs transportés par le fluide, sur leur PROPRE grille
// (« dye »), plus fine que celle de la vélocité : .rgb = les trois densités de fluides,
// .a = la TEMPÉRATURE du feu — advectée exactement comme les densités, mais avec son
// propre taux de refroidissement (dissipation.w) et sa propre injection (outil feu).
// La vélocité est échantillonnée en coordonnées normalisées — les grilles n'ont pas à
// coïncider. Même schéma prédicteur/correcteur clampé que advect_velocity.wgsl.
// Les murs (grille sim) ne contiennent ni densité ni chaleur ; en mode ouvert la bande
// éponge absorbe tout aux bords.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s), w: outil
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s), w: refroidissement du feu (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s²), w: temps simulé (s)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var lin: sampler;

@group(1) @binding(0) var src_vel: texture_2d<f32>;
@group(1) @binding(1) var src_den: texture_2d<f32>;
@group(1) @binding(2) var dst_hat: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var hat_tex: texture_2d<f32>;
@group(1) @binding(4) var dst_den: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var obstacle: texture_2d<f32>;

// Bande éponge du mode ouvert (voir advect_velocity.wgsl). Position en texels sim.
fn sponge(pos: vec2f, dt: f32) -> f32 {
  if (P.extra.x < 1.5) {
    return 1.0;
  }
  let d = min(min(pos.x, P.grid.x - pos.x), min(pos.y, P.grid.y - pos.y));
  let s = 1.0 - clamp(d / (P.grid.x / 32.0), 0.0, 1.0); // bande ∝ grille (16 texels à 512)
  return 1.0 / (1.0 + 40.0 * s * s * dt);
}

// Vecteur vitesse à la position uv (normalisée), reconstruit depuis la grille MAC :
// chaque composante est échantillonnée avec son propre décalage d'un demi-texel.
fn velocity_at(uv: vec2f) -> vec2f {
  let p = uv * P.grid.xy;
  let u = textureSampleLevel(src_vel, lin, (p + vec2f(0.5, 0.0)) * P.grid.zw, 0.0).x;
  let v = textureSampleLevel(src_vel, lin, (p + vec2f(0.0, 0.5)) * P.grid.zw, 0.0).y;
  return vec2f(u, v);
}

fn bound_coord(q: vec2i, size: vec2i) -> vec2i {
  if (P.extra.x > 0.5 && P.extra.x < 1.5) {
    return ((q % size) + size) % size;
  }
  return clamp(q, vec2i(0), size - vec2i(1));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(P.dye.xy);
  if (gid.x >= size.x || gid.y >= size.y) {
    return;
  }
  let uv = (vec2f(gid.xy) + vec2f(0.5)) * P.dye.zw;
  // Vélocité en texels sim/s → déplacement normalisé = v·dt / taille de grille sim.
  let vel = velocity_at(uv);
  let back_uv = uv - vel * P.impulse.x * P.grid.zw;
  let den = textureSampleLevel(src_den, lin, back_uv, 0.0);
  textureStore(dst_hat, vec2i(gid.xy), den);
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(P.dye.xy);
  if (gid.x >= size.x || gid.y >= size.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let uv = (vec2f(gid.xy) + vec2f(0.5)) * P.dye.zw;
  // Le champ d'obstacles vit sur la grille de vélocité.
  let oc = clamp(vec2i(uv * P.grid.xy), vec2i(0), vec2i(P.grid.xy) - vec2i(1));
  if (textureLoad(obstacle, oc, 0).x > 0.5) {
    textureStore(dst_den, c, vec4f(0.0));
    return;
  }
  let vel = velocity_at(uv);
  let back_uv = uv - vel * P.impulse.x * P.grid.zw;
  let fwd_uv = uv + vel * P.impulse.x * P.grid.zw;
  let hat = textureLoad(hat_tex, c, 0);
  let d0 = textureLoad(src_den, c, 0);

  var den: vec4f;
  if (P.extra.w < 0.5) {
    den = hat; // MacCormack désactivé : premier ordre
  } else {
    let back_track = textureSampleLevel(hat_tex, lin, fwd_uv, 0.0);
    den = hat + 0.5 * (d0 - back_track);
    // Clamp aux extrema du stencil bilinéaire du champ d'origine à la position remontée.
    let isize = vec2i(P.dye.xy);
    let g0 = vec2i(floor(back_uv * P.dye.xy - vec2f(0.5)));
    let s00 = textureLoad(src_den, bound_coord(g0, isize), 0);
    let s10 = textureLoad(src_den, bound_coord(g0 + vec2i(1, 0), isize), 0);
    let s01 = textureLoad(src_den, bound_coord(g0 + vec2i(0, 1), isize), 0);
    let s11 = textureLoad(src_den, bound_coord(g0 + vec2i(1, 1), isize), 0);
    den = clamp(den, min(min(s00, s10), min(s01, s11)), max(max(s00, s10), max(s01, s11)));
  }
  // Dissipation par fluide (.xyz) et refroidissement du feu (.w).
  den = den / (vec4f(1.0) + P.dissipation * P.impulse.x);

  // Outils souris, en normalisé (indépendant des résolutions). Outil 0 : injecte le
  // fluide sélectionné ; outil 1 : gomme tout (chaleur comprise) ; outil 4 : feu —
  // injecte de la température et un voile de fumée (canal 2) qui la matérialise.
  if (P.impulse.y > 0.5) {
    let tool = u32(P.misc.w + 0.5);
    let off = uv - P.pointer.xy * P.grid.zw;
    let radius = P.impulse.w * P.grid.z;
    let falloff = exp(-dot(off, off) / (radius * radius));
    if (tool == 0u) {
      var inject = vec3f(0.0);
      inject[min(u32(P.impulse.z), 2u)] = P.misc.z * P.impulse.x * falloff;
      den = vec4f(den.xyz + inject, den.w);
    } else if (tool == 1u) {
      den = den / (1.0 + 12.0 * falloff * P.impulse.x);
    } else if (tool == 4u) {
      // Débits fixes (valeurs artistiques) : ~7 unités de chaleur/s au centre du splat,
      // plus un voile de fumée qui donne un corps à la flamme.
      den.w += 7.0 * P.impulse.x * falloff;
      den.z += 0.5 * P.impulse.x * falloff;
    }
  }
  // Éponge du mode ouvert, puis bornes : ≥ 0 (la correction peut sous-osciller),
  // densités ≤ 8 (float16 confortable), température ≤ 4.
  den = den * sponge(uv * P.grid.xy, P.impulse.x);
  den = clamp(den, vec4f(0.0), vec4f(8.0, 8.0, 8.0, 4.0));
  textureStore(dst_den, c, den);
}
