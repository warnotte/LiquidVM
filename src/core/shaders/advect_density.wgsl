// Advection MacCormack des trois champs de densité (packés en .rgb), sur leur PROPRE
// grille (« dye »), plus fine que celle de la vélocité : la résolution visible est
// celle-ci, le coût de la projection reste celui de la grille de vélocité. La vélocité
// est échantillonnée en coordonnées normalisées — les deux grilles n'ont pas à coïncider.
// Même schéma prédicteur/correcteur clampé que advect_velocity.wgsl, plus la dissipation
// par fluide et l'injection souris (splat gaussien, en normalisé). Les murs (grille sim)
// ne contiennent jamais de densité ; en mode ouvert, ce qui sort du domaine disparaît.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s)
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité
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

// Bande éponge du mode ouvert (voir advect_velocity.wgsl) : la densité qui atteint
// les bords est absorbée — visuellement, elle « sort » du monde. Position en texels sim.
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

// Coordonnée entière bornée : wrap en périodique, clamp sinon.
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
  let den = textureSampleLevel(src_den, lin, back_uv, 0.0).xyz;
  textureStore(dst_hat, vec2i(gid.xy), vec4f(den, 0.0));
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
  let hat = textureLoad(hat_tex, c, 0).xyz;
  let d0 = textureLoad(src_den, c, 0).xyz;

  var den: vec3f;
  if (P.extra.w < 0.5) {
    den = hat; // MacCormack désactivé : premier ordre
  } else {
    let back_track = textureSampleLevel(hat_tex, lin, fwd_uv, 0.0).xyz;
    den = hat + 0.5 * (d0 - back_track);
    // Clamp aux extrema du stencil bilinéaire du champ d'origine à la position remontée.
    let isize = vec2i(P.dye.xy);
    let g0 = vec2i(floor(back_uv * P.dye.xy - vec2f(0.5)));
    let s00 = textureLoad(src_den, bound_coord(g0, isize), 0).xyz;
    let s10 = textureLoad(src_den, bound_coord(g0 + vec2i(1, 0), isize), 0).xyz;
    let s01 = textureLoad(src_den, bound_coord(g0 + vec2i(0, 1), isize), 0).xyz;
    let s11 = textureLoad(src_den, bound_coord(g0 + vec2i(1, 1), isize), 0).xyz;
    den = clamp(den, min(min(s00, s10), min(s01, s11)), max(max(s00, s10), max(s01, s11)));
  }
  den = den / (vec3f(1.0) + P.dissipation.xyz * P.impulse.x);

  // Outils souris sur la densité, en normalisé (indépendant des résolutions).
  // Outil 0 : injecte le fluide sélectionné ; outil 1 : gomme les trois champs.
  if (P.impulse.y > 0.5) {
    let tool = u32(P.misc.w + 0.5);
    let off = uv - P.pointer.xy * P.grid.zw;
    let radius = P.impulse.w * P.grid.z;
    let falloff = exp(-dot(off, off) / (radius * radius));
    if (tool == 0u) {
      var inject = vec3f(0.0);
      inject[min(u32(P.impulse.z), 2u)] = P.misc.z * P.impulse.x * falloff;
      den += inject;
    } else if (tool == 1u) {
      den = den / (1.0 + 12.0 * falloff * P.impulse.x);
    }
  }
  // Éponge du mode ouvert, puis bornes : ≥ 0 (la correction peut sous-osciller)
  // et plage confortable du float16.
  den = den * sponge(uv * P.grid.xy, P.impulse.x);
  den = clamp(den, vec3f(0.0), vec3f(8.0));
  textureStore(dst_den, c, vec4f(den, 0.0));
}
