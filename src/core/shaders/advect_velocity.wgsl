// Advection MacCormack de la vélocité sur GRILLE DÉCALÉE (MAC) :
// le texel (i,j) stocke .x = u sur la face gauche de la cellule (position physique
// (i, j+½)) et .y = v sur la face haute (position (i+½, j)). Chaque composante est
// advectée À SA PROPRE POSITION : le vecteur vitesse y est reconstruit en échantillonnant
// l'autre composante avec un UV décalé d'un demi-texel (voir sample_u / sample_v — c'est
// exactement l'interpolation bilinéaire MAC standard). Schéma prédicteur/correcteur
// d'ordre 2 clampé (Selle et al. 2008) ; extra.w = 0 désactive le correcteur (debug).
// Les faces bloquées (murs dessinés, bords du domaine hors périodique) sont forcées à
// zéro à l'écriture — la divergence en aval lit un champ déjà cohérent.
//
// Mode ouvert = boîte fermée + BANDE ÉPONGE : l'opérateur de pression reste celui des
// parois (symétrique, stable avec le multigrid), et « ce qui sort disparaît » est obtenu
// en amortissant exponentiellement vélocité et densité dans une bande aux bords. Une
// vraie condition de sortie Dirichlet rendrait le système non symétrique (la face
// virtuelle extrapolée découple la dernière colonne) et fait diverger le V-cycle.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s), w: outil
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var lin: sampler;

@group(1) @binding(0) var src_vel: texture_2d<f32>;
@group(1) @binding(1) var dst_hat: texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var hat_tex: texture_2d<f32>;
@group(1) @binding(3) var dst_vel: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var obstacle: texture_2d<f32>;

// Composante u du champ MAC à la position physique p (texel u centré en (i, j+½)).
fn sample_u(t: texture_2d<f32>, p: vec2f) -> f32 {
  return textureSampleLevel(t, lin, (p + vec2f(0.5, 0.0)) * P.grid.zw, 0.0).x;
}
// Composante v du champ MAC à la position physique p (texel v centré en (i+½, j)).
fn sample_v(t: texture_2d<f32>, p: vec2f) -> f32 {
  return textureSampleLevel(t, lin, (p + vec2f(0.0, 0.5)) * P.grid.zw, 0.0).y;
}

fn bound_coord(q: vec2i, size: vec2i) -> vec2i {
  if (P.extra.x > 0.5 && P.extra.x < 1.5) {
    return ((q % size) + size) % size;
  }
  return clamp(q, vec2i(0), size - vec2i(1));
}

fn solid_at(q: vec2i, size: vec2i) -> bool {
  return textureLoad(obstacle, bound_coord(q, size), 0).x > 0.5;
}

// La face u du texel c (entre les cellules c-1x et c) est-elle bloquée ?
fn u_face_blocked(c: vec2i, size: vec2i) -> bool {
  if (solid_at(c, size)) { return true; }
  if (c.x == 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return solid_at(vec2i(size.x - 1, c.y), size); // périodique : voisin d'en face
    }
    return true; // parois ET ouvert : bord fermé (l'éponge absorbe, cf. sponge())
  }
  return solid_at(vec2i(c.x - 1, c.y), size);
}

fn v_face_blocked(c: vec2i, size: vec2i) -> bool {
  if (solid_at(c, size)) { return true; }
  if (c.y == 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return solid_at(vec2i(c.x, size.y - 1), size);
    }
    return true;
  }
  return solid_at(vec2i(c.x, c.y - 1), size);
}

// Bande éponge du mode ouvert : facteur d'amortissement ∈ (0,1], = 1 hors mode ouvert.
fn sponge(pos: vec2f, dt: f32) -> f32 {
  if (P.extra.x < 1.5) {
    return 1.0;
  }
  let d = min(min(pos.x, P.grid.x - pos.x), min(pos.y, P.grid.y - pos.y));
  let s = 1.0 - clamp(d / (P.grid.x / 32.0), 0.0, 1.0); // bande ∝ grille (16 texels à 512)
  return 1.0 / (1.0 + 40.0 * s * s * dt);
}

// Extrema du stencil bilinéaire de la composante `chan` (0 = u, 1 = v) autour de la
// position physique p, pour le clamp anti-overshoot du correcteur.
fn stencil_bounds(p: vec2f, chan: u32) -> vec2f {
  let size = vec2i(P.grid.xy);
  var tc: vec2f;
  if (chan == 0u) { tc = p + vec2f(0.5, 0.0); } else { tc = p + vec2f(0.0, 0.5); }
  let g0 = vec2i(floor(tc - vec2f(0.5)));
  var lo = 1e9;
  var hi = -1e9;
  for (var dy = 0; dy < 2; dy++) {
    for (var dx = 0; dx < 2; dx++) {
      let s = textureLoad(src_vel, bound_coord(g0 + vec2i(dx, dy), size), 0);
      let value = select(s.y, s.x, chan == 0u);
      lo = min(lo, value);
      hi = max(hi, value);
    }
  }
  return vec2f(lo, hi);
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2u(P.grid.xy);
  if (gid.x >= size.x || gid.y >= size.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let dt = P.impulse.x;
  let here = textureLoad(src_vel, c, 0).xy;

  // Face u : backtrace depuis (i, j+½) avec le vecteur complet reconstruit sur place.
  let pu = vec2f(f32(c.x), f32(c.y) + 0.5);
  let vel_u = vec2f(here.x, sample_v(src_vel, pu));
  let u_hat = sample_u(src_vel, pu - vel_u * dt);

  // Face v : backtrace depuis (i+½, j).
  let pv = vec2f(f32(c.x) + 0.5, f32(c.y));
  let vel_v = vec2f(sample_u(src_vel, pv), here.y);
  let v_hat = sample_v(src_vel, pv - vel_v * dt);

  textureStore(dst_hat, c, vec4f(u_hat, v_hat, 0.0, 0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let usize = vec2u(P.grid.xy);
  if (gid.x >= usize.x || gid.y >= usize.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let size = vec2i(P.grid.xy);
  let dt = P.impulse.x;
  let here = textureLoad(src_vel, c, 0).xy;
  let hat = textureLoad(hat_tex, c, 0).xy;

  // Composante u.
  var u: f32;
  let pu = vec2f(f32(c.x), f32(c.y) + 0.5);
  let vel_u = vec2f(here.x, sample_v(src_vel, pu));
  if (P.extra.w < 0.5) {
    u = hat.x; // MacCormack désactivé : premier ordre
  } else {
    let back_track = sample_u(hat_tex, pu + vel_u * dt);
    u = hat.x + 0.5 * (here.x - back_track);
    let b = stencil_bounds(pu - vel_u * dt, 0u);
    u = clamp(u, b.x, b.y);
  }

  // Composante v.
  var v: f32;
  let pv = vec2f(f32(c.x) + 0.5, f32(c.y));
  let vel_v = vec2f(sample_u(src_vel, pv), here.y);
  if (P.extra.w < 0.5) {
    v = hat.y;
  } else {
    let back_track = sample_v(hat_tex, pv + vel_v * dt);
    v = hat.y + 0.5 * (here.y - back_track);
    let b = stencil_bounds(pv - vel_v * dt, 1u);
    v = clamp(v, b.x, b.y);
  }

  // Dissipation exponentielle discrétisée (≈ viscosité) + éponge du mode ouvert,
  // puis faces bloquées à zéro.
  let k = sponge(vec2f(gid.xy) + vec2f(0.5), dt) / (1.0 + P.misc.x * dt);
  u = u * k;
  v = v * k;
  if (u_face_blocked(c, size)) { u = 0.0; }
  if (v_face_blocked(c, size)) { v = 0.0; }
  textureStore(dst_vel, c, vec4f(u, v, 0.0, 0.0));
}
