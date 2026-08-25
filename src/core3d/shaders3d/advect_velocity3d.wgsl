// Advection MacCormack de la vélocité sur grille MAC 3D — deux entry points.
// predict : semi-lagrangien φ̂ = A(φ) vers la texture scratch (aux).
// correct : φ̃ = A⁻¹(φ̂) (re-advection en remontant le temps), φ' = φ̂ + ½(φ − φ̃),
// clampé au min/max du stencil trilinéaire du point rétro-advecté (Selle 2008) —
// l'ordre 2 sans les oscillations. Dissipation et murs appliqués au correcteur.
// Convention MAC : texel (i,j,k) → u à (i, j+½, k+½), v à (i+½, j, k+½), w à (i+½, j+½, k).

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,     // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,
  sphere_vel: vec4f, // xyz: vitesse de la sphère (voxels/s) — condition de bord MOBILE
  // Champs traversés sans être lus ici : la disposition de l'uniforme est
  // COMMUNE à toutes les passes, un shader n'en déclare qu'un préfixe.
  ink_weights: vec4f,
  wind: vec4f,
  field_meta: vec4f,
  field0_a: vec4f, field0_b: vec4f,
  field1_a: vec4f, field1_b: vec4f,
  field2_a: vec4f, field2_b: vec4f,
  burst_a: vec4f, burst_b: vec4f,
  soot: vec4f,
  // x: bande LATÉRALE (voxels, 0 = parois closes), y: sa force,
  // z: bande de PLAFOND, w: sa force. Deux réglages parce que deux rôles.
  open_box: vec4f,
}

// BANDE ÉPONGE (« ciel ouvert ») — la boîte 3D est close de partout (Neumann par
// clamp), donc tout ce qu'on injecte finit par la remplir : un panache plafonne
// et repeint le plafond, une explosion sature le volume en deux secondes.
//
// La solution n'est PAS une vraie sortie Dirichlet avec face virtuelle : elle
// rend le système non symétrique (dernière rangée découplée) et le multigrid
// diverge par construction — leçon déjà payée en 2D, ne pas y revenir. On garde
// la boîte fermée, donc l'opérateur symétrique intact, et on AMORTIT dans une
// bande près des parois : la matière et la vitesse s'y éteignent avant d'avoir
// eu le temps de s'accumuler.
//
// Le SOL est exclu : c'est la seule paroi qui soit un vrai objet de la scène
// (le raymarch l'éclaire et y projette les ombres). Une explosion doit rebondir
// dessus, pas s'y dissoudre.
fn sponge3(p: vec3f, dt: f32) -> f32 {
  // Per-axe : le plafond n'est plus à la même hauteur que la largeur.
  let nv = n_sizef();
  var f = 1.0;
  // PAROIS LATÉRALES : c'est par elles que doit s'évacuer la nappe qui s'étale
  // au sol. Rien d'intéressant ne s'y gare, on peut donc y être franc.
  if (P.open_box.x > 0.0) {
    let ds = min(min(p.x, nv.x - p.x), min(p.z, nv.z - p.z));
    let s = 1.0 - clamp(ds / P.open_box.x, 0.0, 1.0);
    f /= 1.0 + P.open_box.y * s * s * dt;
  }
  // PLAFOND : réglage SÉPARÉ, et bien plus doux. Un panache TRAVERSE la bande ;
  // le chapeau d'un champignon, lui, monte au plafond et s'y GARE. Une
  // absorption calibrée pour un passage le lamine alors sur place — c'est ce qui
  // transformait le chapeau en dalle diffuse. Deux parois, deux rôles opposés :
  // les traiter avec la même sévérité était l'erreur.
  if (P.open_box.z > 0.0) {
    let s = 1.0 - clamp((nv.y - p.y) / P.open_box.z, 0.0, 1.0);
    f /= 1.0 + P.open_box.w * s * s * dt;
  }
  // Le SOL n'est jamais absorbant : c'est la seule paroi qui soit un objet de la
  // scène, et une explosion doit rebondir dessus.
  return f;
}


// Sphère-obstacle analytique : la cellule est solide si son CENTRE est dans la
// sphère — la même règle que le gradient et le lisseur (adjonction exacte).
fn solid_cell(c: vec3i) -> bool {
  if (P.sphere.w <= 0.0) {
    return false;
  }
  return distance(vec3f(c) + vec3f(0.5), P.sphere.xyz) < P.sphere.w;
}

fn face_blocked(c: vec3i, axis: vec3i) -> bool {
  return solid_cell(c) || solid_cell(c - axis);
}

@group(0) @binding(0) var<uniform> P: Params;

// TAILLE DE LA GRILLE, par axe. Le domaine n'est plus forcément cubique :
// Nx = Nz = misc.z, Ny = misc.z × misc.w. Les CELLULES, elles, restent cubiques
// — c'est ce qui permet de ne rien changer à l'opérateur ni à l'advection.
fn n_size() -> vec3i {
  return vec3i(i32(P.misc.z), i32(P.misc.z * P.misc.w), i32(P.misc.z));
}

fn n_sizef() -> vec3f {
  return vec3f(P.misc.z, P.misc.z * P.misc.w, P.misc.z);
}
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var aux: texture_3d<f32>;
@group(1) @binding(2) var vel_dst: texture_storage_3d<rgba16float, write>;

const OFF_U = vec3f(0.5, 0.0, 0.0);
const OFF_V = vec3f(0.0, 0.5, 0.0);
const OFF_W = vec3f(0.0, 0.0, 0.5);

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
}

fn src_u(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_U) * inv_n(), 0.0).x;
}
fn src_v(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_V) * inv_n(), 0.0).y;
}
fn src_w(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + OFF_W) * inv_n(), 0.0).z;
}
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(src_u(p), src_v(p), src_w(p));
}

// Remontée de caractéristique RK2 (point milieu) : en écoulement courbe, l'erreur
// d'un pas d'Euler est en O(dt²·courbure) — le point milieu la réduit d'un ordre.
fn backtrace(p: vec3f, dt: f32) -> vec3f {
  let mid = p - 0.5 * dt * velocity_at(p);
  return p - dt * velocity_at(mid);
}

fn forwardtrace(p: vec3f, dt: f32) -> vec3f {
  let mid = p + 0.5 * dt * velocity_at(p);
  return p + dt * velocity_at(mid);
}

fn aux_u(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_U) * inv_n(), 0.0).x;
}
fn aux_v(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_V) * inv_n(), 0.0).y;
}
fn aux_w(p: vec3f) -> f32 {
  return textureSampleLevel(aux, lin, (p + OFF_W) * inv_n(), 0.0).z;
}

// Min/max d'une composante sur les 8 coins du stencil trilinéaire au point `pos`
// (grille des faces de cette composante : coordonnée texel continue = pos + off).
fn stencil_minmax(pos: vec3f, off: vec3f, comp: i32) -> vec2f {
  let base = vec3i(floor(pos + off - vec3f(0.5)));
  var lo = 1e30;
  var hi = -1e30;
  for (var i = 0; i < 8; i++) {
    let corner = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let c = clamp(base + corner, vec3i(0), n_size() - vec3i(1));
    let v4 = textureLoad(vel_src, c, 0);
    let v = select(select(v4.z, v4.y, comp == 1), v4.x, comp == 0);
    lo = min(lo, v);
    hi = max(hi, v);
  }
  return vec2f(lo, hi);
}

@compute @workgroup_size(4, 4, 4)
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let dt = P.misc.x;
  let fc = vec3f(c);
  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);
  let u = src_u(backtrace(pu, dt));
  let v = src_v(backtrace(pv, dt));
  let w = src_w(backtrace(pw, dt));
  textureStore(vel_dst, gid, vec4f(u, v, w, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let dt = P.misc.x;
  let decay = 1.0 / (1.0 + P.diss.x * dt);
  let fc = vec3f(c);
  let orig = textureLoad(vel_src, c, 0);
  let hat = textureLoad(aux, c, 0);

  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);

  // φ' = φ̂ + ½(φ − φ̃), clampé au stencil du point rétro-advecté (traces RK2).
  var u = hat.x + 0.5 * (orig.x - aux_u(forwardtrace(pu, dt)));
  var v = hat.y + 0.5 * (orig.y - aux_v(forwardtrace(pv, dt)));
  var w = hat.z + 0.5 * (orig.z - aux_w(forwardtrace(pw, dt)));
  let mu = stencil_minmax(backtrace(pu, dt), OFF_U, 0);
  let mv = stencil_minmax(backtrace(pv, dt), OFF_V, 1);
  let mw = stencil_minmax(backtrace(pw, dt), OFF_W, 2);
  // Éponge : chaque composante est amortie à SA position MAC. Amortir la vitesse
  // ici introduit de la divergence, que la projection de la frame suivante
  // reprend — c'est le prix, et il est faible devant le jet qui longeait le
  // plafond quand la boîte était close.
  u = clamp(u, mu.x, mu.y) * decay * sponge3(pu, dt);
  v = clamp(v, mv.x, mv.y) * decay * sponge3(pv, dt);
  w = clamp(w, mw.x, mw.y) * decay * sponge3(pw, dt);

  // Sphère-obstacle MOBILE : les faces bloquées portent SA vitesse (la boule
  // brasse le fluide) ; les parois de la boîte restent immobiles.
  u = select(u, P.sphere_vel.x, face_blocked(c, vec3i(1, 0, 0)));
  v = select(v, P.sphere_vel.y, face_blocked(c, vec3i(0, 1, 0)));
  w = select(w, P.sphere_vel.z, face_blocked(c, vec3i(0, 0, 1)));
  u = select(u, 0.0, c.x == 0);
  v = select(v, 0.0, c.y == 0);
  w = select(w, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(u, v, w, 0.0));
}
