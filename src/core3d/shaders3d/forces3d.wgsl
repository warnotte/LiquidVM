// Forces : poussée thermique (la chaleur monte, +y), VENT horizontal, impulsion
// de l'émetteur (jet vertical avec balancement latéral périodique — l'asymétrie
// qui déclenche les instabilités naturelles du panache) et souffle du pointeur.
// Toutes appliquées aux faces MAC concernées.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,   // émetteur 0 : xyz centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit d'encre, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f, // xyz: origine du rayon du pointeur (voxels), w: rayon du pinceau (voxels)
  blow_dir: vec4f,    // xyz: direction du rayon (normalisée), w: actif (0/1)
  blow_force: vec4f,  // xyz: force du souffle (voxels/s²), w: libre
  sphere: vec4f,      // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,   // x: nombre d'émetteurs actifs (1..4)
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,   // encre (0/1/2) de chaque émetteur
  sphere_vel: vec4f,  // xyz: vitesse de la sphère (voxels/s) — condition de bord mobile
  ink_weights: vec4f, // xyz: poids propre des matières (voxels/s² par unité)
  wind: vec4f,        // x: force, y: amplitude d'oscillation (rad), z: période (s), w: cap (rad)
  // CHAMPS DE FORCE posés dans la scène (« modificateurs »), max 3.
  field_meta: vec4f,  // x: nombre actif, yzw: type de chaque champ (0 tourbillon, 1 vent)
  field0_a: vec4f,    // xyz: centre (voxels), w: rayon (voxels)
  field0_b: vec4f,    // xyz: axe (tourbillon) ou direction (vent), w: force
  field1_a: vec4f,
  field1_b: vec4f,
  field2_a: vec4f,
  field2_b: vec4f,
}

fn field_a(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.field0_a; }
    case 1u: { return P.field1_a; }
    default: { return P.field2_a; }
  }
}

fn field_b(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.field0_b; }
    case 1u: { return P.field1_b; }
    default: { return P.field2_b; }
  }
}

fn field_type(i: u32) -> f32 {
  switch i {
    case 0u: { return P.field_meta.y; }
    case 1u: { return P.field_meta.z; }
    default: { return P.field_meta.w; }
  }
}

fn emitter_pos(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.emitter; }
    case 1u: { return P.emitter1; }
    case 2u: { return P.emitter2; }
    default: { return P.emitter3; }
  }
}

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
@group(1) @binding(1) var den_src: texture_3d<f32>;
@group(1) @binding(2) var vel_dst: texture_storage_3d<rgba16float, write>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
}

fn density_at(p: vec3f) -> vec4f {
  // Les densités sont aux centres des cellules : échantillonnage direct.
  return textureSampleLevel(den_src, lin, p * inv_n(), 0.0);
}

// Poids gaussien cumulé de TOUS les émetteurs actifs à la position p.
fn emitter_gauss(p: vec3f) -> f32 {
  var g = 0.0;
  let count = u32(P.emit_meta.x + 0.5);
  for (var i = 0u; i < count; i++) {
    let e = emitter_pos(i);
    let d = distance(p, e.xyz) / max(e.w, 1e-3);
    g += exp(-d * d * 3.0);
  }
  return min(g, 1.5);
}

// Souffle du pointeur : poids gaussien autour du RAYON caméra→scène (un tube),
// nul derrière la caméra. Le geste pousse le fluide partout où son rayon passe.
fn blow_gauss(p: vec3f) -> f32 {
  if (P.blow_dir.w < 0.5) {
    return 0.0;
  }
  let v = p - P.blow_origin.xyz;
  let t = dot(v, P.blow_dir.xyz);
  if (t < 0.0) {
    return 0.0;
  }
  let d = distance(p, P.blow_origin.xyz + P.blow_dir.xyz * t) / max(P.blow_origin.w, 1e-3);
  return exp(-d * d * 2.0);
}

// Force des CHAMPS placés, évaluée en un point. Deux types, et leur traitement
// diffère pour une raison physique, pas esthétique :
//  - TOURBILLON : rotation solide autour d'un axe. Un champ rotationnel est
//    à divergence nulle, donc la projection le PRÉSERVE — il peut agir sur
//    l'air lui-même, matière ou pas.
//  - VENT LOCAL : direction constante, donc irrotationnel ; uniforme, il serait
//    annulé par la projection (leçon du souffle radial 2D). Il doit être
//    DIFFÉRENTIEL, pondéré par la matière présente, comme la poussée.
fn field_force(p: vec3f, matter: f32) -> vec3f {
  var f = vec3f(0.0);
  let count = u32(P.field_meta.x);
  for (var i = 0u; i < count; i++) {
    let a = field_a(i);
    let b = field_b(i);
    let rel = p - a.xyz;
    let radius = max(a.w, 1e-4);
    let d = length(rel) / radius;
    let fall = exp(-d * d * 2.0);
    if (fall > 1e-3) {
      if (field_type(i) < 0.5) {
        f += cross(b.xyz, rel) / radius * b.w * fall;
      } else {
        f += b.xyz * b.w * fall * matter;
      }
    }
  }
  return f;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let dt = P.misc.x;
  let t = P.misc.y;
  let fc = vec3f(c);
  let pu = fc + vec3f(0.0, 0.5, 0.5);
  let pv = fc + vec3f(0.5, 0.0, 0.5);
  let pw = fc + vec3f(0.5, 0.5, 0.0);

  var vel = textureLoad(vel_src, c, 0).xyz;

  // Boussinesq à deux voies : la chaleur pousse vers le haut, le poids propre des
  // matières tire vers le bas (l'encre lourde retombe, le carburant froid coule).
  let dn = density_at(pv);
  // La poussée SATURE au plafond des flammes. Le modèle de flottabilité est une
  // linéarisation valable sur une plage étroite ; au-delà, la chaleur d'une
  // boule de feu RAYONNE (elle éclaire), elle ne soulève pas proportionnellement.
  // Sans cette saturation, ouvrir le domaine d'incandescence multiplierait la
  // poussée d'autant et enverrait la scène au plafond.
  vel.y += dt * (P.diss.w * min(dn.w, 2.0) - dot(P.ink_weights.xyz, dn.xyz));

  // VENT horizontal. Force DIFFÉRENTIELLE, proportionnelle à la matière
  // présente — exactement comme la poussée. Un vent UNIFORME serait vain : dans
  // une boîte close, un champ constant viole la non-pénétration aux parois, la
  // projection lui oppose un gradient de pression et l'annule presque
  // entièrement (même leçon que le souffle radial du 2D, annulé parce
  // qu'irrotationnel). Proportionnel à la matière, il pousse le panache sans
  // que la pression puisse le compenser globalement.
  // Le cap OSCILLE lentement autour de sa direction moyenne : un vent constant
  // couche le panache une fois pour toutes, un vent qui tourne le fait onduler.
  if (P.wind.x > 0.0) {
    let ang = P.wind.w + P.wind.y * sin(t * 6.28318 / max(P.wind.z, 0.25));
    let du = density_at(pu);
    let dw = density_at(pw);
    vel.x += dt * P.wind.x * cos(ang) * (du.x + du.y + du.z);
    vel.z += dt * P.wind.x * sin(ang) * (dw.x + dw.y + dw.z);
  }

  // CHAMPS DE FORCE posés dans la scène : évalués sur chaque face MAC, à sa
  // propre position (la matière est celle du voisinage de la face).
  if (P.field_meta.x > 0.5) {
    let mu = density_at(pu);
    let mv = density_at(pv);
    let mw = density_at(pw);
    vel.x += dt * field_force(pu, mu.x + mu.y + mu.z).x;
    vel.y += dt * field_force(pv, mv.x + mv.y + mv.z).y;
    vel.z += dt * field_force(pw, mw.x + mw.y + mw.z).z;
  }

  // Impulsion de l'émetteur : jet montant + balancement latéral (fréquences
  // incommensurables pour ne jamais boucler visiblement).
  let lat = P.emit_vals.w * vec2f(sin(t * 2.9 + 1.3), cos(t * 2.3));
  vel.x += dt * emitter_gauss(pu) * lat.x;
  vel.y += dt * emitter_gauss(pv) * P.emit_vals.z;
  vel.z += dt * emitter_gauss(pw) * lat.y;

  // Souffle du pointeur (clic droit + glisser), par face.
  vel.x += dt * blow_gauss(pu) * P.blow_force.x;
  vel.y += dt * blow_gauss(pv) * P.blow_force.y;
  vel.z += dt * blow_gauss(pw) * P.blow_force.z;

  // Sphère mobile : les faces bloquées portent sa vitesse ; parois de boîte à 0.
  vel.x = select(vel.x, P.sphere_vel.x, face_blocked(c, vec3i(1, 0, 0)));
  vel.y = select(vel.y, P.sphere_vel.y, face_blocked(c, vec3i(0, 1, 0)));
  vel.z = select(vel.z, P.sphere_vel.z, face_blocked(c, vec3i(0, 0, 1)));
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
