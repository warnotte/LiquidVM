// Projection 3D : divergence MAC compacte, lisseur Jacobi (Neumann par clamp des
// voisins — la boîte fermée), soustraction du gradient. Trois entry points, un
// layout commun (les bindings inutilisés par un entry point sont ignorés).
// Schéma compact : la face N n'est pas stockée et vaut 0 (mur) dans la divergence,
// exactement comme en 2D — le couple divergence/gradient reste l'adjoint du lisseur.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,     // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f,
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,
  sphere_vel: vec4f, // xyz: vitesse de la sphère (voxels/s)
  // Traversée jusqu'à `shape` : la disposition de l'uniforme est COMMUNE, un
  // shader n'en déclare qu'un PRÉFIXE et doit décrire tout ce qui précède.
  // Neuf vec4 séparent `sphere_vel` (#13) de `shape` (#23 = floats 92-95,
  // l'ancien slot d'explosion unique, libre depuis les charges multiples).
  pad_shape: array<vec4f, 9>,
  shape: vec4f,      // x: type d'obstacle, yzw: paramètres (ratios du rayon)
}

// OBSTACLE — DISTANCE SIGNÉE en voxels, négative dedans. Une seule fonction
// pour toutes les passes : le prédicat n'est plus « dans la sphère » mais
// « sd < 0 », et l'opérateur ne change pas — seul le masque binaire de cellules
// change, donc le lisseur reste l'adjoint exact du couple divergence/gradient.
// `sphere.w` reste LE rayon (≤ 0 = pas d'obstacle) et les paramètres de forme
// sont des RATIOS de ce rayon : le curseur « taille » vaut pour toute forme.
// (Dupliquée dans les passes qui en ont besoin — les shaders sont autonomes.)
fn solid_sd(p: vec3f) -> f32 {
  let r = P.sphere.w;
  if (r <= 0.0) {
    return 1e9;
  }
  let q = p - P.sphere.xyz;
  let kind = i32(P.shape.x + 0.5);
  if (kind == 1) { // BOÎTE : demi-côtés = rayon × ratios
    let d = abs(q) - r * P.shape.yzw;
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  if (kind == 2) { // TORE d'axe vertical : grand rayon r, petit rayon r × y
    return length(vec2f(length(q.xz) - r, q.y)) - r * P.shape.y;
  }
  if (kind == 3) { // CLOCHE : coquille sphérique, OUVERTE PAR LE SOL —
    // pas de plan de coupe : c'est le plancher de la boîte (déjà un mur de
    // non-pénétration) qui ferme le bas. Le gaz est enfermé, la lumière
    // non : le rendu la traite en VERRE (voir raymarch).
    let rr = r * P.shape.z;
    let shell = abs(length(q) - rr) - rr * P.shape.y;
    // CHEMINÉE : un puits vertical percé au sommet (soustraction
    // d'un cylindre). Une cloche HERMÉTIQUE est un piège — la cavité
    // devient une région de fluide isolée, sans sortie pour la
    // pression : une détonation à l'intérieur sature en blanc et
    // éjecte les particules en lignes droites (vu par Renaud). Le
    // cylindre ne perce QUE la calotte haute : la calotte basse est
    // sous le plancher, qui la ferme. Assez étroit pour que l'air ne
    // se renouvelle presque pas — l'étouffement tient (mesuré).
    let vent = length(q.xz) - rr * 0.16;
    return max(shell, -vent);
  }
  return length(q) - r; // SPHÈRE
}

// Même règle « centre de cellule » que les écritures de vélocité.
fn solid_cell(c: vec3i) -> bool {
  return solid_sd(vec3f(c) + vec3f(0.5)) < 0.0;
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
@group(1) @binding(1) var pressure_src: texture_3d<f32>;
@group(1) @binding(2) var divergence_src: texture_3d<f32>;
@group(1) @binding(3) var scalar_dst: texture_storage_3d<r32float, write>;
@group(1) @binding(4) var vel_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(5) var den_src: texture_3d<f32>;
@group(1) @binding(6) var oxy_src: texture_3d<f32>;


fn in_bounds(c: vec3i) -> bool {
  let n = n_size();
  return all(c < n);
}

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let n = n_size();
  let v0 = textureLoad(vel_src, c, 0).xyz;
  // FACES BLOQUÉES : lire la vitesse PRESCRITE, pas la vitesse stockée.
  //
  // Le gradient impose la vitesse de la sphère sur les faces qu'elle bloque, et
  // zéro sur les parois. Si la divergence, elle, lit la valeur stockée à ces
  // mêmes faces, le triplet divergence / lisseur / gradient ne décrit plus le
  // MÊME opérateur autour de l'obstacle : la projection ne peut plus y annuler
  // la divergence, et il reste un écoulement parasite qui s'installe et grandit
  // avec le temps. C'est ce qu'on voyait comme une sphère « attirant » la fumée
  // après quelques secondes — l'obstacle ne l'attirait pas, il la fabriquait.
  //
  // Jacobi le pardonnait en partie (il converge peu) ; le multigrid non, il
  // converge FORT vers la solution du mauvais système. C'est le piège déjà écrit
  // pour le 2D — « le lisseur doit être l'adjoint exact du couple
  // divergence/gradient » — appliqué cette fois à l'OBSTACLE et non aux bords.
  let sv = P.sphere_vel.xyz;
  let solid_c = solid_cell(c);
  // Faces négatives (celles que ce texel stocke).
  var u0 = select(v0.x, sv.x, solid_c || solid_cell(c - vec3i(1, 0, 0)));
  var v_0 = select(v0.y, sv.y, solid_c || solid_cell(c - vec3i(0, 1, 0)));
  var w0 = select(v0.z, sv.z, solid_c || solid_cell(c - vec3i(0, 0, 1)));
  u0 = select(u0, 0.0, c.x == 0);
  v_0 = select(v_0, 0.0, c.y == 0);
  w0 = select(w0, 0.0, c.z == 0);
  // Faces positives (stockées par le voisin) ; hors grille = paroi, donc zéro.
  var ux = select(0.0, textureLoad(vel_src, c + vec3i(1, 0, 0), 0).x, c.x + 1 < n.x);
  var vy = select(0.0, textureLoad(vel_src, c + vec3i(0, 1, 0), 0).y, c.y + 1 < n.y);
  var wz = select(0.0, textureLoad(vel_src, c + vec3i(0, 0, 1), 0).z, c.z + 1 < n.z);
  ux = select(ux, sv.x, c.x + 1 < n.x && (solid_c || solid_cell(c + vec3i(1, 0, 0))));
  vy = select(vy, sv.y, c.y + 1 < n.y && (solid_c || solid_cell(c + vec3i(0, 1, 0))));
  wz = select(wz, sv.z, c.z + 1 < n.z && (solid_c || solid_cell(c + vec3i(0, 0, 1))));
  var div = ux - u0 + vy - v_0 + wz - w0;
  // EXPANSION de combustion : source de divergence au front de flamme (le gaz
  // brûlé gonfle) — mêmes critères que la réaction, oxygène compris.
  let dn = textureLoad(den_src, c, 0);
  let o2 = textureLoad(oxy_src, c, 0).x;
  let ignite = smoothstep(0.28, 0.55, dn.w);
  div -= P.emit_meta.w * min(dn.z, 1.0) * ignite * clamp(o2 / 0.25, 0.0, 1.0);
  textureStore(scalar_dst, gid, vec4f(div, 0.0, 0.0, 0.0));
}

fn p_at(c: vec3i) -> f32 {
  // Clamp = condition de Neumann (le voisin hors boîte vaut la valeur du bord).
  let n = n_size();
  return textureLoad(pressure_src, clamp(c, vec3i(0), n - vec3i(1)), 0).x;
}

// Voisin pour le lisseur : un voisin solide (sphère) contribue la valeur du centre —
// exactement la face fermée du gradient, comme le clamp aux bords de la boîte.
fn p_neighbor(c: vec3i, pc: f32) -> f32 {
  return select(p_at(c), pc, solid_cell(c));
}

@compute @workgroup_size(4, 4, 4)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let pc = p_at(c);
  let sum = p_neighbor(c + vec3i(1, 0, 0), pc) + p_neighbor(c - vec3i(1, 0, 0), pc) +
    p_neighbor(c + vec3i(0, 1, 0), pc) + p_neighbor(c - vec3i(0, 1, 0), pc) +
    p_neighbor(c + vec3i(0, 0, 1), pc) + p_neighbor(c - vec3i(0, 0, 1), pc);
  let div = textureLoad(divergence_src, c, 0).x;
  textureStore(scalar_dst, gid, vec4f((sum - div) / 6.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn gradient(@builtin(global_invocation_id) gid: vec3u) {
  let c = vec3i(gid);
  if (!in_bounds(c)) {
    return;
  }
  let pc = p_at(c);
  let solid_c = solid_cell(c);
  var vel = textureLoad(vel_src, c, 0).xyz;
  // Face ouverte : soustraire le gradient ; face de sphère : vitesse de la sphère
  // (bord mobile prescrit, le gradient n'y touche pas) ; paroi de boîte : 0.
  vel.x = select(
    P.sphere_vel.x,
    vel.x - (pc - p_at(c - vec3i(1, 0, 0))),
    !solid_c && !solid_cell(c - vec3i(1, 0, 0)),
  );
  vel.y = select(
    P.sphere_vel.y,
    vel.y - (pc - p_at(c - vec3i(0, 1, 0))),
    !solid_c && !solid_cell(c - vec3i(0, 1, 0)),
  );
  vel.z = select(
    P.sphere_vel.z,
    vel.z - (pc - p_at(c - vec3i(0, 0, 1))),
    !solid_c && !solid_cell(c - vec3i(0, 0, 1)),
  );
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);
  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
