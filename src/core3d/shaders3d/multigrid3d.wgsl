// Multigrid 3D pour la pression : V-cycle géométrique, pyramide GRID3 → 24³/16³
// (MG3_COARSEST_SIZE). Lisseur rouge-noir : V(1,1) et 2 cycles par frame valent
// mieux que le V(2,2) Jacobi ×4 qu'ils remplacent, pour la moitié du coût.
// Les tailles se déduisent de textureDimensions — les mêmes entry points servent
// tous les niveaux. Leçons du 2D, payées cher, appliquées d'office :
// 1) le couple divergence/gradient compact et ce lisseur partagent EXACTEMENT le
//    même opérateur A p = Σ(p_voisin) − 6p (le clamp des voisins hors boîte ajoute
//    k·p_centre qui reproduit précisément le Neumann des faces manquantes) ;
// 2) en Neumann pur la pression n'est définie qu'à une constante près et la
//    restriction amplifie sa dérive ×4 par niveau → ancrage p(0,0,0) = 0 au
//    niveau le plus grossier (8³), directement dans le lisseur. NE PAS RETIRER.

struct Params {
  misc: vec4f,      // z: Nx fin (remonte aux voxels fins), w: hauteur du domaine
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,    // xyz: centre (voxels fins), w: rayon (voxels fins, ≤0 = absente)
  // Traversée jusqu'à `shape` (#23 = floats 92-95, l'ancien slot
  // d'explosion unique, libre depuis les charges multiples).
  pad_shape: array<vec4f, 15>,
  shape: vec4f,     // x: type d'obstacle, yzw: paramètres
}

@group(0) @binding(0) var<uniform> P: Params;

// OBSTACLE — DISTANCE SIGNÉE en voxels, négative dedans (voir PLAN-OBSTACLES).
// Le prédicat n'est plus « dans la sphère » mais « sd < 0 » : l'opérateur ne
// change pas, seul le masque binaire de cellules change. `sphere.w` reste LE
// rayon (≤ 0 = pas d'obstacle), les paramètres de forme sont des RATIOS de ce
// rayon. Dupliquée dans chaque passe — les shaders sont autonomes.
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
  return length(q) - r; // SPHÈRE
}

@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var src_tex: texture_3d<f32>;  // restrict : résidu fin ; prolong : correction grossière
@group(1) @binding(1) var p_src: texture_3d<f32>;
@group(1) @binding(2) var rhs: texture_3d<f32>;
@group(1) @binding(3) var scalar_dst: texture_storage_3d<r32float, write>;
// Lisseur rouge-noir EN PLACE : la pression est lue ET écrite dans la même
// texture. r32float est le seul format storage garanti en read_write — la
// pression l'est, et c'est ce qui rend le Gauss-Seidel possible sans ping-pong.
@group(1) @binding(4) var fine_src: texture_3d<f32>;  // prolong ping-pong : pression fine courante
@group(1) @binding(5) var p_rw: texture_storage_3d<r32float, read_write>;

// CÔTÉ DU NIVEAU LE PLUS GROSSIER, fourni à la création du pipeline (constante
// override) : l'ancrage de l'espace nul ne doit viser QUE ce niveau, et sa
// taille dépend de la grille (24 pour 384, 16 pour les puissances de deux
// depuis que la pyramide s'arrête à 24³ — voir MG3_COARSEST_SIZE).
override COARSEST_SIDE: i32 = 8;
// Sur-relaxation du lisseur rouge-noir. 1,0 = Gauss-Seidel pur, la bonne valeur
// pour LISSER (un ω > 1 amplifie les hautes fréquences au lieu de les amortir).
// Le niveau le plus grossier, lui, ne lisse pas : il RÉSOUT — et là le SOR
// converge en O(N) balayages où Gauss-Seidel en demande O(N²). Ses pipelines
// sont créés avec ω ≈ 2/(1 + sin(π/N)) : le mode le plus lent d'un 24³ passe
// de 0,993 par balayage (Jacobi pondéré — 16 balayages n'en retiraient que
// 10 %) à une résolution quasi complète sur les 16 mêmes balayages.
override OMEGA_RB: f32 = 1.0;

fn p_at(c: vec3i, n: vec3i) -> f32 {
  return textureLoad(p_src, clamp(c, vec3i(0), n - vec3i(1)), 0).x;
}

// Lecture pour le lisseur EN PLACE (même clamp = même Neumann que p_at).
fn p_rw_at(c: vec3i, n: vec3i) -> f32 {
  return textureLoad(p_rw, clamp(c, vec3i(0), n - vec3i(1))).x;
}

// Sphère-obstacle vue depuis un niveau grossier. RESTRICTION CONSERVATRICE : la
// cellule n'est solide que si elle est ENTIÈREMENT dans la sphère, pas
// seulement son centre.
//
// C'est la règle du journal J4 — « le domaine grossier doit RÉTRÉCIR, jamais
// grandir » — et elle vaut pour l'obstacle comme elle valait pour le fluide.
// Avec le critère du centre, une cellule à moitié dedans devient solide, donc
// l'obstacle GROSSIT à chaque niveau : le niveau grossier corrige alors un
// problème que le niveau fin ne pose pas, et cette incohérence pompe de
// l'énergie tout autour de la boule. Le défaut n'apparaît qu'en HAUTE
// résolution parce que la pyramide y a plus de niveaux, et il met plusieurs
// secondes à devenir visible — d'où l'impression que la sphère « attire » la
// fumée au bout d'un moment.
fn solid_cell(c: vec3i, n: vec3i) -> bool {
  if (P.sphere.w <= 0.0) {
    return false;
  }
  // Les CELLULES sont cubiques et tous les axes se divisent par deux à chaque
  // niveau : un seul facteur d'échelle suffit, quelle que soit la forme du
  // domaine.
  let scale = P.misc.z / f32(n.x);
  // Demi-diagonale de la cellule grossière, en voxels fins : ce qu'il faut
  // retrancher au rayon pour exiger la cellule ENTIÈRE à l'intérieur.
  let half_diag = scale * 0.866025;
  // « Entièrement dedans » s'écrit sd < −demi-diagonale : la même inégalité
  // qu'avec le rayon, valable pour toute forme.
  return solid_sd((vec3f(c) + vec3f(0.5)) * scale) < -half_diag;
}

fn neighbor_sum(c: vec3i, n: vec3i) -> f32 {
  let pc = p_at(c, n);
  var sum = 0.0;
  for (var a = 0; a < 6; a++) {
    var off = vec3i(0);
    let axis = a >> 1;
    let sign = select(-1, 1, (a & 1) == 0);
    if (axis == 0) {
      off.x = sign;
    } else if (axis == 1) {
      off.y = sign;
    } else {
      off.z = sign;
    }
    let nb = c + off;
    sum += select(p_at(nb, n), pc, solid_cell(nb, n));
  }
  return sum;
}

// LISSEUR ROUGE-NOIR EN PING-PONG — la forme des niveaux MINUSCULES.
// Mesuré (bissection __mgLast, 256³, 1 V-cycle) : les balayages rouge-noir en
// place s'effondrent dès que la texture est ≤ 16³ (60 FPS à profondeur 3, 19 à
// profondeur 4, 24 en pyramide complète), pour un motif que ni les barrières ni
// les transitions n'expliquent proprement — pathologie de driver sur les
// dispatches read_write minuscules. À ces tailles le Jacobi ping-pong ne coûte
// rien : on garde donc rouge-noir où il paie (les grands niveaux) et Jacobi où
// il ne coûte pas (les petits). Même opérateur dans les deux cas.
// L'ordre Gauss-Seidel n'exige PAS le read_write : la passe rouge met à jour
// ses cellules et RECOPIE les noires vers l'autre ping ; la passe noire lit
// alors des rouges déjà frais. Deux dispatches par balayage, et la pression du
// niveau retombe sur son ping de départ après un nombre PAIR de dispatches.
//
// UN SEUL pipeline pour toutes les combinaisons : la parité ET le ω arrivent
// par un uniforme à OFFSET DYNAMIQUE (un root-constant côté D3D12), parce
// qu'alterner les PIPELINES entre dispatches minuscules est précisément ce que
// le driver fait payer — mesuré à 384³ : 16 dispatches en deux pipelines
// alternés = 5 FPS, les 16 mêmes en un seul = 21,6.
struct TinyParams {
  // x : parité (0 = rouge, 1 = noir) · y : ω (1 = Gauss-Seidel pour lisser,
  // ≈ 1,7 = SOR au niveau le plus grossier, qui RÉSOUT au lieu de lisser).
  v: vec4f,
}
@group(1) @binding(6) var<uniform> TP: TinyParams;

@compute @workgroup_size(4, 4, 4)
fn smooth_tiny(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(p_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let pc = p_at(c, n);
  var value = pc;
  if (((c.x + c.y + c.z) & 1) == i32(TP.v.x)) {
    let b = textureLoad(rhs, c, 0).x;
    value = mix(pc, (neighbor_sum(c, n) - b) / 6.0, TP.v.y);
  }
  // Ancrage de l'espace nul au niveau le plus grossier.
  if (n.x == COARSEST_SIDE && all(c == vec3i(0))) {
    value = 0.0;
  }
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

// LISSEUR ROUGE-NOIR (Gauss-Seidel par couleurs), EN PLACE.
//
// Pourquoi remplacer le Jacobi pondéré : à 384³ la pression coûtait 4 V-cycles
// par frame (~30 ms). Le facteur de lissage du Jacobi amorti (ω = 0,8) est ~0,5
// par balayage ; celui du Gauss-Seidel rouge-noir est ~0,25, ET il s'exécute en
// place — plus de ping-pong, donc plus d'écriture de la moitié inchangée. Un
// V-cycle rouge-noir vaut la qualité de plusieurs V-cycles Jacobi, pour un coût
// moindre par cycle. MESURÉ, pas raisonné : voir NOTES-DEV (couché du panache à
// 384³, la jauge de sous-convergence connue).
//
// L'OPÉRATEUR EST INCHANGÉ AU BIT PRÈS : même somme de voisins (clamp = Neumann,
// voisin solide = valeur du centre), même second membre, même ancrage. Le piège
// J4 (« le lisseur doit être l'adjoint exact du couple divergence/gradient »)
// porte sur l'opérateur, pas sur l'ordre de parcours — rouge-noir ne change QUE
// l'ordre. Deux dispatches par balayage : les voisins d'une cellule rouge sont
// tous noirs (stencil 7 points), donc aucune course dans un dispatch, et WebGPU
// ordonne les écritures entre dispatches d'une même passe.
fn rb_smooth(gid: vec3u, parity: i32) {
  let n = vec3i(textureDimensions(p_rw));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  if (((c.x + c.y + c.z) & 1) != parity) {
    return;
  }
  let pc = p_rw_at(c, n);
  var sum = 0.0;
  for (var a = 0; a < 6; a++) {
    var off = vec3i(0);
    let axis = a >> 1;
    let sign = select(-1, 1, (a & 1) == 0);
    if (axis == 0) {
      off.x = sign;
    } else if (axis == 1) {
      off.y = sign;
    } else {
      off.z = sign;
    }
    let nb = c + off;
    sum += select(p_rw_at(nb, n), pc, solid_cell(nb, n));
  }
  let b = textureLoad(rhs, c, 0).x;
  var value = mix(pc, (sum - b) / 6.0, OMEGA_RB);
  // Ancrage de l'espace nul au niveau le plus grossier.
  if (n.x == COARSEST_SIDE && all(c == vec3i(0))) {
    value = 0.0;
  }
  textureStore(p_rw, c, vec4f(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn smooth_red(@builtin(global_invocation_id) gid: vec3u) {
  rb_smooth(gid, 0);
}

@compute @workgroup_size(4, 4, 4)
fn smooth_black(@builtin(global_invocation_id) gid: vec3u) {
  rb_smooth(gid, 1);
}

// RÉSIDU + RESTRICTION EN UNE PASSE, dispatchée à la taille GROSSIÈRE. Le
// résidu fin n'était calculé QUE pour être restreint : l'écrire dans une
// texture pleine grille (226 Mo à 384³) pour la relire aussitôt coûtait deux
// allers-retours mémoire par cycle — mesuré au profileur, la pression pesait
// 17,7 ms sur 44. Ici chaque cellule grossière évalue le résidu de ses huit
// enfants à la volée : mêmes lectures de pression, zéro texture intermédiaire.
// Même opérateur, même facteur h² : Σ(8 résidus)/8 × 4 = Σ × 0,5.
@compute @workgroup_size(4, 4, 4)
fn restrict_residual(@builtin(global_invocation_id) gid: vec3u) {
  let nc = vec3i(textureDimensions(scalar_dst));
  let c = vec3i(gid);
  if (any(c >= nc)) {
    return;
  }
  let nf = vec3i(textureDimensions(p_src));
  var sum = 0.0;
  for (var i = 0; i < 8; i++) {
    let f = c * 2 + vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let ap = neighbor_sum(f, nf) - 6.0 * p_at(f, nf);
    sum += textureLoad(rhs, f, 0).x - ap;
  }
  textureStore(scalar_dst, gid, vec4f(sum * 0.5, 0.0, 0.0, 0.0));
}

// Correction grossière → niveau fin : interpolation trilinéaire manuelle
// (r32float non filtrable) ajoutée à la pression fine courante.
// Deux prolongations, choisies par la taille de la CIBLE :
// · prolong_add, EN PLACE (read_write) pour les grands niveaux — chaque thread
//   ne lit et n'écrit QUE sa propre cellule fine, aucune course possible, et
//   chaque texture garde le même rôle d'une frame à l'autre ;
// · prolong_add_pp, PING-PONG, pour les niveaux MINUSCULES : comme pour le
//   lisseur, tout accès read_write sur ces petites textures fait s'effondrer la
//   frame (mesuré à 384³ : la prolongation en place vers le niveau 24³ suffisait
//   à faire passer la frame de 46 à 260 ms). On écrit en write-only partout où
//   la texture est petite, sans chercher la règle exacte du driver.
@compute @workgroup_size(4, 4, 4)
fn prolong_add_pp(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(fine_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let nc = vec3i(textureDimensions(src_tex));
  let q = (vec3f(c) + vec3f(0.5)) * 0.5 - vec3f(0.5);
  let base = vec3i(floor(q));
  let t = q - vec3f(base);
  var corr = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let w = mix(1.0 - t.x, t.x, f32(o.x)) * mix(1.0 - t.y, t.y, f32(o.y)) *
      mix(1.0 - t.z, t.z, f32(o.z));
    corr += w * textureLoad(src_tex, clamp(base + o, vec3i(0), nc - vec3i(1)), 0).x;
  }
  let value = textureLoad(fine_src, c, 0).x + corr;
  textureStore(scalar_dst, gid, vec4f(value, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn prolong_add(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(textureDimensions(p_rw));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let nc = vec3i(textureDimensions(src_tex));
  let q = (vec3f(c) + vec3f(0.5)) * 0.5 - vec3f(0.5);
  let base = vec3i(floor(q));
  let t = q - vec3f(base);
  var corr = 0.0;
  for (var i = 0; i < 8; i++) {
    let o = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let w = mix(1.0 - t.x, t.x, f32(o.x)) * mix(1.0 - t.y, t.y, f32(o.y)) *
      mix(1.0 - t.z, t.z, f32(o.z));
    corr += w * textureLoad(src_tex, clamp(base + o, vec3i(0), nc - vec3i(1)), 0).x;
  }
  let value = textureLoad(p_rw, c).x + corr;
  textureStore(p_rw, c, vec4f(value, 0.0, 0.0, 0.0));
}
