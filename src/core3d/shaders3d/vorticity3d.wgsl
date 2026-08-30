// Vorticity confinement 3D. En 3D la vorticité est un VECTEUR : ω = ∇×v.
// curl : ω aux centres des cellules (différences centrées des vitesses centrées),
//        |ω| dans le canal w.
// blur_curl : |ω| flouté (boîte 3³) — LE fix du grain : le gradient de |ω| brut
//        à ±1 voxel injectait du bruit à l'échelle de la grille (calibré EPS-*,
//        d'où l'ancien défaut ε=0) ; flouter la magnitude AVANT d'en prendre le
//        gradient donne des normales N̂ cohérentes sur ~3 voxels. ω passe tel quel.
// confine : force F = ε (N̂ × ω) avec N̂ = ∇|ω|flou normalisé — pousse vers les
//        cœurs de tourbillons, restitue l'énergie que l'advection dissipe.
// Chaque composante de force est évaluée à SA position de face MAC.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: Nx (= Nz), w: HAUTEUR du domaine (Ny/Nx)
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
  sphere_vel: vec4f, // xyz: vitesse de la sphère (voxels/s), w: force de vorticité ε
                     // (elle a déménagé ici : `misc.w` porte désormais la
                     //  hauteur du domaine, seul slot déclaré par TOUS les
                     //  shaders — c'est ce qui a permis de rendre la grille non
                     //  cubique sans rallonger le préfixe d'uniforme de chacun.)
  // Traversée jusqu'à `shape` (#23 = floats 92-95, l'ancien slot
  // d'explosion unique, libre depuis les charges multiples).
  pad_shape: array<vec4f, 9>,
  shape: vec4f,      // x: type d'obstacle, yzw: paramètres
}

fn solid_cell(c: vec3i) -> bool {
  if (P.sphere.w <= 0.0) {
    return false;
  }
  return solid_sd(vec3f(c) + vec3f(0.5)) < 0.0;
}

fn face_blocked(c: vec3i, axis: vec3i) -> bool {
  return solid_cell(c) || solid_cell(c - axis);
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
@group(1) @binding(1) var curl_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var curl_src: texture_3d<f32>;
@group(1) @binding(3) var vel_dst: texture_storage_3d<rgba16float, write>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
}

// ROTATIONNEL EN MI-RÉSOLUTION (2026-08-28). La chaîne du confinement ne
// consomme ω qu'après un FLOU volontaire (le fix anti-grain) : autant ne jamais
// le calculer plus fin que ce qu'on s'apprête à flouter. Le rotationnel et son
// flou vivent sur une grille MOITIÉ (1/8 du coût — mesuré : la chaîne vorticité
// pesait ~8 ms sur les ~50 d'une frame 384³) ; le confinement, lui, reste à
// pleine grille et échantillonne ce champ en coordonnées NORMALISÉES — il ne
// sait même pas que la texture est plus petite.
//
// L'ÉCHELLE NE BOUGE PAS, et ce n'est pas un détail (cf. le piège du ×SCALE3) :
// les dérivées sont centrées à ±1 cellule grossière (= 2 voxels fins) mais
// exprimées PAR VOXEL FIN — mêmes unités qu'avant, la force de confinement ne
// voit qu'un ω un peu plus lisse, ce qui est précisément l'intention du flou.
// La vitesse est échantillonnée en trilinéaire (le décalage d'une demi-face du
// stockage MAC devient un demi-voxel d'à-peu-près — noyé dans le lissage).
@compute @workgroup_size(4, 4, 4)
fn curl(@builtin(global_invocation_id) gid: vec3u) {
  let nh = vec3i(textureDimensions(curl_dst));
  let c = vec3i(gid);
  if (any(c >= nh)) {
    return;
  }
  let inv = vec3f(1.0) / n_sizef();
  // Centre de la cellule grossière, en voxels fins.
  let p = (vec3f(c) + vec3f(0.5)) * 2.0;
  let d = 2.0; // un pas grossier, en voxels fins (le sampler clampe aux bords)
  let xp = textureSampleLevel(vel_src, lin, (p + vec3f(d, 0.0, 0.0)) * inv, 0.0).xyz;
  let xm = textureSampleLevel(vel_src, lin, (p - vec3f(d, 0.0, 0.0)) * inv, 0.0).xyz;
  let yp = textureSampleLevel(vel_src, lin, (p + vec3f(0.0, d, 0.0)) * inv, 0.0).xyz;
  let ym = textureSampleLevel(vel_src, lin, (p - vec3f(0.0, d, 0.0)) * inv, 0.0).xyz;
  let zp = textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.0, d)) * inv, 0.0).xyz;
  let zm = textureSampleLevel(vel_src, lin, (p - vec3f(0.0, 0.0, d)) * inv, 0.0).xyz;
  let omega = (0.5 / d) * vec3f(
    (yp.z - ym.z) - (zp.y - zm.y),
    (zp.x - zm.x) - (xp.z - xm.z),
    (xp.y - xm.y) - (yp.x - ym.x),
  );
  textureStore(curl_dst, gid, vec4f(omega, length(omega)));
}

// |ω| flouté (boîte 3³, 27 loads), ω du centre passé tel quel : le confinement
// lit UNE texture. Entrée : la texture de rotationnel via le binding 0 (même
// layout que curl), sortie : velScratch (libre à ce point de la frame).
@compute @workgroup_size(4, 4, 4)
fn blur_curl(@builtin(global_invocation_id) gid: vec3u) {
  // Dimensions lues sur la TEXTURE (mi-résolution), pas sur la grille fine.
  let n = vec3i(textureDimensions(vel_src));
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  var sum = 0.0;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(c + vec3i(dx, dy, dz), vec3i(0), n - vec3i(1));
        sum += textureLoad(vel_src, q, 0).w;
      }
    }
  }
  let omega = textureLoad(vel_src, c, 0).xyz;
  textureStore(curl_dst, gid, vec4f(omega, sum / 27.0));
}

fn omega_at(p: vec3f) -> vec3f {
  return textureSampleLevel(curl_src, lin, p * inv_n(), 0.0).xyz;
}

// |ω| FLOUTÉ (canal w écrit par blur_curl) — jamais la magnitude brute.
fn omega_len(p: vec3f) -> f32 {
  return textureSampleLevel(curl_src, lin, p * inv_n(), 0.0).w;
}

// FONDU DU CONFINEMENT AU CONTACT DE L'OBSTACLE.
//
// Le rotationnel ne connaît pas la sphère : il lit la vitesse PRESCRITE sur les
// faces qu'elle bloque à côté de la vitesse du fluide une cellule plus loin. La
// marche d'escalier entre les deux produit un ω énorme — qui n'est pas un
// tourbillon du fluide mais un artefact de la discrétisation du bord. Le
// confinement, lui, RENFORCE ce qu'il trouve : il transforme cette nappe en un
// jet tangent à la boule, qui s'entretient et finit par remplir la boîte.
//
// Et le défaut CROÎT AVEC LA RÉSOLUTION sans que ε y soit pour rien : la marche
// vaut Δv sur UNE cellule, donc |ω| ≈ Δv/h grandit quand h rétrécit. C'est la
// même famille que le tourbillon parasite du ×SCALE3 (le confinement injecte de
// l'énergie sans borne), mais la source n'est plus le facteur : c'est le bord.
// MESURÉ à 384³, boule TRAÎNÉE puis lâchée, preset champignon : la boîte est
// envahie vers 6 s ; à ε = 0 elle ne l'est pas ; à 256³ elle ne l'est pas.
//
// On éteint donc le confinement sur une peau de trois cellules autour de la
// sphère — en fondu, jamais net : une coupure franche serait elle-même une
// discontinuité de force, donc une nouvelle source. Le fluide loin de la boule
// garde exactement le confinement d'avant.
const PEAU_SANS_CONFINEMENT = 3.0;

fn fondu_obstacle(p: vec3f) -> f32 {
  // C'ÉTAIT DÉJÀ une distance signée (distance − rayon) : la peau se
  // généralise sans rien changer d'autre. Absent → sd = 1e9 → fondu 1.
  return smoothstep(0.0, PEAU_SANS_CONFINEMENT, solid_sd(p));
}

// Force de confinement à la position p : ε (N̂ × ω), N̂ = ∇|ω|flou normalisé.
fn confine_force(p: vec3f) -> vec3f {
  let fondu = fondu_obstacle(p);
  if (fondu <= 0.0) {
    return vec3f(0.0);
  }
  let grad = 0.5 * vec3f(
    omega_len(p + vec3f(1.0, 0.0, 0.0)) - omega_len(p - vec3f(1.0, 0.0, 0.0)),
    omega_len(p + vec3f(0.0, 1.0, 0.0)) - omega_len(p - vec3f(0.0, 1.0, 0.0)),
    omega_len(p + vec3f(0.0, 0.0, 1.0)) - omega_len(p - vec3f(0.0, 0.0, 1.0)),
  );
  let nrm = grad / max(length(grad), 1e-5);
  return P.sphere_vel.w * fondu * cross(nrm, omega_at(p));
}

@compute @workgroup_size(4, 4, 4)
fn confine(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let dt = P.misc.x;
  let fc = vec3f(c);
  var vel = textureLoad(vel_src, c, 0).xyz;
  // UNE évaluation au centre de la cellule, appliquée aux trois faces — et non
  // trois évaluations aux positions MAC exactes (21 lectures par cellule). Le
  // champ est flouté sur ~3 voxels ET calculé en mi-résolution : à cette
  // échelle de lissage, un demi-voxel d'écart entre face et centre est du
  // bruit. Mesuré avant de trancher : images de référence indiscernables.
  let f = confine_force(fc + vec3f(0.5));
  vel += dt * f;

  vel.x = select(vel.x, P.sphere_vel.x, face_blocked(c, vec3i(1, 0, 0)));
  vel.y = select(vel.y, P.sphere_vel.y, face_blocked(c, vec3i(0, 1, 0)));
  vel.z = select(vel.z, P.sphere_vel.z, face_blocked(c, vec3i(0, 0, 1)));
  vel.x = select(vel.x, 0.0, c.x == 0);
  vel.y = select(vel.y, 0.0, c.y == 0);
  vel.z = select(vel.z, 0.0, c.z == 0);

  textureStore(vel_dst, gid, vec4f(vel, 0.0));
}
