// Tracé des BRAISES : billboards additifs face caméra dans la passe HDR (après
// le raymarch, avant le bloom — les braises héritent du halo). Pas de matrice
// de projection dans ce moteur : la projection INVERSE la construction des
// rayons du raymarch (clip = [dot·right/(tanf·aspect), dot·up/tanf, ·, dot·fwd]).
// Occlusion approchée par particule (au vertex, pas au pixel) : 4 échantillons
// d'extinction vers la caméra + test du segment contre la boule. Une braise
// morte émet un quad de taille nulle (zéro pixel rasterisé).

struct RenderParams {
  cam_pos: vec4f,   // xyz: caméra (monde), w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect
  cam_up: vec4f,    // xyz, w: exposition
  cam_fwd: vec4f,   // xyz, w: pas de marche
  light: vec4f,
  sphere: vec4f,    // xyz: centre (monde), w: rayon (monde, ≤0 = absente)
  blow_a: vec4f,
  blow_b: vec4f,
  style: vec4f,     // z: débit des braises, w: N (voxels → monde)
  // Traversée jusqu'à la hauteur du domaine (la disposition de l'uniforme est
  // commune ; voir raymarch.wgsl).
  giz0_a: vec4f, giz0_b: vec4f,
  giz1_a: vec4f, giz1_b: vec4f,
  giz2_a: vec4f, giz2_b: vec4f,
  emit0: vec4f, emit1: vec4f, emit2: vec4f, emit3: vec4f,
  opts: vec4f,
  sel: vec4f,
  aim: vec4f,
  soot: vec4f,      // w: hauteur monde du domaine
  shape: vec4f,     // OBSTACLE : x type, yzw paramètres (#23)
}

/** Coordonnée de texture d'un point monde — y normalisé par la hauteur du
 *  domaine, qui n'est plus forcément 1 (voir raymarch.wgsl). */
fn tex_uvw(p: vec3f) -> vec3f {
  return vec3f(p.x + 0.5, (p.y + 0.5) / max(R.soot.w, 1e-3), p.z + 0.5);
}

struct Ember {
  pos: vec4f, // xyz: voxels, w: âge (s)
  vel: vec4f, // xyz: voxels/s, w: durée de vie (s, 0 = jamais née)
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> embers: array<Ember>;
// Occlusion PRÉCALCULÉE par particule (passe compute `occlude` ci-dessous) :
// le facteur final exp(−τ·portée/4), lu tel quel par le vertex.
@group(0) @binding(4) var<storage, read> occ_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> occ_out: array<f32>;

// OBSTACLE — DISTANCE SIGNÉE en unités MONDE, négative dedans (PLAN-OBSTACLES).
// Même fonction que côté simulation, mais la vue travaille en monde : `R.sphere`
// y est déjà converti, et les paramètres de forme sont des RATIOS, sans
// dimension — ils traversent la conversion inchangés.
fn solid_sd(p: vec3f) -> f32 {
  let r = R.sphere.w;
  if (r <= 0.0) {
    return 1e9;
  }
  let q = p - R.sphere.xyz;
  let kind = i32(R.shape.x + 0.5);
  if (kind == 1) {
    let d = abs(q) - r * R.shape.yzw;
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  if (kind == 2) {
    return length(vec2f(length(q.xz) - r, q.y)) - r * R.shape.y;
  }
  if (kind == 3) { // CLOCHE : coquille sphérique, OUVERTE PAR LE SOL —
    // pas de plan de coupe : c'est le plancher de la boîte (déjà un mur de
    // non-pénétration) qui ferme le bas. Le gaz est enfermé, la lumière
    // non : le rendu la traite en VERRE (voir raymarch).
    let rr = r * R.shape.z;
    return abs(length(q) - rr) - rr * R.shape.y;
  }
  return length(q) - r;
}

// Rayon de la sphère ENGLOBANTE (monde) — il borne toute marche.
fn solid_bound() -> f32 {
  let kind = i32(R.shape.x + 0.5);
  if (kind == 1) {
    return R.sphere.w * (length(R.shape.yzw) + 0.02);
  }
  if (kind == 2) {
    return R.sphere.w * (1.0 + R.shape.y + 0.02);
  }
  if (kind == 3) {
    return R.sphere.w * R.shape.z * (1.0 + R.shape.y) + 0.02;
  }
  return R.sphere.w * 1.02;
}

// Intersection par SPHERE TRACING, BORNÉE à l'englobante : hors d'elle on ne
// marche pas du tout, ce qui garde le coût quasi nul pour l'immense majorité
// des rayons — c'était l'objection perf du passage de l'analytique à la marche.
// La SPHÈRE reste résolue exactement, en une évaluation : c'est le cas par
// défaut, il ne doit rien payer à la généralisation.
fn solid_hit_max(ro: vec3f, rd: vec3f, t_max: f32, steps: i32) -> f32 {
  if (R.sphere.w <= 0.0) {
    return 1e9;
  }
  let oc = ro - R.sphere.xyz;
  let rb = solid_bound();
  let b = dot(oc, rd);
  let disc = b * b - (dot(oc, oc) - rb * rb);
  if (disc < 0.0) {
    return 1e9;
  }
  let sq = sqrt(disc);
  if (i32(R.shape.x + 0.5) == 0) {
    let td = -b - sq;
    return select(1e9, td, td > 0.0 && td < t_max);
  }
  var t = max(-b - sq, 0.0);
  let t_end = min(-b + sq, t_max);
  for (var i = 0; i < steps; i++) {
    if (t >= t_end) {
      break;
    }
    let d = solid_sd(ro + rd * t);
    if (d < 0.0015) {
      return t;
    }
    t += max(d, 0.003);
  }
  return 1e9;
}


// Dupliqués (les shaders sont autonomes).
fn blackbody(heat: f32) -> vec3f {
  let h = clamp(heat, 0.0, 1.7);
  return vec3f(h * 1.6, h * h * 0.9, h * h * h * 0.42);
}
fn extinction(s: vec4f) -> f32 {
  return (s.x + s.y + s.z) * 22.0 + s.w * 5.0;
}
fn rand01(v: u32) -> f32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return f32((word >> 22u) ^ word) * (1.0 / 4294967296.0);
}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
}

@vertex
fn vs_embers(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  var out: VSOut;
  out.uv = quad[vi];
  out.color = vec3f(0.0);
  // Sortie dégénérée par défaut (braise morte / derrière la caméra).
  out.pos = vec4f(0.0, 0.0, 0.0, 1.0);

  let e = embers[ii];
  let life = e.vel.w;
  let age = e.pos.w;
  if (life <= 0.0 || age >= life) {
    return out;
  }
  let t = age / life;
  let world = e.pos.xyz / R.style.w - vec3f(0.5);

  // Occlusion : précalculée PAR PARTICULE (passe `occlude`) — la recalculer ici
  // coûtait six fois par braise (une par sommet), ~4 ms à 384³.
  let cam = R.cam_pos.xyz;
  let to_cam = cam - world;
  let dist = length(to_cam);
  let dir = to_cam / max(dist, 1e-4);
  var brightness = (1.0 - t) * (1.0 - t) * occ_in[ii];
  // Boule devant la braise : occultée.
  // OBSTACLE devant la particule : marche courte (16 pas suffisent — c'est une
  // occultation binaire, pas une image), bornée à l'englobante comme partout.
  // (le VERRE n'occulte pas : on voit les braises à travers la cloche)
  if (i32(R.shape.x + 0.5) != 3 && solid_hit_max(world, dir, dist, 16) < 1e8) {
    brightness = 0.0;
  }
  if (brightness < 0.003) {
    return out;
  }

  // Couleur : corps noir propre qui refroidit, variance par particule
  // (1.05–1.45 : doré → orange, jamais blanc crème).
  let h1 = rand01(ii * 2654435761u);
  let heat_e = (1.05 + 0.40 * h1) * (1.0 - 0.55 * t);
  out.color = blackbody(heat_e) * brightness * 2.2;

  // Billboard : coin décalé dans la base caméra, puis projection inverse rayons.
  let size = (0.0028 + 0.0028 * h1) * (1.0 - 0.4 * t);
  let corner = world + (R.cam_right.xyz * out.uv.x + R.cam_up.xyz * out.uv.y) * size;
  let d = corner - cam;
  let df = dot(d, R.cam_fwd.xyz);
  if (df < 0.05) {
    out.color = vec3f(0.0);
    out.pos = vec4f(0.0, 0.0, 0.0, 1.0);
    return out;
  }
  let tanf = R.cam_pos.w;
  out.pos = vec4f(
    dot(d, R.cam_right.xyz) / (tanf * R.cam_right.w),
    dot(d, R.cam_up.xyz) / tanf,
    0.5 * df,
    df,
  );
  return out;
}

@fragment
fn fs_embers(frag: VSOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let fall = max(1.0 - r2, 0.0);
  return vec4f(frag.color * fall * fall, 1.0);
}

// Occlusion par PARTICULE, en compute — une fois par braise et par frame, au
// lieu de six fois (une par sommet du billboard) : 4 échantillons × 6 sommets ×
// 32k particules ≈ 786k lectures trilinéaires par frame, ~4 ms à 384³, ramenés
// à 131k. Tourne aussi en pause : l'orbite de caméra doit garder l'occlusion
// juste même quand les braises sont figées.
@compute @workgroup_size(64)
fn occlude(@builtin(global_invocation_id) gid: vec3u) {
  let ii = gid.x;
  if (ii >= arrayLength(&occ_out)) {
    return;
  }
  let e = embers[ii];
  // Morte ou jamais née : le vertex l'écarte de toute façon.
  if (e.vel.w <= 0.0 || e.pos.w >= e.vel.w) {
    occ_out[ii] = 0.0;
    return;
  }
  let world = e.pos.xyz / R.style.w - vec3f(0.5);
  let to_cam = R.cam_pos.xyz - world;
  let dist = length(to_cam);
  let dir = to_cam / max(dist, 1e-4);
  let span = min(dist, 0.45);
  var tau = 0.0;
  for (var j = 1u; j <= 4u; j++) {
    let sp = world + dir * (f32(j) / 4.0) * span;
    tau += extinction(textureSampleLevel(density_tex, lin, tex_uvw(sp), 0.0));
  }
  occ_out[ii] = exp(-tau * span * 0.25);
}
