// Tracé des ÉTINCELLES de feu d'artifice : billboards additifs face caméra dans
// la passe HDR (après le raymarch, avant le bloom — les étoiles héritent du
// halo), même machinerie que les braises À UNE DIFFÉRENCE PRÈS, qui est toute
// la raison d'être du système : la couleur est PRESCRITE par particule
// (tint.rgb), aucun corps noir. Projection sans matrice (inverse des rayons du
// raymarch), occlusion précalculée par particule (kernel `occlude` ci-dessous),
// morte = quad dégénéré (zéro pixel).

struct RenderParams {
  cam_pos: vec4f,   // xyz: caméra (monde), w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect
  cam_up: vec4f,    // xyz, w: exposition
  cam_fwd: vec4f,   // xyz, w: pas de marche
  light: vec4f,
  sphere: vec4f,    // xyz: centre (monde), w: rayon (monde, ≤0 = absente)
  blow_a: vec4f,
  blow_b: vec4f,
  style: vec4f,     // w: N (voxels → monde)
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
}

/** Coordonnée de texture d'un point monde — y normalisé par la hauteur du
 *  domaine, qui n'est plus forcément 1 (voir raymarch.wgsl). */
fn tex_uvw(p: vec3f) -> vec3f {
  return vec3f(p.x + 0.5, (p.y + 0.5) / max(R.soot.w, 1e-3), p.z + 0.5);
}

struct Spark {
  pos: vec4f,  // xyz: voxels, w: âge (s)
  vel: vec4f,  // xyz: voxels/s, w: durée de vie (s, 0 = jamais née)
  tint: vec4f, // xyz: couleur RGB, w: époque de naissance
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> sparks: array<Spark>;
// Occlusion PRÉCALCULÉE par particule (passe compute `occlude` ci-dessous).
@group(0) @binding(4) var<storage, read> occ_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> occ_out: array<f32>;

// Dupliqués (les shaders sont autonomes).
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
fn vs_sparks(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  var out: VSOut;
  out.uv = quad[vi];
  out.color = vec3f(0.0);
  // Sortie dégénérée par défaut (étoile morte / derrière la caméra).
  out.pos = vec4f(0.0, 0.0, 0.0, 1.0);

  let e = sparks[ii];
  let life = e.vel.w;
  let age = e.pos.w;
  if (life <= 0.0 || age >= life) {
    return out;
  }
  let t = age / life;
  let world = e.pos.xyz / R.style.w - vec3f(0.5);

  let cam = R.cam_pos.xyz;
  let to_cam = cam - world;
  let dist = length(to_cam);
  let dir = to_cam / max(dist, 1e-4);
  var brightness = (1.0 - t) * (1.0 - t) * occ_in[ii];
  // Boule devant l'étoile : occultée.
  if (R.sphere.w > 0.0) {
    let oc = world - R.sphere.xyz;
    let b = dot(oc, dir);
    let d2 = dot(oc, oc) - b * b;
    if (d2 < R.sphere.w * R.sphere.w && b > 0.0 && b < dist) {
      brightness = 0.0;
    }
  }
  if (brightness < 0.003) {
    return out;
  }

  // COULEUR PRESCRITE — c'est ici que les étincelles divergent des braises :
  // la teinte vient de la particule, pas d'un corps noir. Une pointe de blanc
  // à la naissance (le flash), la teinte pure ensuite.
  let h1 = rand01(ii * 2654435761u);
  let flash = max(1.0 - t * 6.0, 0.0);
  let tint = mix(e.tint.xyz, vec3f(1.0), flash * 0.6);
  out.color = tint * brightness * (2.0 + 1.2 * h1);

  // Billboard : coin décalé dans la base caméra, puis projection inverse rayons.
  let size = (0.0030 + 0.0030 * h1) * (1.0 - 0.35 * t);
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
fn fs_sparks(frag: VSOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let fall = max(1.0 - r2, 0.0);
  return vec4f(frag.color * fall * fall, 1.0);
}

// Occlusion par PARTICULE, en compute — une fois par étoile et par frame (même
// économie que les braises : 6× moins de lectures trilinéaires qu'au vertex).
// Tourne aussi en pause : l'orbite de caméra change la ligne de visée.
@compute @workgroup_size(64)
fn occlude(@builtin(global_invocation_id) gid: vec3u) {
  let ii = gid.x;
  if (ii >= arrayLength(&occ_out)) {
    return;
  }
  let e = sparks[ii];
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
