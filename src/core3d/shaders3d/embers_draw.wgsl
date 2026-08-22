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
}

struct Ember {
  pos: vec4f, // xyz: voxels, w: âge (s)
  vel: vec4f, // xyz: voxels/s, w: durée de vie (s, 0 = jamais née)
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var<storage, read> embers: array<Ember>;

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

  // Occlusion : extinction cumulée sur 4 échantillons vers la caméra.
  let cam = R.cam_pos.xyz;
  let to_cam = cam - world;
  let dist = length(to_cam);
  let dir = to_cam / max(dist, 1e-4);
  let span = min(dist, 0.45);
  var tau = 0.0;
  for (var j = 1u; j <= 4u; j++) {
    let sp = world + dir * (f32(j) / 4.0) * span;
    tau += extinction(textureSampleLevel(density_tex, lin, sp + vec3f(0.5), 0.0));
  }
  var brightness = (1.0 - t) * (1.0 - t) * exp(-tau * span * 0.25);
  // Boule devant la braise : occultée.
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
