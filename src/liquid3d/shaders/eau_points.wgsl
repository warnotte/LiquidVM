// J1 — rendu des particules en POINTS additifs (la physique se regarde avant
// de se maquiller — le rendu de surface est le jalon J5). Même projection
// inverse-rayons que les braises du feu : pas de matrice, la base caméra des
// uniforms suffit. Couleur par vitesse : bleu profond au repos → cyan → blanc
// aux vitesses de chute. L'accumulation additive fait lire la densité.

struct UEau {
  sim: vec4f,      // x: N, y: nb particules
  sim2: vec4f,
  cam_pos: vec4f,  // xyz: caméra (monde), w: tan(fov/2)
  cam_right: vec4f,// xyz, w: aspect
  cam_up: vec4f,   // xyz, w: exposition des points
  cam_fwd: vec4f,  // xyz, w: taille des points (monde)
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
}

@vertex
fn vs_points(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0),
  );
  var out: VSOut;
  out.uv = quad[vi];
  out.color = vec3f(0.0);
  out.pos = vec4f(0.0, 0.0, 0.0, 1.0);

  let world = particles[2u * ii].xyz / U.sim.x - vec3f(0.5);
  let speed = length(particles[2u * ii + 1u].xyz);
  // Rampe : eau profonde (bleu) → agitée (cyan) → écumante (blanc cassé).
  let t = clamp(speed / 320.0, 0.0, 1.0);
  let deep = vec3f(0.05, 0.16, 0.42);
  let mid_c = vec3f(0.10, 0.55, 0.85);
  let foam = vec3f(0.85, 0.95, 1.0);
  var col = mix(deep, mid_c, clamp(t * 2.0, 0.0, 1.0));
  col = mix(col, foam, clamp(t * 2.0 - 1.0, 0.0, 1.0));

  let cam = U.cam_pos.xyz;
  let size = U.cam_fwd.w;
  let corner = world + (U.cam_right.xyz * out.uv.x + U.cam_up.xyz * out.uv.y) * size;
  let d = corner - cam;
  let df = dot(d, U.cam_fwd.xyz);
  if (df < 0.05) {
    return out;
  }
  let tanf = U.cam_pos.w;
  out.color = col * U.cam_up.w;
  out.pos = vec4f(
    dot(d, U.cam_right.xyz) / (tanf * U.cam_right.w),
    dot(d, U.cam_up.xyz) / tanf,
    0.5 * df,
    df,
  );
  return out;
}

@fragment
fn fs_points(frag: VSOut) -> @location(0) vec4f {
  let r2 = dot(frag.uv, frag.uv);
  let fall = max(1.0 - r2, 0.0);
  return vec4f(frag.color * fall, 1.0);
}
