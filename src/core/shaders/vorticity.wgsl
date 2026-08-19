// Vorticity confinement (Fedkiw et al., 2001) sur grille MAC : la force
// F = ε·ω·(N.y, −N.x) avec N = ∇|ω| normalisé est évaluée SÉPARÉMENT à la position de
// chaque face (u reçoit F.x à (i, j+½), v reçoit F.y à (i+½, j)), en échantillonnant
// bilinéairement le rotationnel stocké aux nœuds. C'est une force ARTISTIQUE assumée
// (anti-diffusion) : ε = 0 la coupe entièrement — réglable à chaud dans le panneau.
// Les faces bloquées restent à zéro.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: fluide sélectionné, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s)
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s², positif = monte)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var lin: sampler;

@group(1) @binding(0) var src_vel: texture_2d<f32>;
@group(1) @binding(1) var src_curl: texture_2d<f32>;
@group(1) @binding(2) var dst_vel: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var obstacle: texture_2d<f32>;

// ω à la position physique p (les nœuds (i,j) sont aux centres des texels du curl).
fn curl_at(p: vec2f) -> f32 {
  return textureSampleLevel(src_curl, lin, (p + vec2f(0.5)) * P.grid.zw, 0.0).x;
}

// Force de confinement à la position p : ε·ω·(N.y, −N.x), N = ∇|ω| normalisé.
fn confine(p: vec2f) -> vec2f {
  let w = curl_at(p);
  let gx = abs(curl_at(p + vec2f(1.0, 0.0))) - abs(curl_at(p - vec2f(1.0, 0.0)));
  let gy = abs(curl_at(p + vec2f(0.0, 1.0))) - abs(curl_at(p - vec2f(0.0, 1.0)));
  let g = 0.5 * vec2f(gx, gy);
  let n = g / max(length(g), 1e-5);
  return P.extra.z * w * vec2f(n.y, -n.x);
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

fn u_face_blocked(c: vec2i, size: vec2i) -> bool {
  if (solid_at(c, size)) { return true; }
  if (c.x == 0) {
    if (P.extra.x > 0.5 && P.extra.x < 1.5) {
      return solid_at(vec2i(size.x - 1, c.y), size);
    }
    return true; // parois et ouvert : bord fermé (l'éponge de l'advection absorbe)
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

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let usize = vec2u(P.grid.xy);
  if (gid.x >= usize.x || gid.y >= usize.y) {
    return;
  }
  let c = vec2i(gid.xy);
  let size = vec2i(P.grid.xy);
  var vel = textureLoad(src_vel, c, 0).xy;

  let pu = vec2f(f32(c.x), f32(c.y) + 0.5);
  let pv = vec2f(f32(c.x) + 0.5, f32(c.y));
  vel.x += confine(pu).x * P.impulse.x;
  vel.y += confine(pv).y * P.impulse.x;

  if (u_face_blocked(c, size)) { vel.x = 0.0; }
  if (v_face_blocked(c, size)) { vel.y = 0.0; }
  textureStore(dst_vel, c, vec4f(vel, 0.0, 0.0));
}
