// Application du flux optique au champ de vélocité MAC : chaque composante de face
// échantillonne le champ de flux (bilinéaire — c'est lui qui lisse le bruit du flux
// normal) à sa propre position et reçoit une accélération proportionnelle.
// Encodée entre les forces et la vorticité, uniquement quand la caméra est active —
// la projection en aval garde le tout incompressible.

// Doit rester identique à SimUniformWriter (core/uniforms.ts).
struct SimParams {
  grid: vec4f,        // xy: taille de grille vélocité (texels), zw: 1/taille
  pointer: vec4f,     // xy: position pointeur (texels sim), zw: delta du sous-pas (texels)
  impulse: vec4f,     // x: dt (s), y: bouton (0|1), z: matière sélectionnée, w: rayon splat (texels)
  misc: vec4f,        // x: dissipation vélocité (1/s), y: force splat, z: débit densité (1/s), w: outil
  dissipation: vec4f, // xyz: dissipation de densité par fluide (1/s), w: refroidissement du feu (1/s)
  buoyancy: vec4f,    // xyz: poussée par fluide (texels/s²), w: temps simulé (s)
  extra: vec4f,       // x: frontières (0 parois, 1 périodique, 2 ouvert), y: pinceau mur, z: vorticité, w: MacCormack
  dye: vec4f,         // xy: taille de la grille de densités, zw: 1/taille
}

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var lin: sampler;

@group(1) @binding(0) var flow_tex: texture_2d<f32>;
@group(1) @binding(1) var src_vel: texture_2d<f32>;
@group(1) @binding(2) var dst_vel: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var obstacle: texture_2d<f32>;

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
    return true;
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

fn flow_at(p: vec2f) -> vec2f {
  return textureSampleLevel(flow_tex, lin, p * P.grid.zw, 0.0).xy;
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

  // Accélération ∝ flux, mise à l'échelle de la grille comme les autres forces absolues.
  let strength = 420.0 * (P.grid.x / 512.0);
  let pu = vec2f(f32(c.x), f32(c.y) + 0.5);
  let pv = vec2f(f32(c.x) + 0.5, f32(c.y));
  vel.x += flow_at(pu).x * strength * P.impulse.x;
  vel.y += flow_at(pv).y * strength * P.impulse.x;

  if (u_face_blocked(c, size)) { vel.x = 0.0; }
  if (v_face_blocked(c, size)) { vel.y = 0.0; }
  textureStore(dst_vel, c, vec4f(vel, 0.0, 0.0));
}
