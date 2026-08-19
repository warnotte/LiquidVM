// Forces externes sur grille MAC, appliquées à chaque composante À SA POSITION DE FACE :
// 1. Buoyancy — appliquée aux faces v (composante verticale), proportionnelle à la
//    densité échantillonnée à la position de la face (l'axe y descend : poussée en -y).
// 2. Impulsion souris — splat gaussien du déplacement du pointeur, évalué séparément
//    à la position de chaque face (u reçoit delta.x, v reçoit delta.y).
// Les faces bloquées (murs, bords fermés) restent à zéro.

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
@group(1) @binding(1) var src_den: texture_2d<f32>;
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

fn splat(p: vec2f) -> f32 {
  let off = p - P.pointer.xy;
  return exp(-dot(off, off) / (P.impulse.w * P.impulse.w));
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

  // Buoyancy sur la face v : densité échantillonnée à la position de la face
  // (grilles découplées : conversion physique → uv par la taille de grille sim).
  let den = textureSampleLevel(src_den, lin, pv * P.grid.zw, 0.0).xyz;
  vel.y -= dot(den, P.buoyancy.xyz) * P.impulse.x;

  // Impulsion souris, par composante à sa propre position. L'outil (misc.w) choisit
  // le champ de force : 0 = drag directionnel, 2 = tourbillon (tangentiel),
  // 3 = souffle (radial sortant). L'outil 1 (gomme) n'agit que sur la densité.
  if (P.impulse.y > 0.5) {
    let tool = u32(P.misc.w + 0.5);
    switch tool {
      case 0u: {
        vel.x += P.pointer.z * P.misc.y * splat(pu);
        vel.y += P.pointer.w * P.misc.y * splat(pv);
      }
      case 2u: {
        // Tourbillon : vélocité tangentielle autour du curseur, zone élargie et
        // montée rapide (le rotationnel survit intégralement à la projection).
        let r = P.impulse.w * 2.5;
        let ou = pu - P.pointer.xy;
        let ov = pv - P.pointer.xy;
        let k = P.misc.y * 30.0 * P.impulse.x;
        vel.x += (-ou.y / max(length(ou), 2.0)) * k * exp(-dot(ou, ou) / (r * r));
        vel.y += (ov.x / max(length(ov), 2.0)) * k * exp(-dot(ov, ov) / (r * r));
      }
      case 3u: {
        // Souffle : jet dans la direction du GESTE, zone élargie, sans densité.
        // (Une poussée radiale pure est un champ irrotationnel : la projection
        // incompressible l'annulerait exactement — l'outil semblerait mort.)
        let r = P.impulse.w * 2.5;
        let ou = pu - P.pointer.xy;
        let ov = pv - P.pointer.xy;
        vel.x += P.pointer.z * P.misc.y * 2.5 * exp(-dot(ou, ou) / (r * r));
        vel.y += P.pointer.w * P.misc.y * 2.5 * exp(-dot(ov, ov) / (r * r));
      }
      default: {}
    }
  }

  if (u_face_blocked(c, size)) { vel.x = 0.0; }
  if (v_face_blocked(c, size)) { vel.y = 0.0; }
  textureStore(dst_vel, c, vec4f(vel, 0.0, 0.0));
}
