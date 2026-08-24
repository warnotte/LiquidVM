// GIZMOS — couche de lignes 3D dessinée APRÈS la présentation, directement sur
// le canvas. C'est la base d'outillage du moteur : dès qu'on sait tracer des
// segments dans le monde, tout repère se dessine (champs de force, émetteurs,
// boîte, manipulateurs plus tard). Trois choix la rendent solide :
//
//  1. AUCUN tampon de géométrie. Toute la position vient de `vertex_index` et de
//     l'uniforme de rendu — même principe que les braises et les points d'eau,
//     et la doctrine « zéro allocation par frame » est tenue sans effort.
//  2. Après le tone-mapping, pas dedans. Les lignes gardent donc EXACTEMENT la
//     couleur demandée, sans être délavées par l'exposition ni bavées par le
//     bloom — c'est ce qui sépare un repère d'un halo.
//  3. Pas de tampon de profondeur : les gizmos passent devant le volume, comme
//     les repères de Blender. Un repère à moitié caché ne repère plus rien.
//
// Géométrie par champ : trois cercles orthogonaux (la « sphère vide » de
// Blender, qui donne le rayon d'action et se lit sous tous les angles) et un axe
// fléché qui donne l'ORIENTATION — axe de rotation pour un tourbillon, sens du
// courant pour un vent.

struct RenderParams {
  cam_pos: vec4f,   // xyz: caméra (monde), w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect
  cam_up: vec4f,
  cam_fwd: vec4f,
  light: vec4f,
  sphere: vec4f,
  blow_a: vec4f,
  blow_b: vec4f,
  style: vec4f,
  // Deux vec4 par champ : (xyz centre monde, w rayon monde — 0 = pas de champ)
  // et (xyz axe unitaire, w = type + 2 si le champ est actif).
  giz0_a: vec4f,
  giz0_b: vec4f,
  giz1_a: vec4f,
  giz1_b: vec4f,
  giz2_a: vec4f,
  giz2_b: vec4f,
}

@group(0) @binding(0) var<uniform> R: RenderParams;

const SEG: u32 = 64u;
const CIRCLE_VERTS: u32 = SEG * 2u;
const RING_VERTS: u32 = CIRCLE_VERTS * 3u;
const AXIS_VERTS: u32 = 6u;
const FIELD_VERTS: u32 = RING_VERTS + AXIS_VERTS;
/** Sommets à dessiner : 3 champs. */
const TOTAL_VERTS: u32 = FIELD_VERTS * 3u;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
}

fn giz_a(i: u32) -> vec4f {
  switch i {
    case 0u: { return R.giz0_a; }
    case 1u: { return R.giz1_a; }
    default: { return R.giz2_a; }
  }
}

fn giz_b(i: u32) -> vec4f {
  switch i {
    case 0u: { return R.giz0_b; }
    case 1u: { return R.giz1_b; }
    default: { return R.giz2_b; }
  }
}

// Projection : l'INVERSE de la construction des rayons du raymarch — ce moteur
// n'a pas de matrice (même convention que les braises).
fn project(p: vec3f) -> vec4f {
  let d = p - R.cam_pos.xyz;
  let df = dot(d, R.cam_fwd.xyz);
  if (df < 1e-3) {
    return vec4f(0.0, 0.0, 2.0, 1.0); // derrière la caméra : hors du volume de clip
  }
  let tanf = R.cam_pos.w;
  return vec4f(
    dot(d, R.cam_right.xyz) / (tanf * R.cam_right.w),
    dot(d, R.cam_up.xyz) / tanf,
    0.5 * df,
    df,
  );
}

@vertex
fn vs_gizmo(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let fi = vi / FIELD_VERTS;
  let li = vi % FIELD_VERTS;
  let a = giz_a(fi);
  let b = giz_b(fi);
  if (a.w <= 0.0) {
    // Champ absent : sommet dégénéré, aucun pixel rasterisé.
    out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
    out.color = vec4f(0.0);
    return out;
  }

  var p: vec3f;
  if (li < RING_VERTS) {
    // Trois cercles orthogonaux : le rayon d'action, lisible sous tout angle.
    let ring = li / CIRCLE_VERTS;
    let k = li % CIRCLE_VERTS;
    let ang = (f32(k / 2u) + f32(k % 2u)) * 6.2831853 / f32(SEG);
    let c = cos(ang) * a.w;
    let s = sin(ang) * a.w;
    if (ring == 0u) {
      p = a.xyz + vec3f(c, s, 0.0);
    } else if (ring == 1u) {
      p = a.xyz + vec3f(0.0, c, s);
    } else {
      p = a.xyz + vec3f(c, 0.0, s);
    }
  } else {
    // Axe fléché : orientation du champ (axe de rotation ou sens du courant).
    let axis = normalize(b.xyz);
    let tip = a.xyz + axis * (a.w * 1.3);
    let tail = a.xyz - axis * (a.w * 1.3);
    // « ref » et « active » sont des mots RÉSERVÉS WGSL (famille from / target /
    // move / smooth) : d'où up_ref et is_active.
    var up_ref = vec3f(0.0, 0.0, 1.0);
    if (abs(axis.z) > 0.9) {
      up_ref = vec3f(1.0, 0.0, 0.0);
    }
    let perp = normalize(cross(axis, up_ref)) * (a.w * 0.18);
    let back = tip - axis * (a.w * 0.32);
    switch (li - RING_VERTS) {
      case 0u: { p = tail; }
      case 1u: { p = tip; }
      case 2u: { p = tip; }
      case 3u: { p = back + perp; }
      case 4u: { p = tip; }
      default: { p = back - perp; }
    }
  }

  // Bleu froid pour un tourbillon, ambre pour un vent ; le champ ACTIF passe en
  // blanc chaud et opaque — la sélection se voit sans avoir à la deviner.
  let kind = b.w - select(0.0, 2.0, b.w >= 2.0);
  let is_active = b.w >= 2.0;
  var col = select(vec3f(0.42, 0.72, 1.0), vec3f(1.0, 0.70, 0.28), kind > 0.5);
  var alpha = 0.55;
  if (is_active) {
    col = mix(col, vec3f(1.0), 0.55);
    alpha = 0.95;
  }
  out.pos = project(p);
  out.color = vec4f(col, alpha);
  return out;
}

@fragment
fn fs_gizmo(in: VSOut) -> @location(0) vec4f {
  return in.color;
}
