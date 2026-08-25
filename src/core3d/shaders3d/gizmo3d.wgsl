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
// Trois familles de repères, dans cet ordre le long de `vertex_index` :
//
//  CHAMPS DE FORCE — trois cercles orthogonaux (la « sphère vide » de Blender,
//  qui donne le rayon d'action et se lit sous tous les angles) et un axe fléché
//  qui donne l'ORIENTATION : axe de rotation pour un tourbillon, sens du
//  courant pour un vent.
//
//  ÉMETTEURS — un anneau posé à plat au rayon d'émission (la surface qui
//  souffle) et une tige fléchée vers le haut (le sens du panache), à la couleur
//  de l'encre émise. Sans lui, un émetteur posé hors de la flamme est INVISIBLE
//  tant qu'il n'a pas allumé quelque chose ; et rien ne disait lequel des
//  quatre les touches 1/2/3 allaient repeindre.
//
//  BOÎTE — les 12 arêtes du cube unité, très pâles. Les parois sont le seul
//  élément de la scène qu'on ne voit qu'à ses effets (le panache qui s'écrase
//  au plafond, la fumée qui revient) ; les tracer situe tout le reste.
//
// Les têtes de flèche sont BILLBOARDÉES (perpendiculaire prise dans le plan de
// l'écran) : une flèche vue dans l'axe de son propre plan se réduisait à un
// trait, précisément quand on l'orientait vers la caméra.

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
  // Un vec4 par émetteur : (xyz centre monde, w = −1 aucun, sinon numéro
  // d'encre + 4 si l'émetteur est actif).
  emit0: vec4f,
  emit1: vec4f,
  emit2: vec4f,
  emit3: vec4f,
  // x: rayon d'émission (monde) · y: boîte visible (0/1) · z: axe en cours de
  // traînée (−1 aucun, 0/1/2) · w: départ des flèches de poignée, en fraction de
  // leur longueur (le centre reste libre pour le déplacement non contraint).
  opts: vec4f,
  // Poignées de l'objet sélectionné : xyz centre monde, w longueur monde
  // (0 = rien de sélectionné). La longueur vient du CPU pour que le dessin et
  // la saisie partagent exactement le même nombre.
  sel: vec4f,
  // Bouton d'orientation : xyz axe unitaire de l'objet sélectionné, w distance
  // monde du bouton (0 = l'objet n'a pas d'orientation). Un seul nombre sert de
  // drapeau ET de position.
  aim: vec4f,
  soot: vec4f,      // w: HAUTEUR monde du domaine (1 = cube)
}

@group(0) @binding(0) var<uniform> R: RenderParams;

const SEG: u32 = 64u;
const CIRCLE_VERTS: u32 = SEG * 2u;
const RING_VERTS: u32 = CIRCLE_VERTS * 3u;
const ARROW_VERTS: u32 = 6u;
const FIELD_VERTS: u32 = RING_VERTS + ARROW_VERTS;
const FIELDS_TOTAL: u32 = FIELD_VERTS * 3u;

const ESEG: u32 = 32u;
const EMIT_RING: u32 = ESEG * 2u;
const EMIT_VERTS: u32 = EMIT_RING + ARROW_VERTS;
const EMITS_TOTAL: u32 = EMIT_VERTS * 4u;

const BOX_VERTS: u32 = 24u;
const HANDLE_VERTS: u32 = ARROW_VERTS * 3u;

const ASEG: u32 = 24u;
const AIM_RING: u32 = ASEG * 2u;
const AIM_VERTS: u32 = AIM_RING + 2u;

/** Sommets à dessiner : 3 champs + 4 émetteurs + la boîte + les 3 poignées +
 *  le bouton d'orientation. */
const TOTAL_VERTS: u32 =
  FIELDS_TOTAL + EMITS_TOTAL + BOX_VERTS + HANDLE_VERTS + AIM_VERTS;

// Palette des trois matières (canaux xyz) — dupliquée depuis config3d.ts, comme
// dans raymarch.wgsl : fumée grise, encre magenta, vapeur de carburant ambrée.
const INK0 = vec3f(0.55, 0.60, 0.68);
const INK1 = vec3f(1.00, 0.30, 0.80);
const INK2 = vec3f(1.00, 0.80, 0.45);

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

fn emit_slot(i: u32) -> vec4f {
  switch i {
    case 0u: { return R.emit0; }
    case 1u: { return R.emit1; }
    case 2u: { return R.emit2; }
    default: { return R.emit3; }
  }
}

fn ink_color(i: u32) -> vec3f {
  switch i {
    case 0u: { return INK0; }
    case 1u: { return INK1; }
    default: { return INK2; }
  }
}

/** Sommet dégénéré, hors du volume de clip : aucun pixel rasterisé. */
fn hidden() -> VSOut {
  var out: VSOut;
  out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  out.color = vec4f(0.0);
  return out;
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

/** Perpendiculaire à `axis` prise dans le PLAN DE L'ÉCRAN : la tête de flèche
 *  garde sa largeur sous tout angle. Repli sur la verticale caméra quand l'axe
 *  pointe vers l'objectif — là, la flèche est de toute façon un point. */
fn billboard_perp(axis: vec3f) -> vec3f {
  var cp = cross(axis, R.cam_fwd.xyz);
  if (length(cp) < 1e-3) {
    cp = cross(axis, R.cam_up.xyz);
  }
  if (length(cp) < 1e-3) {
    return vec3f(1.0, 0.0, 0.0);
  }
  return normalize(cp);
}

/** Un des 6 sommets de la flèche : tige `origin`→`origin + axis*len`, plus deux
 *  barbes ramenées vers l'arrière. `k` ∈ [0,6). */
fn arrow_point(k: u32, origin: vec3f, axis: vec3f, len: f32) -> vec3f {
  let tip = origin + axis * len;
  let perp = billboard_perp(axis) * (len * 0.07);
  let back = tip - axis * (len * 0.125);
  switch k {
    case 0u: { return origin; }
    case 1u: { return tip; }
    case 2u: { return tip; }
    case 3u: { return back + perp; }
    case 4u: { return tip; }
    default: { return back - perp; }
  }
}

/** Un des 12 côtés du cube unité : `k` ∈ [0,24), deux sommets par arête. */
fn box_point(k: u32) -> vec3f {
  let e = k / 2u;
  let t = f32(k % 2u);
  let axis_i = e / 4u;
  let c = e % 4u;
  let b0 = f32(c & 1u);
  let b1 = f32((c >> 1u) & 1u);
  var p = vec3f(b0, b1, t);
  if (axis_i == 0u) {
    p = vec3f(t, b0, b1);
  } else if (axis_i == 1u) {
    p = vec3f(b0, t, b1);
  }
  // Le domaine n'est plus forcément cubique : le sol reste à −0,5 et la boîte
  // pousse vers le HAUT. Un cube (hauteur 1) redonne exactement l'ancien tracé.
  return vec3f(p.x - 0.5, p.y * max(R.soot.w, 1e-3) - 0.5, p.z - 0.5);
}

@vertex
fn vs_gizmo(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;

  // ---- CHAMPS DE FORCE ----
  if (vi < FIELDS_TOTAL) {
    let fi = vi / FIELD_VERTS;
    let li = vi % FIELD_VERTS;
    let a = giz_a(fi);
    let b = giz_b(fi);
    if (a.w <= 0.0) {
      return hidden();
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
      // Axe fléché : orientation du champ (axe de rotation ou sens du courant),
      // tracé de part et d'autre du centre.
      let axis = normalize(b.xyz);
      p = arrow_point(li - RING_VERTS, a.xyz - axis * (a.w * 1.3), axis, a.w * 2.6);
    }

    // Bleu froid pour un tourbillon, ambre pour un vent ; le champ ACTIF passe
    // en blanc chaud et opaque — la sélection se voit sans avoir à la deviner.
    // « ref » et « active » sont des mots RÉSERVÉS WGSL (famille from / target /
    // move / smooth) : d'où is_active.
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

  // ---- ÉMETTEURS ----
  if (vi < FIELDS_TOTAL + EMITS_TOTAL) {
    let vk = vi - FIELDS_TOTAL;
    let ei = vk / EMIT_VERTS;
    let li = vk % EMIT_VERTS;
    let e = emit_slot(ei);
    let r = R.opts.x;
    if (e.w < 0.0 || r <= 0.0) {
      return hidden();
    }

    var p: vec3f;
    if (li < EMIT_RING) {
      // Anneau posé à plat : la surface qui souffle, à son vrai rayon.
      let ang = (f32(li / 2u) + f32(li % 2u)) * 6.2831853 / f32(ESEG);
      p = e.xyz + vec3f(cos(ang) * r, 0.0, sin(ang) * r);
    } else {
      // Tige fléchée vers le haut : le sens du panache.
      p = arrow_point(li - EMIT_RING, e.xyz, vec3f(0.0, 1.0, 0.0), r * 2.6);
    }

    let is_active = e.w >= 4.0;
    let ink = u32(e.w - select(0.0, 4.0, is_active));
    var col = ink_color(ink);
    var alpha = 0.5;
    if (is_active) {
      col = mix(col, vec3f(1.0), 0.5);
      alpha = 0.9;
    }
    out.pos = project(p);
    out.color = vec4f(col, alpha);
    return out;
  }

  // ---- BOÎTE ----
  if (vi < FIELDS_TOTAL + EMITS_TOTAL + BOX_VERTS) {
    if (R.opts.y < 0.5) {
      return hidden();
    }
    out.pos = project(box_point(vi - FIELDS_TOTAL - EMITS_TOTAL));
    out.color = vec4f(0.62, 0.68, 0.78, 0.22);
    return out;
  }

  // ---- BOUTON D'ORIENTATION ----
  // Un anneau FACE CAMÉRA au bout de l'axe, relié par une tige : le point qu'on
  // tire pour viser. Il est posé juste au-delà des flèches de déplacement (sur
  // le même axe, mais hors de leur portée) — le tourbillon a l'axe vertical par
  // défaut, son bouton tomberait sinon pile sur la pointe de la poignée Y.
  if (vi >= FIELDS_TOTAL + EMITS_TOTAL + BOX_VERTS + HANDLE_VERTS) {
    if (R.aim.w <= 0.0 || R.sel.w <= 0.0) {
      return hidden();
    }
    let k = vi - (FIELDS_TOTAL + EMITS_TOTAL + BOX_VERTS + HANDLE_VERTS);
    let knob = R.sel.xyz + normalize(R.aim.xyz) * R.aim.w;
    var p: vec3f;
    if (k < AIM_RING) {
      let ang = (f32(k / 2u) + f32(k % 2u)) * 6.2831853 / f32(ASEG);
      let rad = R.sel.w * 0.13;
      p = knob + R.cam_right.xyz * (cos(ang) * rad) + R.cam_up.xyz * (sin(ang) * rad);
    } else {
      // Tige : du bout des flèches de déplacement jusqu'au bouton — elle dit
      // « ce bouton appartient à cet axe » sans ajouter une deuxième ligne.
      p = R.sel.xyz + normalize(R.aim.xyz) * select(R.sel.w, R.aim.w, k % 2u == 1u);
    }
    var col = vec3f(0.74, 0.62, 1.0);
    var alpha = 0.8;
    if (R.opts.z >= 2.5) {
      col = mix(col, vec3f(1.0), 0.55);
      alpha = 1.0;
    }
    out.pos = project(p);
    out.color = vec4f(col, alpha);
    return out;
  }

  // ---- POIGNÉES DE MANIPULATION ----
  // Trois flèches aux couleurs d'axe (X rouge, Y vert, Z bleu, la convention
  // que tout le monde connaît), portées par l'objet sélectionné. Attraper une
  // flèche contraint le déplacement à SON axe : la traînée libre sur le plan
  // face caméra reste au corps de l'objet.
  if (R.sel.w <= 0.0) {
    return hidden();
  }
  let hk = vi - FIELDS_TOTAL - EMITS_TOTAL - BOX_VERTS;
  let ai = hk / ARROW_VERTS;
  var axis = vec3f(1.0, 0.0, 0.0);
  var col = vec3f(0.95, 0.36, 0.36);
  if (ai == 1u) {
    axis = vec3f(0.0, 1.0, 0.0);
    col = vec3f(0.48, 0.90, 0.42);
  } else if (ai == 2u) {
    axis = vec3f(0.0, 0.0, 1.0);
    col = vec3f(0.38, 0.60, 1.00);
  }
  var alpha = 0.8;
  if (u32(max(R.opts.z, 0.0)) == ai && R.opts.z >= 0.0) {
    col = mix(col, vec3f(1.0), 0.5);
    alpha = 1.0;
  }
  let inner = R.sel.w * R.opts.w;
  out.pos = project(arrow_point(hk % ARROW_VERTS, R.sel.xyz + axis * inner, axis, R.sel.w - inner));
  out.color = vec4f(col, alpha);
  return out;
}

@fragment
fn fs_gizmo(in: VSOut) -> @location(0) vec4f {
  return in.color;
}
