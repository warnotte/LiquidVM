// Rendu des particules : un quad étiré le long de la vitesse par instance (traînée de
// mouvement), en blending ADDITIF dans la scène HDR — le bloom les attrape donc comme
// les hautes lumières du fluide. Couleur : teinte du fluide sous la particule (densités
// échantillonnées au vertex) + une pointe de blanc, luminosité ∝ vitesse locale.
// Contribution individuelle très faible : c'est l'accumulation dans les filaments du
// champ qui dessine — 1M de poussières traçant l'écoulement.

struct RenderParams {
  color0: vec4f, // rgb: couleur fluide 0, w: intensité des particules (réglage panneau)
  color1: vec4f, // rgb: couleur fluide 1
  color2: vec4f, // rgb: couleur fluide 2
  tone: vec4f,   // x: exposition, y: vue, z: échelle de vitesse (normalisation), w: bloom
}

struct Particle {
  posvel: vec4f, // xy: position normalisée, zw: vitesse normalisée/s
  misc: vec4f,   // x: âge (s), y: durée de vie (s)
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) local: vec2f, // coordonnées locales du quad ∈ [-1,1]², pour le dégradé
  @location(1) color: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let corner = corners[vi];
  let p = particles[ii];

  var out: VSOut;
  out.local = corner;

  // Fondu d'apparition/disparition pour éviter le clignotement des respawns.
  let fade = clamp(p.misc.x * 2.0, 0.0, 1.0) *
    clamp((p.misc.y - p.misc.x) * 2.0, 0.0, 1.0);
  if (p.misc.y <= 0.0 || fade <= 0.0) {
    out.pos = vec4f(2.0, 2.0, 0.0, 1.0); // quad dégénéré hors clip
    out.color = vec3f(0.0);
    return out;
  }

  let vel = p.posvel.zw;
  let speed = length(vel);
  // Axe de la traînée : direction de la vitesse (axe x arbitraire si immobile).
  let dir = select(vec2f(1.0, 0.0), vel / max(speed, 1e-6), speed > 1e-6);
  let half_len = clamp(speed * 0.02, 0.0012, 0.012); // en unités normalisées
  let half_width = 0.0006;
  let offset = dir * (corner.x * half_len) + vec2f(-dir.y, dir.x) * (corner.y * half_width);
  let posn = p.posvel.xy + offset;
  out.pos = vec4f(posn.x * 2.0 - 1.0, 1.0 - posn.y * 2.0, 0.0, 1.0);

  // Teinte du fluide local + pointe de blanc ; luminosité ∝ vitesse (normalisée par
  // l'échelle de la vue debug vélocité, déjà proportionnelle à la grille). Dans les
  // zones chaudes (température .a), les particules deviennent des ÉTINCELLES : la
  // rampe de corps noir domine leur couleur et le bloom fait le reste.
  let den = textureSampleLevel(density_tex, lin, p.posvel.xy, 0.0);
  let tint = R.color0.rgb * den.x + R.color1.rgb * den.y + R.color2.rgb * den.z;
  let heat = clamp(den.w, 0.0, 1.7);
  let spark = vec3f(heat * 1.6, heat * heat * 0.9, heat * heat * heat * 0.42);
  let energy = clamp(speed / (R.tone.z * 0.002), 0.08, 1.0);
  out.color = (tint * 0.7 + spark * 2.5 + vec3f(0.30, 0.32, 0.36)) *
    (energy * fade * 0.14 * R.color0.w);
  return out;
}

@fragment
fn fs_main(v: VSOut) -> @location(0) vec4f {
  // Dégradé doux vers les extrémités et les bords de la traînée.
  let fall = max(1.0 - abs(v.local.x), 0.0) * max(1.0 - abs(v.local.y), 0.0);
  return vec4f(v.color * (fall * fall), 1.0);
}
