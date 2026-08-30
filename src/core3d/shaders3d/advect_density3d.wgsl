// Advection MacCormack des densités (x: fumée, yz: encres futures, w: chaleur),
// aux centres des cellules. predict : φ̂ vers scratch ; correct : correction clampée
// + dissipations + injection de l'émetteur (uniquement à l'écriture finale).

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,   // émetteur 0 : xyz centre (voxels), w: rayon (voxels)
  emit_vals: vec4f, // x: débit chaleur, y: débit d'encre, z: impulsion ↑, w: impulsion latérale
  diss: vec4f,      // x: dissipation vélocité, y: encres, z: refroidissement, w: buoyancy
  blow_origin: vec4f,
  blow_dir: vec4f,
  blow_force: vec4f,
  sphere: vec4f,    // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  emit_meta: vec4f, // x: nb d'émetteurs, y: taux de combustion, z: chaleur dégagée, w: expansion
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f, // encre (0/1/2) de chaque émetteur
  // Champs traversés sans être lus ici : la disposition de l'uniforme est
  // COMMUNE à toutes les passes, un shader n'en déclare qu'un PRÉFIXE et doit
  // donc décrire tout ce qui précède ce qui l'intéresse.
  sphere_vel: vec4f,
  ink_weights: vec4f,
  wind: vec4f,
  field_meta: vec4f,
  field0_a: vec4f,
  field0_b: vec4f,
  field1_a: vec4f,
  field1_b: vec4f,
  field2_a: vec4f,
  field2_b: vec4f,
  // (libres — l'ancien slot d'explosion UNIQUE ; les charges vivent dans
  // `bursts`, en queue de struct.)
  // OBSTACLE : x type, yzw paramètres (ratios du rayon).
  shape: vec4f,
  libre_b: vec4f,
  // x: rendement de suie, y: évanouissement, z: refroidissement PAR la suie.
  soot: vec4f,
  // x: bande LATÉRALE (voxels, 0 = parois closes), y: sa force,
  // z: bande de PLAFOND, w: sa force. Deux réglages parce que deux rôles.
  open_box: vec4f,
  // POUSSIÈRE soulevée : y rayon horizontal (voxels), z épaisseur — communs à
  // toutes les charges (x est libre : le débit, lui, est PAR CHARGE, dans
  // `bursts`), w: PLAFOND du canal de chaleur — 2 pour une flamme, bien plus
  // pour une boule de feu (voir le second domaine de `blackbody`, raymarch.wgsl).
  dust: vec4f,
  // (stratification — lue par l'advection de VITESSE ; déclarée ici pour caler
  // l'offset des charges.)
  strat: vec4f,
  // CHARGES en vol (jusqu'à MAX_BURSTS simultanées) : deux vec4 par charge —
  //  [2k]   xyz centre (voxels), w rayon (voxels) ;
  //  [2k+1] x débit de carburant (1/s), y débit d'amorce en chaleur,
  //         z débit de poussière, w graine des grumeaux. Les débits portent
  //         déjà la fraction de frame de LEUR fenêtre (injection et poussière,
  //         chacune la sienne) ; une charge éteinte a ses trois débits à zéro.
  bursts: array<vec4f, 8>,
}

const MAX_BURSTS = 4u;

// BANDE ÉPONGE (« ciel ouvert ») — la boîte 3D est close de partout (Neumann par
// clamp), donc tout ce qu'on injecte finit par la remplir : un panache plafonne
// et repeint le plafond, une explosion sature le volume en deux secondes.
//
// La solution n'est PAS une vraie sortie Dirichlet avec face virtuelle : elle
// rend le système non symétrique (dernière rangée découplée) et le multigrid
// diverge par construction — leçon déjà payée en 2D, ne pas y revenir. On garde
// la boîte fermée, donc l'opérateur symétrique intact, et on AMORTIT dans une
// bande près des parois : la matière et la vitesse s'y éteignent avant d'avoir
// eu le temps de s'accumuler.
//
// Le SOL est exclu : c'est la seule paroi qui soit un vrai objet de la scène
// (le raymarch l'éclaire et y projette les ombres). Une explosion doit rebondir
// dessus, pas s'y dissoudre.
fn sponge3(p: vec3f, dt: f32) -> f32 {
  // Per-axe : le plafond n'est plus à la même hauteur que la largeur.
  let nv = n_sizef();
  var f = 1.0;
  // PAROIS LATÉRALES : c'est par elles que doit s'évacuer la nappe qui s'étale
  // au sol. Rien d'intéressant ne s'y gare, on peut donc y être franc.
  if (P.open_box.x > 0.0) {
    let ds = min(min(p.x, nv.x - p.x), min(p.z, nv.z - p.z));
    let s = 1.0 - clamp(ds / P.open_box.x, 0.0, 1.0);
    f /= 1.0 + P.open_box.y * s * s * dt;
  }
  // PLAFOND : réglage SÉPARÉ, et bien plus doux. Un panache TRAVERSE la bande ;
  // le chapeau d'un champignon, lui, monte au plafond et s'y GARE. Une
  // absorption calibrée pour un passage le lamine alors sur place — c'est ce qui
  // transformait le chapeau en dalle diffuse. Deux parois, deux rôles opposés :
  // les traiter avec la même sévérité était l'erreur.
  if (P.open_box.z > 0.0) {
    let s = 1.0 - clamp((nv.y - p.y) / P.open_box.z, 0.0, 1.0);
    f /= 1.0 + P.open_box.w * s * s * dt;
  }
  // Le SOL n'est jamais absorbant : c'est la seule paroi qui soit un objet de la
  // scène, et une explosion doit rebondir dessus.
  return f;
}


// Bruit de valeur 3D — uniquement pour DÉCHIRER la charge d'explosion. Une
// gaussienne analytique est parfaitement symétrique, et une boule de feu
// parfaitement symétrique ressemble à une ampoule : les instabilités qui font
// l'aspect d'une détonation n'ont rien à amplifier. Deux octaves suffisent, et
// le coût ne se paie que dans la petite boule, pendant les ~3 frames
// d'injection.
fn hash31(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(hash31(i + vec3f(0.0, 0.0, 0.0)), hash31(i + vec3f(1.0, 0.0, 0.0)), u.x);
  let b = mix(hash31(i + vec3f(0.0, 1.0, 0.0)), hash31(i + vec3f(1.0, 1.0, 0.0)), u.x);
  let c = mix(hash31(i + vec3f(0.0, 0.0, 1.0)), hash31(i + vec3f(1.0, 0.0, 1.0)), u.x);
  let d = mix(hash31(i + vec3f(0.0, 1.0, 1.0)), hash31(i + vec3f(1.0, 1.0, 1.0)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn emitter_pos(i: u32) -> vec4f {
  switch i {
    case 0u: { return P.emitter; }
    case 1u: { return P.emitter1; }
    case 2u: { return P.emitter2; }
    default: { return P.emitter3; }
  }
}

fn emitter_ink(i: u32) -> u32 {
  switch i {
    case 0u: { return u32(P.emit_inks.x + 0.5); }
    case 1u: { return u32(P.emit_inks.y + 0.5); }
    case 2u: { return u32(P.emit_inks.z + 0.5); }
    default: { return u32(P.emit_inks.w + 0.5); }
  }
}

@group(0) @binding(0) var<uniform> P: Params;

// OBSTACLE — DISTANCE SIGNÉE en voxels, négative dedans (voir PLAN-OBSTACLES).
// Le prédicat n'est plus « dans la sphère » mais « sd < 0 » : l'opérateur ne
// change pas, seul le masque binaire de cellules change. `sphere.w` reste LE
// rayon (≤ 0 = pas d'obstacle), les paramètres de forme sont des RATIOS de ce
// rayon. Dupliquée dans chaque passe — les shaders sont autonomes.
fn solid_sd(p: vec3f) -> f32 {
  let r = P.sphere.w;
  if (r <= 0.0) {
    return 1e9;
  }
  let q = p - P.sphere.xyz;
  let kind = i32(P.shape.x + 0.5);
  if (kind == 1) { // BOÎTE : demi-côtés = rayon × ratios
    let d = abs(q) - r * P.shape.yzw;
    return length(max(d, vec3f(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
  }
  if (kind == 2) { // TORE d'axe vertical : grand rayon r, petit rayon r × y
    return length(vec2f(length(q.xz) - r, q.y)) - r * P.shape.y;
  }
  return length(q) - r; // SPHÈRE
}

// LA RÉTRO-TRACE NE DOIT PAS TRAVERSER UN SOLIDE. À grand pas de temps (donc à
// bas framerate) elle saute une paroi de plusieurs voxels et va échantillonner
// DE L'AUTRE CÔTÉ — matière et oxygène passent alors à travers un mur. On
// marche le segment et on s'arrête au dernier point resté dans le fluide.
// (`from` est un mot réservé WGSL — d'où `p0`.)
fn trace_clamp(p0: vec3f, p1: vec3f) -> vec3f {
  let d = p1 - p0;
  let steps = i32(clamp(length(d) / 1.5, 1.0, 12.0));
  var last = p0;
  for (var i = 1; i <= steps; i++) {
    let q = p0 + d * (f32(i) / f32(steps));
    if (solid_sd(q) < 0.0) {
      return last;
    }
    last = q;
  }
  return p1;
}


// TAILLE DE LA GRILLE, par axe. Le domaine n'est plus forcément cubique :
// Nx = Nz = misc.z, Ny = misc.z × misc.w. Les CELLULES, elles, restent cubiques
// — c'est ce qui permet de ne rien changer à l'opérateur ni à l'advection.
fn n_size() -> vec3i {
  return vec3i(i32(P.misc.z), i32(P.misc.z * P.misc.w), i32(P.misc.z));
}

fn n_sizef() -> vec3f {
  return vec3f(P.misc.z, P.misc.z * P.misc.w, P.misc.z);
}
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var den_src: texture_3d<f32>;
@group(1) @binding(1) var vel_src: texture_3d<f32>;
@group(1) @binding(2) var aux: texture_3d<f32>;
@group(1) @binding(3) var den_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(4) var oxy_src: texture_3d<f32>;
// ESPÈCES FUSIONNÉES (2026-08-28) : la passe species3d a été absorbée ici — le
// correcteur calcule déjà la rétro-trace RK2 et la combustion ; refaire les
// deux dans une passe séparée coûtait un plein parcours de grille. Le canal
// x = oxygène, y = SUIE (une flamme qui manque d'air craque son carburant et
// noircit — c'est la signature d'une explosion ; une bougie brûle dans l'air
// libre et reste claire, le même terme donne les deux).
@group(1) @binding(5) var oxy_dst: texture_storage_3d<rgba16float, write>;

// Fraction d'O₂ de référence : en dessous, la flamme faiblit puis s'étouffe.
const O2_REF = 0.25;
// Stœchiométrie : O₂ consommé par unité de carburant brûlé.
const O2_PER_FUEL = 0.55;

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
}

fn sample_u(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x;
}
fn sample_v(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y;
}
fn sample_w(p: vec3f) -> f32 {
  return textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z;
}
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(sample_u(p), sample_v(p), sample_w(p));
}

// Traces de caractéristiques RK2 (point milieu) — voir advect_velocity3d.
fn backtrace(p: vec3f, dt: f32) -> vec3f {
  let mid = p - 0.5 * dt * velocity_at(p);
  return p - dt * velocity_at(mid);
}

fn forwardtrace(p: vec3f, dt: f32) -> vec3f {
  let mid = p + 0.5 * dt * velocity_at(p);
  return p + dt * velocity_at(mid);
}

@compute @workgroup_size(4, 4, 4)
fn predict(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let center = vec3f(c) + vec3f(0.5);
  let back = trace_clamp(center, backtrace(center, P.misc.x));
  textureStore(den_dst, gid, textureSampleLevel(den_src, lin, back * inv_n(), 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (any(c >= n)) {
    return;
  }
  let dt = P.misc.x;
  let center = vec3f(c) + vec3f(0.5);
  let orig = textureLoad(den_src, c, 0);
  let hat = textureLoad(aux, c, 0);
  let tilde = textureSampleLevel(aux, lin, forwardtrace(center, dt) * inv_n(), 0.0);
  var val = hat + 0.5 * (orig - tilde);

  // Clamp au stencil trilinéaire du point rétro-advecté (par canal, trace RK2).
  let back = trace_clamp(center, backtrace(center, dt));
  let base = vec3i(floor(back - vec3f(0.5)));
  var lo = vec4f(1e30);
  var hi = vec4f(-1e30);
  for (var i = 0; i < 8; i++) {
    let corner = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let s = textureLoad(den_src, clamp(base + corner, vec3i(0), n - vec3i(1)), 0);
    lo = min(lo, s);
    hi = max(hi, s);
  }
  val = clamp(val, lo, hi);

  // COMBUSTION (Feldman/Fedkiw simplifié) : au-dessus de la température d'ignition,
  // le carburant (canal z) se consume — il dégage de la chaleur et de la suie grise
  // (canal x). L'expansion volumique associée est injectée par la passe divergence.
  // Le triangle du feu est complet : sans OXYGÈNE, la flamme faiblit puis s'étouffe.
  // L'oxygène est ADVECTÉ ICI (même rétro-trace RK2 que la correction — c'est
  // toute l'économie de la fusion), puis consommé par cette combustion.
  var spec = textureSampleLevel(oxy_src, lin, back * inv_n(), 0.0);
  var o2 = spec.x;
  let oxy_factor = clamp(o2 / O2_REF, 0.0, 1.0);
  let ignite = smoothstep(0.28, 0.55, val.w);
  let burn = min(val.z * P.emit_meta.y * dt * ignite, val.z) * oxy_factor;
  o2 -= O2_PER_FUEL * burn;
  val.z -= burn;
  val.w = min(val.w + P.emit_meta.z * burn, 1.9);
  val.x += 0.35 * burn;

  // Dissipations : la fumée s'estompe lentement, la chaleur se refroidit vite —
  // avec un terme RADIATIF en T⁴ (Fedkiw) : les gaz les plus chauds s'éteignent
  // brutalement (pointes de flammes nettes), la fumée tiède flotte longtemps.
  // La SUIE rayonne : un gaz qui en est chargé perd sa chaleur bien plus vite
  // qu'un gaz clair. C'est ce terme qui fait virer au NOIR le nuage d'une
  // explosion — sans lui, il devient opaque tout en restant chaud, donc il émet
  // en orange sous une peau impénétrable : une braise géante, pas de la suie.
  var heat = val.w / (1.0 + (P.diss.z + P.soot.z * spec.y) * dt);
  heat = max(heat - dt * 0.30 * heat * heat * heat * heat, 0.0);
  val = vec4f(val.xyz / (1.0 + P.diss.y * dt), heat);

  // Émetteurs gaussiens : chaleur + l'encre propre à chacun (canal xyz selon l'index).
  let count = u32(P.emit_meta.x + 0.5);
  for (var i = 0u; i < count; i++) {
    let e = emitter_pos(i);
    let d = distance(center, e.xyz) / max(e.w, 1e-3);
    let g = exp(-d * d * 3.0);
    let ink = emitter_ink(i);
    var inject = vec3f(0.0);
    if (ink == 0u) {
      inject.x = 1.0;
    } else if (ink == 1u) {
      inject.y = 1.0;
    } else {
      inject.z = 1.0;
    }
    // Le CARBURANT est émis froid : pour l'embraser, il faut l'amener au contact
    // d'une flamme (chaleur d'un autre panache, souffle...). Émission de chaleur
    // pour les autres matières uniquement.
    let heat_rate = select(P.emit_vals.x, 0.0, ink == 2u);
    val = vec4f(
      val.xyz + inject * (P.emit_vals.y * dt * g),
      val.w + heat_rate * dt * g,
    );
  }
  // EXPLOSIONS : chaque charge en vol injecte une bouffée de CARBURANT avec son
  // AMORCE en chaleur. Rien d'autre — la combustion trois lignes plus haut fait
  // la boule de feu, la suie et (par l'expansion, dans la passe de divergence)
  // le souffle. L'amorce est plus resserrée que le carburant : le cœur part le
  // premier et le front gagne le bord, ce qui donne une boule qui S'OUVRE au
  // lieu de s'allumer d'un bloc. Les débits d'une charge éteinte sont à zéro :
  // le contrôle est uniforme, les groupes sautent ses blocs d'un même pas.
  for (var k = 0u; k < MAX_BURSTS; k++) {
    let a = P.bursts[k * 2u];
    let b = P.bursts[k * 2u + 1u];
    if (b.x + b.y > 0.0) {
      let dq = distance(center, a.xyz) / max(a.w, 1e-3);
      let gauss = exp(-dq * dq * 3.0);
      if (gauss > 1e-3) {
        // Grumeaux de charge : le bruit est ancré sur le CENTRE de la bouffée
        // (il ne glisse donc pas d'une frame à l'autre) et décalé par une
        // graine, pour que deux détonations ne soient pas jumelles. Deux
        // octaves, taille de motif ≈ un tiers du rayon.
        let q = (center - a.xyz) / max(a.w, 1e-3) * 3.4 + b.w;
        let n = vnoise(q) * 0.65 + vnoise(q * 2.7) * 0.35;
        // Le CARBURANT est fortement modulé (c'est lui qui dessine les langues
        // de flamme), l'AMORCE beaucoup moins : le cœur doit partir à coup sûr,
        // sinon la charge couve au lieu de détoner.
        val.z += b.x * dt * gauss * mix(0.25, 1.75, n);
        val.w += b.y * dt * exp(-dq * dq * 7.0) * mix(0.8, 1.2, n);
      }
    }
    // POUSSIÈRE SOULEVÉE : une galette de matière FROIDE plaquée au sol, autour
    // du point d'impact. Le pied d'un champignon n'est pas fait de la charge —
    // il est fait du sol arraché, que le courant ascendant de la boule aspire
    // ensuite derrière elle. Sans ce terme, on obtient un chapeau qui flotte
    // sans colonne. Froide, donc : elle ne monte que parce qu'on l'entraîne, ce
    // qui est exactement ce qui donne au pied sa lenteur et son étranglement.
    // La fenêtre d'arrachement est PROPRE à chaque charge.
    if (b.z > 0.0) {
      let dr = length(center.xz - a.xz) / max(P.dust.y, 1e-3);
      let dh = center.y / max(P.dust.z, 1e-3);
      val.x += b.z * dt * exp(-dr * dr * 1.4) * exp(-dh * dh * 2.0);
    }
  }

  val = min(val, vec4f(3.0, 3.0, 3.0, max(P.dust.w, 2.0)));

  // Éponge : la matière s'éteint près des parois ouvertes au lieu de s'y empiler.
  val *= sponge3(center, dt);

  // Jamais d'encre ni de chaleur dans la sphère-obstacle.
  if (solid_sd(center) < 0.0) {
    val = vec4f(0.0);
  }

  textureStore(den_dst, gid, max(val, vec4f(0.0)));

  // SUIE : le carburant CHAUD que l'air ne suffit plus à brûler craque en noir.
  // La mesure de MANQUE D'AIR se prend sur le rapport de ce que le carburant
  // présent RÉCLAME à ce qui est disponible — surtout pas sur `oxy_factor`, qui
  // sature à 1 dès 0,25 d'oxygène et vaut donc 1 presque partout (leçon payée :
  // première version à suie invisible). Une bougie brûle du carburant dilué :
  // propre ; une charge concentre des unités de carburant sur une unité d'air :
  // elle noircit. Le même terme donne les deux.
  let demande = val.z * O2_PER_FUEL;
  let manque = clamp(1.0 - o2 / max(demande, 1e-3), 0.0, 1.0);
  var soot = spec.y + P.soot.x * dt * val.z * ignite * manque;
  soot = soot / (1.0 + P.soot.y * dt);

  // Récupération lente de l'oxygène (la boîte fuit) + apport du souffle.
  o2 += dt * P.ink_weights.w * (1.0 - o2);
  if (P.blow_dir.w > 0.5) {
    let bv = center - P.blow_origin.xyz;
    let bt = dot(bv, P.blow_dir.xyz);
    if (bt > 0.0) {
      let bd = distance(center, P.blow_origin.xyz + P.blow_dir.xyz * bt) /
        max(P.blow_origin.w, 1e-3);
      o2 += dt * P.blow_force.w * exp(-bd * bd * 2.0) * (1.0 - o2);
    }
  }

  // UN SOLIDE NE CONTIENT PAS DE GAZ — même convention que les densités, qui
  // sont mises à zéro dans l'obstacle quelques lignes plus haut. Sans cela les
  // cellules de l'obstacle gardent leur oxygène initial (1,0) pour toujours :
  // rien ne les consomme, et l'échantillonnage trilinéaire des cellules
  // voisines le réinjecte dans le fluide — un obstacle devient un RÉSERVOIR
  // D'AIR INFINI. Mesuré au chantier obstacles : l'air d'une poche fermée
  // plafonnait au lieu de s'épuiser, et rétrécir la poche n'y changeait rien
  // (la source suivait la SURFACE, pas le volume).
  if (solid_sd(center) < 0.0) {
    textureStore(oxy_dst, gid, vec4f(0.0));
    return;
  }
  textureStore(oxy_dst, gid, vec4f(clamp(o2, 0.0, 1.0), min(soot, 4.0), spec.zw));
}
