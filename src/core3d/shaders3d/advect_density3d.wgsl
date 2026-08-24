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
  // EXPLOSION : xyz centre (voxels), w rayon (voxels).
  burst_a: vec4f,
  // x: débit de carburant (1/s), y: débit de l'amorce en chaleur (1/s),
  // z: injection en cours (0/1) — les deux débits sont déjà à zéro sinon,
  // w: graine des grumeaux (décalage du bruit, change à chaque détonation).
  burst_b: vec4f,
  // x: rendement de suie, y: évanouissement, z: refroidissement PAR la suie.
  soot: vec4f,
  // x: largeur de la bande éponge (voxels, 0 = boîte close), y: sa force.
  open_box: vec4f,
  // POUSSIÈRE soulevée : x débit, y rayon horizontal (voxels), z épaisseur.
  dust: vec4f,
}

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
  let band = P.open_box.x;
  if (band <= 0.0) {
    return 1.0;
  }
  let n = P.misc.z;
  let d = min(
    min(min(p.x, n - p.x), min(p.z, n - p.z)),
    n - p.y, // plafond seulement — le sol reste dur
  );
  let s = 1.0 - clamp(d / band, 0.0, 1.0);
  return 1.0 / (1.0 + P.open_box.y * s * s * dt);
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
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var den_src: texture_3d<f32>;
@group(1) @binding(1) var vel_src: texture_3d<f32>;
@group(1) @binding(2) var aux: texture_3d<f32>;
@group(1) @binding(3) var den_dst: texture_storage_3d<rgba16float, write>;
@group(1) @binding(4) var oxy_src: texture_3d<f32>;

fn n_size() -> i32 {
  return i32(P.misc.z);
}
fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(P.misc.z);
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
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let center = vec3f(c) + vec3f(0.5);
  let back = backtrace(center, P.misc.x);
  textureStore(den_dst, gid, textureSampleLevel(den_src, lin, back * inv_n(), 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn correct(@builtin(global_invocation_id) gid: vec3u) {
  let n = n_size();
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let center = vec3f(c) + vec3f(0.5);
  let orig = textureLoad(den_src, c, 0);
  let hat = textureLoad(aux, c, 0);
  let tilde = textureSampleLevel(aux, lin, forwardtrace(center, dt) * inv_n(), 0.0);
  var val = hat + 0.5 * (orig - tilde);

  // Clamp au stencil trilinéaire du point rétro-advecté (par canal, trace RK2).
  let back = backtrace(center, dt);
  let base = vec3i(floor(back - vec3f(0.5)));
  var lo = vec4f(1e30);
  var hi = vec4f(-1e30);
  for (var i = 0; i < 8; i++) {
    let corner = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
    let s = textureLoad(den_src, clamp(base + corner, vec3i(0), vec3i(n - 1)), 0);
    lo = min(lo, s);
    hi = max(hi, s);
  }
  val = clamp(val, lo, hi);

  // COMBUSTION (Feldman/Fedkiw simplifié) : au-dessus de la température d'ignition,
  // le carburant (canal z) se consume — il dégage de la chaleur et de la suie grise
  // (canal x). L'expansion volumique associée est injectée par la passe divergence.
  // Le triangle du feu est complet : sans OXYGÈNE, la flamme faiblit puis s'étouffe.
  let spec = textureSampleLevel(oxy_src, lin, center * inv_n(), 0.0);
  let o2 = spec.x;
  let oxy_factor = clamp(o2 / 0.25, 0.0, 1.0);
  let ignite = smoothstep(0.28, 0.55, val.w);
  let burn = min(val.z * P.emit_meta.y * dt * ignite, val.z) * oxy_factor;
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
  // EXPLOSION : une bouffée de CARBURANT avec son AMORCE en chaleur. Rien
  // d'autre — la combustion trois lignes plus haut fait la boule de feu, la suie
  // et (par l'expansion, dans la passe de divergence) le souffle. L'amorce est
  // plus resserrée que le carburant : le cœur part le premier et le front gagne
  // le bord, ce qui donne une boule qui S'OUVRE au lieu de s'allumer d'un bloc.
  if (P.burst_b.z > 0.5) {
    let dq = distance(center, P.burst_a.xyz) / max(P.burst_a.w, 1e-3);
    let gauss = exp(-dq * dq * 3.0);
    if (gauss > 1e-3) {
      // Grumeaux de charge : le bruit est ancré sur le CENTRE de la bouffée (il
      // ne glisse donc pas d'une frame à l'autre) et décalé par une graine, pour
      // que deux détonations ne soient pas jumelles. Deux octaves, taille de
      // motif ≈ un tiers du rayon.
      let q = (center - P.burst_a.xyz) / max(P.burst_a.w, 1e-3) * 3.4 + P.burst_b.w;
      let n = vnoise(q) * 0.65 + vnoise(q * 2.7) * 0.35;
      // Le CARBURANT est fortement modulé (c'est lui qui dessine les langues de
      // flamme), l'AMORCE beaucoup moins : le cœur doit partir à coup sûr, sinon
      // la charge couve au lieu de détoner.
      val.z += P.burst_b.x * dt * gauss * mix(0.25, 1.75, n);
      val.w += P.burst_b.y * dt * exp(-dq * dq * 7.0) * mix(0.8, 1.2, n);
    }
  }

  // POUSSIÈRE SOULEVÉE : une galette de matière FROIDE plaquée au sol, autour du
  // point d'impact. Le pied d'un champignon n'est pas fait de la charge — il est
  // fait du sol arraché, que le courant ascendant de la boule aspire ensuite
  // derrière elle. Sans ce terme, on obtient un chapeau qui flotte sans colonne.
  // Froide, donc : elle ne monte que parce qu'on l'entraîne, ce qui est
  // exactement ce qui donne au pied sa lenteur et son étranglement.
  if (P.dust.x > 0.0) {
    let dr = length(center.xz - P.burst_a.xz) / max(P.dust.y, 1e-3);
    let dh = center.y / max(P.dust.z, 1e-3);
    val.x += P.dust.x * dt * exp(-dr * dr * 1.4) * exp(-dh * dh * 2.0);
  }

  val = min(val, vec4f(3.0, 3.0, 3.0, 2.0));

  // Éponge : la matière s'éteint près des parois ouvertes au lieu de s'y empiler.
  val *= sponge3(center, dt);

  // Jamais d'encre ni de chaleur dans la sphère-obstacle.
  if (P.sphere.w > 0.0 && distance(center, P.sphere.xyz) < P.sphere.w) {
    val = vec4f(0.0);
  }

  textureStore(den_dst, gid, max(val, vec4f(0.0)));
}
