// ÉTINCELLES de feu d'artifice (chantier PLAN-ARTIFICES) : particules à
// COULEUR PRESCRITE — aucune ne vient du corps noir, c'est toute la raison
// d'être du système. Elles naissent par ÉVÉNEMENT DE TIR : le CPU désigne une
// tranche [base, base+count) de l'anneau (curseur côté CPU, déterministe, pas
// d'atomics) et ces particules-là renaissent d'un coup à vitesse radiale — la
// pivoine. Pas de renaissance spontanée : sans tir, une morte reste morte.
// Positions en VOXELS (l'espace des vitesses). Buffer zéro-initialisé = toutes
// mortes (life 0). L'ÉPOQUE (spark_c.z, recopiée dans tint.w à la naissance)
// tue au premier update tout ce qui vient d'avant un reset.

struct Params {
  misc: vec4f, // x: dt, y: temps, z: N, w: hauteur du domaine (Ny/Nx)
  // Champs traversés sans être lus : la disposition de l'uniforme est COMMUNE
  // à toutes les passes, un shader n'en déclare que ce qu'il adresse — ici par
  // BLOCS DE PADDING (le tampon fait 640 o = 40 vec4, l'événement de tir vit
  // dans les trois derniers, au-delà des 37 vec4 de la simulation).
  pad0: array<vec4f, 6>,
  sphere: vec4f, // xyz: centre (voxels), w: rayon (voxels, ≤0 = absente)
  pad1: array<vec4f, 29>,
  // L'ÉVÉNEMENT DE TIR — consommé en une frame (le patron des charges) :
  spark_a: vec4f, // xyz: centre de l'éclat (voxels), w: vitesse radiale (voxels/s)
  spark_b: vec4f, // xyz: couleur RGB prescrite, w: nombre à faire naître (0 = pas de tir)
  spark_c: vec4f, // x: base de l'anneau, y: graine du tir, z: époque (reset), w: patron
}

// PATRONS de tir (spark_c.w à la naissance, puis encodés PAR PARTICULE dans
// tint.w = époque × 8 + patron — deux patrons peuvent coexister en vol) :
//  0 PIVOINE : la sphère classique — traînée moyenne, gravité douce.
//  1 SAULE : longue retombée — vie longue, traînée lâche, gravité marquée.
//  2 ÉCLAT : bref et nerveux — vif, vie courte, presque sans chute.
//  3 TRAÇANTE : la fusée qui monte — balistique PURE (aucune traînée, gravité
//    0,55·N dupliquée côté CPU), meurt à l'apogée (vy ≤ 0) où le CPU tire
//    l'éclat : la comète s'éteint exactement là où la sphère naît.
struct Spark {
  pos: vec4f,  // xyz: voxels, w: âge (s)
  vel: vec4f,  // xyz: voxels/s, w: durée de vie (s, 0 = jamais née)
  tint: vec4f, // xyz: couleur RGB, w: époque × 8 + patron
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var vel_src: texture_3d<f32>;
@group(1) @binding(1) var<storage, read_write> sparks: array<Spark>;
// LUEUR TEINTÉE : chaque étoile vivante dépose sa couleur dans ce tampon à la
// résolution du volume de lueur (3 × u32 par cellule, virgule fixe ×1024,
// remis à zéro par le CPU avant chaque update). L'injection de lueur l'ajoute
// à l'émission corps noir — c'est ce qui fait qu'une pivoine ÉCLAIRE la fumée
// et le sol au lieu de briller dans le vide.
@group(1) @binding(2) var<storage, read_write> glow_splat: array<atomic<u32>>;

// Grille non cubique : positions et bornes per-axe, vitesses en N horizontal
// (les cellules sont cubiques) — même règle que les braises.
fn n_sizef() -> vec3f {
  return vec3f(P.misc.z, P.misc.z * P.misc.w, P.misc.z);
}

fn inv_n() -> vec3f {
  return vec3f(1.0) / n_sizef();
}

// Convention MAC identique à advect_density3d.wgsl.
fn velocity_at(p: vec3f) -> vec3f {
  return vec3f(
    textureSampleLevel(vel_src, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x,
    textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y,
    textureSampleLevel(vel_src, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z,
  );
}

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return f32(pcg(seed)) * (1.0 / 4294967296.0);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&sparks)) {
    return;
  }
  var e = sparks[i];
  let dt = P.misc.x;
  let n = P.misc.z;

  // NAISSANCE PAR ANNEAU : si un tir est en cours et que cet index tombe dans
  // sa tranche, la particule renaît — vivante ou pas (l'anneau écrase, c'est
  // son contrat : un barrage recycle ses étoiles les plus anciennes).
  let count = u32(P.spark_b.w + 0.5);
  if (count > 0u) {
    let total = arrayLength(&sparks);
    let base = u32(P.spark_c.x + 0.5);
    let rel = (i + total - base) % total;
    if (rel < count) {
      let seed = i * 1664525u + bitcast<u32>(P.spark_c.y);
      // Direction UNIFORME sur la sphère (hauteur uniforme + angle uniforme) :
      // c'est elle qui fait la pivoine — un biais donnerait un oursin.
      let z = rand01(seed) * 2.0 - 1.0;
      let a = rand01(seed ^ 0x9e3779b9u) * 6.28318;
      let rxy = sqrt(max(1.0 - z * z, 0.0));
      var dir = vec3f(rxy * cos(a), z, rxy * sin(a));
      // Vitesse étalée (0,55–1,2×) : le cœur reste dense, le front file — la
      // sphère a une épaisseur au lieu d'être une coquille.
      var speed = P.spark_a.w * (0.55 + 0.65 * rand01(seed ^ 0x85ebca6bu));
      var life = 1.4 + 1.2 * rand01(seed ^ 0xc2b2ae35u);
      var carried = velocity_at(P.spark_a.xyz);
      let patt = u32(P.spark_c.w + 0.5);
      if (patt == 1u) { // saule : vie longue, départ un peu plus doux
        life = 2.6 + 1.4 * rand01(seed ^ 0xc2b2ae35u);
        speed *= 0.85;
      }
      if (patt == 2u) { // éclat : vif et bref
        life = 0.45 + 0.45 * rand01(seed ^ 0xc2b2ae35u);
        speed *= 1.5;
      }
      if (patt == 3u) { // traçante : la comète presque verticale, tête au
        // nominal (le CPU intègre LA MÊME trajectoire et tire l'éclat à
        // l'apogée — ni vent ni étalement vers le haut, sinon elles divergent).
        life = 9.0; // meurt à l'apogée (vy ≤ 0), jamais par l'âge
        speed = P.spark_a.w * (1.0 - 0.10 * rand01(seed ^ 0x85ebca6bu));
        dir = normalize(vec3f(
          (rand01(seed) - 0.5) * 0.06,
          1.0,
          (rand01(seed ^ 0x9e3779b9u) - 0.5) * 0.06,
        ));
        carried = vec3f(0.0);
      }
      e.pos = vec4f(P.spark_a.xyz, 0.0);
      e.vel = vec4f(carried + dir * speed, life);
      e.tint = vec4f(P.spark_b.xyz, P.spark_c.z * 8.0 + f32(patt));
      sparks[i] = e;
      return;
    }
  }

  // ÉPOQUE : un reset est passé par là — tout ce qui est né avant meurt au
  // premier update qui suit (l'update précède le tracé dans la frame, donc
  // aucune étoile rassie ne réapparaît à l'image). L'époque vit dans les bits
  // hauts de tint.w (× 8), le patron dans les bits bas.
  if (e.vel.w > 0.0 && abs(floor(e.tint.w * 0.125) - P.spark_c.z) > 0.5) {
    e.vel.w = 0.0;
    sparks[i] = e;
    return;
  }

  let alive = e.vel.w > 0.0 && e.pos.w < e.vel.w;
  if (!alive) {
    return;
  }

  // BALISTIQUE D'UNE ÉTOILE : gravité + traînée exponentielle vers le fluide,
  // toutes deux PAR PATRON. La traînée est plus lâche que celle des braises :
  // l'élan radial de l'éclat doit survivre assez longtemps pour dessiner la
  // sphère avant que le vent ne reprenne la main. Le saule tombe fort et
  // freine peu (les retombées pendantes) ; l'éclat vit trop peu pour chuter ;
  // la traçante est en balistique PURE (traînée 0 : sa trajectoire doit
  // rester confondue avec l'intégration CPU, qui tire l'éclat à l'apogée —
  // gravité 0,55 dupliquée dans ROCKET_G, sim3d.ts).
  let patt = u32(e.tint.w + 0.5) & 7u;
  var drags = array<f32, 4>(1.8, 0.9, 2.6, 0.0);
  var gravs = array<f32, 4>(0.12, 0.20, 0.06, 0.55);
  let k = 1.0 - exp(-drags[min(patt, 3u)] * dt);
  var v = mix(e.vel.xyz, velocity_at(e.pos.xyz), k);
  v.y -= gravs[min(patt, 3u)] * n * dt;
  var pos = e.pos.xyz + v * dt;
  var age = e.pos.w + dt;
  // TRAÇANTE : l'apogée est sa mort — le même critère (vy ≤ 0) que le CPU,
  // qui y fait naître la sphère : la comète s'éteint là où l'éclat surgit.
  if (patt == 3u && v.y <= 0.0) {
    age = e.vel.w + 1.0;
  }
  // Hors boîte ou dans la boule : morte (ne renaîtra qu'au prochain tir).
  let margin = 1.0;
  if (any(pos < vec3f(margin)) || any(pos > n_sizef() - vec3f(margin)) ||
      (P.sphere.w > 0.0 && distance(pos, P.sphere.xyz) < P.sphere.w)) {
    age = e.vel.w + 1.0;
  }
  sparks[i] = Spark(vec4f(pos, age), vec4f(v, e.vel.w), e.tint);

  // SPLAT de lueur teintée — le fondu de fin de vie est celui du tracé
  // (quadratique, linéaire pour le saule), sans strobo ni occlusion : la
  // grille grossière et les trois blurs lissent tout de toute façon.
  if (age < e.vel.w) {
    let t = age / e.vel.w;
    var fade = (1.0 - t) * (1.0 - t);
    if (patt == 1u) {
      fade = 1.0 - t;
    }
    let gn = max(n / 8.0, 16.0);
    let gny = round(gn * P.misc.w);
    let cellf = clamp(floor(pos / 8.0), vec3f(0.0), vec3f(gn - 1.0, gny - 1.0, gn - 1.0));
    let cell = vec3u(cellf);
    let idx = ((cell.z * u32(gny) + cell.y) * u32(gn) + cell.x) * 3u;
    let lum = e.tint.xyz * (fade * 1024.0);
    atomicAdd(&glow_splat[idx], u32(lum.x));
    atomicAdd(&glow_splat[idx + 1u], u32(lum.y));
    atomicAdd(&glow_splat[idx + 2u], u32(lum.z));
  }
}
