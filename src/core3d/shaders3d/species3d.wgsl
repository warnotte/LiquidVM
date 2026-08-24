// Transport des ESPÈCES (texture rgba16float : x = oxygène, y = SUIE, zw =
// réservés — le cadre extensible du modèle de réaction).
//
// SUIE : une flamme qui manque d'air ne brûle pas proprement, elle CRAQUE son
// carburant et noircit. C'est la signature d'une explosion — la charge dévore
// l'oxygène de son propre volume en quelques dizaines de millisecondes, et ce
// qu'elle recrache est noir. Une bougie, elle, brûle dans l'air libre et reste
// claire : le même terme donne les deux, sans réglage séparé.
// La suie vit dans le canal y parce qu'il était libre ET déjà advecté par cette
// passe — elle voyage avec le gaz sans coûter un transport de plus.
// Advection semi-lagrangienne RK2, puis chimie de l'oxygène : la combustion le
// consomme (stœchiométrie, mêmes critères que la réaction des densités qui lit
// les densités POST-advection de cette frame), il revient lentement (boîte qui
// fuit), et le souffle du pointeur en injecte le long de son rayon.

struct Params {
  misc: vec4f,      // x: dt, y: temps, z: N, w: force de vorticité
  emitter: vec4f,
  emit_vals: vec4f,
  diss: vec4f,
  blow_origin: vec4f, // xyz: origine du rayon (voxels), w: rayon du pinceau
  blow_dir: vec4f,    // xyz: direction, w: actif (0/1)
  blow_force: vec4f,  // w: débit d'oxygène du souffle (1/s au cœur du fuseau)
  sphere: vec4f,
  emit_meta: vec4f,   // y: taux de combustion
  emitter1: vec4f,
  emitter2: vec4f,
  emitter3: vec4f,
  emit_inks: vec4f,
  sphere_vel: vec4f,
  ink_weights: vec4f, // w: taux de récupération d'oxygène (1/s)
  wind: vec4f,
  field_meta: vec4f,
  field0_a: vec4f,
  field0_b: vec4f,
  field1_a: vec4f,
  field1_b: vec4f,
  field2_a: vec4f,
  field2_b: vec4f,
  burst_a: vec4f,
  burst_b: vec4f,
  // x: rendement de suie (1/s par unité de carburant chaud privé d'air),
  // y: son évanouissement (1/s). Le reste est libre.
  soot: vec4f,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var oxy_src: texture_3d<f32>;
@group(1) @binding(1) var vel_src: texture_3d<f32>;
@group(1) @binding(2) var den_src: texture_3d<f32>;
@group(1) @binding(3) var oxy_dst: texture_storage_3d<rgba16float, write>;

// Fraction d'O₂ de référence : en dessous, la flamme faiblit puis s'étouffe.
const O2_REF = 0.25;
// Stœchiométrie : O₂ consommé par unité de carburant brûlé.
const O2_PER_FUEL = 0.55;

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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(P.misc.z);
  let c = vec3i(gid);
  if (c.x >= n || c.y >= n || c.z >= n) {
    return;
  }
  let dt = P.misc.x;
  let center = vec3f(c) + vec3f(0.5);
  // Advection RK2 (point milieu).
  let mid = center - 0.5 * dt * velocity_at(center);
  let back = center - dt * velocity_at(mid);
  var species = textureSampleLevel(oxy_src, lin, back * inv_n(), 0.0);
  var o2 = species.x;

  // Consommation par la combustion (densités post-advection de cette frame).
  let den = textureLoad(den_src, c, 0);
  let ignite = smoothstep(0.28, 0.55, den.w);
  let oxy_factor = clamp(o2 / O2_REF, 0.0, 1.0);
  let burn = min(den.z * P.emit_meta.y * dt * ignite, den.z) * oxy_factor;
  o2 -= O2_PER_FUEL * burn;

  // SUIE : le carburant CHAUD que l'air ne suffit plus à brûler craque en noir.
  // La mesure de MANQUE D'AIR se prend sur le rapport de ce que le carburant
  // présent RÉCLAME à ce qui est disponible — surtout pas sur `oxy_factor`, qui
  // sature à 1 dès 0,25 d'oxygène et vaut donc 1 presque partout (première
  // version : suie invisible, la bascule au rendu ne changeait rien).
  // Une bougie brûle dans l'air libre, du carburant très dilué : elle reste
  // propre. Une charge concentre des unités de carburant dans un volume qui n'a
  // qu'une unité d'oxygène : elle noircit. Le même terme donne les deux.
  let demande = den.z * O2_PER_FUEL;
  let manque = clamp(1.0 - o2 / max(demande, 1e-3), 0.0, 1.0);
  var soot = species.y + P.soot.x * dt * den.z * ignite * manque;
  soot = soot / (1.0 + P.soot.y * dt);

  // Récupération lente (la boîte fuit) + apport du souffle le long de son rayon.
  o2 += dt * P.ink_weights.w * (1.0 - o2);
  if (P.blow_dir.w > 0.5) {
    let v = center - P.blow_origin.xyz;
    let t = dot(v, P.blow_dir.xyz);
    if (t > 0.0) {
      let d = distance(center, P.blow_origin.xyz + P.blow_dir.xyz * t) /
        max(P.blow_origin.w, 1e-3);
      o2 += dt * P.blow_force.w * exp(-d * d * 2.0) * (1.0 - o2);
    }
  }

  textureStore(oxy_dst, gid, vec4f(clamp(o2, 0.0, 1.0), min(soot, 4.0), species.zw));
}
