// Volume de LUEUR : la flamme éclaire sa propre fumée. Grille grossière
// (GRID3/8) : inject = émission corps noir de la chaleur moyennée par pavé,
// blur ×3 (boîte 3³) = diffusion de la lumière — répété, le noyau boîte
// approche une gaussienne qui s'élargit : approximation de scattering multiple,
// la radiance du feu baigne les volutes voisines en s'atténuant. Le raymarch,
// le sol et la sphère l'échantillonnent en trilinéaire (la basse résolution
// donne exactement le flou voulu d'une lumière de zone).
// Layouts partagés avec curl : source échantillonnée (0) + sortie storage (1) ;
// du groupe 0 commun, seul le sampler sert.

@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var src: texture_3d<f32>;
@group(1) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
// LUEUR TEINTÉE : la lumière des étincelles, splattée en virgule fixe (×1024)
// par leur passe d'update, cellule pour cellule — l'injection seule le lit
// (les blurs restent sur le layout partagé avec curl).
@group(1) @binding(2) var<storage, read> spark_glow: array<u32>;

// Poids de la lumière d'étincelle face à l'émission corps noir — LE réglage
// « somptueux ou criard » de l'essai lueur teintée.
const SPARK_GLOW = 1.2;

// Dupliqué de raymarch.wgsl (les shaders sont autonomes).
fn blackbody(heat: f32) -> vec3f {
  let h = clamp(heat, 0.0, 1.7);
  return vec3f(h * 1.6, h * h * 0.9, h * h * h * 0.42);
}

// Injection : émission moyenne du pavé fin couvert par le voxel grossier
// (8 prises trilinéaires aux centres des sous-octants).
@compute @workgroup_size(4, 4, 4)
fn inject(@builtin(global_invocation_id) gid: vec3u) {
  let m = textureDimensions(dst);
  if (gid.x >= m.x || gid.y >= m.y || gid.z >= m.z) {
    return;
  }
  let inv = vec3f(1.0) / vec3f(m);
  let base = (vec3f(gid) + vec3f(0.5)) * inv;
  var emis = vec3f(0.0);
  for (var i = 0u; i < 8u; i++) {
    let corner = vec3f(vec3u(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u));
    let heat = textureSampleLevel(src, lin, base + (corner - vec3f(0.5)) * 0.5 * inv, 0.0).w;
    emis += blackbody(heat);
  }
  var light = emis * 0.125;
  // La lumière des étincelles s'AJOUTE au corps noir (hors moyenne des 8
  // prises : le splat est déjà par cellule grossière) — le seul endroit du
  // moteur où une lumière verte ou bleue peut naître.
  let idx = ((gid.z * m.y + gid.y) * m.x + gid.x) * 3u;
  light += vec3f(f32(spark_glow[idx]), f32(spark_glow[idx + 1u]), f32(spark_glow[idx + 2u])) *
    (SPARK_GLOW / 1024.0);
  textureStore(dst, gid, vec4f(light, 0.0));
}

// Diffusion : boîte 3³ (27 loads), bords clampés.
@compute @workgroup_size(4, 4, 4)
fn blur(@builtin(global_invocation_id) gid: vec3u) {
  let m = vec3i(textureDimensions(dst));
  let c = vec3i(gid);
  if (c.x >= m.x || c.y >= m.y || c.z >= m.z) {
    return;
  }
  var sum = vec3f(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(c + vec3i(dx, dy, dz), vec3i(0), m - vec3i(1));
        sum += textureLoad(src, q, 0).rgb;
      }
    }
  }
  textureStore(dst, gid, vec4f(sum / 27.0, 0.0));
}
