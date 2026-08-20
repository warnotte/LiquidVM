// Marbrure mathématique (Lu 2011 / Jaffer) : les outils du marbreur sont des
// DÉFORMATIONS EXACTES du plan qui préservent les aires — appliquées ici en warp
// inverse de la texture d'encres (pour chaque texel destination, on échantillonne
// l'encre à la position d'origine). Aucune physique : c'est pour ça que les motifs
// de marbrure sont nets et laminaires, même bain figé.
//   Goutte  : l'encre nouvelle emplit le disque, l'existante est repoussée en anneaux
//             concentriques — p_src = C + d·√(|d|²−r²)/|d| (aire préservée).
//   Stylet  : traînée le long du geste, décroissant avec la distance à la ligne —
//             déplacement = α·λ/(d+λ), α = longueur du segment de geste.
//   Peigne  : idem avec la distance à la DENT la plus proche (dents parallèles au
//             geste, espacées régulièrement).

struct MarbleOp {
  a: vec4f,      // xy: point de départ du geste (normalisé), zw: point d'arrivée
  params: vec4f, // x: outil (0 goutte, 1 stylet, 2 peigne), y: rayon de goutte,
                 // z: λ (portée), w: espacement des dents
  ink: vec4f,    // encre déposée par la goutte (densités rgb + chaleur w)
}

@group(0) @binding(0) var<uniform> OP: MarbleOp;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var src_den: texture_2d<f32>;
@group(0) @binding(3) var dst_den: texture_storage_2d<rgba16float, write>;

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(dst_den);
  if (any(gid.xy >= dims)) {
    return;
  }
  let c = vec2i(gid.xy);
  let p = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(dims);
  let tool = u32(OP.params.x + 0.5);

  // Outil 3 : « fond » — couvre tout le bain de l'encre sélectionnée.
  if (tool == 3u) {
    textureStore(dst_den, c, OP.ink);
    return;
  }

  var src = p;
  if (tool == 0u) {
    // Goutte : disque d'encre neuve, anneaux repoussés autour.
    let d = p - OP.a.xy;
    let l = length(d);
    let r = OP.params.y;
    if (l <= r) {
      textureStore(dst_den, c, OP.ink);
      return;
    }
    src = OP.a.xy + d * (sqrt(l * l - r * r) / l);
  } else {
    let stroke = OP.a.zw - OP.a.xy;
    let len = length(stroke);
    if (len > 1e-5) {
      let dir = stroke / len;
      let n = vec2f(-dir.y, dir.x);
      var d: f32;
      if (tool == 1u) {
        d = abs(dot(p - OP.a.xy, n)); // distance à la ligne du geste
      } else {
        // Distance à la dent la plus proche du peigne.
        let t = dot(p - OP.a.xy, n);
        let s = OP.params.w;
        d = abs(t - round(t / s) * s);
      }
      let shift = len * OP.params.z / (d + OP.params.z);
      src = p - dir * shift;
    }
  }
  textureStore(dst_den, c, textureSampleLevel(src_den, lin, src, 0.0));
}
