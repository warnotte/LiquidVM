// Vue debug « résidu MG » : mosaïque de la pyramide multigrid, composée à la demande
// (vue 6 active uniquement — zéro coût sinon). Un dispatch par niveau : chaque niveau
// remplit sa cellule d'une grille 4 colonnes × 2 rangées (niveau 0 en haut à gauche,
// puis de plus en plus grossier en lisant comme un texte). Dans chaque cellule :
// moitié HAUTE = résidu r = rhs − A·p laissé par la descente du dernier V-cycle
// (au niveau le plus grossier, qui n'a pas de résidu : son second membre) ;
// moitié BASSE = pression du niveau à la fin du cycle.
// Diagnostic : un résidu qui flambe à un niveau désigne l'étage qui ne converge pas ;
// une pression qui explose désigne l'étage où la stabilité se perd en premier.

@group(0) @binding(0) var residual_tex: texture_2d<f32>;
@group(0) @binding(1) var pressure_tex: texture_2d<f32>;
@group(0) @binding(2) var mosaic: texture_storage_2d<rgba16float, write>;

// Même colormap divergente que les vues pression/divergence (scene.wgsl) :
// bleu (négatif) ↔ noir (zéro) ↔ orange (positif).
fn diverging(t: f32) -> vec3f {
  let a = clamp(abs(t), 0.0, 1.0);
  let warm = vec3f(1.0, 0.45, 0.12);
  let cold = vec3f(0.15, 0.5, 1.0);
  return select(cold, warm, t >= 0.0) * a + vec3f(0.02, 0.022, 0.03);
}

@compute @workgroup_size(16, 16)
fn compose(@builtin(global_invocation_id) gid: vec3u) {
  let grid = i32(textureDimensions(mosaic).x);
  let cell_w = grid / 4;
  let cell_h = grid / 2;
  if (i32(gid.x) >= cell_w || i32(gid.y) >= cell_h) {
    return;
  }
  // Le niveau se déduit du rapport des tailles : pas d'uniform, pas d'état CPU.
  let src = i32(textureDimensions(residual_tex).x);
  let level = i32(firstTrailingBit(u32(grid / src)));
  let origin = vec2i((level % 4) * cell_w, (level / 4) * cell_h);
  let local = vec2i(gid.xy);
  let half_h = cell_h / 2;
  let show_pressure = local.y >= half_h;
  let ly = select(local.y, local.y - half_h, show_pressure);
  let sp = clamp(
    vec2i(vec2f(vec2i(local.x, ly)) / vec2f(f32(cell_w), f32(half_h)) * f32(src)),
    vec2i(0),
    vec2i(src - 1),
  );
  var rgb: vec3f;
  if (show_pressure) {
    rgb = diverging(textureLoad(pressure_tex, sp, 0).x * 0.05);
  } else {
    rgb = diverging(textureLoad(residual_tex, sp, 0).x * 0.2);
  }
  // Liserés de séparation des cellules et des moitiés résidu/pression.
  if (local.x == 0 || local.y == 0 || local.y == half_h) {
    rgb = vec3f(0.16, 0.17, 0.2);
  }
  textureStore(mosaic, origin + local, vec4f(rgb, 1.0));
}
