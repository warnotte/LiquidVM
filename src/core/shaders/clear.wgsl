// Remise à zéro d'un champ (reset « R »). Deux entry points car les storage textures
// ont des formats différents (rgba16float pour vélocité/densité, r32float pour
// pression/divergence) ; chaque pipeline n'utilise que son binding.
// Pas d'uniforms : les dimensions viennent de la texture elle-même.

@group(0) @binding(0) var dst_rgba: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var dst_scalar: texture_storage_2d<r32float, write>;

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn clear_rgba16f(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= textureDimensions(dst_rgba))) {
    return;
  }
  textureStore(dst_rgba, vec2i(gid.xy), vec4f(0.0));
}

// Synchronisé avec WORKGROUP_SIZE (core/config.ts).
@compute @workgroup_size(16, 16)
fn clear_r32f(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= textureDimensions(dst_scalar))) {
    return;
  }
  textureStore(dst_scalar, vec2i(gid.xy), vec4f(0.0));
}
