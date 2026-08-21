// Remise à zéro des champs 3D (reset). Deux entry points, un par format.

@group(0) @binding(0) var rgba_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var scalar_dst: texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn clear_rgba(@builtin(global_invocation_id) gid: vec3u) {
  if (all(gid < textureDimensions(rgba_dst))) {
    textureStore(rgba_dst, gid, vec4f(0.0));
  }
}

@compute @workgroup_size(4, 4, 4)
fn clear_scalar(@builtin(global_invocation_id) gid: vec3u) {
  if (all(gid < textureDimensions(scalar_dst))) {
    textureStore(scalar_dst, gid, vec4f(0.0));
  }
}

// Initialisation de la texture d'espèces : oxygène = 1 partout, le reste à 0.
@compute @workgroup_size(4, 4, 4)
fn clear_one(@builtin(global_invocation_id) gid: vec3u) {
  if (all(gid < textureDimensions(rgba_dst))) {
    textureStore(rgba_dst, gid, vec4f(1.0, 0.0, 0.0, 0.0));
  }
}
