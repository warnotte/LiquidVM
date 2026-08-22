// J1/J2 — transfert grille → particules (G2P) + advection.
// FLIP/PIC : v = mix(v_grille, v_particule + Δv_grille, α) où Δv = velNew −
// velOld (la grille avant forces/projection). α = 1 → FLIP pur (vif, bruité),
// α = 0 → PIC pur (amorti).
// APIC (J2, sim2.z = 1) : v = v_grille (PIC) ET C = ∇v_grille au point de la
// particule (différences centrées à ½ voxel sur l'interpolant trilinéaire MAC),
// rendue à la grille par le scatter (v + C·dx) : l'information sous-cellule
// survit sans le bruit du FLIP. Layout particule : voir sim_p2g.wgsl.
// Advection RK2 point-milieu sur velNew, déplacement borné (garde CFL), clamp
// aux parois avec annulation de la composante normale.

struct UEau {
  sim: vec4f,  // x: N, y: nb particules, z: dt sous-pas, w: gravité
  sim2: vec4f, // x: mélange FLIP, y: largeur colonne, z: 1 = APIC
  cam_pos: vec4f,
  cam_right: vec4f,
  cam_up: vec4f,
  cam_fwd: vec4f,
  render: vec4f,
  sphere: vec4f,     // xyz: centre (voxels), w: rayon (≤ 0 = absente)
  sphere_vel: vec4f, // xyz: vitesse de la boule (voxels/s)
}

@group(0) @binding(0) var<uniform> U: UEau;
@group(0) @binding(1) var lin: sampler;
@group(1) @binding(0) var<storage, read_write> particles: array<vec4f>;
@group(1) @binding(1) var vel_new: texture_3d<f32>;
@group(1) @binding(2) var vel_old: texture_3d<f32>;
@group(1) @binding(3) var<storage, read_write> census: array<atomic<u32>>;

fn inv_n() -> vec3f {
  return vec3f(1.0) / vec3f(U.sim.x);
}

// Les faces N n'existent pas (murs gratuits du schéma compact) : dans la
// dernière demi-cellule le sampler clampe sur la face N−1 — on fond
// linéairement vers 0 au mur, sinon la vitesse normale contre la paroi est
// surestimée et les particules s'y collent.
fn wall_fade(p: vec3f) -> vec3f {
  return clamp(vec3f(U.sim.x) - p, vec3f(0.0), vec3f(1.0));
}

// Convention MAC du moteur (identique à advect_density3d.wgsl du feu).
fn sample_new(p: vec3f) -> vec3f {
  return vec3f(
    textureSampleLevel(vel_new, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x,
    textureSampleLevel(vel_new, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y,
    textureSampleLevel(vel_new, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z,
  ) * wall_fade(p);
}

fn sample_old(p: vec3f) -> vec3f {
  return vec3f(
    textureSampleLevel(vel_old, lin, (p + vec3f(0.5, 0.0, 0.0)) * inv_n(), 0.0).x,
    textureSampleLevel(vel_old, lin, (p + vec3f(0.0, 0.5, 0.0)) * inv_n(), 0.0).y,
    textureSampleLevel(vel_old, lin, (p + vec3f(0.0, 0.0, 0.5)) * inv_n(), 0.0).z,
  ) * wall_fade(p);
}

const MAX_MOVE = 3.0; // voxels/sous-pas — garde CFL (plan : « clamp + sous-pas »)
// Marge aux parois ≈ 0. LEÇON (2026-08-23, trouvée par le recensement des
// cellules) : avec une marge de 1 voxel, la rangée 0 (sol) et les colonnes 0
// restaient VIDES donc classées AIR (p = 0) par le solveur — le sol était une
// surface libre, aucune pression hydrostatique ne pouvait s'établir et toute
// l'eau s'écrasait en une galette d'UNE cellule contre le clamp (128
// particules/cellule). Les particules doivent pouvoir occuper les cellules de
// bord : ce sont leurs faces 0 / N qui sont les murs.
const MARGIN = 0.01;

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let n = U.sim.x;
  let dt = U.sim.z;
  let pos = particles[4u * i].xyz;
  let vel_p = particles[4u * i + 1u].xyz;

  let g_new = sample_new(pos);
  var v: vec3f;
  var c_u = vec3f(0.0);
  var c_v = vec3f(0.0);
  var c_w = vec3f(0.0);
  // Filet float16 : les textures de vitesse sont en rgba16float — au-delà de
  // ~65k tout devient inf puis NaN (leçon durement payée du 2D). Une vitesse
  // physique ne dépasse jamais ~600 voxels/s ici : tout excès est une erreur
  // numérique qu'on écrase avant qu'elle ne contamine.
  if (U.sim2.z > 0.5) {
    // APIC : vitesse PIC + gradient de vitesse (colonnes = ∂/∂x, ∂/∂y, ∂/∂z).
    let h = 0.5;
    let gx = (sample_new(pos + vec3f(h, 0.0, 0.0)) - sample_new(pos - vec3f(h, 0.0, 0.0))) / (2.0 * h);
    let gy = (sample_new(pos + vec3f(0.0, h, 0.0)) - sample_new(pos - vec3f(0.0, h, 0.0))) / (2.0 * h);
    let gz = (sample_new(pos + vec3f(0.0, 0.0, h)) - sample_new(pos - vec3f(0.0, 0.0, h))) / (2.0 * h);
    let lim = vec3f(2000.0);
    c_u = clamp(vec3f(gx.x, gy.x, gz.x), -lim, lim);
    c_v = clamp(vec3f(gx.y, gy.y, gz.y), -lim, lim);
    c_w = clamp(vec3f(gx.z, gy.z, gz.z), -lim, lim);
    v = clamp(g_new, vec3f(-600.0), vec3f(600.0));
  } else {
    // FLIP/PIC.
    let g_old = sample_old(pos);
    let v_flip = vel_p + (g_new - g_old);
    v = clamp(mix(g_new, v_flip, U.sim2.x), vec3f(-600.0), vec3f(600.0));
  }

  // Advection RK2 (point milieu) sur le champ projeté, déplacement borné.
  // NB : « move » est un mot réservé WGSL (comme « from » et « target »).
  let mid = pos + 0.5 * dt * sample_new(pos);
  var disp = dt * sample_new(mid);
  let disp_len = length(disp);
  if (disp_len > MAX_MOVE) {
    disp *= MAX_MOVE / disp_len;
  }
  var p_new = pos + disp;

  // Parois : clamp + annulation de la composante normale entrante.
  let lo = vec3f(MARGIN);
  let hi = vec3f(n - MARGIN);
  if (p_new.x < lo.x || p_new.x > hi.x) {
    v.x = 0.0;
  }
  if (p_new.y < lo.y || p_new.y > hi.y) {
    v.y = 0.0;
  }
  if (p_new.z < lo.z || p_new.z > hi.z) {
    v.z = 0.0;
  }
  p_new = clamp(p_new, lo, hi);

  // BOULE (J3) : aucune particule ne reste dedans — on la repose sur la surface
  // (marge d'un demi-voxel, le rayon que « voit » la grille) et on annule la
  // composante entrante de la vitesse RELATIVE : la boule pousse l'eau au lieu
  // de l'aspirer, et une boule immobile ne colle pas les particules.
  if (U.sphere.w > 0.0) {
    let rel_pos = p_new - U.sphere.xyz;
    let dist = length(rel_pos);
    let surf = U.sphere.w + 0.5;
    if (dist < surf) {
      let nrm = select(vec3f(0.0, 1.0, 0.0), rel_pos / max(dist, 1e-5), dist > 1e-5);
      p_new = clamp(U.sphere.xyz + nrm * surf, lo, hi);
      let vn = dot(v - U.sphere_vel.xyz, nrm);
      if (vn < 0.0) {
        v -= vn * nrm;
      }
    }
  }

  particles[4u * i] = vec4f(p_new, c_w.x);
  particles[4u * i + 1u] = vec4f(v, c_w.y);
  particles[4u * i + 2u] = vec4f(c_u, c_w.z);
  particles[4u * i + 3u] = vec4f(c_v, 0.0);
}

// RECENSEMENT (l'instrument du jalon J1 : « compteur de particules au HUD ») :
// [0] valides, [1] positions NaN/hors monde, [2] vitesses suspectes (> 550),
// [3] particules DANS la boule (critère de sortie de J3 : doit rester 0).
// NaN se détecte par x != x (un NaN n'est jamais égal à lui-même).
@compute @workgroup_size(64)
fn census_pass(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (f32(i) >= U.sim.y) {
    return;
  }
  let pos = particles[4u * i].xyz;
  let vel = particles[4u * i + 1u].xyz;
  let pos_ok = pos.x == pos.x && pos.y == pos.y && pos.z == pos.z &&
    all(pos >= vec3f(0.0)) && all(pos <= vec3f(U.sim.x));
  let vel_ok = vel.x == vel.x && vel.y == vel.y && vel.z == vel.z;
  if (pos_ok && vel_ok) {
    atomicAdd(&census[0], 1u);
  } else {
    atomicAdd(&census[1], 1u);
  }
  if (vel_ok && length(vel) > 550.0) {
    atomicAdd(&census[2], 1u);
  }
  if (U.sphere.w > 0.0 && pos_ok && distance(pos, U.sphere.xyz) < U.sphere.w) {
    atomicAdd(&census[3], 1u);
  }
  // Échantillon brut : 8 particules espacées, pos+vel dans census[4..68]
  // (u32 = bits float, décodés côté CPU) — voir les données, pas les deviner.
  if (i % 262144u == 0u) {
    let slot = 4u + (i / 262144u) * 8u;
    atomicStore(&census[slot], bitcast<u32>(pos.x));
    atomicStore(&census[slot + 1u], bitcast<u32>(pos.y));
    atomicStore(&census[slot + 2u], bitcast<u32>(pos.z));
    atomicStore(&census[slot + 3u], bitcast<u32>(vel.x));
    atomicStore(&census[slot + 4u], bitcast<u32>(vel.y));
    atomicStore(&census[slot + 5u], bitcast<u32>(vel.z));
  }
}
