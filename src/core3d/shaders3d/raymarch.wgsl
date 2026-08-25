// Rendu volumétrique : ray-marching à travers la boîte [-½, ½]³, absorption
// Beer-Lambert + émission corps noir (chaleur) + diffusion éclairée par une
// lumière directionnelle (phase Henyey-Greenstein, ombre interne à DEUX octaves
// d'extinction — le « powder » des moteurs de nuages : la lumière pénètre plus
// profond qu'une Beer-Lambert simple) + IN-SCATTERING du volume de lueur :
// la flamme éclaire sa propre fumée, le sol et la sphère (glow_tex, voir
// glow3d.wgsl). Sortie : HDR LINÉAIRE — tone-mapping et gamma vivent dans
// post3d.wgsl (présentation), après le bloom.

struct RenderParams {
  cam_pos: vec4f,   // xyz: position caméra, w: tan(fov/2)
  cam_right: vec4f, // xyz, w: aspect (largeur/hauteur)
  cam_up: vec4f,    // xyz, w: exposition (consommée par la présentation)
  cam_fwd: vec4f,   // xyz, w: pas de marche (nombre)
  light: vec4f,     // xyz: direction VERS la lumière, w: intensité
  sphere: vec4f,    // xyz: centre (monde), w: rayon (monde, ≤0 = absente)
  blow_a: vec4f,    // retour visuel du souffle : xyz origine (monde), w rayon
  blow_b: vec4f,    // xyz direction, w actif (0/1)
  style: vec4f,     // x: lueur du feu, y: bloom (présentation), zw: libres
  // Les gizmos occupent les vec4 suivants (voir gizmo3d.wgsl) : ils ne servent
  // pas ici, mais la disposition de l'uniforme est COMMUNE et il faut donc les
  // traverser pour atteindre ce qui suit.
  giz0_a: vec4f, giz0_b: vec4f,
  giz1_a: vec4f, giz1_b: vec4f,
  giz2_a: vec4f, giz2_b: vec4f,
  emit0: vec4f, emit1: vec4f, emit2: vec4f, emit3: vec4f,
  opts: vec4f,
  sel: vec4f,
  aim: vec4f,
  soot: vec4f,      // x: MODE PRISE DE VUE (0 = atelier, 1 = extérieur),
                    // y: hauteur du soleil, z: densité de suie au rendu,
                    // w: HAUTEUR MONDE du domaine (1 = cube)
  // Suivent les gizmos des champs de force (2 vec4 par champ) — dessinés par
  // gizmo3d.wgsl en LIGNES après la présentation, pas ici : un repère doit être
  // net, pas un halo noyé dans le volume.
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var density_tex: texture_3d<f32>;
@group(0) @binding(3) var glow_tex: texture_3d<f32>;
// Espèces : on n'en lit que le canal y, la SUIE (voir species3d.wgsl).
@group(0) @binding(4) var species_tex: texture_3d<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = corners[vi];
  var out: VSOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv = p;
  return out;
}

// Intersection rayon / boîte centrée (slab). Retourne (t_entrée, t_sortie).
// HAUTEUR MONDE du domaine (1 = le cube d'avant). Le sol reste à y = −0,5 : la
// boîte pousse vers le HAUT, ce qui laisse intacts le plancher, la boule et
// tous les repères posés au sol.
fn box_top() -> f32 {
  return -0.5 + max(R.soot.w, 1e-3);
}

// Coordonnée de TEXTURE d'un point monde. Elle n'est plus symétrique : le
// volume compte HEIGHT × plus de voxels en hauteur, donc y se normalise par la
// hauteur du domaine. Les cellules restant CUBIQUES, x et z sont inchangés.
fn tex_uvw(p: vec3f) -> vec3f {
  return vec3f(p.x + 0.5, (p.y + 0.5) / max(R.soot.w, 1e-3), p.z + 0.5);
}

fn box_hit(ro: vec3f, rd: vec3f) -> vec2f {
  let inv = 1.0 / rd;
  let t0 = (vec3f(-0.5) - ro) * inv;
  let t1 = (vec3f(0.5, box_top(), 0.5) - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  return vec2f(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// Palette des trois matières (canaux xyz) — dupliquée depuis config3d.ts :
// fumée grise, encre magenta, vapeur de carburant ambrée.
const INK0 = vec3f(0.55, 0.60, 0.68);
const INK1 = vec3f(1.00, 0.30, 0.80);
const INK2 = vec3f(1.00, 0.80, 0.45);

// La SUIE éteint bien plus que la fumée à concentration égale — c'est ce qui
// fait qu'un panache d'explosion est OPAQUE là où une fumée de bougie est un
// voile. Son albédo est presque noir : elle absorbe au lieu de diffuser.
const SOOT_EXT = 46.0;
const SOOT_ALBEDO = vec3f(0.055, 0.050, 0.047);

fn extinction(s: vec4f) -> f32 {
  return (s.x + s.y + s.z) * 22.0 + s.w * 5.0;
}

fn soot_at(pos: vec3f) -> f32 {
  return textureSampleLevel(species_tex, lin, tex_uvw(pos), 0.0).y;
}

// Albédo du voxel : mélange des couleurs d'encres pondéré par leurs concentrations.
fn ink_albedo(s: vec4f) -> vec3f {
  let total = s.x + s.y + s.z;
  return (INK0 * s.x + INK1 * s.y + INK2 * s.z) / max(total, 1e-4);
}

// Deux DOMAINES, pas une seule courbe. Jusqu'à 1,7 c'est celui des FLAMMES, et
// la courbe n'y change pas d'un poil. Au-dessus commence celui des BOULES DE
// FEU : plusieurs ordres de grandeur plus haut, où l'émission part en loi de
// puissance et la couleur sature vers le blanc.
// Sans ce second domaine, la plus grosse explosion imaginable reste aussi
// lumineuse qu'un feu de camp — c'est CE plafond qui empêchait la détonation de
// ressembler à autre chose qu'à un gros feu, et aucun réglage de charge ne
// pouvait le contourner puisque la chaleur était écrêtée avant d'arriver ici.
fn blackbody(heat: f32) -> vec3f {
  let h = min(heat, 1.7);
  let flame = vec3f(h * 1.6, h * h * 0.9, h * h * h * 0.42);
  let over = max(heat - 1.7, 0.0);
  return flame + vec3f(1.0, 0.97, 0.92) * (over * over * 3.5);
}

// Lueur du feu diffusée (volume grossier, trilinéaire = flou voulu).
fn glow_at(pos: vec3f) -> vec3f {
  return textureSampleLevel(glow_tex, lin, tex_uvw(pos), 0.0).rgb * R.style.x;
}

// Phase Henyey-Greenstein (g = 0.45), ~0.85 à 90°, ~2.7 plein contre-jour :
// le liseré argenté des fumées entre caméra et lumière.
fn phase_hg(mu: f32) -> f32 {
  let g = 0.45;
  let g2 = g * g;
  return 0.55 + 0.45 * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * mu, 1.5);
}

// Intersection rayon / sphère-obstacle : t d'entrée, ou 1e9 si manquée/absente.
fn sphere_hit(ro: vec3f, rd: vec3f) -> f32 {
  if (R.sphere.w <= 0.0) {
    return 1e9;
  }
  let oc = ro - R.sphere.xyz;
  let b = dot(oc, rd);
  let disc = b * b - (dot(oc, oc) - R.sphere.w * R.sphere.w);
  if (disc < 0.0) {
    return 1e9;
  }
  let t = -b - sqrt(disc);
  return select(1e9, t, t > 0.0);
}

// Hachage rapide pour décaler le départ de chaque rayon (casse le banding).
fn hash12(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 345.45));
  q += dot(q, q + 34.345);
  return fract(q.x * q.y);
}

// PRISE DE VUE EXTÉRIEURE. Une simulation parfaite dans une boîte de verre sur
// fond gris ressemble à une bouffée de fumée dans une boîte de verre : rien n'y
// dit l'ÉCHELLE. Ce qui rend une photo d'essai reconnaissable, c'est d'abord
// qu'elle est immense — et l'échelle se lit sur un horizon, un ciel, un soleil
// rasant, une silhouette à contre-jour. C'est du cadrage, pas de la physique,
// et c'est ce qui manquait le plus.
fn is_outdoor() -> bool {
  return R.soot.x > 0.5;
}

fn sun_dir() -> vec3f {
  // Soleil BAS : un éclairage rasant sculpte les volutes et creuse les
  // dessous, là où un éclairage zénithal les aplatit.
  let h = max(R.soot.y, 0.02);
  return normalize(vec3f(0.62, h, 0.42));
}

/** Ciel : bleu profond au zénith, blanchi et chaud près de l'horizon, avec le
 *  halo du soleil. C'est le fond CLAIR qui fait lire un nuage sombre comme une
 *  masse, au lieu d'un voile gris sur du gris. */
fn sky(rd: vec3f) -> vec3f {
  let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  let zenith = vec3f(0.20, 0.34, 0.62);
  let horizon = vec3f(0.72, 0.74, 0.76);
  var c = mix(horizon, zenith, pow(up, 0.65));
  let mu = clamp(dot(rd, sun_dir()), 0.0, 1.0);
  c += vec3f(1.0, 0.86, 0.62) * pow(mu, 220.0) * 6.0;   // le disque
  c += vec3f(1.0, 0.88, 0.70) * pow(mu, 6.0) * 0.28;    // le halo
  return c;
}

/** Sol vu de l'extérieur : plan qui court jusqu'à l'horizon, avec une brume de
 *  distance. C'est LUI qui donne l'échelle — sans horizon, aucune taille n'est
 *  lisible. */
/** Bruit de valeur 2D pour le moucheté du terrain (haché, sans périodicité). */
fn mottle(q: vec2f) -> f32 {
  let i = floor(q);
  let f = q - i;
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash12(i);
  let b = hash12(i + vec2f(1.0, 0.0));
  let c = hash12(i + vec2f(0.0, 1.0));
  let e = hash12(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, e, u.x), u.y);
}

fn ground(p: vec3f, rd: vec3f) -> vec3f {
  let d = length(p.xz);
  // Terrain SOMBRE : contre un ciel clair, un sol clair aplatit tout. C'est le
  // contraste sol/ciel qui donne sa masse au nuage.
  var tint = mix(vec3f(0.115, 0.100, 0.082), vec3f(0.075, 0.070, 0.065), clamp(d * 0.05, 0.0, 1.0));
  // Moucheté IRRÉGULIER. Sans aucun détail au sol l'œil n'a pas de référence de
  // taille et le nuage flotte hors d'échelle — mais un motif RÉGULIER (une
  // grille) est pire : il crie « scène de test » et détruit l'illusion. Deux
  // échelles de bruit haché, aucune périodicité visible.
  let m = mottle(p.xz * 1.7) * 0.6 + mottle(p.xz * 7.3) * 0.4;
  tint *= mix(0.62, 1.18, m);
  // Brume de distance : le sol se fond dans l'horizon, ce qui creuse la profondeur.
  return mix(tint, sky(vec3f(rd.x, 0.03, rd.z)), clamp(d / 26.0, 0.0, 1.0));
}

// Ombrage du sol sous la boîte : tapis sombre qui reçoit l'ombre volumétrique de
// la fumée (marche courte vers la lumière à travers la boîte) et celle de la boule.
fn floor_shade(fp: vec3f, bg: vec3f, ldir: vec3f) -> vec3f {
  let reach = smoothstep(1.15, 0.40, length(fp.xz));
  if (reach <= 0.0) {
    return bg;
  }
  var occ = 0.0;
  let bh = box_hit(fp, ldir);
  let s0 = max(bh.x, 0.0);
  if (bh.y > s0) {
    let step_len = (bh.y - s0) / 12.0;
    for (var j = 0u; j < 12u; j++) {
      let sp = fp + ldir * (s0 + (f32(j) + 0.5) * step_len);
      occ += extinction(textureSampleLevel(density_tex, lin, tex_uvw(sp), 0.0)) * step_len;
    }
  }
  if (sphere_hit(fp, ldir) < 1e8) {
    occ += 3.0;
  }
  let shadow = exp(-occ * 0.8);
  let base = vec3f(0.075, 0.080, 0.098);
  // Flaque de lumière chaude sous la flamme : la lueur échantillonnée juste
  // au-dessus du sol éclaire le tapis (clamp-to-edge au-delà de la boîte).
  let pool = glow_at(vec3f(fp.x, -0.46, fp.z)) * 0.5;
  return mix(bg, base * (0.30 + 0.70 * shadow) + pool * shadow, reach);
}

@fragment
fn fs_main(frag: VSOut) -> @location(0) vec4f {
  let tanf = R.cam_pos.w;
  let ndc = frag.uv * vec2f(1.0, 1.0);
  let rd = normalize(
    R.cam_fwd.xyz + R.cam_right.xyz * ndc.x * tanf * R.cam_right.w +
      R.cam_up.xyz * ndc.y * tanf,
  );
  let ro = R.cam_pos.xyz;

  // Fond : atelier (dégradé sombre + tapis) ou EXTÉRIEUR (ciel + sol à
  // l'horizon). Le second ne change rien à la simulation — seulement ce qui
  // permet de lire une échelle.
  var bg = mix(vec3f(0.045, 0.05, 0.065), vec3f(0.10, 0.105, 0.13), frag.uv.y * 0.5 + 0.5);
  if (is_outdoor()) {
    bg = sky(rd);
  }
  if (rd.y < -1e-4) {
    let t_floor = (-0.502 - ro.y) / rd.y;
    if (t_floor > 0.0 && is_outdoor()) {
      bg = ground(ro + rd * t_floor, rd);
    } else if (t_floor > 0.0) {
      bg = floor_shade(ro + rd * t_floor, bg, R.light.xyz);
    }
  }

  let hit = box_hit(ro, rd);
  let t0 = max(hit.x, 0.0);
  // La marche s'arrête à la sphère-obstacle si le rayon la touche.
  let t_sphere = sphere_hit(ro, rd);
  let t_end = min(hit.y, t_sphere);
  var col: vec3f;
  if (hit.y <= t0) {
    col = bg;
  } else {
    let steps = max(R.cam_fwd.w, 16.0);
    let step_len = (max(t_end, t0 + 1e-4) - t0) / steps;
    var t = t0 + step_len * hash12(frag.uv * 917.0);
    var transmit = 1.0;
    var acc = vec3f(0.0);

    // Liseré des arêtes de la boîte au point d'entrée : ancre la « boîte de verre ».
    let entry = ro + rd * hit.x;
    // Distance aux parois NORMALISÉE par les demi-côtés : le domaine n'est plus
    // un cube, donc y ne se mesure plus contre 0,5.
    // Sans cette normalisation, TOUT point au-dessus de l'ancien plafond du cube
    // a |y| > 0,5, le liseré sature, et il peint la partie haute entière — un
    // bloc gris uniforme qu'on lit à tort comme « la simulation remplit la
    // boîte ». Le solveur n'y était pour rien.
    let half = vec3f(0.5, max(R.soot.w, 1e-3) * 0.5, 0.5);
    let e = abs(entry - vec3f(0.0, -0.5 + half.y, 0.0)) / half * 0.5;
    let edge = smoothstep(0.485, 0.499, max(min(e.x, e.y), min(max(e.x, e.y), e.z)));
    acc += vec3f(0.10, 0.11, 0.14) * edge * 0.5 * select(1.0, 0.0, is_outdoor());

    let ldir = select(R.light.xyz, sun_dir(), is_outdoor());
    let shadow_step = 0.5 / 6.0;
    // Phase directionnelle : constante par rayon (mu = angle vue/lumière).
    let phase = phase_hg(dot(rd, ldir));
    // PLAFOND DUR de la boucle. Il doit rester ≥ au maximum du curseur « pas de
    // marche » (main3d.ts) : au-delà, le rayon s'arrête AVANT la sortie de boîte
    // et tout ce qui est derrière disparaît — pas d'erreur, juste un nuage qui
    // s'éclaircit, ce qui se lit à tort comme un problème de rendu. La sortie se
    // fait normalement par `t >= t_end` ou par extinction, donc ce plafond ne
    // coûte rien tant qu'on ne le touche pas.
    for (var i = 0u; i < 1024u; i++) {
      if (t >= t_end || transmit < 0.015) {
        break;
      }
      let pos = ro + rd * t;
      let s = textureSampleLevel(density_tex, lin, tex_uvw(pos), 0.0);
      // Suie : lue au même point, elle s'ajoute à l'extinction et TIRE L'ALBÉDO
      // VERS LE NOIR. Le pas adaptatif ci-dessous doit en tenir compte, sinon un
      // nuage de suie sans fumée serait traversé à double enjambée.
      let soot = soot_at(pos) * R.soot.z;
      let ext = extinction(s) + soot * SOOT_EXT;
      // Pas ADAPTATIF : l'air vide se traverse à double enjambée (gros gain quand
      // la boîte est peu remplie), les intégrales utilisent la longueur réelle.
      var adv = step_len;
      if (ext <= 0.005) {
        adv = step_len * 2.0;
      }
      // Retour visuel du souffle : fuseau lumineux le long du rayon du geste —
      // visible même dans l'air vide, pour voir OÙ le souffle agit.
      if (R.blow_b.w > 0.5) {
        let bv = pos - R.blow_a.xyz;
        let bt = dot(bv, R.blow_b.xyz);
        if (bt > 0.0) {
          let bd = distance(pos, R.blow_a.xyz + R.blow_b.xyz * bt) / max(R.blow_a.w, 1e-4);
          acc += transmit * exp(-bd * bd * 2.0) * vec3f(0.05, 0.08, 0.13) * adv * 22.0;
        }
      }
      if (ext > 0.005) {
        // Ombre interne : marche courte vers la lumière.
        var occ = 0.0;
        for (var j = 1u; j <= 6u; j++) {
          let sp = pos + ldir * (f32(j) * shadow_step);
          let ss = textureSampleLevel(density_tex, lin, tex_uvw(sp), 0.0);
          // La suie compte AUSSI dans l'ombre interne : c'est elle qui rend un
          // panache d'explosion sombre en son cœur. L'ignorer ici donnerait un
          // nuage noir de face et translucide de dos.
          occ += extinction(ss) + soot_at(sp) * R.soot.z * SOOT_EXT;
        }
        // DEUX octaves d'extinction (« powder ») : la seconde, 4× plus
        // transparente, laisse la lumière pénétrer les cœurs épais — la fumée
        // dense devient lumineuse en profondeur au lieu de virer au noir.
        var shade = 0.62 * exp(-occ * shadow_step * 0.9) + 0.38 * exp(-occ * shadow_step * 0.22);
        // La boule projette son ombre dans la fumée.
        if (sphere_hit(pos, ldir) < 1e8) {
          shade *= 0.25;
        }
        // Mélange des albédos pondéré par l'extinction que chacun apporte : là
        // où la suie domine, elle impose sa noirceur ; là où elle est absente,
        // rien ne change par rapport à avant.
        let soot_ext = soot * SOOT_EXT;
        let albedo = mix(ink_albedo(s), SOOT_ALBEDO, clamp(soot_ext / max(ext, 1e-4), 0.0, 1.0));
        let lo = blackbody(s.w) * 2.2 +
          albedo * (
            R.light.w * phase * (shade * 0.92 + 0.08) * 0.28 +
            // In-scattering de la lueur du feu : les volutes voisines de la
            // flamme baignent dans sa lumière (sans direction — déjà diffusée).
            glow_at(pos)
          ) * min(ext, 3.0);
        let a = 1.0 - exp(-ext * adv);
        acc += transmit * lo * a;
        transmit *= exp(-ext * adv);
      }
      t += adv;
    }
    // Sphère-obstacle : surface mate ardoise, diffuse + lueur de contour —
    // et elle ROUGEOIE au corps noir quand la flamme la lèche.
    if (t_sphere < hit.y && transmit > 0.005) {
      let sp = ro + rd * t_sphere;
      let nrm = normalize(sp - R.sphere.xyz);
      let diff = max(dot(nrm, R.light.xyz), 0.0);
      let rim = pow(1.0 - max(dot(nrm, -rd), 0.0), 3.0);
      let heat_here = textureSampleLevel(density_tex, lin, tex_uvw(sp + nrm * 0.012), 0.0).w;
      var sphere_col = vec3f(0.30, 0.32, 0.38) * (0.30 + 0.70 * diff) + vec3f(0.10) * rim;
      sphere_col += blackbody(heat_here) * 0.9;
      // La lueur du feu éclaire aussi la boule (lumière de zone diffusée).
      sphere_col += glow_at(sp) * 0.7;
      acc += transmit * sphere_col;
      transmit = 0.0;
    }
    col = acc + transmit * bg;
  }

  // Sortie HDR LINÉAIRE : tone-mapping et gamma dans post3d.wgsl, après bloom.
  return vec4f(col, 1.0);
}
