/**
 * Encodeur OpenVDB minimal, en TypeScript pur (aucune dépendance, aucun DOM) :
 * écrit un .vdb version 224 contenant des grilles float denses N³ (arbre 5-4-3,
 * sans compression), directement importable dans Blender / Houdini.
 *
 * Le format n'a pas de spécification officielle — cette implémentation suit la
 * rétro-ingénierie publiée par JangaFX (« VDB: a deep dive »,
 * https://jangafx.com/insights/vdb-a-deep-dive) : en-tête magique « ␣BDV »,
 * descripteurs avec trois u64 d'offsets (écrits même quand le flag est à 0),
 * octet 6 = « non compressé » devant chaque tableau de valeurs, u32 constant 1
 * en tête d'arbre. Topologie pour N ≤ 128 : racine → un nœud interne 32³ (un
 * seul enfant actif) → un nœud interne 16³ dense → feuilles 8³ denses.
 *
 * Repère : l'axe Y (vertical simulation) est mappé sur +Z monde (convention
 * Blender, la fumée monte) via la matrice de transformation — rotation, pas
 * de miroir. La boîte est centrée sur l'origine, taille 1.
 */

export interface VdbGrid {
  /** Nom de la grille (« density », « temperature »…). */
  readonly name: string;
  /** Valeurs denses N³, X le plus rapide : index = (z·N + y)·N + x. */
  readonly values: Float32Array;
}

const LEAF_LOG2 = 3; // feuilles 8³
const INTERNAL_LOG2 = 4; // nœud interne bas : 16³ feuilles → 128 voxels par axe

class Writer {
  private buf: Uint8Array<ArrayBuffer>;
  private view: DataView;
  private pos = 0;

  constructor(capacity: number) {
    this.buf = new Uint8Array(capacity);
    this.view = new DataView(this.buf.buffer);
  }

  private ensure(bytes: number): void {
    if (this.pos + bytes > this.buf.length) {
      const grown = new Uint8Array(Math.max(this.buf.length * 2, this.pos + bytes));
      grown.set(this.buf);
      this.buf = grown;
      this.view = new DataView(grown.buffer);
    }
  }

  u8(v: number): void {
    this.ensure(1);
    this.view.setUint8(this.pos, v);
    this.pos += 1;
  }
  u32(v: number): void {
    this.ensure(4);
    this.view.setUint32(this.pos, v, true);
    this.pos += 4;
  }
  i32(v: number): void {
    this.ensure(4);
    this.view.setInt32(this.pos, v, true);
    this.pos += 4;
  }
  u64(v: number): void {
    this.ensure(8);
    this.view.setBigUint64(this.pos, BigInt(v), true);
    this.pos += 8;
  }
  f32(v: number): void {
    this.ensure(4);
    this.view.setFloat32(this.pos, v, true);
    this.pos += 4;
  }
  f64(v: number): void {
    this.ensure(8);
    this.view.setFloat64(this.pos, v, true);
    this.pos += 8;
  }
  bytes(b: Uint8Array): void {
    this.ensure(b.length);
    this.buf.set(b, this.pos);
    this.pos += b.length;
  }
  zeros(count: number): void {
    this.ensure(count);
    this.pos += count; // le buffer est déjà zéro-initialisé (ou étendu à zéro)
  }
  /** Chaîne préfixée par sa longueur (u32), sans terminateur. */
  str(s: string): void {
    const b = new TextEncoder().encode(s);
    this.u32(b.length);
    this.bytes(b);
  }
  get offset(): number {
    return this.pos;
  }
  finish(): Uint8Array<ArrayBuffer> {
    return this.buf.subarray(0, this.pos);
  }
}

/** Métadonnée de grille : nom, type, valeur (taille u32 + octets bruts). */
function meta(w: Writer, name: string, type: 'string' | 'bool', value: string | boolean): void {
  w.str(name);
  w.str(type);
  if (type === 'string') {
    w.str(value as string);
  } else {
    w.u32(1);
    w.u8(value ? 1 : 0);
  }
}

/**
 * Encode un fichier .vdb contenant les grilles float denses fournies (côté n ≤ 128,
 * puissance de deux). `voxelSize` en unités monde (1/n → boîte de taille 1).
 */
export function encodeVdb(
  grids: readonly VdbGrid[],
  n: number,
  voxelSize: number,
): Uint8Array<ArrayBuffer> {
  const leavesPerAxis = n >> LEAF_LOG2;
  if (leavesPerAxis > 1 << INTERNAL_LOG2) {
    throw new Error(`encodeVdb : n=${n} dépasse un nœud interne 16³ de feuilles (128 max)`);
  }
  const leafCount = leavesPerAxis ** 3;
  const estimate = 4096 + grids.length * (2048 + 32768 * 4 + 4096 * 4 + leafCount * (17 + 512 * 4));
  const w = new Writer(estimate);

  // En-tête du fichier.
  w.bytes(new Uint8Array([0x20, 0x42, 0x44, 0x56, 0, 0, 0, 0])); // « ␣BDV » + padding
  w.u32(224); // version du format
  w.u32(8); // bibliothèque majeure
  w.u32(1); // bibliothèque mineure
  w.u8(0); // offsets de grille désactivés
  // UUID : 36 caractères ASCII BRUTS, sans préfixe de longueur (vérifié dans les
  // lecteurs — un préfixe décale tout le fichier de 4 octets et le rend illisible).
  w.bytes(new TextEncoder().encode('4c696b75-6964-564d-2033-442065787021'));
  w.u32(0); // métadonnées de fichier : aucune
  w.u32(grids.length);

  for (const grid of grids) {
    // Descripteur.
    w.str(grid.name);
    w.str('Tree_float_5_4_3');
    w.str(''); // pas de parent d'instance
    w.u64(w.offset + 24); // position du contenu (juste après les trois u64)
    w.u64(0);
    w.u64(0);
    w.u32(0); // compression du flux : aucune

    // Métadonnées de la grille.
    w.u32(4);
    meta(w, 'class', 'string', 'unknown');
    meta(w, 'file_compression', 'string', 'none');
    meta(w, 'is_saved_as_half_float', 'bool', false);
    meta(w, 'name', 'string', grid.name);

    // Transformation : Y simulation (vertical) → +Z monde, boîte centrée, taille 1.
    // Vecteurs-lignes (p' = p·M) : ligne 3 = translation.
    w.str('AffineMap');
    const s = voxelSize;
    const m = [s, 0, 0, 0, /**/ 0, 0, s, 0, /**/ 0, -s, 0, 0, /**/ -0.5, 0.5, -0.5, 1];
    for (const v of m) {
      w.f64(v);
    }

    // Arbre : topologie.
    w.u32(1); // constante (cf. article JangaFX)
    w.f32(0); // valeur de fond
    w.u32(0); // tuiles racine
    w.u32(1); // enfants racine
    w.i32(0);
    w.i32(0);
    w.i32(0); // origine du nœud 32³

    // Nœud interne 32³ : seul l'enfant (0,0,0) est actif (bit 0).
    // Masques de 32768 bits = 4096 octets chacun (512 mots u64).
    w.u8(1);
    w.zeros(4095); // masque d'enfants : bit 0
    w.zeros(4096); // masque de valeurs : vide
    w.u8(6); // non compressé
    w.zeros(32768 * 4); // tuiles f32 (toutes inactives)

    // Nœud interne 16³ : les leavesPerAxis³ premières feuilles actives.
    // Indexation des bits : z | y<<4 | x<<8.
    const childMask = new Uint8Array(512);
    for (let x = 0; x < leavesPerAxis; x++) {
      for (let y = 0; y < leavesPerAxis; y++) {
        for (let z = 0; z < leavesPerAxis; z++) {
          const bit = z | (y << 4) | (x << 8);
          childMask[bit >> 3]! |= 1 << (bit & 7);
        }
      }
    }
    w.bytes(childMask.subarray(0, 512));
    w.zeros(512); // masque de valeurs : vide
    w.u8(6);
    w.zeros(4096 * 4); // tuiles f32

    // Feuilles (topologie) : masque de valeurs plein (512 bits = 64 octets = 8 u64),
    // par index de bit croissant.
    const fullMask = new Uint8Array(64).fill(0xff);
    for (let i = 0; i < leafCount; i++) {
      w.bytes(fullMask);
    }

    // Données : pour chaque feuille (même ordre), masque + octet 6 + 512 f32.
    // Ordre des voxels dans la feuille : index = vx<<6 | vy<<3 | vz.
    for (let bit = 0; bit < 4096; bit++) {
      const lx = bit >> 8;
      const ly = (bit >> 4) & 15;
      const lz = bit & 15;
      if (lx >= leavesPerAxis || ly >= leavesPerAxis || lz >= leavesPerAxis) {
        continue;
      }
      w.bytes(fullMask);
      w.u8(6);
      const ox = lx << LEAF_LOG2;
      const oy = ly << LEAF_LOG2;
      const oz = lz << LEAF_LOG2;
      for (let vx = 0; vx < 8; vx++) {
        for (let vy = 0; vy < 8; vy++) {
          for (let vz = 0; vz < 8; vz++) {
            w.f32(grid.values[((oz + vz) * n + (oy + vy)) * n + (ox + vx)]!);
          }
        }
      }
    }
  }
  return w.finish();
}
