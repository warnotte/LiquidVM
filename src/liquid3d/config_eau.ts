/**
 * Configuration du chantier EAU (voir PLAN-EAU.md — à lire avant tout code ici).
 * Doctrine identique au reste du moteur : zéro DOM dans liquid3d/, zéro
 * allocation par frame, un CommandEncoder, zéro readback en boucle.
 */

/** Côté de la grille MAC (voxels). 128³ tant que J1–J4 ne sont pas verts. */
export const GRID_EAU = 128;

/** Nombre de particules : 8 par cellule d'eau initiale, ~1/8 de boîte
 *  (64³ cellules × 8 = exactement 2 M). Buffer fixe 32 o/particule = 64 Mo. */
export const PARTICLES_EAU = 2_097_152;

/** Échelle de virgule fixe des accumulations atomiques P2G (i32) :
 *  vitesses ≤ ~500 voxels/s × poids ≤ 1 × 256 → marge i32 très confortable
 *  même avec ~64 particules contribuant à une même face. */
export const FIXED_POINT_SCALE = 256;

/** Workgroup du scatter (1D sur les particules) et des passes de grille (3D). */
export const WG_PARTICLES = 64;
export const WG_GRID = 4;
