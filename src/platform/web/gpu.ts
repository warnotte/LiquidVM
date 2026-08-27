/**
 * Acquisition adapter/device — spécifique navigateur.
 * Équivalent natif : wgpuInstanceRequestAdapter / wgpuAdapterRequestDevice (Dawn/wgpu).
 */

export class WebGPUUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGPUUnavailableError';
  }
}

export interface GpuInit {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
}

export async function acquireDevice(): Promise<GpuInit> {
  if (!('gpu' in navigator) || navigator.gpu === undefined) {
    throw new WebGPUUnavailableError(
      "WebGPU n'est pas disponible dans ce navigateur. " +
        'Utilisez un Chrome/Edge récent (ou Firefox/Safari avec WebGPU activé).',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (adapter === null) {
    throw new WebGPUUnavailableError(
      "Aucun adapter WebGPU n'a été accordé (GPU non supporté ou accès refusé).",
    );
  }
  // Features de base uniquement — aucune feature optionnelle demandée, par portabilité.
  // Deux limites relevées au maximum de l'adapter. Relever une limite n'alloue
  // rien par soi-même — c'est un plafond, pas une réservation.
  //  - maxBufferSize : le défaut (256 Mio) plafonne les staging buffers INTERNES
  //    de Dawn (zéro-init des textures 3D : une rgba16float 384³ = 432 Mo →
  //    chaque submit échouait EN SILENCE, écran noir) et le readback VDB.
  //  - maxStorageBufferBindingSize : le défaut (128 Mio) plafonne un BINDING, pas
  //    l'allocation. Le buffer de particules de l'eau vaut n³ × 64 o, soit
  //    exactement 128 Mio à 128³ (ça passait à zéro octet près) et 250 Mo à
  //    160³ — d'où l'échec de validation qui a révélé la limite.
  const device = await adapter.requestDevice({
    label: 'liquidvm-device',
    // timestamp-query : demandée si l'adapter l'offre — elle ne coûte rien tant
    // qu'aucun QuerySet n'existe, et c'est elle qui permet le profil GPU par
    // passe (?profile sur la page 3D). Les chiffres du chantier de perf sortent
    // de là, pas d'imprécisions rAF.
    requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  return { device, format: navigator.gpu.getPreferredCanvasFormat() };
}
