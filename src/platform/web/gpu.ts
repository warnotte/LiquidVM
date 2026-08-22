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
  // maxBufferSize : le défaut (256 Mio) plafonne les staging buffers INTERNES de
  // Dawn (zéro-init des textures 3D : une rgba16float 384³ = 432 Mo → chaque
  // submit échouait en silence, écran noir) et le readback de l'export VDB. On
  // demande le maximum de l'adapter — toujours valide par construction, et
  // relever une limite n'alloue rien par soi-même.
  const device = await adapter.requestDevice({
    label: 'liquidvm-device',
    requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize },
  });
  return { device, format: navigator.gpu.getPreferredCanvasFormat() };
}
