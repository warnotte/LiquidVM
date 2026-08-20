/**
 * Capture webcam pour le flux optique : getUserMedia → <video> → OffscreenCanvas
 * carré (drawImage rééchantillonne quelle que soit la résolution de la caméra) →
 * copyExternalImageToTexture vers la texture caméra exposée par le core.
 * Toute la partie navigateur (permission, périphérique) vit ici — le core ne voit
 * qu'une texture qui se remplit.
 */

import { FLOW_SIZE } from '../../core/config';

export class CameraFlow {
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private readonly canvas = new OffscreenCanvas(FLOW_SIZE, FLOW_SIZE);
  private readonly ctx = this.canvas.getContext('2d', { willReadFrequently: false });

  get active(): boolean {
    return this.video !== null;
  }

  async start(): Promise<void> {
    if (this.video) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 320 }, height: { ideal: 240 } },
      audio: false,
    });
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    this.stream = stream;
    this.video = video;
  }

  stop(): void {
    this.video?.pause();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.video = null;
    this.stream = null;
  }

  /** Copie l'image courante vers la texture caméra (appelé chaque frame si actif). */
  copyFrame(device: GPUDevice, texture: GPUTexture): void {
    if (!this.video || !this.ctx || this.video.readyState < 2) {
      return;
    }
    this.ctx.drawImage(this.video, 0, 0, FLOW_SIZE, FLOW_SIZE);
    device.queue.copyExternalImageToTexture(
      { source: this.canvas },
      { texture },
      [FLOW_SIZE, FLOW_SIZE],
    );
  }
}
