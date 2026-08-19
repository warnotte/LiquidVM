import { defineConfig } from 'vite';

// Configuration minimale : pas de plugin nécessaire.
// Les shaders WGSL sont importés comme texte via le suffixe `?raw` (natif Vite).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
