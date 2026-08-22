import { defineConfig } from 'vite';

// Configuration minimale : pas de plugin nécessaire.
// Les shaders WGSL sont importés comme texte via le suffixe `?raw` (natif Vite).
// Deux pages : l'appli 2D (index.html) et le prototype 3D volumétrique (3d.html).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      // Chemins relatifs à la racine du projet (résolus par Vite/Rollup).
      input: {
        main: 'index.html',
        volume: '3d.html',
        eau: 'eau.html',
      },
    },
  },
});
