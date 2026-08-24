import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [svelte()],
  // Relative, because the page is served from mozo's own static directory
  base: './',
  build: {
    outDir: '../mozo/workflow/static',
    emptyOutDir: true,
    // Let Vite auto-generate asset paths - no manual editing needed
    rollupOptions: {
      output: {
        // Assets will be in static/assets/ with hashed names
        // Vite automatically updates references in index.html
      }
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});