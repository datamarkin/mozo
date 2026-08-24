import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [svelte()],
  // Absolute, and matching the route the page is served at. Relative would resolve against
  // `/workflow` -- which has no trailing slash, so the browser drops the last segment and asks for
  // `/assets/...`, which nothing serves. A blank page with one 404 in the console.
  base: '/workflow/',
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