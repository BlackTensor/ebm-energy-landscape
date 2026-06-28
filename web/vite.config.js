import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base so the built bundle works under any GitHub Pages path
// (user site or project subpath) without rewrites. See Phase 7.1.
// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
})
