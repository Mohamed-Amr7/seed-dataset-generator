/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'
import { defineConfig, type PluginOption } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The app is served two ways:
//   * dev   -> Vite dev server (:5173), proxying /api + /ws to the FastAPI backend (:8000)
//   * prod  -> built into the Python package's static/ dir and served by FastAPI from
//              `/`, with hashed assets under `/static/` (hence base='/static/').
export default defineConfig(({ mode }) => {
  const plugins: PluginOption[] = [react()]
  // The Tailwind plugin only matters for the real build/dev server; skip it under Vitest
  // so component tests don't pay for (or trip over) CSS processing.
  if (!process.env.VITEST) plugins.push(tailwindcss())

  return {
    base: mode === 'production' ? '/static/' : '/',
    plugins,
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    build: {
      // emit straight into the FastAPI package; emptyOutDir replaces the old vanilla UI.
      outDir: '../multiseedgen/web/static',
      emptyOutDir: true,
    },
    server: {
      proxy: {
        '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
        '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
      },
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      include: ['src/**/*.test.{ts,tsx}'],
    },
  }
})
