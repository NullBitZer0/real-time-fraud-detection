import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/demo': {
        target: 'http://localhost:8888',
      },
      '/predict': {
        target: 'http://localhost:8888',
      },
      '/health': {
        target: 'http://localhost:8888',
      },
      '/readyz': {
        target: 'http://localhost:8888',
      },
      '/metrics': {
        target: 'http://localhost:8888',
      },
      '/audit': {
        target: 'http://localhost:8888',
      },
      '/static': {
        target: 'http://localhost:8888',
      },
      '/ws': {
        target: 'ws://localhost:8888',
        ws: true,
      },
    },
  },
})
