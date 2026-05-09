import { defineConfig } from 'vite'

export default defineConfig(async () => {
  const reactPlugin = (await import('@vitejs/plugin-react')).default
  return {
    base: process.env.NODE_ENV === 'production' ? '/blindhelp/' : '/',
    plugins: [reactPlugin()],
    server: { port: 5173 }
  }
})
