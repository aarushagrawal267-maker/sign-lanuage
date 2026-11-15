/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend = process.env.SLR_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  // Static (browser-only) build is served from a sub-path-agnostic location with hash routing.
  base: process.env.VITE_STATIC === "1" ? "./" : "/",
  plugins: [react(), tailwindcss()],
  server: {
    host: "localhost",
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true, ws: true },
      "/assets/mediapipe": { target: backend, changeOrigin: true },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": { target: backend, changeOrigin: true, ws: true },
      "/assets/mediapipe": { target: backend, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test-setup.ts"],
  },
});
