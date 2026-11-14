import { defineConfig } from "@playwright/test";

// Expects backend (:8000) and frontend dev server (:5173) to be running
// (start.ps1, or scripts/e2e_env.sh in CI). The fake camera plays E2E_FAKE_VIDEO
// (a .y4m file) when provided, otherwise Chromium's synthetic test pattern.
const fakeVideo = process.env.E2E_FAKE_VIDEO;

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:5173",
    trace: "off",
    permissions: ["camera"],
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        ...(fakeVideo ? [`--use-file-for-fake-video-capture=${fakeVideo}`] : []),
        "--enable-unsafe-swiftshader",
      ],
    },
  },
  reporter: [["list"]],
});
