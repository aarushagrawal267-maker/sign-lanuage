// Copies the MediaPipe Tasks Vision WASM runtime into public/ so it is served
// locally (no CDN at runtime).
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const dst = join(root, "public", "mediapipe", "wasm");
if (!existsSync(src)) {
  console.warn("[copy-mediapipe-wasm] @mediapipe/tasks-vision not installed yet; skipping");
} else {
  mkdirSync(dst, { recursive: true });
  cpSync(src, dst, { recursive: true });
  console.log("[copy-mediapipe-wasm] copied WASM runtime to public/mediapipe/wasm");
}

// onnxruntime-web WASM runtime for the static (browser-only) build.
const ortSrc = join(root, "node_modules", "onnxruntime-web", "dist");
if (existsSync(ortSrc)) {
  const ortDst = join(root, "public", "ort");
  mkdirSync(ortDst, { recursive: true });
  for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) cpSync(join(ortSrc, f), join(ortDst, f));
  console.log("[copy-mediapipe-wasm] copied onnxruntime-web WASM to public/ort");
}
