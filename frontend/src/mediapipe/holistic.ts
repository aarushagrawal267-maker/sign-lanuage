// Browser-side MediaPipe Holistic. Runs entirely locally (WASM served from
// /mediapipe/wasm, model from the local backend) - no frames leave the browser.
import { FilesetResolver, HolisticLandmarker, type HolisticLandmarkerResult, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { RawFrame } from "../api/client";

const STATIC = import.meta.env.VITE_STATIC === "1";
export const MODEL_URL = STATIC ? `${import.meta.env.BASE_URL}static-data/holistic_landmarker.task` : "/assets/mediapipe/holistic_landmarker.task";
export const WASM_BASE = `${import.meta.env.BASE_URL}mediapipe/wasm`;

export type DelegatePref = "auto" | "GPU" | "CPU";
let landmarkerPromise: Promise<{ landmarker: HolisticLandmarker; delegate: "GPU" | "CPU" }> | null = null;
let loadedPref: DelegatePref | null = null;

export function loadHolistic(pref: DelegatePref = "auto"): Promise<{ landmarker: HolisticLandmarker; delegate: "GPU" | "CPU" }> {
  if (landmarkerPromise && loadedPref === pref) return landmarkerPromise;
  if (landmarkerPromise) landmarkerPromise.then(({ landmarker }) => landmarker.close()).catch(() => {});
  loadedPref = pref;
  landmarkerPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    const opts = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO" as const,
      minPoseDetectionConfidence: 0.5,
      minPoseLandmarksConfidence: 0.5,
      minHandLandmarksConfidence: 0.5,
      minFaceDetectionConfidence: 0.5,
    });
    if (pref === "CPU" || (pref === "auto" && softwareWebGL())) return { landmarker: await HolisticLandmarker.createFromOptions(fileset, opts("CPU")), delegate: "CPU" as const };
    try {
      return { landmarker: await HolisticLandmarker.createFromOptions(fileset, opts("GPU")), delegate: "GPU" as const };
    } catch (gpuErr) {
      console.warn("MediaPipe GPU delegate unavailable, falling back to CPU", gpuErr);
      return { landmarker: await HolisticLandmarker.createFromOptions(fileset, opts("CPU")), delegate: "CPU" as const };
    }
  })();
  landmarkerPromise.catch(() => {
    landmarkerPromise = null; // allow retry
  });
  return landmarkerPromise;
}

/** True when WebGL is rendered in software (no real GPU) - the GPU delegate is then far slower than WASM CPU. */
export function softwareWebGL(): boolean {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") ?? document.createElement("canvas").getContext("webgl");
    if (!gl) return true;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    return /swiftshader|llvmpipe|software|basic render/i.test(renderer);
  } catch {
    return false;
  }
}

const first = (a: NormalizedLandmark[][] | undefined) => (a && a.length ? a[0] : null);

/** Convert a Holistic result into the raw frame layout the backend expects (see /api/features/spec). */
export function toRawFrame(res: HolisticLandmarkerResult, t: number, faceIdx: number[]): RawFrame {
  const pose = first(res.poseLandmarks);
  const left = first(res.leftHandLandmarks);
  const right = first(res.rightHandLandmarks);
  const face = first(res.faceLandmarks);
  return {
    t,
    pose: pose && pose.length === 33 ? pose.map((p) => [p.x, p.y, p.z, p.visibility ?? 1]) : null,
    left: left && left.length === 21 ? left.map((p) => [p.x, p.y, p.z]) : null,
    right: right && right.length === 21 ? right.map((p) => [p.x, p.y, p.z]) : null,
    face: face && face.length > Math.max(...faceIdx) ? faceIdx.map((i) => [face[i].x, face[i].y, face[i].z]) : null,
  };
}

/** Same subset as backend app/features/spec.py FACE_IDX; replaced at runtime by /api/features/spec. */
export const DEFAULT_FACE_IDX = [1, 152, 234, 454, 61, 291, 0, 17, 13, 14, 70, 105, 107, 336, 334, 300, 33, 133, 362, 263];
