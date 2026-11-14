export type LiveStatus =
  | "camera_inactive"
  | "camera_starting"
  | "camera_error"
  | "mediapipe_loading"
  | "mediapipe_error"
  | "preview"
  | "paused"
  | "connecting"
  | "unavailable"
  | "no_body"
  | "waiting_for_hands"
  | "collecting"
  | "analyzing"
  | "recognized"
  | "uncertain";

type Meta = { label: string; tone: "idle" | "active" | "busy" | "good" | "warn" | "error"; help: string };

export const STATUS: Record<LiveStatus, Meta> = {
  camera_inactive: { label: "Camera inactive", tone: "idle", help: "Start the camera to begin. Nothing is captured until you do." },
  camera_starting: { label: "Requesting camera", tone: "busy", help: "Allow camera access in the browser prompt." },
  camera_error: { label: "Camera unavailable", tone: "error", help: "Camera access was denied or no camera was found." },
  mediapipe_loading: { label: "Initialising MediaPipe", tone: "busy", help: "Loading the Holistic landmark model locally in your browser." },
  mediapipe_error: { label: "MediaPipe failed to load", tone: "error", help: "Run setup.ps1 so the model file is available, then reload." },
  preview: { label: "Detecting landmarks", tone: "active", help: "Landmarks are tracked locally. Start recognition to classify signs." },
  paused: { label: "Recognition paused", tone: "idle", help: "Landmark tracking continues; no frames are sent for classification." },
  connecting: { label: "Connecting to model", tone: "busy", help: "Opening a local connection to the recognition server." },
  unavailable: { label: "Recognition unavailable", tone: "error", help: "No active model, or the backend is not reachable." },
  no_body: { label: "Body not visible", tone: "warn", help: "Sit so your head and shoulders are in frame." },
  waiting_for_hands: { label: "Waiting for hands", tone: "active", help: "Raise your hands into view to begin a sign; lower them when finished." },
  collecting: { label: "Collecting sequence", tone: "busy", help: "Recording the movement of the current sign." },
  analyzing: { label: "Model inference", tone: "busy", help: "Classifying the sign so far. Lower your hands to finish." },
  recognized: { label: "Recognised sign", tone: "good", help: "The sign passed confidence, margin and agreement checks." },
  uncertain: { label: "Uncertain sign", tone: "warn", help: "Not confident enough to output text. Try again, or the sign may be outside the vocabulary." },
};

export const TONE_CLASS: Record<Meta["tone"], string> = {
  idle: "border-ink-600 bg-ink-800/90 text-fg-2",
  active: "border-accent/50 bg-[#0f1d36]/90 text-[#a9c8ff]",
  busy: "border-accent/50 bg-[#0f1d36]/90 text-[#a9c8ff]",
  good: "border-teal/60 bg-[#0b2b28]/90 text-teal",
  warn: "border-warn/50 bg-[#2b2210]/90 text-warn",
  error: "border-danger/50 bg-[#2d1316]/90 text-danger",
};

export const REJECT_REASON: Record<string, string> = {
  low_confidence: "confidence below threshold",
  ambiguous: "too close to the runner-up",
  unstable: "temporal crops disagree",
  cooldown: "inside cooldown window",
  repeat_suppressed: "same sign repeated",
  too_short: "too few frames with hands",
  ok: "accepted",
};
