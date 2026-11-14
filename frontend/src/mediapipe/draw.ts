import type { RawFrame } from "../api/client";

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

export const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [0, 2], [0, 5], [2, 7], [5, 8], [9, 10],
];

export const COLORS = {
  left: "#2dd4bf", // teal
  right: "#4c8dff", // electric blue
  pose: "rgba(232,235,239,0.55)",
  face: "rgba(250,178,25,0.85)",
};

export type DrawOptions = { pose: boolean; hands: boolean; face: boolean; mirror: boolean; trail?: { x: number; y: number; side: "left" | "right" }[] };

/** Draw a raw frame onto a canvas sized to the displayed video. */
export function drawFrame(ctx: CanvasRenderingContext2D, f: RawFrame | null, w: number, h: number, o: DrawOptions) {
  ctx.clearRect(0, 0, w, h);
  if (!f) return;
  const X = (x: number) => (o.mirror ? (1 - x) * w : x * w);
  const Y = (y: number) => y * h;
  const lw = Math.max(1.5, w / 480);

  if (o.trail && o.trail.length > 1) {
    for (const side of ["left", "right"] as const) {
      const pts = o.trail.filter((p) => p.side === side);
      for (let i = 1; i < pts.length; i++) {
        ctx.strokeStyle = side === "left" ? COLORS.left : COLORS.right;
        ctx.globalAlpha = (i / pts.length) * 0.6;
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.moveTo(X(pts[i - 1].x), Y(pts[i - 1].y));
        ctx.lineTo(X(pts[i].x), Y(pts[i].y));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  if (o.pose && f.pose) {
    ctx.strokeStyle = COLORS.pose;
    ctx.lineWidth = lw;
    for (const [a, b] of POSE_CONNECTIONS) {
      const p = f.pose[a], q = f.pose[b];
      if ((p[3] ?? 1) < 0.3 || (q[3] ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(X(p[0]), Y(p[1]));
      ctx.lineTo(X(q[0]), Y(q[1]));
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(232,235,239,0.9)";
    for (const i of [0, 11, 12, 13, 14, 15, 16]) {
      const p = f.pose[i];
      if ((p[3] ?? 1) < 0.3) continue;
      ctx.beginPath();
      ctx.arc(X(p[0]), Y(p[1]), lw * 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (o.face && f.face) {
    ctx.fillStyle = COLORS.face;
    for (const p of f.face) {
      ctx.beginPath();
      ctx.arc(X(p[0]), Y(p[1]), lw * 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (o.hands) {
    for (const side of ["left", "right"] as const) {
      const hand = f[side];
      if (!hand) continue;
      ctx.strokeStyle = COLORS[side];
      ctx.lineWidth = lw * 1.3;
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(X(hand[a][0]), Y(hand[a][1]));
        ctx.lineTo(X(hand[b][0]), Y(hand[b][1]));
        ctx.stroke();
      }
      for (let i = 0; i < hand.length; i++) {
        ctx.fillStyle = i === 0 ? "#ffffff" : COLORS[side];
        ctx.beginPath();
        ctx.arc(X(hand[i][0]), Y(hand[i][1]), lw * (i === 0 ? 2.2 : 1.5), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
