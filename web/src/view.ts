/** Pure view helpers: glow layout and result formatting (unit-tested). */

import { EMOTIONS, type Emotion, type Prediction } from "./api";

export interface GlowSpec {
  emotion: Emotion;
  c1: string;
  c2: string;
  x: number; // % of the glow field
  y: number;
  size: string;
  period: string;
}

// Two Aurora tokens per emotion; positions spread the field so no two neighbours share a hue.
export const GLOWS: readonly GlowSpec[] = [
  {
    emotion: "angry",
    c1: "var(--magenta)",
    c2: "var(--magenta)",
    x: 14,
    y: 46,
    size: "36vmax",
    period: "17s",
  },
  {
    emotion: "calm",
    c1: "var(--cyan)",
    c2: "var(--violet)",
    x: 82,
    y: 40,
    size: "34vmax",
    period: "21s",
  },
  {
    emotion: "disgust",
    c1: "var(--orange)",
    c2: "var(--magenta)",
    x: 30,
    y: 70,
    size: "30vmax",
    period: "19s",
  },
  {
    emotion: "fearful",
    c1: "var(--violet)",
    c2: "var(--cyan)",
    x: 64,
    y: 72,
    size: "32vmax",
    period: "23s",
  },
  {
    emotion: "happy",
    c1: "var(--orange)",
    c2: "var(--orange)",
    x: 48,
    y: 52,
    size: "36vmax",
    period: "16s",
  },
  {
    emotion: "neutral",
    c1: "var(--hot)",
    c2: "var(--violet)",
    x: 90,
    y: 78,
    size: "26vmax",
    period: "25s",
  },
  {
    emotion: "sad",
    c1: "var(--violet)",
    c2: "var(--violet)",
    x: 6,
    y: 80,
    size: "32vmax",
    period: "20s",
  },
  {
    emotion: "surprised",
    c1: "var(--hot)",
    c2: "var(--orange)",
    x: 58,
    y: 34,
    size: "24vmax",
    period: "18s",
  },
];

export const AMBIENT = 0.45;

/** Glow opacity for a probability: a dim floor so the field never goes fully dark. */
export function glowLevel(p: number): number {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(p) ? p : 0));
  return 0.06 + 0.94 * Math.sqrt(clamped); // sqrt: a 20% second guess still shows
}

export function percent(p: number): string {
  return `${String(Math.round(p * 100))}%`;
}

/** Emotions ordered by probability, highest first (ties in the model's class order). */
export function ranked(pred: Prediction): [Emotion, number][] {
  return EMOTIONS.map((e) => [e, pred.probabilities[e]] as [Emotion, number]).sort(
    (a, b) => b[1] - a[1],
  );
}

export function announcement(pred: Prediction): string {
  const [first, second] = ranked(pred);
  if (!first) return "";
  const lead = `It sounds ${first[0]}, ${percent(first[1])}.`;
  return second && second[1] >= 0.15 ? `${lead} Next: ${second[0]}, ${percent(second[1])}.` : lead;
}
