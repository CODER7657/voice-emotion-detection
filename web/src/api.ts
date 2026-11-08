/** Client for POST /predict (same origin in production). */

export const EMOTIONS = [
  "angry",
  "calm",
  "disgust",
  "fearful",
  "happy",
  "neutral",
  "sad",
  "surprised",
] as const;
export type Emotion = (typeof EMOTIONS)[number];

export interface Prediction {
  emotion: Emotion;
  probabilities: Record<Emotion, number>;
  inferenceMs: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterS?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const TIMEOUT_MS = 30_000; // a cold start loads the model (about 7 s on Vercel)

function isEmotion(v: unknown): v is Emotion {
  return typeof v === "string" && (EMOTIONS as readonly string[]).includes(v);
}

/** Validate the response instead of trusting it: the UI renders these numbers. */
export function parsePrediction(json: unknown): Prediction {
  if (typeof json !== "object" || json === null) throw new ApiError("Unexpected response.", 502);
  const { emotion, probabilities, inference_ms } = json as Record<string, unknown>;
  if (!isEmotion(emotion) || typeof probabilities !== "object" || probabilities === null) {
    throw new ApiError("Unexpected response.", 502);
  }
  const probs = {} as Record<Emotion, number>;
  for (const e of EMOTIONS) {
    const p = (probabilities as Record<string, unknown>)[e];
    if (typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) {
      throw new ApiError("Unexpected response.", 502);
    }
    probs[e] = p;
  }
  return {
    emotion,
    probabilities: probs,
    inferenceMs: typeof inference_ms === "number" ? inference_ms : NaN,
  };
}

function message(status: number, retryAfterS?: number): string {
  if (status === 429) {
    return `Too many tries. Wait ${String(retryAfterS ?? 60)} s and record again.`;
  }
  if (status === 503) return "The model is starting or unavailable. Try again in a moment.";
  if (status >= 500) return "The server had a problem. Try again.";
  return "That clip could not be analysed. Record again.";
}

export async function predict(
  body: ArrayBuffer,
  fetchImpl: typeof fetch = fetch,
  timeoutMs = TIMEOUT_MS,
): Promise<Prediction> {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);
  let res: Response;
  try {
    res = await fetchImpl("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
      signal: controller.signal,
    });
  } catch {
    throw new ApiError(
      controller.signal.aborted
        ? "The server took too long. Try again."
        : "You appear to be offline.",
      0,
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    const header = Number(res.headers.get("Retry-After"));
    const retry = Number.isFinite(header) && header > 0 ? Math.ceil(header) : undefined;
    throw new ApiError(message(res.status, retry), res.status, retry);
  }
  let json: unknown;
  try {
    json = await res.json();
  } catch {
    throw new ApiError("Unexpected response.", 502);
  }
  return parsePrediction(json);
}
