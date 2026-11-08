import { describe, expect, it } from "vitest";
import { EMOTIONS, type Prediction } from "./api";
import { GLOWS, announcement, glowLevel, percent, ranked } from "./view";
import { micErrorMessage, pickMimeType } from "./recorder";

function pred(p: Partial<Record<(typeof EMOTIONS)[number], number>>): Prediction {
  const probabilities = Object.fromEntries(
    EMOTIONS.map((e) => [e, p[e] ?? 0]),
  ) as Prediction["probabilities"];
  const emotion = EMOTIONS.reduce((a, b) => (probabilities[b] > probabilities[a] ? b : a));
  return { emotion, probabilities, inferenceMs: 1 };
}

describe("glows", () => {
  it("has exactly one glow per emotion, coloured only with Aurora tokens", () => {
    expect(GLOWS.map((g) => g.emotion).sort()).toEqual([...EMOTIONS].sort());
    const tokens = /^var\(--(magenta|orange|hot|violet|cyan)\)$/;
    for (const g of GLOWS) {
      expect(g.c1).toMatch(tokens);
      expect(g.c2).toMatch(tokens);
    }
  });

  it("maps probability to intensity monotonically, within [0.06, 1]", () => {
    expect(glowLevel(0)).toBeCloseTo(0.06);
    expect(glowLevel(1)).toBe(1);
    expect(glowLevel(0.2)).toBeGreaterThan(glowLevel(0.1));
    expect(glowLevel(-1)).toBeCloseTo(0.06);
    expect(glowLevel(7)).toBe(1);
    expect(glowLevel(Number.NaN)).toBeCloseTo(0.06);
  });
});

describe("result text", () => {
  it("ranks highest first and rounds percentages", () => {
    const r = ranked(pred({ calm: 0.6, sad: 0.3, happy: 0.1 }));
    expect(r.slice(0, 3)).toEqual([
      ["calm", 0.6],
      ["sad", 0.3],
      ["happy", 0.1],
    ]);
    expect(r).toHaveLength(8);
    expect(percent(0.6049)).toBe("60%");
  });

  it("announces a runner-up only when it is a real contender", () => {
    expect(announcement(pred({ calm: 0.6, sad: 0.3 }))).toBe(
      "It sounds calm, 60%. Next: sad, 30%.",
    );
    expect(announcement(pred({ angry: 0.9, sad: 0.1 }))).toBe("It sounds angry, 90%.");
  });
});

describe("recorder helpers", () => {
  it("prefers Opus WebM and falls back to Safari's MP4", () => {
    expect(pickMimeType(() => true)).toBe("audio/webm;codecs=opus");
    expect(pickMimeType((t) => t === "audio/mp4")).toBe("audio/mp4");
    expect(pickMimeType(() => false)).toBeUndefined();
  });

  it("explains each microphone failure", () => {
    const err = (name: string) => new DOMException("x", name);
    expect(micErrorMessage(err("NotAllowedError"))).toMatch(/blocked/);
    expect(micErrorMessage(err("NotFoundError"))).toMatch(/No microphone/);
    expect(micErrorMessage(err("NotReadableError"))).toMatch(/busy/);
    expect(micErrorMessage(new Error("?"))).toMatch(/choose an audio file/i);
  });
});
