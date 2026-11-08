import { describe, expect, it } from "vitest";
import {
  N_SAMPLES,
  isSilent,
  mixToMono,
  resample,
  toBody,
  toModelWindow,
  trimBounds,
} from "./audio";
import fixture from "./trim-fixture.json";

type Segment = [number, number, number];

function build(segments: Segment[], sr = 16_000): Float32Array {
  const parts = segments.map(([n, amp, freq]) =>
    Float32Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / sr)),
  );
  const out = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function tone(n: number, freq: number, sr: number, amp = 0.5): Float32Array {
  return Float32Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * freq * i) / sr));
}

function rms(x: Float32Array, from = 0, to = x.length): number {
  let s = 0;
  for (let i = from; i < to; i++) s += (x[i] ?? 0) ** 2;
  return Math.sqrt(s / (to - from));
}

describe("trimBounds matches src/vemo/trim.py", () => {
  for (const [name, c] of Object.entries(fixture)) {
    it(name, () => {
      expect(trimBounds(build(c.segments as Segment[]))).toEqual(c.bounds);
    });
  }

  it("returns (0, 0) for empty input", () => {
    expect(trimBounds(new Float32Array(0))).toEqual([0, 0]);
  });
});

describe("toModelWindow", () => {
  it("is always exactly 3 s", () => {
    for (const n of [0, 100, 30_000, 48_000, 90_000]) {
      expect(toModelWindow(tone(n, 220, 16_000)).length).toBe(N_SAMPLES);
    }
  });

  it("moves speech after leading silence to the start of the window", () => {
    const y = build([
      [24_000, 0, 0],
      [40_000, 0.5, 220],
    ]);
    const w = toModelWindow(y);
    // librosa keeps up to one frame (1024 samples) of lead-in; speech follows, not 1.5 s of silence.
    expect(rms(w, 1024, 4096)).toBeGreaterThan(0.3);
  });

  it("zero-pads a short clip at the end", () => {
    const w = toModelWindow(build([[8000, 0.5, 220]]));
    expect(rms(w, 0, 8000)).toBeGreaterThan(0.3);
    expect(rms(w, 9000, N_SAMPLES)).toBe(0);
  });
});

describe("mixToMono", () => {
  it("averages channels like librosa.load(mono=True)", () => {
    const m = mixToMono([Float32Array.from([1, 0, -1]), Float32Array.from([0, 1, -1])]);
    expect(Array.from(m)).toEqual([0.5, 0.5, -1]);
  });

  it("passes one channel through and handles none", () => {
    const one = Float32Array.from([0.1, 0.2]);
    expect(mixToMono([one])).toBe(one);
    expect(mixToMono([]).length).toBe(0);
  });
});

describe("resample", () => {
  it("keeps an in-band tone (44.1 kHz -> 16 kHz, 440 Hz)", () => {
    const out = resample(tone(44_100, 440, 44_100), 44_100, 16_000);
    expect(out.length).toBe(16_000);
    const ref = tone(16_000, 440, 16_000);
    let maxErr = 0;
    for (let i = 2000; i < 14_000; i++)
      maxErr = Math.max(maxErr, Math.abs((out[i] ?? 0) - (ref[i] ?? 0)));
    expect(maxErr).toBeLessThan(0.01);
  });

  it("removes content above the new Nyquist instead of aliasing it", () => {
    // 10 kHz would alias to 6 kHz at 16 kHz without a low-pass.
    const out = resample(tone(48_000, 10_000, 48_000), 48_000, 16_000);
    expect(rms(out, 1000, 15_000)).toBeLessThan(0.005);
  });

  it("is the identity at equal rates and rejects bad rates", () => {
    const x = tone(100, 300, 16_000);
    expect(Array.from(resample(x, 16_000, 16_000))).toEqual(Array.from(x));
    expect(() => resample(x, 0, 16_000)).toThrow(RangeError);
    expect(() => resample(x, Number.NaN, 16_000)).toThrow(RangeError);
  });
});

describe("isSilent", () => {
  it("flags a muted mic and a click, not speech", () => {
    expect(isSilent(new Float32Array(48_000))).toBe(true);
    expect(isSilent(tone(48_000, 220, 16_000, 0.0005))).toBe(true);
    expect(
      isSilent(
        build([
          [2000, 0.5, 300],
          [46_000, 0, 0],
        ]),
      ),
    ).toBe(true); // 0.125 s
    expect(
      isSilent(
        build([
          [8000, 0, 0],
          [16_000, 0.3, 220],
        ]),
      ),
    ).toBe(false);
  });
});

describe("toBody", () => {
  it("encodes 48,000 little-endian float32 samples", () => {
    const w = new Float32Array(N_SAMPLES);
    w[0] = 0.5;
    w[N_SAMPLES - 1] = -0.25;
    const view = new DataView(toBody(w));
    expect(view.byteLength).toBe(192_000);
    expect(view.getFloat32(0, true)).toBe(0.5);
    expect(view.getFloat32((N_SAMPLES - 1) * 4, true)).toBe(-0.25);
  });

  it("rejects the wrong length and non-finite samples", () => {
    expect(() => toBody(new Float32Array(10))).toThrow(RangeError);
    const w = new Float32Array(N_SAMPLES);
    w[5] = Number.NaN;
    expect(() => toBody(w)).toThrow(/NaN/);
  });
});
