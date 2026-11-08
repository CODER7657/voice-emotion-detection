/**
 * Browser side of the model's input contract: any recording or file becomes exactly
 * 48,000 float32 samples (3 s, 16 kHz mono), prepared the way the model was trained:
 * mono mix, resample to 16 kHz, trim silence (librosa.effects.trim, top_db 30), then
 * crop or zero-pad to 3 s. The server trims again; trimming here first keeps leading
 * silence from eating the speech inside the 3 s window.
 *
 * Everything except `decodeTo16k` is pure and unit-tested in Node.
 */

export const SAMPLE_RATE = 16_000;
export const N_SAMPLES = 48_000;
export const FRAME_LENGTH = 2048;
export const HOP_LENGTH = 512;
const AMIN = 1e-5;

/** (start, end) of the non-silent span, mirroring src/vemo/trim.py (librosa's arithmetic). */
export function trimBounds(y: Float32Array, topDb = 30): [number, number] {
  if (y.length === 0) return [0, 0];
  const pad = FRAME_LENGTH / 2;
  const nFrames = 1 + Math.floor(y.length / HOP_LENGTH);
  const power = new Float64Array(nFrames);
  let ref = 0;
  for (let f = 0; f < nFrames; f++) {
    // Frame f covers padded[f*hop, f*hop + frame), i.e. y[f*hop - pad, f*hop + pad).
    const lo = f * HOP_LENGTH - pad;
    let sum = 0;
    for (let i = Math.max(0, lo); i < Math.min(y.length, lo + FRAME_LENGTH); i++) {
      const v = y[i] ?? 0;
      sum += v * v;
    }
    const p = sum / FRAME_LENGTH;
    power[f] = p;
    if (p > ref) ref = p;
  }
  const amin = AMIN * AMIN;
  const refDb = 10 * Math.log10(Math.max(amin, ref));
  let first = -1;
  let last = -1;
  for (let f = 0; f < nFrames; f++) {
    if (10 * Math.log10(Math.max(amin, power[f] ?? 0)) - refDb > -topDb) {
      if (first < 0) first = f;
      last = f;
    }
  }
  if (first < 0) return [0, 0];
  return [first * HOP_LENGTH, Math.min(y.length, (last + 1) * HOP_LENGTH)];
}

/** Trim, then exactly 3 s: the model's input window. */
export function toModelWindow(y16k: Float32Array): Float32Array {
  const [start, end] = trimBounds(y16k);
  const out = new Float32Array(N_SAMPLES); // zero-padded
  out.set(y16k.subarray(start, Math.min(end, start + N_SAMPLES)));
  return out;
}

/** Mean of the channels, as librosa.load(mono=True) does. */
export function mixToMono(channels: readonly Float32Array[]): Float32Array {
  const first = channels[0];
  if (!first) return new Float32Array(0);
  if (channels.length === 1) return first;
  const out = new Float32Array(first.length);
  for (const ch of channels)
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (ch[i] ?? 0);
  for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / channels.length;
  return out;
}

function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 50; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

/**
 * Band-limited resampling with a Kaiser-windowed sinc (beta 8.6, 32 zero crossings).
 * Used when the browser cannot render an OfflineAudioContext at 16 kHz.
 */
export function resample(x: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate <= 0 || toRate <= 0 || !Number.isFinite(fromRate) || !Number.isFinite(toRate)) {
    throw new RangeError(`invalid sample rates ${String(fromRate)} -> ${String(toRate)}`);
  }
  if (fromRate === toRate) return x.slice();
  const ratio = toRate / fromRate;
  const cutoff = 0.95 * Math.min(1, ratio); // of the input Nyquist, below the output Nyquist
  const zeros = 32;
  const half = Math.ceil(zeros / cutoff);
  const beta = 8.6;
  const i0b = besselI0(beta);
  const n = Math.round(x.length * ratio);
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const t = j / ratio; // position in input samples
    const center = Math.floor(t);
    let acc = 0;
    for (let k = center - half + 1; k <= center + half; k++) {
      if (k < 0 || k >= x.length) continue;
      const d = t - k;
      const u = d / half;
      if (u <= -1 || u >= 1) continue;
      const arg = Math.PI * cutoff * d;
      const sinc = arg === 0 ? 1 : Math.sin(arg) / arg;
      const w = besselI0(beta * Math.sqrt(1 - u * u)) / i0b;
      acc += (x[k] ?? 0) * cutoff * sinc * w;
    }
    out[j] = acc;
  }
  return out;
}

export interface Loudness {
  peak: number;
  speechSeconds: number;
}

export function loudness(y16k: Float32Array): Loudness {
  let peak = 0;
  for (const v of y16k) peak = Math.max(peak, Math.abs(v));
  const [start, end] = trimBounds(y16k);
  return { peak, speechSeconds: (end - start) / SAMPLE_RATE };
}

/** Too quiet or too short to say anything about (a muted mic, a click). */
export function isSilent(y16k: Float32Array): boolean {
  const { peak, speechSeconds } = loudness(y16k);
  return peak < 1e-3 || speechSeconds < 0.3;
}

/** Little-endian float32 bytes, the body /predict expects. */
export function toBody(window: Float32Array): ArrayBuffer {
  if (window.length !== N_SAMPLES) throw new RangeError(`expected ${String(N_SAMPLES)} samples`);
  const view = new DataView(new ArrayBuffer(N_SAMPLES * 4));
  for (let i = 0; i < N_SAMPLES; i++) {
    const v = window[i] ?? 0;
    if (!Number.isFinite(v)) throw new RangeError("audio contains NaN or infinity");
    view.setFloat32(i * 4, v, true);
  }
  return view.buffer;
}

/** Web Audio is needed to decode and resample (every current browser; not every test engine). */
export function canProcessAudio(): boolean {
  return typeof OfflineAudioContext === "function";
}

/** Decode any browser-supported file or recording to 16 kHz mono (browser only). */
export async function decodeTo16k(data: ArrayBuffer): Promise<Float32Array> {
  let ctx: OfflineAudioContext | undefined;
  try {
    // decodeAudioData resamples to its context's rate: one browser resample, straight to
    // 16 kHz. An OfflineAudioContext needs no user gesture and no audio device.
    ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  } catch (err) {
    if (!(err instanceof DOMException) || err.name !== "NotSupportedError") throw err;
  }
  if (ctx) {
    const decoded = await ctx.decodeAudioData(data);
    return mixToMono(channelsOf(decoded));
  }
  // Some engines refuse a 16 kHz context: decode at 44.1 kHz and resample in JS.
  const decoded = await new OfflineAudioContext(1, 1, 44_100).decodeAudioData(data);
  return resample(mixToMono(channelsOf(decoded)), decoded.sampleRate, SAMPLE_RATE);
}

function channelsOf(buffer: AudioBuffer): Float32Array[] {
  return Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
}
