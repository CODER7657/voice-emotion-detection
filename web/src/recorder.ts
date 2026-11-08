/** Microphone capture with MediaRecorder: Chrome/Firefox give WebM/Opus, Safari MP4/AAC. */

export const RECORD_S = 3.5; // a little over the 3 s window, so trimming leaves enough speech
export const MIN_S = 1;

const TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];

export class RecorderError extends Error {
  override name = "RecorderError";
}

export function canRecord(): boolean {
  return (
    typeof MediaRecorder !== "undefined" &&
    typeof navigator !== "undefined" &&
    "mediaDevices" in navigator &&
    typeof navigator.mediaDevices.getUserMedia === "function"
  );
}

export function pickMimeType(isSupported: (t: string) => boolean): string | undefined {
  return TYPES.find((t) => isSupported(t));
}

export function micErrorMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone access was blocked. Allow it in your browser settings, or choose a file.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone was found. Choose an audio file instead.";
  }
  if (name === "NotReadableError") {
    return "The microphone is busy in another app. Close it and try again.";
  }
  return "Recording failed. Choose an audio file instead.";
}

export interface Recording {
  /** Resolves with the encoded clip once recording stops (by timer or `stop`). */
  done: Promise<ArrayBuffer>;
  stop: () => void;
}

export async function startRecording(seconds = RECORD_S): Promise<Recording> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      // Raw speech, as the training clips were: no voice processing.
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    throw new RecorderError(micErrorMessage(err));
  }
  const mimeType = pickMimeType((t) => MediaRecorder.isTypeSupported(t));
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks: Blob[] = [];
  recorder.addEventListener("dataavailable", (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  });
  const release = () => {
    for (const track of stream.getTracks()) track.stop();
  };
  const done = new Promise<ArrayBuffer>((resolve, reject) => {
    recorder.addEventListener("stop", () => {
      release();
      new Blob(chunks, { type: recorder.mimeType }).arrayBuffer().then(resolve, reject);
    });
    recorder.addEventListener("error", () => {
      release();
      reject(new RecorderError("Recording failed. Choose an audio file instead."));
    });
  });
  const timer = setTimeout(stop, seconds * 1000);
  function stop() {
    clearTimeout(timer);
    if (recorder.state !== "inactive") recorder.stop();
  }
  recorder.start();
  return { done, stop };
}
