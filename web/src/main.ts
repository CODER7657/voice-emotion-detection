import "@fontsource/syne/800.css";
import "@fontsource/space-grotesk/400.css";
import "@fontsource/space-grotesk/500.css";
import "./app.css";

import { createElement, Mic, Square, Upload, LoaderCircle } from "lucide";
import { ApiError, predict, type Prediction } from "./api";
import { canProcessAudio, decodeTo16k, isSilent, toBody, toModelWindow } from "./audio";
import {
  MIN_S,
  RECORD_S,
  RecorderError,
  canRecord,
  startRecording,
  type Recording,
} from "./recorder";
import { AMBIENT, GLOWS, announcement, glowLevel, percent, ranked } from "./view";

type State = "idle" | "recording" | "analysing" | "done" | "error";
const MAX_FILE_BYTES = 25 * 1024 * 1024;

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el;
}

const app = $("app");
const recordBtn = $("record") as HTMLButtonElement;
const recordIcon = $("record-icon");
const recordLabel = $("record-label");
const fileInput = $("file") as HTMLInputElement;
const status = $("status");
const result = $("result");
const verdict = $("verdict");
const bars = $("bars");
const timing = $("timing");
const glowField = $("glows");

function icon(node: Parameters<typeof createElement>[0]): SVGElement {
  const svg = createElement(node);
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("stroke-width", "1.5");
  return svg;
}

const glowEls = new Map(
  GLOWS.map((g) => {
    const el = document.createElement("span");
    el.className = "glow";
    for (const [k, v] of Object.entries({
      c1: g.c1,
      c2: g.c2,
      x: `${String(g.x)}%`,
      y: `${String(g.y)}%`,
      size: g.size,
      period: g.period,
    })) {
      el.style.setProperty(`--${k}`, v);
    }
    glowField.append(el);
    return [g.emotion, el] as const;
  }),
);

function setLevels(pred?: Prediction) {
  for (const [emotion, el] of glowEls) {
    el.style.setProperty(
      "--level",
      String(pred ? glowLevel(pred.probabilities[emotion]) : AMBIENT),
    );
  }
}

let recording: Recording | undefined;
let recordStarted = 0;

function setState(state: State, message = "") {
  app.dataset.state = state;
  status.textContent = message;
  const busy = state === "analysing";
  recordBtn.disabled = busy;
  fileInput.disabled = busy || state === "recording";
  recordBtn.setAttribute("aria-busy", String(busy));
  const [node, label] =
    state === "recording"
      ? [Square, "Stop recording"]
      : busy
        ? [LoaderCircle, "Analysing"]
        : [Mic, "Record 3 seconds"];
  recordIcon.replaceChildren(icon(node));
  recordLabel.textContent = label;
}

function showResult(pred: Prediction) {
  const order = ranked(pred);
  verdict.textContent = pred.emotion;
  bars.replaceChildren(
    ...order.map(([emotion, p], i) => {
      const li = document.createElement("li");
      li.className = "bar";
      if (i === 0) li.dataset.top = "";
      li.style.setProperty("--p", String(p));
      const name = document.createElement("span");
      name.className = "bar-name";
      name.textContent = emotion;
      const track = document.createElement("span");
      track.className = "bar-track";
      track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.className = "bar-fill";
      track.append(fill);
      const value = document.createElement("span");
      value.className = "bar-value";
      value.textContent = percent(p);
      li.append(name, track, value);
      return li;
    }),
  );
  timing.textContent = Number.isFinite(pred.inferenceMs)
    ? `model time ${String(Math.round(pred.inferenceMs))} ms`
    : "";
  result.hidden = false;
  setLevels(pred);
  setState("done", announcement(pred));
}

async function analyse(encoded: ArrayBuffer) {
  setState("analysing", "Listening closely…");
  try {
    let y: Float32Array;
    try {
      y = await decodeTo16k(encoded);
    } catch {
      throw new Error("That file could not be read as audio. Try a WAV, MP3 or M4A file.");
    }
    if (isSilent(y)) throw new Error("We could not hear any speech. Move closer and try again.");
    showResult(await predict(toBody(toModelWindow(y))));
  } catch (err) {
    setLevels();
    const msg =
      err instanceof ApiError || err instanceof Error ? err.message : "Something went wrong.";
    setState("error", msg);
  }
}

async function onRecord() {
  if (recording) {
    if (performance.now() - recordStarted >= MIN_S * 1000) recording.stop();
    return;
  }
  result.hidden = true;
  setLevels();
  try {
    recording = await startRecording(RECORD_S);
  } catch (err) {
    setState("error", err instanceof RecorderError ? err.message : "Recording failed.");
    return;
  }
  recordStarted = performance.now();
  app.style.setProperty("--record-s", `${String(RECORD_S)}s`);
  setState("recording", "Listening… speak now.");
  try {
    const encoded = await recording.done;
    recording = undefined;
    await analyse(encoded);
  } catch (err) {
    recording = undefined;
    setState("error", err instanceof Error ? err.message : "Recording failed.");
  }
}

function onFile() {
  const file = fileInput.files?.[0];
  fileInput.value = ""; // choosing the same file again still fires `change`
  if (!file) return;
  if (file.size > MAX_FILE_BYTES) {
    setState("error", "That file is over 25 MB. Choose a shorter clip.");
    return;
  }
  result.hidden = true;
  file.arrayBuffer().then(analyse, () => {
    setState("error", "That file could not be opened.");
  });
}

$("file-icon").append(icon(Upload));
setLevels();
setState("idle");
if (!canProcessAudio()) {
  recordBtn.disabled = true;
  fileInput.disabled = true;
  $("record-hint").textContent =
    "(this browser cannot process audio; try Chrome, Firefox or Safari)";
} else if (!canRecord()) {
  recordBtn.disabled = true;
  $("record-hint").textContent = "(recording is not supported here; choose a file)";
}
recordBtn.addEventListener("click", () => void onRecord());
fileInput.addEventListener("change", onFile);
// Warm the model while the visitor reads: a cold start loads it (about 7 s on Vercel).
void fetch("/ready").catch(() => undefined);
