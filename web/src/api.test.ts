import { describe, expect, it } from "vitest";
import { ApiError, EMOTIONS, parsePrediction, predict } from "./api";

const probabilities = Object.fromEntries(EMOTIONS.map((e) => [e, 0.125]));
const good = { emotion: "calm", probabilities, inference_ms: 12.5, model_sha256: "x", note: "n" };

function fakeFetch(res: Response | Error, seen: RequestInit[] = []): typeof fetch {
  return (_input, init) => {
    seen.push(init ?? {});
    return res instanceof Error ? Promise.reject(res) : Promise.resolve(res);
  };
}

const body = new ArrayBuffer(192_000);

describe("predict", () => {
  it("posts raw float32 bytes and parses the prediction", async () => {
    const seen: RequestInit[] = [];
    const p = await predict(body, fakeFetch(Response.json(good), seen));
    expect(p).toEqual({ emotion: "calm", probabilities, inferenceMs: 12.5 });
    expect(seen[0]?.method).toBe("POST");
    expect(seen[0]?.headers).toEqual({ "Content-Type": "application/octet-stream" });
    expect(seen[0]?.body).toBe(body);
  });

  it("turns 429 into a wait message using Retry-After", async () => {
    const res = new Response("{}", { status: 429, headers: { "Retry-After": "17" } });
    await expect(predict(body, fakeFetch(res))).rejects.toMatchObject({
      status: 429,
      retryAfterS: 17,
      message: expect.stringContaining("17 s") as unknown,
    });
  });

  it("explains 503 and other server errors without leaking details", async () => {
    await expect(predict(body, fakeFetch(new Response("{}", { status: 503 })))).rejects.toThrow(
      /starting or unavailable/,
    );
    const res = new Response('{"detail":"Traceback ..."}', { status: 500 });
    await expect(predict(body, fakeFetch(res))).rejects.toThrow(
      "The server had a problem. Try again.",
    );
  });

  it("reports a network failure as offline", async () => {
    await expect(predict(body, fakeFetch(new TypeError("Failed to fetch")))).rejects.toMatchObject({
      status: 0,
      message: "You appear to be offline.",
    });
  });

  it("times out a request that hangs", async () => {
    const hang: typeof fetch = (_i, init) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        });
      });
    await expect(predict(body, hang, 20)).rejects.toThrow(/took too long/);
  });

  it("rejects a body that is not JSON", async () => {
    await expect(predict(body, fakeFetch(new Response("<html>")))).rejects.toBeInstanceOf(ApiError);
  });
});

describe("parsePrediction", () => {
  it.each([
    ["null", null],
    ["unknown emotion", { ...good, emotion: "bored" }],
    ["missing class", { ...good, probabilities: { calm: 1 } }],
    ["probability out of range", { ...good, probabilities: { ...probabilities, calm: 1.5 } }],
    ["probability not a number", { ...good, probabilities: { ...probabilities, calm: "0.5" } }],
  ])("rejects %s", (_name, json) => {
    expect(() => parsePrediction(json)).toThrow(ApiError);
  });

  it("tolerates a missing inference time", () => {
    const rest: Partial<typeof good> = { ...good };
    delete rest.inference_ms;
    expect(parsePrediction(rest).inferenceMs).toBeNaN();
  });
});
