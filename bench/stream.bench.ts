import { test } from "vitest";
import { createSyntheticEEG } from "../src/io/synthetic";
import { ExponentialSmoother } from "../src/metrics/smoothing";
import { RingBuffer } from "../src/stream/ring-buffer";
import { SlidingWindow } from "../src/stream/window";

// Workload: 8 channels at 256 Hz cut into 2 s windows every 0.25 s.
// Each iteration pushes 10 s of data (2560 frames, 40 windows).
const FS = 256;
const CHANNELS = 8;
const SECONDS = 10;
const chunk = createSyntheticEEG({ sampleRate: FS, channels: CHANNELS, seed: 1 }).next(
  FS * SECONDS,
);
const frames = Array.from({ length: FS * SECONDS }, (_, i) =>
  Float64Array.from(chunk, (ch) => ch[i] as number),
);
const options = { channels: CHANNELS, size: 2 * FS, step: FS / 4 };

test("SlidingWindow: 10 s of 8 channels @ 256 Hz, 2 s windows every 0.25 s", async ({ bench }) => {
  const plain = new SlidingWindow(options);
  const listened = new SlidingWindow(options);
  let sink = 0;
  listened.onWindow((w) => {
    sink += (w[0] as Float64Array)[0] as number;
  });
  const chunked = new SlidingWindow(options);

  await bench.compare(
    bench("pushFrame × 2560", () => {
      for (const frame of frames) plain.pushFrame(frame);
    }),
    bench("pushFrame × 2560 with an onWindow listener (40 window copies)", () => {
      for (const frame of frames) listened.pushFrame(frame);
    }),
    bench("pushChunk (one 8 × 2560 chunk)", () => {
      chunked.pushChunk(chunk);
    }),
  );
  if (Number.isNaN(sink)) throw new Error("unreachable; keeps the listener's work observable");
});

test("stream primitives", async ({ bench }) => {
  const ring = new RingBuffer(512);
  const out = new Float64Array(512);
  const smoother = new ExponentialSmoother({ timeConstant: 1, rate: 4 });
  await bench.compare(
    bench("RingBuffer.push × 2560", () => {
      for (let i = 0; i < 2560; i++) ring.push(i);
    }),
    bench("RingBuffer.toArray(out), 512 samples", () => {
      ring.toArray(out);
    }),
    bench("ExponentialSmoother.update × 2560", () => {
      for (let i = 0; i < 2560; i++) smoother.update(i);
    }),
  );
});
