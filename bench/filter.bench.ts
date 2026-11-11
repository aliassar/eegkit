import { test } from "vitest";
import {
  butterworthBandpass,
  cascade,
  FilterBank,
  filtfilt,
  IIRFilter,
  notch,
} from "../src/filter";

// Streaming workload: 8 channels at 256 Hz through a 1–40 Hz band-pass (order 4)
// plus a 50 Hz notch = 5 biquads per channel. Each iteration filters one second.
const fs = 256;
const channels = 8;
const sos = cascade(butterworthBandpass(4, 1, 40, fs), notch(50, fs));
const frames = Array.from({ length: fs }, (_, t) =>
  Float64Array.from({ length: channels }, (_, ch) => 20 * Math.sin(t / (5 + ch)) + ch),
);
const chunk = Array.from({ length: channels }, (_, ch) =>
  Float64Array.from({ length: fs }, (_, t) => 20 * Math.sin(t / (5 + ch)) + ch),
);

test("streaming filter: 1 s of 8 channels @ 256 Hz, 5 biquads", async ({ bench }) => {
  const bank = new FilterBank(sos, channels);
  const out = new Float64Array(channels);
  const filters = Array.from({ length: channels }, () => new IIRFilter(sos));
  const chunkBank = new FilterBank(sos, channels);

  await bench.compare(
    bench("FilterBank.processFrame × 256 frames", () => {
      for (const frame of frames) bank.processFrame(frame, out);
    }),
    bench("IIRFilter.process × 256 samples × 8 channels", () => {
      for (const frame of frames) {
        for (let ch = 0; ch < channels; ch++) {
          out[ch] = filters[ch]?.process(frame[ch] ?? 0) ?? 0;
        }
      }
    }),
    bench("FilterBank.processChunk (8 × 256)", () => {
      chunkBank.processChunk(chunk);
    }),
  );
});

test("offline zero-phase filtering", async ({ bench }) => {
  const tenSeconds = Float64Array.from({ length: fs * 10 }, (_, t) => 20 * Math.sin(t / 5));
  await bench("filtfilt, 10 s @ 256 Hz, 1 channel", () => {
    filtfilt(sos, tenSeconds);
  }).run();
});
