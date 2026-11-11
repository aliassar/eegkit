import { test } from "vitest";
import { bandPowers, fft, peakFrequency, rfft, welch } from "../src/spectral";

const fs = 256;
const channels = 8;
// 2 s analysis windows, the usual update unit for neurofeedback.
const windows = Array.from({ length: channels }, (_, ch) =>
  Float64Array.from(
    { length: 2 * fs },
    (_, t) => 20 * Math.sin((2 * Math.PI * 10 * t) / fs) + Math.sin(t * (ch + 1)),
  ),
);
const at250 = Float64Array.from({ length: 500 }, (_, t) => Math.sin(t / 3));

test("welch on 2 s windows (256-sample Hann segments, 50 % overlap)", async ({ bench }) => {
  await bench.compare(
    bench("welch, 1 channel @ 256 Hz", () => {
      welch(windows[0] ?? [], { sampleRate: fs });
    }),
    bench("welch, 1 channel @ 250 Hz (250-sample segments, Bluestein)", () => {
      welch(at250, { sampleRate: 250, segmentLength: 250 });
    }),
    bench("welch + bandPowers + peakFrequency, 8 channels @ 256 Hz", () => {
      for (const w of windows) {
        const psd = welch(w, { sampleRate: fs });
        bandPowers(psd);
        peakFrequency(psd, [7, 14]);
      }
    }),
  );
});

test("FFT", async ({ bench }) => {
  const cases = [256, 250, 1024, 1000].flatMap((n) => {
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i));
    const im = new Float64Array(n);
    const x = Float64Array.from(re);
    return [
      bench(`fft complex, n = ${n}`, () => {
        fft(re, im);
      }),
      bench(`rfft, n = ${n}`, () => {
        rfft(x);
      }),
    ];
  });
  await bench.compare(...cases);
});
