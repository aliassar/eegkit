import { test } from "vitest";
import { createSyntheticEEG } from "../src/io/synthetic";
import { detectBlinks } from "../src/quality/blinks";
import { goertzelPower } from "../src/quality/goertzel";
import { assessQuality } from "../src/quality/quality";

const FS = 256;
const eeg = createSyntheticEEG({ sampleRate: FS, channels: 8, seed: 1, blinkRate: 15 });
const window2s = eeg.next(2 * FS);
const minute = eeg.next(60 * FS)[0] as Float64Array;

test("signal quality on 2 s windows @ 256 Hz", async ({ bench }) => {
  await bench.compare(
    bench("assessQuality, 1 channel", () => {
      assessQuality(window2s[0] as Float64Array, { sampleRate: FS });
    }),
    bench("assessQuality, 8 channels", () => {
      for (const ch of window2s) assessQuality(ch, { sampleRate: FS });
    }),
    bench("goertzelPower (Hann), 512 samples", () => {
      goertzelPower(window2s[0] as Float64Array, 50, FS);
    }),
  );
});

test("blink detection and synthetic data", async ({ bench }) => {
  const gen = createSyntheticEEG({ sampleRate: FS, channels: 8, seed: 2 });
  await bench.compare(
    bench("detectBlinks, 60 s @ 256 Hz", () => {
      detectBlinks(minute, { sampleRate: FS });
    }),
    bench("createSyntheticEEG.next, 8 channels × 1 s", () => {
      gen.next(FS);
    }),
  );
});
