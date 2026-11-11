import { test } from "vitest";
import { createSyntheticEEG } from "../src/io/synthetic";
import { createPipeline } from "../src/pipeline";

// Workload: the default pipeline (band-pass + 50 Hz notch, 2 s windows every
// 0.25 s, Welch, band powers, peak alpha, quality) on 10 s of data, i.e. 40
// windows per iteration. Divide 10 s by the mean time to get "× real time".
const FS = 256;
const SECONDS = 10;

test("createPipeline: 10 s of data @ 256 Hz, 40 windows", async ({ bench }) => {
  const chunk8 = createSyntheticEEG({ sampleRate: FS, channels: 8, seed: 1 }).next(FS * SECONDS);
  const chunk32 = createSyntheticEEG({ sampleRate: FS, channels: 32, seed: 1 }).next(FS * SECONDS);
  const p8 = createPipeline({ sampleRate: FS, channels: 8, notch: 50 });
  const p8NoQuality = createPipeline({ sampleRate: FS, channels: 8, notch: 50, quality: false });
  const p32 = createPipeline({ sampleRate: FS, channels: 32, notch: 50 });

  await bench.compare(
    bench("8 channels", () => p8.pushChunk(chunk8)),
    bench("8 channels, quality off", () => p8NoQuality.pushChunk(chunk8)),
    bench("32 channels", () => p32.pushChunk(chunk32)),
  );
});
