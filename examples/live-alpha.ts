/**
 * A terminal neurofeedback loop: live synthetic EEG in, a smoothed alpha bar out.
 *
 *   npx tsx examples/live-alpha.ts
 *
 * Swap `startSyntheticStream` for `connectWebSocket("ws://…", { onFrame: pipeline.push })`
 * to drive it from a real amplifier bridge.
 */
import { Baseline, createPipeline, ExponentialSmoother, startSyntheticStream } from "../src/index";

const sampleRate = 256;
const pipeline = createPipeline({ sampleRate, channels: ["O1", "O2"], notch: 50 });
const smoother = new ExponentialSmoother({ timeConstant: 1.5, rate: 4 });
const baseline = new Baseline({ minSamples: 20 });

pipeline.subscribe(({ time, channels }) => {
  const alpha =
    channels.reduce((sum, c) => sum + (c.relativeBandPower.alpha ?? 0), 0) / channels.length;
  const smoothed = smoother.update(alpha);
  baseline.add(smoothed);
  const z = baseline.ready ? baseline.zScore(smoothed) : 0;
  const bar = "█".repeat(Math.max(0, Math.round(20 + z * 8)));
  const quality = channels.map((c) => `${c.label}:${c.quality?.level}`).join(" ");
  process.stdout.write(
    `\r${time.toFixed(1).padStart(6)}s  alpha ${(smoothed * 100).toFixed(1).padStart(5)}%  ` +
      `peak ${channels[0]?.peakAlpha.toFixed(1)} Hz  ${quality}  ${bar.padEnd(45)}`,
  );
});

const stop = startSyntheticStream(pipeline.push, {
  sampleRate,
  channels: 2,
  seed: 42,
  bands: { alpha: 14 },
  lineNoise: { frequency: 50, amplitude: 4 },
  blinkRate: 12,
});

process.on("SIGINT", () => {
  stop();
  process.stdout.write("\n");
  process.exit(0);
});
