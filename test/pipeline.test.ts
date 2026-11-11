import { describe, expect, test } from "vitest";
import { createSyntheticEEG } from "../src/io/synthetic";
import { createPipeline, type PipelineResult } from "../src/pipeline";

const fs = 256;

function collect(pipeline: ReturnType<typeof createPipeline>): PipelineResult[] {
  const results: PipelineResult[] = [];
  pipeline.subscribe((r) => results.push(r));
  return results;
}

describe("createPipeline", () => {
  test("emits one result per step once the first window is full", () => {
    const pipeline = createPipeline({ sampleRate: fs, channels: 2, quality: false });
    const results = collect(pipeline);
    const eeg = createSyntheticEEG({ sampleRate: fs, channels: 2, seed: 1 });
    const seconds = 10;
    pipeline.pushChunk(eeg.next(seconds * fs));
    const size = 2 * fs;
    const step = fs / 4;
    expect(results).toHaveLength(Math.floor((seconds * fs - size) / step) + 1);
    expect(results.map((r) => r.index)).toEqual(results.map((_, i) => i));
    expect(results.at(-1)?.time).toBe(seconds);
    expect(pipeline.latest).toBe(results.at(-1));
  });

  test("finds dominant alpha and its peak frequency", () => {
    const pipeline = createPipeline({ sampleRate: fs, channels: ["O1", "O2"] });
    const eeg = createSyntheticEEG({
      sampleRate: fs,
      channels: 2,
      seed: 7,
      bands: { delta: 2, theta: 2, alpha: 25, beta: 2, gamma: 1 },
    });
    pipeline.pushChunk(eeg.next(6 * fs));
    const result = pipeline.latest;
    expect(result).toBeDefined();
    for (const channel of result?.channels ?? []) {
      const relative = channel.relativeBandPower;
      const strongest = Object.entries(relative).sort((a, b) => b[1] - a[1])[0]?.[0];
      expect(strongest).toBe("alpha");
      expect(channel.peakAlpha).toBeGreaterThan(8);
      expect(channel.peakAlpha).toBeLessThan(13);
      expect(channel.quality?.level).toBe("good");
    }
    expect(result?.channels.map((c) => c.label)).toEqual(["O1", "O2"]);
  });

  test("push and pushChunk give identical results", () => {
    const eeg = createSyntheticEEG({ sampleRate: fs, channels: 3, seed: 3 });
    const chunk = eeg.next(4 * fs);
    const a = createPipeline({ sampleRate: fs, channels: 3, notch: 50 });
    const b = createPipeline({ sampleRate: fs, channels: 3, notch: 50 });
    const ra = collect(a);
    const rb = collect(b);
    a.pushChunk(chunk);
    for (let i = 0; i < 4 * fs; i++) b.push(chunk.map((ch) => ch[i] as number));
    expect(ra).toHaveLength(rb.length);
    expect(ra.at(-1)?.channels[1]?.bandPower).toEqual(rb.at(-1)?.channels[1]?.bandPower);
  });

  test("quality sees mains noise on the raw signal that the notch removes", () => {
    const eeg = createSyntheticEEG({
      sampleRate: fs,
      channels: 1,
      seed: 5,
      lineNoise: { frequency: 50, amplitude: 40 },
    });
    const chunk = eeg.next(4 * fs);
    const withNotch = createPipeline({ sampleRate: fs, channels: 1, notch: 50, bandpass: [1, 70] });
    const withoutNotch = createPipeline({ sampleRate: fs, channels: 1, bandpass: [1, 70] });
    withNotch.pushChunk(chunk);
    withoutNotch.pushChunk(chunk);
    const notched = withNotch.latest?.channels[0];
    const plain = withoutNotch.latest?.channels[0];
    expect(notched?.quality?.lineNoiseRatio).toBeGreaterThan(0.1);
    const at50 = (spectrum: { frequencies: Float64Array; power: Float64Array } | undefined) =>
      spectrum?.power[spectrum.frequencies.indexOf(50)] ?? Number.NaN;
    expect(at50(notched?.spectrum)).toBeLessThan(at50(plain?.spectrum) / 100);
  });

  test("a DC offset does not ring through the first windows", () => {
    const eeg = createSyntheticEEG({ sampleRate: fs, channels: 1, seed: 9, dcOffset: 500 });
    const pipeline = createPipeline({ sampleRate: fs, channels: 1 });
    const results = collect(pipeline);
    pipeline.pushChunk(eeg.next(4 * fs));
    const deltaFirst = results[0]?.channels[0]?.bandPower.delta ?? Number.NaN;
    const deltaLast = results.at(-1)?.channels[0]?.bandPower.delta ?? Number.NaN;
    expect(deltaFirst).toBeLessThan(deltaLast * 5);
  });

  test("reset clears state but keeps subscribers; unsubscribe stops delivery", () => {
    const pipeline = createPipeline({ sampleRate: fs, channels: 1, quality: false });
    const results = collect(pipeline);
    const eeg = createSyntheticEEG({ sampleRate: fs, channels: 1, seed: 2 });
    pipeline.pushChunk(eeg.next(3 * fs));
    const before = results.length;
    pipeline.reset();
    expect(pipeline.latest).toBeUndefined();
    pipeline.pushChunk(eeg.next(2 * fs));
    expect(results.length).toBe(before + 1);
    expect(results.at(-1)?.index).toBe(0);

    const seen: PipelineResult[] = [];
    const unsubscribe = pipeline.subscribe((r) => seen.push(r));
    unsubscribe();
    pipeline.pushChunk(eeg.next(fs));
    expect(seen).toHaveLength(0);
  });

  test("rejects invalid input", () => {
    expect(() => createPipeline({ sampleRate: 0, channels: 1 })).toThrow(RangeError);
    expect(() => createPipeline({ sampleRate: fs, channels: 0 })).toThrow(RangeError);
    expect(() => createPipeline({ sampleRate: fs, channels: [] })).toThrow(RangeError);
    const pipeline = createPipeline({ sampleRate: fs, channels: 2 });
    expect(() => pipeline.push([1, 2, 3])).toThrow(RangeError);
    expect(() => pipeline.pushChunk([[1]])).toThrow(RangeError);
  });
});
