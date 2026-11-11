import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSyntheticEEG, startSyntheticStream } from "../src/io/synthetic";
import { STANDARD_BANDS } from "../src/types";
import { bandPowerDft, sinePower } from "./helpers/naive-dft";

const FS = 256;
const silent = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };

const std = (x: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < x.length; i++) m += x[i] as number;
  m /= x.length;
  let v = 0;
  for (let i = 0; i < x.length; i++) v += ((x[i] as number) - m) ** 2;
  return Math.sqrt(v / x.length);
};

describe("createSyntheticEEG", () => {
  it("is deterministic for a seed and differs between seeds", () => {
    const a = createSyntheticEEG({ sampleRate: FS, channels: 3, seed: 42, blinkRate: 20 }).next(
      1000,
    );
    const b = createSyntheticEEG({ sampleRate: FS, channels: 3, seed: 42, blinkRate: 20 }).next(
      1000,
    );
    const c = createSyntheticEEG({ sampleRate: FS, channels: 3, seed: 43, blinkRate: 20 }).next(
      1000,
    );
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    expect(a.length).toBe(3);
    expect(a[0]?.length).toBe(1000);
    // Channels are not copies of each other.
    expect(a[0]).not.toEqual(a[1]);
  });

  it("does not depend on how samples are requested, and reset rewinds", () => {
    const opts = {
      sampleRate: FS,
      channels: 2,
      seed: 9,
      blinkRate: 30,
      lineNoise: { frequency: 50, amplitude: 5 },
    } as const;
    const whole = createSyntheticEEG(opts).next(1500);
    const gen = createSyntheticEEG(opts);
    const parts = [gen.next(1), gen.next(0), gen.next(499), gen.next(1000)];
    for (let c = 0; c < 2; c++) {
      const joined = Float64Array.from(parts.flatMap((p) => Array.from(p[c] ?? [])));
      expect(joined).toEqual(whole[c]);
    }
    gen.reset();
    expect(gen.next(1500)).toEqual(whole);
  });

  it("puts the requested rhythm's peak inside its band", () => {
    for (const band of STANDARD_BANDS) {
      const x = createSyntheticEEG({
        sampleRate: FS,
        channels: 1,
        seed: 5,
        noise: 2,
        bands: { ...silent, [band.name]: 20 },
      }).next(8 * FS)[0] as Float64Array;
      // Dominant DFT bin over 0.5-60 Hz must fall inside the band.
      let best = 0;
      let bestF = 0;
      for (let f = 0.5; f < 60; f += 1 / 8) {
        const p = sinePower(x, f, FS);
        if (p > best) {
          best = p;
          bestF = f;
        }
      }
      expect(bestF, band.name).toBeGreaterThanOrEqual(band.low);
      expect(bestF, band.name).toBeLessThan(band.high);
      // ... and the band holds most of the power: A²/2 = 200 µV², modulation
      // adds ~3 %, while 2 µV RMS of pink noise contributes at most 4 µV².
      const inBand = bandPowerDft(x, band.low, band.high, FS);
      expect(inBand, band.name).toBeGreaterThan(150);
      expect(inBand, band.name).toBeLessThan(260);
    }
  });

  it("scales the 1/f background to the requested RMS, with more power at low frequencies", () => {
    const x = createSyntheticEEG({
      sampleRate: FS,
      channels: 1,
      seed: 2,
      noise: 10,
      bands: silent,
    }).next(120 * FS)[0] as Float64Array;
    expect(std(x)).toBeGreaterThan(9);
    expect(std(x)).toBeLessThan(11);
    const seg = x.subarray(0, 16 * FS);
    // 1/f: power per Hz falls with frequency (2-4 Hz vs 20-40 Hz, normalised by width).
    const low = bandPowerDft(seg, 2, 4, FS) / 2;
    const high = bandPowerDft(seg, 20, 40, FS) / 20;
    expect(low / high).toBeGreaterThan(4);
  });

  it("injects mains interference at the requested frequency and amplitude", () => {
    const x = createSyntheticEEG({
      sampleRate: FS,
      channels: 2,
      seed: 1,
      lineNoise: { frequency: 60, amplitude: 12 },
    }).next(4 * FS);
    for (const ch of x) {
      expect(sinePower(ch, 60, FS)).toBeCloseTo(72, -0.5); // 12²/2, ±~1.6
      expect(sinePower(ch, 50, FS)).toBeLessThan(1);
    }
  });

  it("adds blinks on top of an unchanged background", () => {
    const base = { sampleRate: FS, channels: 2, seed: 8 };
    const clean = createSyntheticEEG(base).next(60 * FS);
    const blinky = createSyntheticEEG({ ...base, blinkRate: 20, blinkAmplitude: 120 }).next(
      60 * FS,
    );
    for (let c = 0; c < 2; c++) {
      const d = (blinky[c] as Float64Array).map(
        (v, i) => v - ((clean[c] as Float64Array)[i] as number),
      );
      const max = Math.max(...d);
      expect(max).toBeGreaterThan(119);
      expect(max).toBeLessThanOrEqual(120 + 1e-9);
      expect(Math.min(...d)).toBeGreaterThanOrEqual(-1e-9);
      // Count blink onsets: rate 20/min over 1 min (Poisson, so a wide margin).
      let onsets = 0;
      for (let i = 1; i < d.length; i++) if (d[i - 1] === 0 && (d[i] as number) > 0) onsets++;
      expect(onsets).toBeGreaterThan(10);
      expect(onsets).toBeLessThan(32);
    }
    // Same blinks on every channel.
    const d0 = (blinky[0] as Float64Array).map(
      (v, i) => v - ((clean[0] as Float64Array)[i] as number),
    );
    const d1 = (blinky[1] as Float64Array).map(
      (v, i) => v - ((clean[1] as Float64Array)[i] as number),
    );
    for (let i = 0; i < d0.length; i++) expect(d0[i]).toBeCloseTo(d1[i] as number, 9);
  });

  it("applies a DC offset", () => {
    const x = createSyntheticEEG({ sampleRate: FS, channels: 1, dcOffset: 5000 }).next(10 * FS)[0];
    const mean = (x as Float64Array).reduce((a, b) => a + b, 0) / FS / 10;
    expect(mean).toBeGreaterThan(4990);
    expect(mean).toBeLessThan(5010);
  });

  it("validates options", () => {
    expect(() => createSyntheticEEG({ sampleRate: 0, channels: 1 })).toThrow(RangeError);
    expect(() => createSyntheticEEG({ sampleRate: FS, channels: 0 })).toThrow(RangeError);
    expect(() => createSyntheticEEG({ sampleRate: FS, channels: 1, noise: -1 })).toThrow(
      RangeError,
    );
    expect(() => createSyntheticEEG({ sampleRate: FS, channels: 1, blinkRate: -1 })).toThrow(
      RangeError,
    );
    expect(() => createSyntheticEEG({ sampleRate: FS, channels: 1, bands: { alpha: -2 } })).toThrow(
      RangeError,
    );
    expect(() =>
      createSyntheticEEG({
        sampleRate: 100,
        channels: 1,
        lineNoise: { frequency: 50, amplitude: 1 },
      }),
    ).toThrow(RangeError);
    expect(() => createSyntheticEEG({ sampleRate: FS, channels: 1 }).next(-1)).toThrow(RangeError);
  });
});

describe("startSyntheticStream", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("delivers frames in real time, matching the generator's output", () => {
    const frames: Float64Array[] = [];
    const opts = { sampleRate: FS, channels: 3, seed: 6 };
    const stop = startSyntheticStream((f) => frames.push(f), { ...opts, chunkMs: 50 });
    expect(frames.length).toBe(0);
    vi.advanceTimersByTime(1000);
    expect(frames.length).toBe(FS);
    vi.advanceTimersByTime(500);
    expect(frames.length).toBe(FS * 1.5);
    stop();
    vi.advanceTimersByTime(1000);
    expect(frames.length).toBe(FS * 1.5);

    const expected = createSyntheticEEG(opts).next(frames.length);
    frames.forEach((frame, i) => {
      expect(frame.length).toBe(3);
      for (let c = 0; c < 3; c++) expect(frame[c]).toBe((expected[c] as Float64Array)[i]);
    });
    // Frames are independent arrays, safe to keep.
    expect(frames[0]).not.toBe(frames[1]);
  });

  it("keeps the long-run rate exact with a chunk period that doesn't divide the sample period", () => {
    let count = 0;
    const stop = startSyntheticStream(() => count++, { sampleRate: 250, channels: 1, chunkMs: 33 });
    vi.advanceTimersByTime(9999); // 303 ticks, last at 9999 ms
    expect(count).toBe(Math.floor((9999 * 250) / 1000));
    stop();
  });

  it("drops a backlog longer than one second instead of bursting", () => {
    let count = 0;
    const stop = startSyntheticStream(() => count++, { sampleRate: FS, channels: 1, chunkMs: 40 });
    vi.advanceTimersByTime(40);
    const before = count;
    // Simulate a suspended tab: the clock jumps 10 s without timers firing.
    vi.setSystemTime(Date.now() + 10_000);
    vi.advanceTimersByTime(40);
    expect(count - before).toBe(FS);
    // ... and then continues at the normal pace.
    const after = count;
    vi.advanceTimersByTime(1000);
    expect(count - after).toBe(FS);
    stop();
  });

  it("stopping from inside the callback stops immediately", () => {
    let count = 0;
    const stop = startSyntheticStream(
      () => {
        count++;
        if (count === 5) stop();
      },
      { sampleRate: FS, channels: 1, chunkMs: 100 },
    );
    vi.advanceTimersByTime(1000);
    expect(count).toBe(5);
  });

  it("validates chunkMs", () => {
    expect(() =>
      startSyntheticStream(() => {}, { sampleRate: FS, channels: 1, chunkMs: 0 }),
    ).toThrow(RangeError);
  });
});

describe("createSyntheticEEG band overrides", () => {
  it("treats an undefined band amplitude as the default", () => {
    const a = createSyntheticEEG({ sampleRate: FS, channels: 1, seed: 3 }).next(FS);
    const b = createSyntheticEEG({
      sampleRate: FS,
      channels: 1,
      seed: 3,
      bands: { alpha: undefined as unknown as number },
    }).next(FS);
    expect(b).toEqual(a);
  });
});
