import { describe, expect, it } from "vitest";
import { createSyntheticEEG } from "../src/io/synthetic";
import { detectBlinks } from "../src/quality/blinks";
import { goertzel, goertzelPower } from "../src/quality/goertzel";
import { assessQuality } from "../src/quality/quality";
import { dftBin } from "./helpers/naive-dft";

const FS = 256;

/** Deterministic pseudo-random signal (LCG), independent of the code under test. */
function randomSignal(n: number, seed = 1): Float64Array {
  let s = seed;
  return Float64Array.from({ length: n }, () => {
    s = (s * 1664525 + 1013904223) % 2 ** 32;
    return (s / 2 ** 32 - 0.5) * 100;
  });
}

const sine = (n: number, f: number, amp: number, phase = 0, fs = FS) =>
  Float64Array.from({ length: n }, (_, i) => amp * Math.sin((2 * Math.PI * f * i) / fs + phase));

/** 2 s of clean synthetic EEG (seeded), one channel. */
const cleanEEG = (seed = 3, samples = 2 * FS) =>
  createSyntheticEEG({ sampleRate: FS, channels: 1, seed }).next(samples)[0] as Float64Array;

describe("goertzel", () => {
  it("matches a direct DFT bin for on-bin and off-bin frequencies", () => {
    const x = randomSignal(500);
    for (const f of [0, 1.024, 10, 50, 50.3, 77.7, 128]) {
      const ref = dftBin(x, f, FS).power;
      // Goertzel's recursion accumulates rounding error that grows with N
      // near DC and Nyquist (|2cos ω| → 2), hence 1e-8 rather than 1e-12.
      expect(Math.abs(goertzel(x, f, FS) - ref) / ref).toBeLessThan(1e-8);
    }
  });

  it("goertzelPower returns A²/2 for a sine on a bin, with and without Hann", () => {
    for (const [f, amp] of [
      [10, 20],
      [50, 3],
      [60, 7.5],
    ] as const) {
      const x = sine(512, f, amp, 0.7);
      expect(goertzelPower(x, f, FS)).toBeCloseTo((amp * amp) / 2, 9);
      expect(goertzelPower(x, f, FS, { hann: false })).toBeCloseTo((amp * amp) / 2, 9);
    }
  });

  it("ignores a DC offset by default and stays within Hann scalloping off-bin", () => {
    const x = sine(512, 50, 10).map((v) => v + 5000);
    expect(goertzelPower(x, 50, FS)).toBeCloseTo(50, 6);
    // Half a bin away (0.25 Hz at 0.5 Hz resolution): Hann loses at most 1.42 dB.
    const off = sine(512, 50.25, 10);
    const p = goertzelPower(off, 50.25, FS);
    expect(p).toBeLessThanOrEqual(50 * 1.0001);
    expect(p).toBeGreaterThan(50 * 10 ** (-1.43 / 10));
  });

  it("handles DC and Nyquist without the one-sided factor", () => {
    const n = 64;
    const nyq = Float64Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 3 : -3));
    expect(goertzelPower(nyq, FS / 2, FS, { hann: false })).toBeCloseTo(9, 9);
    const dc = new Float64Array(n).fill(4);
    expect(goertzelPower(dc, 0, FS, { hann: false, removeMean: false })).toBeCloseTo(16, 9);
  });

  it("validates arguments", () => {
    expect(() => goertzel([1, 2], 200, FS)).toThrow(RangeError);
    expect(() => goertzel([1, 2], -1, FS)).toThrow(RangeError);
    expect(() => goertzel([1, 2], 10, 0)).toThrow(RangeError);
    expect(() => goertzel([], 10, FS)).toThrow(RangeError);
  });
});

describe("assessQuality", () => {
  it("rates clean synthetic EEG as good, across seeds and windows", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const eeg = createSyntheticEEG({ sampleRate: FS, channels: 4, seed });
      for (let w = 0; w < 10; w++) {
        for (const channel of eeg.next(2 * FS)) {
          const q = assessQuality(channel, { sampleRate: FS });
          expect(q.level).toBe("good");
          expect(q.score).toBe(1);
          expect(q.flatline).toBe(false);
          expect(q.artifact).toBe(false);
          expect(q.clipped).toBe(0);
          expect(q.lineNoiseRatio).toBeLessThan(0.02);
        }
      }
    }
  });

  it("is unaffected by DC offset and slow linear drift", () => {
    const x = cleanEEG().map((v, i) => v + 20_000 + 0.8 * i); // 400 µV ramp over 2 s
    const q = assessQuality(x, { sampleRate: FS });
    expect(q.level).toBe("good");
    expect(q.peakToPeak).toBeCloseTo(assessQuality(cleanEEG(), { sampleRate: FS }).peakToPeak, 6);
  });

  it("flags a flat channel (constant, or below the noise threshold)", () => {
    for (const x of [
      new Float64Array(512).fill(-1234.5),
      randomSignal(512).map((v) => v * 0.002),
    ]) {
      const q = assessQuality(x, { sampleRate: FS });
      expect(q.flatline).toBe(true);
      expect(q.score).toBe(0);
      expect(q.level).toBe("poor");
      expect(q.clipped).toBe(0); // a flat line is not called clipping without known rails
    }
    // Custom threshold
    const quiet = randomSignal(512).map((v) => v * 0.05); // std ≈ 1.4 µV
    expect(assessQuality(quiet, { sampleRate: FS }).flatline).toBe(false);
    expect(assessQuality(quiet, { sampleRate: FS, flatlineThreshold: 2 }).flatline).toBe(true);
  });

  it("flags large artifacts by peak-to-peak amplitude", () => {
    const x = cleanEEG();
    // 300 µV electrode pop lasting 50 ms
    for (let i = 200; i < 213; i++) x[i] = (x[i] as number) + 300;
    const q = assessQuality(x, { sampleRate: FS });
    expect(q.artifact).toBe(true);
    expect(q.peakToPeak).toBeGreaterThan(300);
    expect(q.level).toBe("poor");
    // A milder excursion just over the limit is "fair", not "poor".
    const mild = cleanEEG();
    for (let i = 200; i < 213; i++) mild[i] = (mild[i] as number) + 110;
    const qm = assessQuality(mild, { sampleRate: FS });
    expect(qm.artifact).toBe(true);
    expect(qm.peakToPeak).toBeGreaterThan(150);
    expect(qm.peakToPeak).toBeLessThan(195);
    expect(qm.level).toBe("fair");
    // A higher limit accepts it.
    expect(assessQuality(mild, { sampleRate: FS, amplitudeLimit: 250 }).artifact).toBe(false);
  });

  it("measures clipping against known rails", () => {
    const x = cleanEEG().map((v) => Math.max(-20, Math.min(20, v)));
    const expected = x.filter((v) => v === 20 || v === -20).length / x.length;
    expect(expected).toBeGreaterThan(0.01);
    const q = assessQuality(x, { sampleRate: FS, range: [-20, 20] });
    expect(q.clipped).toBeCloseTo(expected, 12);
    expect(q.level).not.toBe("good");
    // Unclipped data with wide rails: nothing clipped.
    expect(assessQuality(cleanEEG(), { sampleRate: FS, range: [-1000, 1000] }).clipped).toBe(0);
  });

  it("infers clipping from runs at the extremes when rails are unknown", () => {
    const x = cleanEEG().map((v) => Math.min(25, v)); // saturates at +25 µV
    const atRail = x.filter((v) => v === 25).length / x.length;
    const q = assessQuality(x, { sampleRate: FS });
    expect(q.clipped).toBeGreaterThan(0);
    expect(q.clipped).toBeLessThanOrEqual(atRail);
    expect(q.level).not.toBe("good");
  });

  it("flags line noise and reports its share of the variance", () => {
    const pure = sine(512, 50, 30, 1.1);
    expect(assessQuality(pure, { sampleRate: FS }).lineNoiseRatio).toBeCloseTo(1, 6);

    const eeg = cleanEEG();
    const hum = sine(512, 50, 30);
    const noisy = eeg.map((v, i) => v + (hum[i] as number));
    const q = assessQuality(noisy, { sampleRate: FS });
    let varEEG = 0;
    const m = eeg.reduce((a, b) => a + b, 0) / eeg.length;
    for (const v of eeg) varEEG += (v - m) ** 2;
    varEEG /= eeg.length;
    const expected = 450 / (450 + varEEG); // 30²/2 = 450 µV²
    expect(q.lineNoiseRatio).toBeGreaterThan(expected - 0.05);
    expect(q.lineNoiseRatio).toBeLessThan(expected + 0.05);
    expect(q.level).toBe("poor");
    // 60 Hz mains is invisible to a 50 Hz check and caught by a 60 Hz one.
    const us = eeg.map((v, i) => v + 30 * Math.sin((2 * Math.PI * 60 * i) / FS));
    expect(assessQuality(us, { sampleRate: FS }).lineNoiseRatio).toBeLessThan(0.02);
    expect(assessQuality(us, { sampleRate: FS, lineFrequency: 60 }).lineNoiseRatio).toBeGreaterThan(
      0.5,
    );
    // Mild hum lowers the score less.
    const mild = eeg.map((v, i) => v + 0.4 * (hum[i] as number));
    const qm = assessQuality(mild, { sampleRate: FS });
    expect(qm.score).toBeGreaterThan(q.score);
  });

  it("reports NaN line noise when the mains frequency is not below Nyquist", () => {
    const q = assessQuality(randomSignal(200), { sampleRate: 100 });
    expect(q.lineNoiseRatio).toBeNaN();
    expect(q.level).toBe("good");
  });

  it("treats non-finite samples as a poor, artifactual window", () => {
    const x = cleanEEG();
    x[10] = Number.NaN;
    const q = assessQuality(x, { sampleRate: FS });
    expect(q).toMatchObject({ score: 0, level: "poor", artifact: true });
    expect(q.peakToPeak).toBeNaN();
  });

  it("validates options", () => {
    const x = cleanEEG();
    expect(() => assessQuality(x, { sampleRate: 0 })).toThrow(RangeError);
    expect(() => assessQuality(x, { sampleRate: FS, amplitudeLimit: 0 })).toThrow(RangeError);
    expect(() => assessQuality(x, { sampleRate: FS, flatlineThreshold: -1 })).toThrow(RangeError);
    expect(() => assessQuality(x, { sampleRate: FS, range: [5, 5] })).toThrow(RangeError);
    expect(() => assessQuality([1], { sampleRate: FS })).toThrow(RangeError);
  });
});

/** Adds a 0.3 s raised-cosine blink of `amp` µV peaking at `peak`. */
function addBlink(x: Float64Array, peak: number, amp: number, fs = FS) {
  const len = Math.round(0.3 * fs);
  const start = peak - Math.floor(len / 2);
  for (let k = 0; k < len; k++) {
    const i = start + k;
    if (i >= 0 && i < x.length)
      x[i] = (x[i] as number) + amp * 0.5 * (1 - Math.cos((2 * Math.PI * k) / len));
  }
}

describe("detectBlinks", () => {
  it("finds injected blinks at the right place with no false positives", () => {
    const x = cleanEEG(11, 60 * FS);
    // Irregular intervals between 0.8 and 6 s, including near both edges.
    const peaks = [40, 700, 1100, 2400, 3900, 4200, 6100, 7000, 9800, 11000, 13500, 15200];
    for (const p of peaks) addBlink(x, p, 140);
    const found = detectBlinks(x, { sampleRate: FS });
    expect(found.length).toBe(peaks.length);
    found.forEach((f, i) => {
      expect(Math.abs(f - (peaks[i] as number))).toBeLessThanOrEqual(Math.round(0.05 * FS));
    });
  });

  it("produces no detections on long blink-free EEG", () => {
    for (const seed of [1, 2, 3]) {
      const x = createSyntheticEEG({ sampleRate: FS, channels: 1, seed }).next(300 * FS)[0];
      expect(detectBlinks(x as Float64Array, { sampleRate: FS })).toEqual([]);
    }
  });

  it("detects the synthetic generator's blinks at roughly the requested rate", () => {
    const minutes = 5;
    const x = createSyntheticEEG({ sampleRate: FS, channels: 1, seed: 4, blinkRate: 18 }).next(
      minutes * 60 * FS,
    )[0] as Float64Array;
    const rate = detectBlinks(x, { sampleRate: FS }).length / minutes;
    expect(rate).toBeGreaterThan(18 * 0.7);
    expect(rate).toBeLessThan(18 * 1.3);
  });

  it("merges peaks inside the refractory period, keeping the larger", () => {
    // Two separate excursions 0.45 s apart: one blink under the default
    // 0.5 s refractory period, two under a 0.2 s one.
    const second = 1000 + Math.round(0.45 * FS);
    const x = new Float64Array(10 * FS);
    addBlink(x, 1000, 160);
    addBlink(x, second, 200);
    const merged = detectBlinks(x, { sampleRate: FS });
    expect(merged.length).toBe(1);
    expect(Math.abs((merged[0] as number) - second)).toBeLessThanOrEqual(1);
    expect(detectBlinks(x, { sampleRate: FS, refractory: 0.2 }).length).toBe(2);
  });

  it("respects threshold and polarity", () => {
    const x = new Float64Array(10 * FS);
    addBlink(x, 600, 60);
    addBlink(x, 1600, -150);
    // The raised-cosine peak is flat across two samples, so allow ±1.
    const near = (found: number[], expected: number[]) => {
      expect(found.length).toBe(expected.length);
      found.forEach((f, i) => {
        expect(Math.abs(f - (expected[i] as number))).toBeLessThanOrEqual(1);
      });
    };
    expect(detectBlinks(x, { sampleRate: FS })).toEqual([]);
    near(detectBlinks(x, { sampleRate: FS, threshold: 40 }), [600]);
    near(detectBlinks(x, { sampleRate: FS, polarity: "negative" }), [1600]);
    near(detectBlinks(x, { sampleRate: FS, polarity: "both", threshold: 40 }), [600, 1600]);
  });

  it("handles empty input and validates options", () => {
    expect(detectBlinks([], { sampleRate: FS })).toEqual([]);
    expect(() => detectBlinks([1], { sampleRate: 0 })).toThrow(RangeError);
    expect(() => detectBlinks([1], { sampleRate: FS, threshold: 0 })).toThrow(RangeError);
    expect(() => detectBlinks([1], { sampleRate: FS, refractory: -1 })).toThrow(RangeError);
  });
});
