import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  bandPower,
  bandPowers,
  blackman,
  fft,
  getWindow,
  hamming,
  hann,
  peakFrequency,
  periodogram,
  rectangular,
  relativeBandPowers,
  rfft,
  type WelchOptions,
  welch,
} from "../src/spectral";
import { STANDARD_BANDS } from "../src/types";

// ------------------------------------------------------------------ fixtures

interface SpectralFixtures {
  scipy: string;
  sampleRate: number;
  signal: number[];
  tukey200: number[];
  welch: {
    name: string;
    options: Omit<WelchOptions, "sampleRate" | "window"> & { window?: string };
    frequencies: number[];
    power: number[];
  }[];
  periodogram: {
    name: string;
    length: number;
    options: { window?: "hann" | "hamming"; scaling?: "spectrum"; detrend?: "linear" };
    df: number;
    bins: number;
    power: number[];
  }[];
  bandPower: {
    bands: [string, number, number][];
    absolute: Record<string, number>;
    relative: Record<string, number>;
    relativeRange: [number, number];
  };
  fft: { n: number; re: number[]; im: number[]; outRe: number[]; outIm: number[] }[];
  rfft: { n: number; signal: number[]; outRe: number[]; outIm: number[] }[];
  windows: { type: "hann" | "hamming" | "blackman"; n: number; sym: boolean; values: number[] }[];
}

const fx: SpectralFixtures = JSON.parse(
  readFileSync(new URL("./fixtures/spectral.json", import.meta.url), "utf8"),
);

function num(v: number | undefined): number {
  if (v === undefined) throw new Error("value missing");
  return v;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? NaN) - (b[i] ?? NaN)));
  return m;
}

/** Seeded PRNG (mulberry32) with Box–Muller normals, for reproducible noise. */
function gaussian(seed: number): () => number {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
}

function sine(n: number, freq: number, fs: number, amplitude = 1, phase = 0): Float64Array {
  return Float64Array.from(
    { length: n },
    (_, i) => amplitude * Math.sin((2 * Math.PI * freq * i) / fs + phase),
  );
}

/** O(n²) reference DFT. */
function naiveDft(re: ArrayLike<number>, im: ArrayLike<number>) {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * ((j * k) % n)) / n;
      outRe[k] = num(outRe[k]) + num(re[j]) * Math.cos(a) - num(im[j]) * Math.sin(a);
      outIm[k] = num(outIm[k]) + num(re[j]) * Math.sin(a) + num(im[j]) * Math.cos(a);
    }
  }
  return { re: outRe, im: outIm };
}

// Observed worst-case errors vs NumPy/SciPy: FFT ~7e-14, rFFT ~1.4e-12 (|X| up to ~800),
// windows ~5e-16, Welch/periodogram ~4e-12 (PSD up to ~1200 µV²/Hz), band power ~4e-14.
const TOL = 1e-9;

describe(`FFT vs NumPy (fixtures from SciPy ${fx.scipy})`, () => {
  for (const c of fx.fft) {
    it(`fft, n = ${c.n}${(c.n & (c.n - 1)) === 0 ? " (radix-2)" : " (Bluestein)"}`, () => {
      const re = Float64Array.from(c.re);
      const im = Float64Array.from(c.im);
      fft(re, im);
      expect(maxAbsDiff(re, c.outRe)).toBeLessThan(TOL);
      expect(maxAbsDiff(im, c.outIm)).toBeLessThan(TOL);
    });
  }
  for (const c of fx.rfft) {
    it(`rfft, n = ${c.n}`, () => {
      const { re, im } = rfft(c.signal);
      expect(re).toHaveLength(Math.floor(c.n / 2) + 1);
      expect(maxAbsDiff(re, c.outRe)).toBeLessThan(TOL);
      expect(maxAbsDiff(im, c.outIm)).toBeLessThan(TOL);
    });
  }
});

describe("FFT properties", () => {
  const rand = gaussian(1);
  const sizes = [
    2, 3, 4, 6, 7, 8, 12, 17, 31, 32, 60, 100, 127, 128, 200, 250, 256, 500, 509, 1000, 1024,
  ];

  it("matches a naive DFT for many sizes", () => {
    for (const n of sizes) {
      const re = Float64Array.from({ length: n }, rand);
      const im = Float64Array.from({ length: n }, rand);
      const ref = naiveDft(re, im);
      fft(re, im);
      const scale = Math.sqrt(n);
      expect(maxAbsDiff(re, ref.re) / scale).toBeLessThan(1e-12);
      expect(maxAbsDiff(im, ref.im) / scale).toBeLessThan(1e-12);
    }
  });

  it("satisfies Parseval: Σ|x|² = Σ|X|² / n", () => {
    for (const n of sizes) {
      const re = Float64Array.from({ length: n }, rand);
      const im = Float64Array.from({ length: n }, rand);
      let timeEnergy = 0;
      for (let i = 0; i < n; i++) timeEnergy += num(re[i]) ** 2 + num(im[i]) ** 2;
      fft(re, im);
      let freqEnergy = 0;
      for (let i = 0; i < n; i++) freqEnergy += num(re[i]) ** 2 + num(im[i]) ** 2;
      expect(freqEnergy / n / timeEnergy).toBeCloseTo(1, 12);
    }
  });

  it("rfft equals the first n/2 + 1 bins of the complex FFT, for even and odd n", () => {
    for (const n of [1, 2, 3, 5, 10, 64, 99, 250, 256, 255]) {
      const x = Float64Array.from({ length: n }, rand);
      const re = Float64Array.from(x);
      const im = new Float64Array(n);
      fft(re, im);
      const half = rfft(x);
      const bins = Math.floor(n / 2) + 1;
      expect(maxAbsDiff(half.re, re.subarray(0, bins))).toBeLessThan(1e-12 * n);
      expect(maxAbsDiff(half.im, im.subarray(0, bins))).toBeLessThan(1e-12 * n);
    }
  });

  it("stays correct when the plan cache evicts sizes", () => {
    // More distinct sizes than the cache holds, then revisit the first ones.
    const many = Array.from({ length: 40 }, (_, i) => 10 + i * 3);
    for (const n of [...many, ...many.slice(0, 5)]) {
      const re = Float64Array.from({ length: n }, rand);
      const im = Float64Array.from({ length: n }, rand);
      const ref = naiveDft(re, im);
      fft(re, im);
      expect(maxAbsDiff(re, ref.re)).toBeLessThan(1e-10);
    }
  });

  it("puts a pure tone in its bin", () => {
    const { re, im } = rfft(sine(250, 10, 250, 3));
    const mag = Array.from(re, (r, k) => Math.hypot(r, num(im[k])));
    expect(mag.indexOf(Math.max(...mag))).toBe(10);
    expect(num(mag[10])).toBeCloseTo((3 * 250) / 2, 9);
  });

  it("handles trivial sizes and rejects mismatched buffers", () => {
    const re = Float64Array.of(4);
    const im = Float64Array.of(-1);
    fft(re, im);
    expect([re[0], im[0]]).toEqual([4, -1]);
    fft(new Float64Array(0), new Float64Array(0));
    expect(() => fft(new Float64Array(4), new Float64Array(3))).toThrow(RangeError);
    expect(() => rfft([])).toThrow(RangeError);
  });
});

describe("windows vs scipy.signal.get_window", () => {
  const fns = { hann, hamming, blackman };
  it("match for periodic and symmetric variants", () => {
    for (const c of fx.windows) {
      expect(maxAbsDiff(fns[c.type](c.n, c.sym), c.values)).toBeLessThan(1e-15);
    }
  });

  it("rectangular is all ones; getWindow dispatches by name; lengths are validated", () => {
    expect(Array.from(rectangular(4))).toEqual([1, 1, 1, 1]);
    expect(getWindow("hann", 16)).toEqual(hann(16));
    expect(getWindow("rectangular", 3)).toEqual(rectangular(3));
    expect(hann(0)).toHaveLength(0);
    expect(() => hann(-1)).toThrow(RangeError);
    expect(() => hamming(2.5)).toThrow(RangeError);
    // @ts-expect-error unknown window name
    expect(() => getWindow("kaiser", 8)).toThrow(TypeError);
  });

  it("periodic Hann sums to a constant under 50 % overlap", () => {
    const w = hann(64);
    for (let i = 0; i < 32; i++) expect(num(w[i]) + num(w[i + 32])).toBeCloseTo(1, 15);
  });
});

describe(`welch vs scipy.signal.welch ${fx.scipy}`, () => {
  for (const c of fx.welch) {
    it(c.name, () => {
      const { window, ...rest } = c.options;
      const options: WelchOptions = {
        ...rest,
        sampleRate: fx.sampleRate,
        ...(window === undefined
          ? {}
          : { window: window === "tukey200" ? fx.tukey200 : (window as "hann") }),
      };
      const psd = welch(fx.signal, options);
      expect(maxAbsDiff(psd.frequencies, c.frequencies)).toBeLessThan(1e-12);
      expect(maxAbsDiff(psd.power, c.power)).toBeLessThan(TOL);
    });
  }
});

describe(`periodogram vs scipy.signal.periodogram ${fx.scipy}`, () => {
  for (const c of fx.periodogram) {
    it(c.name, () => {
      const psd = periodogram(fx.signal.slice(0, c.length), {
        sampleRate: fx.sampleRate,
        ...c.options,
      });
      expect(psd.power).toHaveLength(c.bins);
      expect(num(psd.frequencies[1])).toBeCloseTo(c.df, 12);
      expect(maxAbsDiff(psd.power, c.power)).toBeLessThan(TOL);
    });
  }
});

describe("welch properties", () => {
  const fs = 256;

  it("peaks at the frequency of a pure sine", () => {
    for (const f0 of [3, 10, 17, 40, 60]) {
      const { frequencies, power } = welch(sine(2048, f0, fs, 10), { sampleRate: fs });
      const k = power.indexOf(Math.max(...power));
      expect(frequencies[k]).toBe(f0);
    }
  });

  it("density integrates to the signal power (A²/2 for a sine)", () => {
    const amplitude = 7;
    const psd = welch(sine(4096, 12, fs, amplitude), { sampleRate: fs });
    expect(bandPower(psd, [0, fs / 2])).toBeCloseTo(amplitude ** 2 / 2, 6);
  });

  it("spectrum scaling reads the squared RMS amplitude at the peak", () => {
    const amplitude = 5;
    const { power } = welch(sine(4096, 20, fs, amplitude), { sampleRate: fs, scaling: "spectrum" });
    expect(Math.max(...power)).toBeCloseTo(amplitude ** 2 / 2, 9);
  });

  it("white noise has a flat density equal to σ² / (fs / 2)", () => {
    const rand = gaussian(3);
    const sigma = 4;
    const x = Float64Array.from({ length: 256 * 600 }, () => sigma * rand());
    const { power } = welch(x, { sampleRate: fs });
    const inner = Array.from(power.subarray(1, -1));
    const mean = inner.reduce((a, b) => a + b, 0) / inner.length;
    expect(mean / (sigma ** 2 / (fs / 2))).toBeCloseTo(1, 1);
  });

  it("median averaging resists a burst artifact that ruins the mean", () => {
    const rand = gaussian(4);
    const x = Float64Array.from(
      { length: 256 * 20 },
      (_, i) => rand() + Math.sin((2 * Math.PI * 10 * i) / fs),
    );
    for (let i = 2560; i < 2816; i++)
      x[i] = num(x[i]) + 500 * Math.sin((2 * Math.PI * 30 * i) / fs);
    const mean = welch(x, { sampleRate: fs });
    const median = welch(x, { sampleRate: fs, average: "median" });
    expect(num(mean.power[30]) / num(median.power[30])).toBeGreaterThan(100);
  });

  it("clamps segmentLength to the signal length, like SciPy", () => {
    const x = sine(100, 10, fs);
    expect(welch(x, { sampleRate: fs }).power).toHaveLength(51);
    expect(welch(x, { sampleRate: fs, segmentLength: 1000 }).power).toHaveLength(51);
  });

  it("validates its options", () => {
    const x = sine(512, 10, fs);
    expect(() => welch([], { sampleRate: fs })).toThrow(/empty/);
    expect(() => welch(x, { sampleRate: 0 })).toThrow(/sampleRate/);
    expect(() => welch(x, { sampleRate: fs, segmentLength: 0 })).toThrow(RangeError);
    expect(() => welch(x, { sampleRate: fs, segmentLength: 100.5 })).toThrow(RangeError);
    expect(() => welch(x, { sampleRate: fs, overlap: 256 })).toThrow(/overlap/);
    expect(() => welch(x, { sampleRate: fs, overlap: -1 })).toThrow(RangeError);
    expect(() => welch(x, { sampleRate: fs, window: hann(64), segmentLength: 32 })).toThrow(
      /must equal the window length/,
    );
    expect(() => welch(x, { sampleRate: fs, window: hann(1024) })).toThrow(/exceeds signal length/);
    expect(() => welch(x, { sampleRate: fs, window: [] })).toThrow(RangeError);
    // @ts-expect-error invalid option value
    expect(() => welch(x, { sampleRate: fs, average: "max" })).toThrow(TypeError);
    // @ts-expect-error invalid option value
    expect(() => welch(x, { sampleRate: fs, detrend: "quadratic" })).toThrow(TypeError);
    // @ts-expect-error invalid option value
    expect(() => welch(x, { sampleRate: fs, scaling: "power" })).toThrow(TypeError);
    expect(() => periodogram(x, { sampleRate: fs, window: hann(64) })).toThrow(RangeError);
  });
});

describe("band power", () => {
  const psd = welch(fx.signal, { sampleRate: fx.sampleRate });

  it("matches trapezoid integration of the interpolated SciPy PSD", () => {
    for (const [name, low, high] of fx.bandPower.bands) {
      expect(Math.abs(bandPower(psd, [low, high]) - num(fx.bandPower.absolute[name]))).toBeLessThan(
        1e-12,
      );
    }
    const powers = bandPowers(psd);
    for (const band of STANDARD_BANDS) {
      expect(Math.abs(num(powers[band.name]) - num(fx.bandPower.absolute[band.name]))).toBeLessThan(
        1e-12,
      );
    }
    const relative = relativeBandPowers(psd);
    for (const band of STANDARD_BANDS) {
      expect(
        Math.abs(num(relative[band.name]) - num(fx.bandPower.relative[band.name])),
      ).toBeLessThan(1e-14);
    }
  });

  it("accepts a Band object or a [low, high] pair", () => {
    expect(bandPower(psd, { name: "alpha", low: 8, high: 13 })).toBe(bandPower(psd, [8, 13]));
  });

  it("adjacent bands tile exactly, even with edges between bins", () => {
    for (const edges of [
      [1, 4, 8, 13, 30, 45],
      [0.7, 3.3, 7.9, 12.1, 29.5, 44.4],
    ]) {
      let sum = 0;
      for (let i = 0; i + 1 < edges.length; i++)
        sum += bandPower(psd, [num(edges[i]), num(edges[i + 1])]);
      expect(sum).toBeCloseTo(bandPower(psd, [num(edges[0]), num(edges.at(-1))]), 10);
    }
  });

  it("is proportional to bandwidth for white noise", () => {
    // An exactly flat density, then a real white-noise estimate.
    const flat = {
      frequencies: Float64Array.from({ length: 129 }, (_, k) => k),
      power: new Float64Array(129).fill(0.25),
    };
    for (const [low, high] of [
      [1, 4],
      [4, 8],
      [8, 13],
      [13, 30],
      [2.5, 3.1],
    ] as const) {
      expect(bandPower(flat, [low, high])).toBeCloseTo(0.25 * (high - low), 12);
    }
    const rand = gaussian(9);
    const fs = 256;
    const noise = welch(
      Float64Array.from({ length: fs * 600 }, () => 3 * rand()),
      { sampleRate: fs },
    );
    const density = 9 / (fs / 2);
    for (const [low, high] of [
      [4, 8],
      [8, 13],
      [13, 30],
      [30, 45],
      [10, 100],
    ] as const) {
      expect(bandPower(noise, [low, high]) / (density * (high - low))).toBeCloseTo(1, 1);
    }
  });

  it("clamps bands to the spectrum's range and returns 0 outside it", () => {
    const flat = { frequencies: [0, 1, 2, 3], power: [1, 1, 1, 1] };
    expect(bandPower(flat, [2, 10])).toBeCloseTo(1, 15);
    expect(bandPower(flat, [5, 10])).toBe(0);
    expect(bandPower({ frequencies: [0], power: [1] }, [0, 1])).toBe(0);
  });

  it("relative powers sum to 1 over the bands' span and are NaN for zero total", () => {
    const relative = relativeBandPowers(psd);
    const sum = Object.values(relative).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 12);
    const custom = relativeBandPowers(psd, STANDARD_BANDS, [0, 125]);
    expect(num(custom.alpha)).toBeLessThan(num(relative.alpha));
    const silent = { frequencies: [0, 1, 2], power: [0, 0, 0] };
    expect(relativeBandPowers(silent, [{ name: "x", low: 0, high: 1 }]).x).toBeNaN();
    expect(relativeBandPowers(psd, [])).toEqual({});
  });

  it("validates bands and spectra", () => {
    expect(() => bandPower(psd, [13, 8])).toThrow(/low < high/);
    expect(() => bandPower(psd, [-1, 4])).toThrow(RangeError);
    expect(() => bandPower(psd, [1, Number.NaN])).toThrow(RangeError);
    expect(() => bandPower({ frequencies: [0, 1], power: [1] }, [0, 1])).toThrow(/same length/);
  });
});

describe("peakFrequency", () => {
  const fs = 256;

  it("finds a sine's frequency to a small fraction of a bin (1 Hz bins)", () => {
    for (let f0 = 8; f0 <= 12; f0 += 0.1) {
      const psd = welch(sine(2048, f0, fs, 10, 0.3), { sampleRate: fs, segmentLength: 256 });
      expect(Math.abs(peakFrequency(psd, [7, 14]) - f0)).toBeLessThan(0.03);
    }
  });

  it("finds the individual alpha peak in noisy data", () => {
    const rand = gaussian(11);
    const x = Float64Array.from(
      { length: fs * 60 },
      (_, i) => 8 * Math.sin((2 * Math.PI * 10.4 * i) / fs) + 5 * rand(),
    );
    const psd = welch(x, { sampleRate: fs, segmentLength: 512 });
    expect(peakFrequency(psd, [7, 14])).toBeCloseTo(10.4, 1);
  });

  it("returns the edge bin unrefined when the peak lies outside the range, and NaN with no bins", () => {
    // A 1/f-like spectrum rising towards low frequencies: the range's lowest bin wins.
    const frequencies = Float64Array.from({ length: 30 }, (_, k) => k);
    const psd = { frequencies, power: frequencies.map((f) => 1 / (1 + f)) };
    expect(peakFrequency(psd, [7, 14])).toBe(7);
    expect(peakFrequency(psd, [10.2, 10.8])).toBeNaN();
    expect(() => peakFrequency(psd, [14, 7])).toThrow(RangeError);
  });
});
