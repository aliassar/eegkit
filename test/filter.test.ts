import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  butterworthBandpass,
  butterworthBandstop,
  butterworthHighpass,
  butterworthLowpass,
  cascade,
  FilterBank,
  filtfilt,
  filtfiltPadLength,
  frequencyResponse,
  IIRFilter,
  notch,
  type SOS,
  sosfilt,
  sosfiltZi,
} from "../src/filter";

// ------------------------------------------------------------------ fixtures

interface FilterCase {
  name: string;
  type: "lowpass" | "highpass" | "bandpass" | "bandstop" | "notch";
  order?: number;
  cutoff?: number;
  low?: number;
  high?: number;
  frequency?: number;
  quality?: number;
  padlen: number;
  sosfilt: number[];
  sosfiltPrimed: number[];
  sosfiltfilt: number[];
  sosfiltfiltShort: number[];
  freqzMagnitude: number[];
  freqzPhase: number[];
  dcGain: number;
}

interface FilterFixtures {
  scipy: string;
  sampleRate: number;
  signal: number[];
  freqzPoints: number[];
  cases: FilterCase[];
}

const fx: FilterFixtures = JSON.parse(
  readFileSync(new URL("./fixtures/filter.json", import.meta.url), "utf8"),
);

function num(v: number | undefined): number {
  if (v === undefined) throw new Error("fixture field missing");
  return v;
}

function design(c: FilterCase, fs: number): SOS {
  switch (c.type) {
    case "lowpass":
      return butterworthLowpass(num(c.order), num(c.cutoff), fs);
    case "highpass":
      return butterworthHighpass(num(c.order), num(c.cutoff), fs);
    case "bandpass":
      return butterworthBandpass(num(c.order), num(c.low), num(c.high), fs);
    case "bandstop":
      return butterworthBandstop(num(c.order), num(c.low), num(c.high), fs);
    case "notch":
      return notch(num(c.frequency), fs, num(c.quality));
  }
}

/** Largest absolute elementwise difference; Infinity on a length mismatch. */
function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? NaN) - (b[i] ?? NaN)));
  return m;
}

/** Wrapped phase difference in (−π, π]. */
function phaseDiff(a: number, b: number): number {
  return Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
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

// Tolerances. Signal amplitudes are ~±45 µV; observed worst-case errors vs
// SciPy are ~6e-12 for outputs and ~1e-12 for |H|, so these leave margin
// while staying at the contract's 1e-9.
const OUTPUT_TOL = 1e-9;
const RESPONSE_TOL = 1e-11;

describe(`filters vs SciPy ${fx.scipy} (fs = ${fx.sampleRate} Hz, n = ${fx.signal.length})`, () => {
  const fs = fx.sampleRate;
  for (const c of fx.cases) {
    describe(c.name, () => {
      const sos = design(c, fs);

      it("sosfilt matches scipy.signal.sosfilt", () => {
        expect(maxAbsDiff(sosfilt(sos, fx.signal), c.sosfilt)).toBeLessThan(OUTPUT_TOL);
      });

      it("a primed IIRFilter matches sosfilt with zi = sosfilt_zi * x[0]", () => {
        const filter = new IIRFilter(sos);
        filter.prime(num(fx.signal[0]));
        expect(maxAbsDiff(filter.processBlock(fx.signal), c.sosfiltPrimed)).toBeLessThan(
          OUTPUT_TOL,
        );
      });

      it("filtfilt matches scipy.signal.sosfiltfilt (including its default padlen)", () => {
        expect(filtfiltPadLength(sos)).toBe(c.padlen);
        expect(maxAbsDiff(filtfilt(sos, fx.signal), c.sosfiltfilt)).toBeLessThan(OUTPUT_TOL);
        const short = fx.signal.slice(0, c.padlen + 1);
        expect(maxAbsDiff(filtfilt(sos, short), c.sosfiltfiltShort)).toBeLessThan(OUTPUT_TOL);
        expect(() => filtfilt(sos, short.slice(0, -1))).toThrow(RangeError);
      });

      it("frequencyResponse matches scipy.signal.sosfreqz", () => {
        const { magnitude, phase } = frequencyResponse(sos, fx.freqzPoints, fs);
        expect(maxAbsDiff(magnitude, c.freqzMagnitude)).toBeLessThan(RESPONSE_TOL);
        c.freqzPhase.forEach((expected, i) => {
          // Phase is undefined at exact zeros of H (DC of a band-pass, the notch frequency).
          if (num(c.freqzMagnitude[i]) < 1e-9) return;
          expect(phaseDiff(num(phase[i]), expected)).toBeLessThan(1e-9);
        });
        expect(
          Math.abs(num(frequencyResponse(sos, [0], fs).magnitude[0]) - Math.abs(c.dcGain)),
        ).toBeLessThan(RESPONSE_TOL);
      });
    });
  }
});

// ------------------------------------------------------------------ design properties

describe("Butterworth design properties", () => {
  const rates = [128, 250, 256, 500, 1000];
  const db = (m: number) => 20 * Math.log10(m);

  it("|H| = 1/√2 at the cutoff of low-pass and high-pass filters, orders 1–10", () => {
    for (const fs of rates) {
      for (let order = 1; order <= 10; order++) {
        for (const cutoff of [0.5, 1, 4, 30, fs * 0.45]) {
          const lp = frequencyResponse(butterworthLowpass(order, cutoff, fs), [cutoff], fs);
          const hp = frequencyResponse(butterworthHighpass(order, cutoff, fs), [cutoff], fs);
          expect(num(lp.magnitude[0])).toBeCloseTo(Math.SQRT1_2, 9);
          expect(num(hp.magnitude[0])).toBeCloseTo(Math.SQRT1_2, 9);
        }
      }
    }
  });

  it("|H| = 1/√2 at both edges of band-pass and band-stop filters", () => {
    for (const fs of rates) {
      for (let order = 1; order <= 6; order++) {
        for (const [low, high] of [
          [1, 40],
          [8, 13],
          [0.5, fs * 0.4],
        ] as const) {
          for (const sos of [
            butterworthBandpass(order, low, high, fs),
            butterworthBandstop(order, low, high, fs),
          ]) {
            const { magnitude } = frequencyResponse(sos, [low, high], fs);
            expect(num(magnitude[0])).toBeCloseTo(Math.SQRT1_2, 8);
            expect(num(magnitude[1])).toBeCloseTo(Math.SQRT1_2, 8);
          }
        }
      }
    }
  });

  it("has unit gain where the passband is flat: DC (low-pass), Nyquist (high-pass), band centre (band-pass)", () => {
    for (const fs of rates) {
      for (let order = 1; order <= 8; order++) {
        const lp = frequencyResponse(butterworthLowpass(order, 10, fs), [0], fs);
        expect(num(lp.magnitude[0])).toBeCloseTo(1, 12);
        const hp = frequencyResponse(butterworthHighpass(order, 10, fs), [fs / 2], fs);
        expect(num(hp.magnitude[0])).toBeCloseTo(1, 12);
        // The digital band centre is the geometric mean of the pre-warped edges.
        const [low, high] = [8, 13];
        const centre =
          (fs / Math.PI) *
          Math.atan(Math.sqrt(Math.tan((Math.PI * low) / fs) * Math.tan((Math.PI * high) / fs)));
        const bp = frequencyResponse(butterworthBandpass(order, low, high, fs), [centre, 0], fs);
        expect(num(bp.magnitude[0])).toBeCloseTo(1, 10);
        expect(num(bp.magnitude[1])).toBeLessThan(1e-12);
        const bs = frequencyResponse(butterworthBandstop(order, low, high, fs), [0, centre], fs);
        expect(num(bs.magnitude[0])).toBeCloseTo(1, 12);
        expect(num(bs.magnitude[1])).toBeLessThan(1e-9);
      }
    }
  });

  it("rolls off at 6·order dB per octave well above a low-pass cutoff", () => {
    // A high rate keeps bilinear frequency warping negligible at 200–400 Hz.
    const fs = 100000;
    for (let order = 1; order <= 6; order++) {
      const { magnitude } = frequencyResponse(butterworthLowpass(order, 10, fs), [200, 400], fs);
      const slope = db(num(magnitude[1])) - db(num(magnitude[0]));
      expect(slope).toBeCloseTo(-6.0206 * order, 1);
    }
  });

  it("notch removes its frequency (> 40 dB) and leaves distant frequencies alone", () => {
    for (const fs of rates) {
      for (const f0 of [50, 60]) {
        if (f0 >= fs / 2) continue;
        for (const q of [10, 30, 35]) {
          const sos = notch(f0, fs, q);
          const { magnitude } = frequencyResponse(sos, [f0, f0 / 2, 0, 10], fs);
          expect(db(num(magnitude[0]))).toBeLessThan(-40);
          expect(num(magnitude[1])).toBeGreaterThan(0.99);
          expect(num(magnitude[2])).toBeCloseTo(1, 12);
          expect(num(magnitude[3])).toBeGreaterThan(0.99);
        }
      }
    }
  });

  it("notch attenuates a 50 Hz sine by > 40 dB once settled, in the time domain", () => {
    const fs = 250;
    const x = sine(5000, 50, fs, 100);
    const y = sosfilt(notch(50, fs), x);
    const tail = y.subarray(4000);
    const peak = tail.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    expect(db(peak / 100)).toBeLessThan(-40);
  });

  it("notch matches scipy.signal.iirnotch's closed form", () => {
    const [s] = notch(50, 250, 30);
    // b = g·[1, −2cos w0, 1], a = [1, −2g·cos w0, 2g − 1], g = 1 / (1 + tan(bw / 2))
    const w0 = (2 * Math.PI * 50) / 250;
    const g = 1 / (1 + Math.tan(w0 / 30 / 2));
    expect(s).toBeDefined();
    expect(s?.b0).toBeCloseTo(g, 15);
    expect(s?.b1).toBeCloseTo(-2 * g * Math.cos(w0), 15);
    expect(s?.b2).toBeCloseTo(g, 15);
    expect(s?.a1).toBeCloseTo(-2 * g * Math.cos(w0), 15);
    expect(s?.a2).toBeCloseTo(2 * g - 1, 15);
  });

  it("uses ceil(order / 2) sections for low/high-pass and `order` sections for band filters", () => {
    for (let order = 1; order <= 9; order++) {
      expect(butterworthLowpass(order, 10, 256)).toHaveLength(Math.ceil(order / 2));
      expect(butterworthHighpass(order, 10, 256)).toHaveLength(Math.ceil(order / 2));
      expect(butterworthBandpass(order, 8, 13, 256)).toHaveLength(order);
      expect(butterworthBandstop(order, 48, 52, 256)).toHaveLength(order);
    }
  });

  it("returns frozen SOS arrays", () => {
    const sos = butterworthLowpass(4, 10, 256);
    expect(Object.isFrozen(sos)).toBe(true);
    expect(Object.isFrozen(sos[0])).toBe(true);
  });
});

// ------------------------------------------------------------------ application properties

describe("filtfilt", () => {
  it("has zero phase: a passband sine comes back aligned, scaled by |H|²", () => {
    const fs = 256;
    const sos = butterworthBandpass(4, 1, 40, fs);
    for (const f0 of [6, 10, 22]) {
      const x = sine(8192, f0, fs, 20, 0.7);
      const y = filtfilt(sos, x);
      const gain2 = num(frequencyResponse(sos, [f0], fs).magnitude[0]) ** 2;
      // Away from the edge transients the output equals the input times |H|², with no shift.
      let err = 0;
      for (let i = 3072; i < 5120; i++)
        err = Math.max(err, Math.abs(num(y[i]) - gain2 * num(x[i])));
      expect(err).toBeLessThan(1e-6 * 20);
      // The causal filter, by contrast, shifts the sine noticeably.
      const causal = sosfilt(sos, x);
      let causalErr = 0;
      for (let i = 3072; i < 5120; i++)
        causalErr = Math.max(causalErr, Math.abs(num(causal[i]) - num(x[i])));
      expect(causalErr).toBeGreaterThan(1);
    }
  });

  it("keeps a constant signal constant through a low-pass (no edge transients)", () => {
    const y = filtfilt(butterworthLowpass(4, 10, 256), new Float64Array(100).fill(-73.5));
    expect(maxAbsDiff(y, new Float64Array(100).fill(-73.5))).toBeLessThan(1e-10);
  });

  it("accepts plain arrays and does not modify its input", () => {
    const x = Array.from(sine(200, 5, 100));
    const copy = [...x];
    const y = filtfilt(butterworthLowpass(2, 10, 100), x);
    expect(y).toBeInstanceOf(Float64Array);
    expect(y).toHaveLength(200);
    expect(x).toEqual(copy);
  });

  it("throws a clear RangeError when the signal is not longer than the padding", () => {
    const sos = butterworthLowpass(4, 10, 256); // 2 sections → 3 · (2·2 + 1) = 15
    expect(filtfiltPadLength(sos)).toBe(15);
    expect(filtfiltPadLength(butterworthLowpass(3, 10, 256))).toBe(12); // one first-order section
    expect(() => filtfilt(sos, new Float64Array(15))).toThrow(/longer than its padding of 15/);
    expect(() => filtfilt(sos, [])).toThrow(RangeError);
    expect(() => filtfilt(sos, new Float64Array(16))).not.toThrow();
  });
});

describe("IIRFilter", () => {
  const fs = 256;
  const sos = cascade(butterworthBandpass(4, 1, 40, fs), notch(50, fs));
  const rand = gaussian(42);
  const x = Float64Array.from({ length: 3000 }, (_, i) => 30 * Math.sin(i / 7) + 5 * rand() + 400);

  it("process() one sample at a time == processBlock() == sosfilt(), bit for bit", () => {
    const reference = sosfilt(sos, x);
    const perSample = new IIRFilter(sos);
    const streamed = Float64Array.from(x, (v) => perSample.process(v));
    expect(streamed).toEqual(reference);
    expect(new IIRFilter(sos).processBlock(x)).toEqual(reference);
  });

  it("gives the same output for any block partitioning", () => {
    const reference = sosfilt(sos, x);
    const filter = new IIRFilter(sos);
    const out = new Float64Array(x.length);
    let pos = 0;
    for (const size of [1, 7, 256, 3, 1000, 0, 33]) {
      const block = x.subarray(pos, Math.min(pos + size, x.length));
      out.set(filter.processBlock(block), pos);
      pos += block.length;
    }
    out.set(filter.processBlock(x.subarray(pos)), pos);
    expect(out).toEqual(reference);
  });

  it("processBlock writes into `out`, including in place", () => {
    const filter = new IIRFilter(sos);
    const buf = Float64Array.from(x);
    const result = filter.processBlock(buf, buf);
    expect(result).toBe(buf);
    expect(buf).toEqual(sosfilt(sos, x));
    expect(() => new IIRFilter(sos).processBlock(x, new Float64Array(10))).toThrow(RangeError);
  });

  it("reset() clears state and clone() starts fresh with the same coefficients", () => {
    const filter = new IIRFilter(sos);
    const first = filter.processBlock(x.subarray(0, 100));
    const twin = filter.clone();
    expect(twin.sos).toBe(filter.sos);
    expect(twin.processBlock(x.subarray(0, 100))).toEqual(first);
    filter.reset();
    expect(filter.processBlock(x.subarray(0, 100))).toEqual(first);
  });

  it("prime(x0) removes the startup transient of a DC offset", () => {
    const hp = butterworthHighpass(2, 0.5, fs);
    const offset = new Float64Array(500).fill(800);
    const cold = new IIRFilter(hp).processBlock(offset);
    expect(Math.abs(num(cold[0]))).toBeGreaterThan(700);
    const primed = new IIRFilter(hp);
    primed.prime(800);
    expect(maxAbsDiff(primed.processBlock(offset), new Float64Array(500))).toBeLessThan(1e-9);
  });

  it("prime(x0) settles a low-pass at x0 · DC gain from the first sample", () => {
    const lp = cascade(butterworthLowpass(5, 30, fs), notch(50, fs));
    const filter = new IIRFilter(lp);
    filter.prime(-12.5);
    const y = filter.processBlock(new Float64Array(50).fill(-12.5));
    expect(maxAbsDiff(y, new Float64Array(50).fill(-12.5))).toBeLessThan(1e-11);
  });

  it("rejects malformed SOS", () => {
    expect(() => new IIRFilter([])).toThrow(TypeError);
    expect(() => new IIRFilter([{ b0: 1, b1: 0, b2: 0, a1: Number.NaN, a2: 0 }])).toThrow(
      TypeError,
    );
  });
});

describe("sosfiltZi", () => {
  it("gives the steady state: a constant input then produces a constant output", () => {
    const sos = butterworthBandstop(3, 45, 55, 250);
    const zi = sosfiltZi(sos);
    expect(zi).toHaveLength(2 * sos.length);
    const filter = new IIRFilter(sos);
    filter.prime(1);
    const y = filter.processBlock(new Float64Array(20).fill(1));
    expect(maxAbsDiff(y, new Float64Array(20).fill(1))).toBeLessThan(1e-12);
  });

  it("throws for a section with a pole at z = 1", () => {
    expect(() => sosfiltZi([{ b0: 1, b1: 0, b2: 0, a1: -1, a2: 0 }])).toThrow(RangeError);
  });
});

describe("cascade", () => {
  it("is equivalent to filtering with each SOS in turn", () => {
    const a = butterworthHighpass(2, 1, 256);
    const b = notch(50, 256);
    const x = sine(1000, 50, 256, 10);
    expect(sosfilt(cascade(a, b), x)).toEqual(sosfilt(b, sosfilt(a, x)));
    expect(cascade(a, b)).toHaveLength(a.length + b.length);
  });
});

describe("FilterBank", () => {
  const fs = 256;
  const channels = 8;
  const sos = cascade(butterworthBandpass(2, 1, 40, fs), notch(50, fs));
  const rand = gaussian(7);
  const chunk = Array.from({ length: channels }, (_, ch) =>
    Float64Array.from({ length: 700 }, (_, i) => 10 * Math.sin(i / (3 + ch)) + rand() + ch * 100),
  );

  it("processFrame and processChunk both equal sosfilt per channel", () => {
    const frameBank = new FilterBank(sos, channels);
    const frame = new Float64Array(channels);
    const out = new Float64Array(channels);
    const byFrame = chunk.map(() => new Float64Array(700));
    for (let t = 0; t < 700; t++) {
      for (let ch = 0; ch < channels; ch++) frame[ch] = num(chunk[ch]?.[t]);
      const result = frameBank.processFrame(frame, out);
      expect(result).toBe(out);
      for (let ch = 0; ch < channels; ch++) (byFrame[ch] as Float64Array)[t] = num(out[ch]);
    }
    const chunkBank = new FilterBank(sos, channels);
    const first = chunkBank.processChunk(chunk.map((c) => c.subarray(0, 300)));
    const rest = chunkBank.processChunk(chunk.map((c) => c.subarray(300)));
    for (let ch = 0; ch < channels; ch++) {
      const reference = sosfilt(sos, chunk[ch] as Float64Array);
      expect(byFrame[ch]).toEqual(reference);
      const joined = new Float64Array(700);
      joined.set(first[ch] as Float64Array);
      joined.set(rest[ch] as Float64Array, 300);
      expect(joined).toEqual(reference);
    }
  });

  it("keeps channels independent, primes per channel, and resets", () => {
    const bank = new FilterBank(butterworthHighpass(2, 0.5, fs), 3);
    expect(bank.channels).toBe(3);
    bank.prime([100, -200, 0]);
    const y = bank.processFrame([100, -200, 0]);
    expect(maxAbsDiff(y, [0, 0, 0])).toBeLessThan(1e-9);
    bank.reset();
    const cold = bank.processFrame([100, -200, 0]);
    expect(num(cold[0])).toBeGreaterThan(90);
    expect(num(cold[1])).toBeLessThan(-180);
    expect(cold[2]).toBe(0);
  });

  it("validates its arguments", () => {
    expect(() => new FilterBank(sos, 0)).toThrow(RangeError);
    expect(() => new FilterBank(sos, 2.5)).toThrow(RangeError);
    const bank = new FilterBank(sos, 2);
    expect(bank.sos).toBe(sos);
    expect(() => bank.processFrame([1, 2, 3])).toThrow(/expected 2/);
    expect(() => bank.processFrame([1, 2], new Float64Array(1))).toThrow(RangeError);
    expect(() => bank.processChunk([[1]])).toThrow(RangeError);
    expect(() => bank.prime([1])).toThrow(RangeError);
  });
});

// ------------------------------------------------------------------ validation

describe("parameter validation", () => {
  it("rejects cutoffs outside (0, Nyquist)", () => {
    for (const cutoff of [0, -1, 128, 200, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => butterworthLowpass(4, cutoff, 256)).toThrow(RangeError);
      expect(() => butterworthHighpass(4, cutoff, 256)).toThrow(RangeError);
    }
    expect(() => butterworthLowpass(4, 128, 256)).toThrow(/Nyquist frequency \(128 Hz\)/);
  });

  it("rejects bad orders", () => {
    for (const order of [0, -1, 1.5, Number.NaN]) {
      expect(() => butterworthLowpass(order, 10, 256)).toThrow(/order must be an integer >= 1/);
      expect(() => butterworthBandpass(order, 1, 10, 256)).toThrow(RangeError);
    }
  });

  it("rejects bad bands", () => {
    expect(() => butterworthBandpass(2, 13, 8, 256)).toThrow(
      /low \(13 Hz\) must be less than high/,
    );
    expect(() => butterworthBandpass(2, 8, 8, 256)).toThrow(RangeError);
    expect(() => butterworthBandstop(2, 0, 10, 256)).toThrow(RangeError);
    expect(() => butterworthBandstop(2, 10, 128, 256)).toThrow(RangeError);
  });

  it("rejects bad sample rates and notch parameters", () => {
    expect(() => butterworthLowpass(2, 10, 0)).toThrow(/sampleRate/);
    expect(() => butterworthLowpass(2, 10, -256)).toThrow(RangeError);
    expect(() => notch(50, 256, 0)).toThrow(/quality/);
    expect(() => notch(50, 256, -3)).toThrow(RangeError);
    expect(() => notch(0, 256)).toThrow(RangeError);
    expect(() => notch(128, 256)).toThrow(RangeError);
    expect(() => frequencyResponse(notch(50, 256), [1], 0)).toThrow(RangeError);
  });
});
