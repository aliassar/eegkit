import { describe, expect, it } from "vitest";
import { frontalAlphaAsymmetry } from "../src/metrics/asymmetry";
import { Baseline } from "../src/metrics/baseline";
import { alphaThetaRatio, bandRatio, engagementIndex, thetaBetaRatio } from "../src/metrics/ratios";
import { ExponentialSmoother } from "../src/metrics/smoothing";

const powers = { delta: 20, theta: 12, alpha: 18, beta: 6, gamma: 1 };

describe("band ratios", () => {
  it("computes the named ratios", () => {
    expect(bandRatio(powers, "alpha", "delta")).toBeCloseTo(0.9, 15);
    expect(thetaBetaRatio(powers)).toBe(2);
    expect(alphaThetaRatio(powers)).toBe(1.5);
    expect(engagementIndex(powers)).toBeCloseTo(6 / 30, 15);
  });

  it("returns NaN (never Infinity) when the denominator is zero or invalid", () => {
    expect(thetaBetaRatio({ ...powers, beta: 0 })).toBeNaN();
    expect(alphaThetaRatio({ ...powers, theta: -1 })).toBeNaN();
    expect(engagementIndex({ ...powers, alpha: 0, theta: 0 })).toBeNaN();
    expect(bandRatio({ a: 1, b: Number.POSITIVE_INFINITY }, "a", "b")).toBeNaN();
    expect(bandRatio({ a: Number.NaN, b: 1 }, "a", "b")).toBeNaN();
    expect(bandRatio({ a: 0, b: 1 }, "a", "b")).toBe(0);
  });

  it("throws a TypeError naming the missing band", () => {
    expect(() => bandRatio(powers, "alpha", "mu")).toThrow(/"mu" not found.*alpha/);
    expect(() => thetaBetaRatio({ theta: 1 })).toThrow(TypeError);
  });
});

describe("frontalAlphaAsymmetry", () => {
  it("is ln(right) - ln(left)", () => {
    expect(frontalAlphaAsymmetry(4, 6)).toBeCloseTo(Math.log(1.5), 15);
    expect(frontalAlphaAsymmetry(6, 4)).toBeCloseTo(-Math.log(1.5), 15);
    expect(frontalAlphaAsymmetry(5, 5)).toBe(0);
  });

  it("accepts band powers and uses their alpha entry", () => {
    expect(frontalAlphaAsymmetry({ alpha: 2, beta: 99 }, { alpha: 8 })).toBeCloseTo(
      Math.log(4),
      15,
    );
    expect(() => frontalAlphaAsymmetry({ beta: 1 }, 1)).toThrow(TypeError);
  });

  it("is invariant to a common gain", () => {
    expect(frontalAlphaAsymmetry(3, 7)).toBeCloseTo(frontalAlphaAsymmetry(300, 700), 12);
  });

  it("returns NaN for non-positive or non-finite power", () => {
    expect(frontalAlphaAsymmetry(0, 1)).toBeNaN();
    expect(frontalAlphaAsymmetry(1, -1)).toBeNaN();
    expect(frontalAlphaAsymmetry(1, Number.POSITIVE_INFINITY)).toBeNaN();
  });
});

describe("ExponentialSmoother", () => {
  it("validates options", () => {
    expect(() => new ExponentialSmoother({ alpha: 0 })).toThrow(RangeError);
    expect(() => new ExponentialSmoother({ alpha: 1.5 })).toThrow(RangeError);
    expect(() => new ExponentialSmoother({ timeConstant: 0, rate: 4 })).toThrow(RangeError);
    expect(() => new ExponentialSmoother({ timeConstant: 1, rate: -4 })).toThrow(RangeError);
  });

  it("initialises from the first value and applies y += alpha (x - y)", () => {
    const s = new ExponentialSmoother({ alpha: 0.5 });
    expect(s.value).toBeNaN();
    expect(s.update(10)).toBe(10);
    expect(s.update(20)).toBe(15);
    expect(s.update(20)).toBe(17.5);
    expect(s.value).toBe(17.5);
  });

  it("alpha = 1 passes values through", () => {
    const s = new ExponentialSmoother({ alpha: 1 });
    s.update(3);
    expect(s.update(-7)).toBe(-7);
  });

  it("reaches 1 - 1/e of a step after one time constant, at any update rate", () => {
    for (const rate of [1, 4, 10, 256]) {
      for (const timeConstant of [1, 2, 10]) {
        const s = new ExponentialSmoother({ timeConstant, rate });
        s.update(0);
        const steps = timeConstant * rate;
        let y = 0;
        for (let i = 0; i < steps; i++) y = s.update(1);
        // after exactly timeConstant seconds of updates
        expect(y).toBeCloseTo(1 - Math.exp(-1), 12);
        // ... and converges to the new level
        for (let i = 0; i < steps * 40; i++) y = s.update(1);
        expect(y).toBeCloseTo(1, 12);
      }
    }
  });

  it("ignores non-finite inputs and can be reset", () => {
    const s = new ExponentialSmoother({ alpha: 0.5 });
    s.update(4);
    expect(s.update(Number.NaN)).toBe(4);
    expect(s.update(Number.POSITIVE_INFINITY)).toBe(4);
    s.reset();
    expect(s.value).toBeNaN();
    expect(s.update(Number.NaN)).toBeNaN();
    expect(s.update(9)).toBe(9);
  });
});

/** Two-pass reference: mean, then sum of squared deviations. */
function twoPass(xs: number[]) {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const ss = xs.reduce((a, x) => a + (x - mean) ** 2, 0);
  return { mean, std: Math.sqrt(ss / (xs.length - 1)) };
}

describe("Baseline", () => {
  it("matches a two-pass mean and sample std", () => {
    let seed = 7;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    const xs = Array.from({ length: 1000 }, () => 50 + 10 * (rand() - 0.5));
    const b = new Baseline();
    for (const x of xs) b.add(x);
    const ref = twoPass(xs);
    expect(b.count).toBe(1000);
    expect(b.mean).toBeCloseTo(ref.mean, 12);
    expect(b.std).toBeCloseTo(ref.std, 12);
    expect(b.variance).toBeCloseTo(ref.std ** 2, 10);
  });

  it("stays accurate with a huge offset where the naive Σx² formula fails", () => {
    const xs = Array.from({ length: 100 }, (_, i) => 1e9 + (i % 10));
    const b = new Baseline();
    for (const x of xs) b.add(x);
    const ref = twoPass(xs);
    // ~1e-9 relative: what float64 allows when values carry a 1e9 offset.
    expect(Math.abs(b.std - ref.std) / ref.std).toBeLessThan(1e-8);
    // The naive formula, for contrast, is visibly wrong here.
    const n = xs.length;
    const s1 = xs.reduce((a, x) => a + x, 0);
    const s2 = xs.reduce((a, x) => a + x * x, 0);
    const naive = Math.sqrt(Math.max(0, (s2 - (s1 * s1) / n) / (n - 1)));
    expect(Math.abs(naive - ref.std)).toBeGreaterThan(1e-3);
  });

  it("is ready after minSamples and gives z-scores", () => {
    const b = new Baseline({ minSamples: 4 });
    expect(b.mean).toBeNaN();
    expect(b.std).toBeNaN();
    for (const x of [2, 4, 4]) b.add(x);
    expect(b.ready).toBe(false);
    expect(b.zScore(4)).toBeNaN();
    b.add(6);
    expect(b.ready).toBe(true);
    // mean 4, sample std sqrt(8/3)
    expect(b.zScore(4 + Math.sqrt(8 / 3))).toBeCloseTo(1, 12);
    expect(b.zScore(4)).toBe(0);
  });

  it("ignores non-finite values, handles zero spread, and resets", () => {
    const b = new Baseline({ minSamples: 2 });
    b.add(Number.NaN);
    b.add(Number.NEGATIVE_INFINITY);
    expect(b.count).toBe(0);
    b.add(3);
    b.add(3);
    expect(b.std).toBe(0);
    expect(b.zScore(5)).toBeNaN();
    b.reset();
    expect(b.count).toBe(0);
    expect(b.mean).toBeNaN();
  });

  it("validates minSamples", () => {
    expect(() => new Baseline({ minSamples: 1 })).toThrow(RangeError);
    expect(() => new Baseline({ minSamples: 2.5 })).toThrow(RangeError);
  });
});
