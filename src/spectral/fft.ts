// biome-ignore-all lint/style/noNonNullAssertion: FFT butterflies index typed arrays within bounds fixed by the plan size.
/**
 * Fast Fourier transforms for any length.
 *
 * Powers of two use an iterative radix-2 Cooley–Tukey transform. Other
 * lengths (EEG windows are often 250 or 500 samples) use Bluestein's chirp-z
 * algorithm, which rewrites the DFT as a convolution evaluated with a
 * power-of-two FFT, so every length costs O(n log n) rather than O(n²).
 *
 * Twiddle factors, bit-reversal tables, chirps and scratch buffers are
 * computed once per size and cached, so repeated transforms of the same size
 * (the usual case for streaming analysis) allocate nothing.
 */

import type { Signal } from "../types";

const MAX_CACHED_SIZES = 16;

/** Small LRU cache keyed by transform size. */
class PlanCache<T> {
  private readonly map = new Map<number, T>();

  constructor(private readonly create: (n: number) => T) {}

  get(n: number): T {
    let plan = this.map.get(n);
    if (plan !== undefined) {
      // Re-insert to mark as most recently used.
      this.map.delete(n);
    } else {
      plan = this.create(n);
      if (this.map.size >= MAX_CACHED_SIZES) {
        const oldest = this.map.keys().next().value;
        if (oldest !== undefined) this.map.delete(oldest);
      }
    }
    this.map.set(n, plan);
    return plan;
  }
}

const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0;

// ---------------------------------------------------------------- radix-2

interface Radix2Plan {
  readonly n: number;
  /** cos(2πk/n) and −sin(2πk/n) for k < n/2: the twiddles e^{−2πik/n}. */
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly rev: Uint32Array;
}

function createRadix2(n: number): Radix2Plan {
  const half = n >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    // Each twiddle is computed directly (not by recurrence) to keep full precision.
    const angle = (2 * Math.PI * k) / n;
    cos[k] = Math.cos(angle);
    sin[k] = -Math.sin(angle);
  }
  const rev = new Uint32Array(n);
  const bits = Math.round(Math.log2(n));
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0, v = i; b < bits; b++, v >>= 1) r = (r << 1) | (v & 1);
    rev[i] = r;
  }
  return { n, cos, sin, rev };
}

const radix2Plans = new PlanCache(createRadix2);

function radix2(re: Float64Array, im: Float64Array, plan: Radix2Plan): void {
  const { n, cos, sin, rev } = plan;
  for (let i = 0; i < n; i++) {
    const j = rev[i]!;
    if (j > i) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0, t = 0; k < half; k++, t += step) {
        const wr = cos[t]!;
        const wi = sin[t]!;
        const a = start + k;
        const b = a + half;
        const br = re[b]!;
        const bi = im[b]!;
        const xr = br * wr - bi * wi;
        const xi = br * wi + bi * wr;
        const ar = re[a]!;
        const ai = im[a]!;
        re[b] = ar - xr;
        im[b] = ai - xi;
        re[a] = ar + xr;
        im[a] = ai + xi;
      }
    }
  }
}

// ---------------------------------------------------------------- Bluestein

interface BluesteinPlan {
  readonly n: number;
  readonly m: number;
  /** Chirp w_k = e^{−iπk²/n}. */
  readonly wRe: Float64Array;
  readonly wIm: Float64Array;
  /** FFT of the conjugate chirp, wrapped for circular convolution of size m. */
  readonly bRe: Float64Array;
  readonly bIm: Float64Array;
  readonly aRe: Float64Array;
  readonly aIm: Float64Array;
  readonly inner: Radix2Plan;
}

function createBluestein(n: number): BluesteinPlan {
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const wRe = new Float64Array(n);
  const wIm = new Float64Array(n);
  // k² mod 2n, updated incrementally ((k+1)² = k² + 2k + 1) so the angle stays
  // small and exact even when k² itself would lose precision.
  for (let k = 0, k2 = 0; k < n; k++) {
    const angle = (Math.PI * k2) / n;
    wRe[k] = Math.cos(angle);
    wIm[k] = -Math.sin(angle);
    k2 = (k2 + 2 * k + 1) % (2 * n);
  }
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = wRe[0]!;
  bIm[0] = -wIm[0]!;
  for (let k = 1; k < n; k++) {
    bRe[k] = bRe[m - k] = wRe[k]!;
    bIm[k] = bIm[m - k] = -wIm[k]!;
  }
  const inner = radix2Plans.get(m);
  radix2(bRe, bIm, inner);
  return {
    n,
    m,
    wRe,
    wIm,
    bRe,
    bIm,
    aRe: new Float64Array(m),
    aIm: new Float64Array(m),
    inner,
  };
}

const bluesteinPlans = new PlanCache(createBluestein);

function bluestein(re: Float64Array, im: Float64Array, plan: BluesteinPlan): void {
  const { n, m, wRe, wIm, bRe, bIm, aRe, aIm, inner } = plan;
  for (let k = 0; k < n; k++) {
    const xr = re[k]!;
    const xi = im[k]!;
    const cr = wRe[k]!;
    const ci = wIm[k]!;
    aRe[k] = xr * cr - xi * ci;
    aIm[k] = xr * ci + xi * cr;
  }
  aRe.fill(0, n);
  aIm.fill(0, n);
  radix2(aRe, aIm, inner);
  // Pointwise product, conjugated so the forward FFT below acts as an inverse.
  for (let k = 0; k < m; k++) {
    const ar = aRe[k]!;
    const ai = aIm[k]!;
    const br = bRe[k]!;
    const bi = bIm[k]!;
    aRe[k] = ar * br - ai * bi;
    aIm[k] = -(ar * bi + ai * br);
  }
  radix2(aRe, aIm, inner);
  const scale = 1 / m;
  for (let k = 0; k < n; k++) {
    // Undo the conjugation and scaling of the inverse transform, then apply the output chirp.
    const cr = aRe[k]! * scale;
    const ci = -aIm[k]! * scale;
    const wr = wRe[k]!;
    const wi = wIm[k]!;
    re[k] = cr * wr - ci * wi;
    im[k] = cr * wi + ci * wr;
  }
}

// ---------------------------------------------------------------- public API

/**
 * In-place forward complex FFT of any length:
 * `X[k] = Σ x[j]·e^{−2πijk/n}` (unnormalised, same convention as `numpy.fft.fft`).
 *
 * @param re - Real parts; overwritten with the real part of the result.
 * @param im - Imaginary parts (same length); overwritten likewise.
 * @example
 * ```ts
 * const re = Float64Array.from(signal);
 * const im = new Float64Array(re.length);
 * fft(re, im);
 * ```
 */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (im.length !== n) {
    throw new RangeError(`re and im must have the same length, got ${n} and ${im.length}`);
  }
  if (n <= 1) return;
  if (isPowerOfTwo(n)) radix2(re, im, radix2Plans.get(n));
  else bluestein(re, im, bluesteinPlans.get(n));
}

/** One-sided spectrum of a real signal. */
export interface ComplexSpectrum {
  /** Real parts of bins `0 … floor(n/2)`. */
  readonly re: Float64Array;
  /** Imaginary parts of bins `0 … floor(n/2)`. */
  readonly im: Float64Array;
}

interface RealPlan {
  readonly n: number;
  /** e^{−2πik/n} for k ≤ n/2, used to split the half-length transform (even n only). */
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly re: Float64Array;
  readonly im: Float64Array;
}

function createRealPlan(n: number): RealPlan {
  const even = n % 2 === 0;
  const half = n >> 1;
  const cos = new Float64Array(even ? half + 1 : 0);
  const sin = new Float64Array(even ? half + 1 : 0);
  for (let k = 0; k < cos.length; k++) {
    const angle = (2 * Math.PI * k) / n;
    cos[k] = Math.cos(angle);
    sin[k] = -Math.sin(angle);
  }
  const work = even ? half : n;
  return { n, cos, sin, re: new Float64Array(work), im: new Float64Array(work) };
}

const realPlans = new PlanCache(createRealPlan);

/**
 * Real FFT into caller-provided buffers of length `floor(n/2) + 1`, reading
 * `n = input.length` samples. Allocation-free after the first call for a size.
 * @internal
 */
export function rfftInto(input: Signal, outRe: Float64Array, outIm: Float64Array): void {
  const n = input.length;
  const bins = (n >> 1) + 1;
  if (outRe.length < bins || outIm.length < bins) {
    throw new RangeError(`output buffers need ${bins} elements`);
  }
  if (n === 0) throw new RangeError("cannot transform an empty signal");
  if (n === 1) {
    outRe[0] = input[0]!;
    outIm[0] = 0;
    return;
  }
  const plan = realPlans.get(n);
  const { re, im } = plan;
  if (n % 2 === 1) {
    for (let i = 0; i < n; i++) re[i] = input[i]!;
    im.fill(0);
    fft(re, im);
    for (let k = 0; k < bins; k++) {
      outRe[k] = re[k]!;
      outIm[k] = im[k]!;
    }
    return;
  }
  // Even n: pack even/odd samples as one complex signal of length n/2,
  // transform it, then separate the two spectra. Half the work of a full FFT.
  const half = n >> 1;
  for (let j = 0; j < half; j++) {
    re[j] = input[2 * j]!;
    im[j] = input[2 * j + 1]!;
  }
  fft(re, im);
  const { cos, sin } = plan;
  for (let k = 0; k <= half; k++) {
    const k1 = k === half ? 0 : k;
    const k2 = k === 0 ? 0 : half - k;
    const zr = re[k1]!;
    const zi = im[k1]!;
    const cr = re[k2]!;
    const ci = -im[k2]!;
    // E = (Z[k] + conj Z[n/2−k]) / 2, O = (Z[k] − conj Z[n/2−k]) / 2i
    const er = 0.5 * (zr + cr);
    const ei = 0.5 * (zi + ci);
    const or = 0.5 * (zi - ci);
    const oi = -0.5 * (zr - cr);
    const wr = cos[k]!;
    const wi = sin[k]!;
    outRe[k] = er + or * wr - oi * wi;
    outIm[k] = ei + or * wi + oi * wr;
  }
}

/**
 * FFT of a real signal, returning the `floor(n/2) + 1` non-negative-frequency
 * bins (like `numpy.fft.rfft`). Bin `k` is at `k * sampleRate / n` Hz.
 *
 * @example
 * ```ts
 * const { re, im } = rfft(signal);
 * const magnitude = re.map((r, k) => Math.hypot(r, im[k]!));
 * ```
 */
export function rfft(signal: Signal): ComplexSpectrum {
  const bins = (signal.length >> 1) + 1;
  const re = new Float64Array(bins);
  const im = new Float64Array(bins);
  rfftInto(signal, re, im);
  return { re, im };
}
