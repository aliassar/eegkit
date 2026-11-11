/**
 * Butterworth and notch filter design.
 *
 * Butterworth filters follow the same route as `scipy.signal.butter`: analog
 * prototype poles → frequency transformation (low/high/band-pass/band-stop)
 * on pre-warped edges → bilinear transform → zeros/poles/gain → pairing into
 * second-order sections. Designing in zpk form and pairing straight into
 * sections never forms the high-order polynomial, so precision does not
 * degrade with order (see `sos.ts`).
 */

import type { Biquad, SOS } from "./sos";

// ---------------------------------------------------------------- validation

function assertSampleRate(sampleRate: number): void {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`sampleRate must be a positive finite number (Hz), got ${sampleRate}`);
  }
}

function assertOrder(order: number): void {
  if (!Number.isInteger(order) || order < 1) {
    throw new RangeError(`order must be an integer >= 1, got ${order}`);
  }
}

function assertFrequency(name: string, f: number, sampleRate: number): void {
  const nyquist = sampleRate / 2;
  if (!Number.isFinite(f) || f <= 0 || f >= nyquist) {
    throw new RangeError(
      `${name} must be between 0 and the Nyquist frequency (${nyquist} Hz), exclusive; got ${f}`,
    );
  }
}

function assertBand(low: number, high: number, sampleRate: number): void {
  assertSampleRate(sampleRate);
  assertFrequency("low", low, sampleRate);
  assertFrequency("high", high, sampleRate);
  if (low >= high) {
    throw new RangeError(`low (${low} Hz) must be less than high (${high} Hz)`);
  }
}

// ---------------------------------------------------------------- complex helpers

interface Complex {
  re: number;
  im: number;
}

const cx = (re: number, im = 0): Complex => ({ re, im });
const cadd = (a: Complex, b: Complex): Complex => cx(a.re + b.re, a.im + b.im);
const csub = (a: Complex, b: Complex): Complex => cx(a.re - b.re, a.im - b.im);
const cmul = (a: Complex, b: Complex): Complex =>
  cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const cscale = (a: Complex, s: number): Complex => cx(a.re * s, a.im * s);
const cabs = (a: Complex): number => Math.hypot(a.re, a.im);
const conj = (a: Complex): Complex => cx(a.re, -a.im);

function cdiv(a: Complex, b: Complex): Complex {
  const d = b.re * b.re + b.im * b.im;
  return cx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}

/** Principal square root. */
function csqrt(a: Complex): Complex {
  const r = cabs(a);
  const re = Math.sqrt((r + a.re) / 2);
  const im = Math.sqrt(Math.max(0, (r - a.re) / 2));
  return cx(re, a.im < 0 ? -im : im);
}

function cprod(values: readonly Complex[]): Complex {
  let acc = cx(1);
  for (const v of values) acc = cmul(acc, v);
  return acc;
}

// ---------------------------------------------------------------- zpk pipeline

interface Zpk {
  z: Complex[];
  p: Complex[];
  k: number;
}

/** Analog Butterworth prototype: poles evenly spaced on the left unit half-circle. */
function buttap(order: number): Zpk {
  const p: Complex[] = [];
  for (let m = -order + 1; m < order; m += 2) {
    const theta = (Math.PI * m) / (2 * order);
    p.push(cx(-Math.cos(theta), -Math.sin(theta)));
  }
  return { z: [], p, k: 1 };
}

function lp2lp({ z, p, k }: Zpk, wo: number): Zpk {
  const degree = p.length - z.length;
  return { z: z.map((v) => cscale(v, wo)), p: p.map((v) => cscale(v, wo)), k: k * wo ** degree };
}

function lp2hp({ z, p, k }: Zpk, wo: number): Zpk {
  const degree = p.length - z.length;
  const w = cx(wo);
  const zz = z.map((v) => cdiv(w, v));
  for (let i = 0; i < degree; i++) zz.push(cx(0));
  const gain = cdiv(cprod(z.map((v) => cscale(v, -1))), cprod(p.map((v) => cscale(v, -1))));
  return { z: zz, p: p.map((v) => cdiv(w, v)), k: k * gain.re };
}

/** Maps each root r to the pair r ± sqrt(r² − wo²) (all `+` first, as SciPy does). */
function splitRoots(roots: readonly Complex[], wo: number): Complex[] {
  const wo2 = cx(wo * wo);
  const roots2 = roots.map((r) => csqrt(csub(cmul(r, r), wo2)));
  return [
    ...roots.map((r, i) => cadd(r, roots2[i] as Complex)),
    ...roots.map((r, i) => csub(r, roots2[i] as Complex)),
  ];
}

function lp2bp({ z, p, k }: Zpk, wo: number, bw: number): Zpk {
  const degree = p.length - z.length;
  const zz = splitRoots(
    z.map((v) => cscale(v, bw / 2)),
    wo,
  );
  for (let i = 0; i < degree; i++) zz.push(cx(0));
  const pp = splitRoots(
    p.map((v) => cscale(v, bw / 2)),
    wo,
  );
  return { z: zz, p: pp, k: k * bw ** degree };
}

function lp2bs({ z, p, k }: Zpk, wo: number, bw: number): Zpk {
  const degree = p.length - z.length;
  const half = cx(bw / 2);
  const zz = splitRoots(
    z.map((v) => cdiv(half, v)),
    wo,
  );
  for (let i = 0; i < degree; i++) zz.push(cx(0, wo));
  for (let i = 0; i < degree; i++) zz.push(cx(0, -wo));
  const pp = splitRoots(
    p.map((v) => cdiv(half, v)),
    wo,
  );
  const gain = cdiv(cprod(z.map((v) => cscale(v, -1))), cprod(p.map((v) => cscale(v, -1))));
  return { z: zz, p: pp, k: k * gain.re };
}

/** Bilinear transform at the normalised rate fs = 2 used throughout (SciPy's convention). */
function bilinear({ z, p, k }: Zpk): Zpk {
  const fs2 = cx(4);
  const degree = p.length - z.length;
  const zz = z.map((v) => cdiv(cadd(fs2, v), csub(fs2, v)));
  for (let i = 0; i < degree; i++) zz.push(cx(-1));
  const gain = cdiv(cprod(z.map((v) => csub(fs2, v))), cprod(p.map((v) => csub(fs2, v))));
  return { z: zz, p: p.map((v) => cdiv(cadd(fs2, v), csub(fs2, v))), k: k * gain.re };
}

/** Pre-warps a frequency in Hz for the bilinear transform at the normalised rate fs = 2. */
function warp(f: number, sampleRate: number): number {
  return 4 * Math.tan((Math.PI * f) / sampleRate);
}

// ---------------------------------------------------------------- zpk → sos

const isReal = (v: Complex): boolean => v.im === 0;

/**
 * Keeps one root of each complex-conjugate pair (positive imaginary part) and
 * snaps near-real roots to the real axis, like SciPy's `_cplxreal`.
 */
function cplxreal(roots: readonly Complex[]): Complex[] {
  const complex: Complex[] = [];
  const real: Complex[] = [];
  for (const r of roots) {
    const tol = 100 * Number.EPSILON * cabs(r);
    if (Math.abs(r.im) <= tol) real.push(cx(r.re));
    else if (r.im > 0) complex.push(r);
  }
  complex.sort((a, b) => a.re - b.re || a.im - b.im);
  real.sort((a, b) => a.re - b.re);
  return [...complex, ...real];
}

function takeAt(list: Complex[], index: number): Complex {
  const [v] = list.splice(index, 1);
  if (v === undefined) throw new Error("internal: root index out of range");
  return v;
}

/** Removes and returns the root in `from` nearest to `to`, optionally restricted to real/complex roots. */
function takeNearest(from: Complex[], to: Complex, which: "any" | "real" | "complex"): Complex {
  let best = -1;
  let bestDist = Number.POSITIVE_INFINITY;
  from.forEach((v, i) => {
    if (which === "real" && !isReal(v)) return;
    if (which === "complex" && isReal(v)) return;
    const d = cabs(csub(v, to));
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return takeAt(from, best);
}

/** Expands roots into monic polynomial coefficients (length roots.length + 1). */
function poly(roots: readonly Complex[]): number[] {
  let c: Complex[] = [cx(1)];
  for (const r of roots) {
    const next: Complex[] = c.map((v) => cx(v.re, v.im));
    next.push(cx(0));
    for (let i = 1; i < next.length; i++) {
      next[i] = csub(next[i] as Complex, cmul(r, c[i - 1] as Complex));
    }
    c = next;
  }
  return c.map((v) => v.re);
}

function section(zeros: readonly Complex[], poles: readonly Complex[]): Biquad {
  // Right-align like SciPy: a single zero gives b = [0, 1, -z].
  const b = [0, 0, 0, ...poly(zeros)].slice(-3);
  const a = [0, 0, 0, ...poly(poles)].slice(-3);
  const a0 = a[0] as number;
  return {
    b0: (b[0] as number) / a0,
    b1: (b[1] as number) / a0,
    b2: (b[2] as number) / a0,
    a1: (a[1] as number) / a0,
    a2: (a[2] as number) / a0,
  };
}

/**
 * Pairs zeros and poles into biquads using SciPy's "nearest" strategy: poles
 * closest to the unit circle are paired first with their nearest zeros, and
 * those sections are placed last in the cascade, where their high gain
 * cannot overflow the sections that follow.
 */
function zpk2sos({ z: zIn, p: pIn, k }: Zpk): SOS {
  const zeros = [...zIn];
  const poles = [...pIn];
  while (zeros.length < poles.length) zeros.push(cx(0));
  while (poles.length < zeros.length) poles.push(cx(0));
  const nSections = (poles.length + 1) >> 1;
  if (poles.length % 2 === 1) {
    poles.push(cx(0));
    zeros.push(cx(0));
  }
  const z = cplxreal(zeros);
  const p = cplxreal(poles);
  const sections: Biquad[] = [];
  const countReal = (list: readonly Complex[]) => list.filter(isReal).length;

  for (let s = 0; s < nSections; s++) {
    let worst = 0;
    p.forEach((v, i) => {
      if (Math.abs(1 - cabs(v)) < Math.abs(1 - cabs(p[worst] as Complex))) worst = i;
    });
    const p1 = takeAt(p, worst);

    if (isReal(p1) && countReal(p) === 0) {
      // Last real pole: first-order section with the nearest real zero.
      const z1 = takeNearest(z, p1, "real");
      sections.push(section([z1, cx(0)], [p1, cx(0)]));
    } else if (
      p.length + 1 === z.length &&
      !isReal(p1) &&
      countReal(p) === 1 &&
      countReal(z) === 1
    ) {
      // One real pole and one real zero must stay together, so take a complex zero here.
      const z1 = takeNearest(z, p1, "complex");
      sections.push(section([z1, conj(z1)], [p1, conj(p1)]));
    } else {
      let p2: Complex;
      if (isReal(p1)) {
        let best = -1;
        p.forEach((v, i) => {
          if (!isReal(v)) return;
          if (best < 0 || Math.abs(cabs(v) - 1) < Math.abs(cabs(p[best] as Complex) - 1)) best = i;
        });
        p2 = takeAt(p, best);
      } else {
        p2 = conj(p1);
      }
      if (z.length > 0) {
        const z1 = takeNearest(z, p1, "any");
        if (!isReal(z1)) {
          sections.push(section([z1, conj(z1)], [p1, p2]));
        } else if (z.length > 0) {
          const z2 = takeNearest(z, p1, "real");
          sections.push(section([z1, z2], [p1, p2]));
        } else {
          sections.push(section([z1], [p1, p2]));
        }
      } else {
        sections.push(section([], [p1, p2]));
      }
    }
  }

  sections.reverse();
  const first = sections[0];
  if (first !== undefined) {
    sections[0] = { ...first, b0: first.b0 * k, b1: first.b1 * k, b2: first.b2 * k };
  }
  return Object.freeze(sections.map((s) => Object.freeze(s)));
}

// ---------------------------------------------------------------- public API

/**
 * Butterworth low-pass filter, equivalent to
 * `scipy.signal.butter(order, cutoff, "lowpass", fs=sampleRate, output="sos")`.
 *
 * @param order - Filter order (≥ 1). The roll-off is 6·order dB/octave.
 * @param cutoff - −3 dB frequency in Hz, strictly between 0 and `sampleRate / 2`.
 * @param sampleRate - Sampling rate in Hz.
 * @example
 * ```ts
 * const sos = butterworthLowpass(4, 40, 256); // anti-alias / smoothing before display
 * ```
 */
export function butterworthLowpass(order: number, cutoff: number, sampleRate: number): SOS {
  assertOrder(order);
  assertSampleRate(sampleRate);
  assertFrequency("cutoff", cutoff, sampleRate);
  return zpk2sos(bilinear(lp2lp(buttap(order), warp(cutoff, sampleRate))));
}

/**
 * Butterworth high-pass filter, equivalent to
 * `scipy.signal.butter(order, cutoff, "highpass", fs=sampleRate, output="sos")`.
 *
 * @param order - Filter order (≥ 1).
 * @param cutoff - −3 dB frequency in Hz, strictly between 0 and `sampleRate / 2`.
 * @param sampleRate - Sampling rate in Hz.
 * @example
 * ```ts
 * const sos = butterworthHighpass(2, 1, 256); // remove electrode drift
 * ```
 */
export function butterworthHighpass(order: number, cutoff: number, sampleRate: number): SOS {
  assertOrder(order);
  assertSampleRate(sampleRate);
  assertFrequency("cutoff", cutoff, sampleRate);
  return zpk2sos(bilinear(lp2hp(buttap(order), warp(cutoff, sampleRate))));
}

/**
 * Butterworth band-pass filter, equivalent to
 * `scipy.signal.butter(order, [low, high], "bandpass", fs=sampleRate, output="sos")`.
 *
 * `order` is the prototype order, so the resulting filter has `2 * order`
 * poles (`order` sections), exactly as in SciPy.
 *
 * @param low - Lower −3 dB edge in Hz.
 * @param high - Upper −3 dB edge in Hz; must exceed `low` and be below `sampleRate / 2`.
 * @example
 * ```ts
 * const sos = butterworthBandpass(4, 1, 40, 256); // typical EEG pre-processing
 * ```
 */
export function butterworthBandpass(
  order: number,
  low: number,
  high: number,
  sampleRate: number,
): SOS {
  assertOrder(order);
  assertBand(low, high, sampleRate);
  const lo = warp(low, sampleRate);
  const hi = warp(high, sampleRate);
  return zpk2sos(bilinear(lp2bp(buttap(order), Math.sqrt(lo * hi), hi - lo)));
}

/**
 * Butterworth band-stop filter, equivalent to
 * `scipy.signal.butter(order, [low, high], "bandstop", fs=sampleRate, output="sos")`.
 *
 * `order` is the prototype order (the filter has `2 * order` poles). For
 * removing mains interference alone, {@link notch} is narrower and cheaper.
 *
 * @param low - Lower −3 dB edge in Hz.
 * @param high - Upper −3 dB edge in Hz; must exceed `low` and be below `sampleRate / 2`.
 * @example
 * ```ts
 * const sos = butterworthBandstop(2, 48, 52, 256);
 * ```
 */
export function butterworthBandstop(
  order: number,
  low: number,
  high: number,
  sampleRate: number,
): SOS {
  assertOrder(order);
  assertBand(low, high, sampleRate);
  const lo = warp(low, sampleRate);
  const hi = warp(high, sampleRate);
  return zpk2sos(bilinear(lp2bs(buttap(order), Math.sqrt(lo * hi), hi - lo)));
}

/**
 * Second-order IIR notch filter, equivalent to
 * `scipy.signal.iirnotch(frequency, quality, fs=sampleRate)`.
 *
 * The response is exactly zero at `frequency` and the −3 dB bandwidth is
 * `frequency / quality` Hz. Use it to remove 50/60 Hz mains interference.
 *
 * @param frequency - Frequency to remove, in Hz.
 * @param sampleRate - Sampling rate in Hz.
 * @param quality - Quality factor `Q = frequency / bandwidth` (default 30).
 * @example
 * ```ts
 * const sos = notch(50, 256); // or notch(60, 256) in the Americas
 * ```
 */
export function notch(frequency: number, sampleRate: number, quality = 30): SOS {
  assertSampleRate(sampleRate);
  assertFrequency("frequency", frequency, sampleRate);
  if (!Number.isFinite(quality) || quality <= 0) {
    throw new RangeError(`quality must be a positive finite number, got ${quality}`);
  }
  const w0 = (2 * Math.PI * frequency) / sampleRate;
  const bw = w0 / quality;
  // SciPy uses a -3 dB gain at the band edges: beta = tan(bw / 2) * sqrt(1 - gb²) / gb, gb = 1/√2.
  const beta = Math.tan(bw / 2);
  const gain = 1 / (1 + beta);
  const cos = Math.cos(w0);
  return Object.freeze([
    Object.freeze({
      b0: gain,
      b1: -2 * gain * cos,
      b2: gain,
      a1: -2 * gain * cos,
      a2: 2 * gain - 1,
    }),
  ]);
}
