import { type Band, type BandPowers, STANDARD_BANDS } from "../types";

/**
 * Any one-sided spectrum: ascending bin frequencies (Hz) and the power at
 * each bin. {@link welch} and {@link periodogram} return one.
 */
export interface SpectrumLike {
  readonly frequencies: ArrayLike<number>;
  readonly power: ArrayLike<number>;
}

/** Reads `a[i]`; callers keep `i` in range, so the NaN fallback never shows. */
function at(a: ArrayLike<number>, i: number): number {
  return a[i] ?? Number.NaN;
}

type BandInput = Band | readonly [low: number, high: number];

function bandEdges(band: BandInput): [number, number] {
  const [low, high] = "low" in band ? [band.low, band.high] : [band[0], band[1]];
  if (!Number.isFinite(low) || !Number.isFinite(high) || low < 0 || low >= high) {
    throw new RangeError(`band must satisfy 0 <= low < high (Hz), got [${low}, ${high}]`);
  }
  return [low, high];
}

function checkSpectrum(spectrum: SpectrumLike): number {
  const n = spectrum.frequencies.length;
  if (spectrum.power.length !== n) {
    throw new RangeError(
      `frequencies (${n}) and power (${spectrum.power.length}) must have the same length`,
    );
  }
  return n;
}

/** Index of the last bin with frequency <= x (binary search; frequencies ascending). */
function binAtOrBelow(f: ArrayLike<number>, x: number): number {
  let lo = 0;
  let hi = f.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (at(f, mid) <= x) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Linear interpolation of the spectrum at x, which must lie within [f[i], f[i+1]]. */
function interpolate(f: ArrayLike<number>, p: ArrayLike<number>, i: number, x: number): number {
  const f0 = at(f, i);
  const p0 = at(p, i);
  if (i + 1 >= f.length || x === f0) return p0;
  const f1 = at(f, i + 1);
  return p0 + ((at(p, i + 1) - p0) * (x - f0)) / (f1 - f0);
}

/**
 * Absolute power in a frequency band: the integral of the power spectral
 * density from `low` to `high`, in µV² when the spectrum is a density in
 * µV²/Hz.
 *
 * Method: trapezoid rule on the piecewise-linear spectrum, with the band
 * edges themselves interpolated rather than snapped to the nearest bins. That
 * way the result does not jump as an edge crosses a bin, and adjacent bands
 * (`[4, 8)`, `[8, 13)`) tile exactly: they sum to the power of their union
 * with no gap or double counting.
 *
 * The common alternative of summing bins times the bin width (the rectangle
 * rule) is simpler but charges a whole bin to whichever band claims its
 * centre, which biases narrow bands by up to one bin width; the trapezoid
 * over just the bins inside the band (`np.trapezoid(psd[mask], f[mask])`)
 * loses up to one bin width per band instead. Both errors matter for the
 * 1–4 Hz resolution typical of short EEG windows.
 *
 * Parts of the band outside the spectrum's frequency range contribute
 * nothing; a band entirely outside it has power 0.
 *
 * @param band - A {@link Band} or a `[low, high]` pair in Hz.
 * @example
 * ```ts
 * const psd = welch(signal, { sampleRate: 256 });
 * const alpha = bandPower(psd, [8, 13]); // µV²
 * ```
 */
export function bandPower(spectrum: SpectrumLike, band: BandInput): number {
  const [low, high] = bandEdges(band);
  const n = checkSpectrum(spectrum);
  if (n < 2) return 0;
  const f = spectrum.frequencies;
  const p = spectrum.power;
  const lo = Math.max(low, at(f, 0));
  const hi = Math.min(high, at(f, n - 1));
  if (hi <= lo) return 0;

  let i = binAtOrBelow(f, lo);
  let x0 = lo;
  let y0 = interpolate(f, p, i, lo);
  let total = 0;
  i++;
  while (i < n && at(f, i) < hi) {
    const x1 = at(f, i);
    const y1 = at(p, i);
    total += 0.5 * (y0 + y1) * (x1 - x0);
    x0 = x1;
    y0 = y1;
    i++;
  }
  const yHi = interpolate(f, p, i - 1, hi);
  total += 0.5 * (y0 + yHi) * (hi - x0);
  return total;
}

/**
 * Absolute power in each band (see {@link bandPower} for the method).
 *
 * @param bands - Bands to measure (default {@link STANDARD_BANDS}: delta … gamma).
 * @returns Power per band name, in µV² for a density spectrum.
 * @example
 * ```ts
 * const { alpha, theta } = bandPowers(welch(signal, { sampleRate: 256 }));
 * ```
 */
export function bandPowers(
  spectrum: SpectrumLike,
  bands: readonly Band[] = STANDARD_BANDS,
): BandPowers {
  const out: BandPowers = {};
  for (const band of bands) out[band.name] = bandPower(spectrum, band);
  return out;
}

/**
 * Power in each band as a fraction of the total power over `range`.
 *
 * Relative power cancels the overall amplitude scale (electrode impedance,
 * skull thickness, amplifier gain), which makes it comparable across people
 * and sessions where absolute power is not.
 *
 * @param bands - Bands to measure (default {@link STANDARD_BANDS}).
 * @param range - Frequency range for the total, in Hz (default: from the lowest
 *   band edge to the highest, e.g. 1–45 Hz for the standard bands).
 * @returns Fraction per band name; `NaN` for every band when the total power is 0.
 */
export function relativeBandPowers(
  spectrum: SpectrumLike,
  bands: readonly Band[] = STANDARD_BANDS,
  range?: readonly [low: number, high: number],
): BandPowers {
  let totalRange = range;
  if (totalRange === undefined) {
    if (bands.length === 0) return {};
    let low = Number.POSITIVE_INFINITY;
    let high = Number.NEGATIVE_INFINITY;
    for (const band of bands) {
      low = Math.min(low, band.low);
      high = Math.max(high, band.high);
    }
    totalRange = [low, high];
  }
  const total = bandPower(spectrum, totalRange);
  const out: BandPowers = {};
  for (const band of bands) {
    out[band.name] = total > 0 ? bandPower(spectrum, band) / total : Number.NaN;
  }
  return out;
}

/**
 * Frequency of the largest spectral peak within `range`, e.g. the individual
 * alpha frequency (IAF) in `[7, 14]` Hz.
 *
 * The highest bin is refined by fitting a parabola through the logarithm of
 * its power and its two neighbours' (Gaussian interpolation), which locates
 * the peak to a small fraction of a bin; with the 0.5–1 Hz resolution of
 * typical EEG windows, picking the bin alone would quantise the IAF too
 * coarsely to track. The log matters: a windowed sinusoid's main lobe is
 * close to a parabola in dB but not in linear power, so fitting linear power
 * is biased by up to ~0.1 bin with a Hann window, the log fit by ~0.02. If a
 * power value is zero the fit falls back to linear power. If the highest bin
 * is not a local maximum (the peak lies outside `range`), its frequency is
 * returned unrefined.
 *
 * @param range - `[low, high]` in Hz, inclusive.
 * @returns The peak frequency in Hz, or `NaN` if no bin lies in `range`.
 * @example
 * ```ts
 * const iaf = peakFrequency(welch(signal, { sampleRate: 256, segmentLength: 512 }), [7, 14]);
 * ```
 */
export function peakFrequency(
  spectrum: SpectrumLike,
  range: readonly [low: number, high: number],
): number {
  const [low, high] = bandEdges(range);
  const n = checkSpectrum(spectrum);
  const f = spectrum.frequencies;
  const p = spectrum.power;
  let best = -1;
  for (let i = 0; i < n; i++) {
    const fi = at(f, i);
    if (fi < low || fi > high) continue;
    if (best < 0 || at(p, i) > at(p, best)) best = i;
  }
  if (best < 0) return Number.NaN;
  const fBest = at(f, best);
  if (best === 0 || best === n - 1) return fBest;
  let a = at(p, best - 1);
  let b = at(p, best);
  let c = at(p, best + 1);
  if (b < a || b < c) return fBest;
  if (a > 0 && c > 0) {
    a = Math.log(a);
    b = Math.log(b);
    c = Math.log(c);
  }
  const curvature = a - 2 * b + c;
  if (!(curvature < 0)) return fBest;
  const delta = (0.5 * (a - c)) / curvature;
  return fBest + (delta * (at(f, best + 1) - at(f, best - 1))) / 2;
}
