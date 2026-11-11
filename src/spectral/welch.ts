// biome-ignore-all lint/style/noNonNullAssertion: inner loops index typed arrays within bounds checked before the loops.
import type { Signal } from "../types";
import { rfftInto } from "./fft";
import { getWindow, type WindowName } from "./window";

/** A one-sided power spectrum. */
export interface PowerSpectrum {
  /** Bin frequencies in Hz, from 0 to the Nyquist frequency (inclusive for even lengths). */
  readonly frequencies: Float64Array;
  /** Power per bin: µV²/Hz for `scaling: "density"`, µV² for `scaling: "spectrum"`. */
  readonly power: Float64Array;
}

/** Options shared by {@link welch} and {@link periodogram}. */
export interface SpectrumOptions {
  /** Sampling rate in Hz. */
  readonly sampleRate: number;
  /**
   * Taper applied to each segment: a window name, or the window values
   * themselves (their length then sets the segment length).
   */
  readonly window?: WindowName | ArrayLike<number>;
  /** Per-segment detrending before windowing (default `"constant"`: remove the mean). */
  readonly detrend?: "constant" | "linear" | false;
  /**
   * `"density"` (default) gives a power spectral density in µV²/Hz, whose
   * integral over a band is the band power. `"spectrum"` gives µV² per bin,
   * whose peak height reads directly as the squared RMS amplitude of a sine.
   */
  readonly scaling?: "density" | "spectrum";
}

/** Options for {@link welch}. */
export interface WelchOptions extends SpectrumOptions {
  /**
   * Segment length in samples (default 256). Clamped to the signal length,
   * as SciPy does. Frequency resolution is `sampleRate / segmentLength` Hz.
   */
  readonly segmentLength?: number;
  /** Overlap between segments in samples (default `floor(segmentLength / 2)`). */
  readonly overlap?: number;
  /**
   * How to combine segment spectra (default `"mean"`). `"median"` is robust to
   * segments hit by artifacts; it is divided by SciPy's bias factor so it
   * estimates the same quantity as the mean for stationary noise.
   */
  readonly average?: "mean" | "median";
}

function assertSampleRate(sampleRate: number): void {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`sampleRate must be a positive finite number (Hz), got ${sampleRate}`);
  }
}

const windowCache = new Map<string, Float64Array>();

/** Named windows are cached so repeated analyses of the same size don't recompute them. */
function namedWindow(name: WindowName, n: number): Float64Array {
  const key = `${name}:${n}`;
  let w = windowCache.get(key);
  if (w === undefined) {
    w = getWindow(name, n);
    if (windowCache.size >= 32) windowCache.clear();
    windowCache.set(key, w);
  }
  return w;
}

/** Removes the least-squares line from `x` in place. */
function detrendLinear(x: Float64Array): void {
  const n = x.length;
  const tMean = (n - 1) / 2;
  let xMean = 0;
  for (let i = 0; i < n; i++) xMean += x[i]!;
  xMean /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    const dt = i - tMean;
    sxy += dt * (x[i]! - xMean);
    sxx += dt * dt;
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  for (let i = 0; i < n; i++) x[i] = x[i]! - xMean - slope * (i - tMean);
}

function detrendConstant(x: Float64Array): void {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i]!;
  mean /= x.length;
  for (let i = 0; i < x.length; i++) x[i] = x[i]! - mean;
}

/**
 * Bias of the median of n χ²₂ variables relative to their mean, as in
 * SciPy's `_median_bias`: 1 + Σ_{k=1}^{(n−1)/2} (1/(2k+1) − 1/(2k)).
 */
function medianBias(n: number): number {
  let bias = 1;
  for (let k = 1; k <= (n - 1) >> 1; k++) bias += 1 / (2 * k + 1) - 1 / (2 * k);
  return bias;
}

function median(values: Float64Array): number {
  values.sort();
  const mid = values.length >> 1;
  return values.length % 2 === 1 ? values[mid]! : (values[mid - 1]! + values[mid]!) / 2;
}

/** Shared Welch / periodogram engine. */
function averagedSpectrum(
  signal: Signal,
  sampleRate: number,
  win: Float64Array,
  overlap: number,
  detrend: "constant" | "linear" | false,
  scaling: "density" | "spectrum",
  average: "mean" | "median",
): PowerSpectrum {
  const n = signal.length;
  const segLen = win.length;
  const step = segLen - overlap;
  const segments = Math.floor((n - overlap) / step);
  const bins = (segLen >> 1) + 1;

  let sumW = 0;
  let sumW2 = 0;
  for (let i = 0; i < segLen; i++) {
    sumW += win[i]!;
    sumW2 += win[i]! * win[i]!;
  }
  const scale = scaling === "density" ? 1 / (sampleRate * sumW2) : 1 / (sumW * sumW);
  // One-sided spectrum: fold negative frequencies onto positive ones, except
  // DC and (for even lengths) the Nyquist bin, which have no mirror image.
  const lastDoubled = segLen % 2 === 0 ? bins - 2 : bins - 1;

  const seg = new Float64Array(segLen);
  const re = new Float64Array(bins);
  const im = new Float64Array(bins);
  const perSegment = average === "median" ? new Float64Array(bins * segments) : null;
  const power = new Float64Array(bins);

  for (let s = 0; s < segments; s++) {
    const start = s * step;
    for (let i = 0; i < segLen; i++) seg[i] = signal[start + i]!;
    if (detrend === "constant") detrendConstant(seg);
    else if (detrend === "linear") detrendLinear(seg);
    for (let i = 0; i < segLen; i++) seg[i] = seg[i]! * win[i]!;
    rfftInto(seg, re, im);
    for (let k = 0; k < bins; k++) {
      let p = (re[k]! * re[k]! + im[k]! * im[k]!) * scale;
      if (k >= 1 && k <= lastDoubled) p *= 2;
      if (perSegment === null) power[k] = power[k]! + p;
      else perSegment[k * segments + s] = p;
    }
  }

  if (perSegment === null) {
    for (let k = 0; k < bins; k++) power[k] = power[k]! / segments;
  } else {
    const bias = medianBias(segments);
    for (let k = 0; k < bins; k++) {
      power[k] = median(perSegment.subarray(k * segments, (k + 1) * segments)) / bias;
    }
  }

  const frequencies = new Float64Array(bins);
  for (let k = 0; k < bins; k++) frequencies[k] = (k * sampleRate) / segLen;
  return { frequencies, power };
}

function resolveCommon(options: SpectrumOptions) {
  const detrend = options.detrend ?? "constant";
  if (detrend !== "constant" && detrend !== "linear" && detrend !== false) {
    throw new TypeError(`detrend must be "constant", "linear" or false, got ${String(detrend)}`);
  }
  const scaling = options.scaling ?? "density";
  if (scaling !== "density" && scaling !== "spectrum") {
    throw new TypeError(`scaling must be "density" or "spectrum", got ${String(scaling)}`);
  }
  return { detrend, scaling };
}

/**
 * Power spectral density by Welch's method: split the signal into
 * overlapping windowed segments, take each one's periodogram, and average.
 * Averaging trades frequency resolution for a much lower-variance estimate
 * than a single periodogram, which is what makes band power usable on noisy
 * EEG.
 *
 * Matches `scipy.signal.welch` defaults: 256-sample segments (clamped to the
 * signal length), 50 % overlap, periodic Hann window, constant detrend,
 * one-sided density scaling, mean averaging.
 *
 * @param signal - One channel, in µV.
 * @returns Frequencies in Hz and power in µV²/Hz (or µV² with `scaling: "spectrum"`).
 * @example
 * ```ts
 * // 2 s window at 256 Hz, 1-s segments → 1 Hz resolution, 3 averaged segments
 * const psd = welch(window, { sampleRate: 256, segmentLength: 256 });
 * const alpha = bandPower(psd, [8, 13]);
 * ```
 */
export function welch(signal: Signal, options: WelchOptions): PowerSpectrum {
  assertSampleRate(options.sampleRate);
  const n = signal.length;
  if (n === 0) throw new RangeError("signal is empty");
  const { detrend, scaling } = resolveCommon(options);
  const average = options.average ?? "mean";
  if (average !== "mean" && average !== "median") {
    throw new TypeError(`average must be "mean" or "median", got ${String(average)}`);
  }

  const { window = "hann", segmentLength } = options;
  let win: Float64Array;
  if (typeof window === "string") {
    let length = segmentLength ?? 256;
    if (!Number.isInteger(length) || length < 1) {
      throw new RangeError(`segmentLength must be an integer >= 1, got ${length}`);
    }
    if (length > n) length = n;
    win = namedWindow(window, length);
  } else {
    win = Float64Array.from(window);
    if (win.length === 0) throw new RangeError("window must not be empty");
    if (segmentLength !== undefined && segmentLength !== win.length) {
      throw new RangeError(
        `segmentLength (${segmentLength}) must equal the window length (${win.length})`,
      );
    }
    if (win.length > n) {
      throw new RangeError(`window length (${win.length}) exceeds signal length (${n})`);
    }
  }

  const overlap = options.overlap ?? win.length >> 1;
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= win.length) {
    throw new RangeError(
      `overlap must be an integer in [0, segmentLength) = [0, ${win.length}), got ${overlap}`,
    );
  }
  return averagedSpectrum(signal, options.sampleRate, win, overlap, detrend, scaling, average);
}

/**
 * Periodogram: the spectrum of the whole signal as one segment, matching
 * `scipy.signal.periodogram` (rectangular window by default, constant
 * detrend, one-sided density scaling).
 *
 * Highest frequency resolution, but its variance does not shrink with signal
 * length; prefer {@link welch} for band power.
 *
 * @param signal - One channel, in µV.
 * @returns Frequencies in Hz and power in µV²/Hz (or µV² with `scaling: "spectrum"`).
 */
export function periodogram(signal: Signal, options: SpectrumOptions): PowerSpectrum {
  // One segment spanning the whole signal; an explicit window must match its length.
  return welch(signal, {
    ...options,
    window: options.window ?? "rectangular",
    segmentLength: signal.length,
    overlap: 0,
  });
}
