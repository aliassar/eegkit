import type { Signal } from "../types";

/** Options for {@link assessQuality}. */
export interface QualityOptions {
  /** Sample rate in Hz. */
  readonly sampleRate: number;
  /**
   * Mains frequency in Hz (default 50). 60 Hz in the Americas, Taiwan,
   * South Korea, the Philippines and parts of Japan and Saudi Arabia.
   */
  readonly lineFrequency?: 50 | 60;
  /**
   * Peak-to-peak amplitude, in µV, above which the window is flagged as an
   * artifact. Default 150 µV.
   *
   * Why 150: background EEG rarely exceeds ~100 µV peak-to-peak in awake
   * adults, while blinks, movement and electrode pops reach hundreds of µV.
   * Peak-to-peak rejection thresholds of 100–150 µV are the usual choice in
   * epoch rejection (e.g. MNE-Python's `reject` examples and ERP practice of
   * ±75–100 µV absolute limits); 150 errs on the side of keeping data, as a
   * streaming quality indicator should not flicker on every large alpha burst.
   */
  readonly amplitudeLimit?: number;
  /**
   * Standard deviation, in µV, below which the channel is considered flat
   * (disconnected, shorted or saturated). Default 0.5 µV.
   *
   * Why 0.5: even quiet eyes-open EEG has a standard deviation of several µV,
   * while the noise floor of a typical EEG amplifier with the input shorted
   * is about 0.2–1 µV RMS. Below 0.5 µV there is no physiological signal.
   */
  readonly flatlineThreshold?: number;
  /**
   * The amplifier's input range `[min, max]` in µV, if known. Samples at or
   * beyond either rail count as clipped. Without it, clipping is inferred
   * from runs of three or more identical samples at the window's extreme
   * value, which real (noisy, high-resolution) EEG essentially never produces.
   */
  readonly range?: readonly [min: number, max: number];
}

/** Quality assessment of one channel over one window. */
export interface ChannelQuality {
  /** Overall quality from 0 (unusable) to 1 (clean); see {@link assessQuality}. */
  readonly score: number;
  /** `score` bucketed: good ≥ 0.8, fair ≥ 0.4, poor below. */
  readonly level: "good" | "fair" | "poor";
  /** True when the detrended standard deviation is below `flatlineThreshold`. */
  readonly flatline: boolean;
  /** Fraction (0..1) of samples judged clipped at the amplifier rails. */
  readonly clipped: number;
  /** Peak-to-peak amplitude in µV after removing a linear trend. */
  readonly peakToPeak: number;
  /**
   * Fraction (0..1) of the signal's variance at the mains frequency; `NaN`
   * when the mains frequency is at or above Nyquist and cannot be measured.
   */
  readonly lineNoiseRatio: number;
  /** True when `peakToPeak > amplitudeLimit` or the window has non-finite samples. */
  readonly artifact: boolean;
}

/** Score thresholds for the `level` buckets. */
const GOOD = 0.8;
const FAIR = 0.4;
/** Line-noise ratio up to which the line component doesn't lower the score. */
const LINE_OK = 0.1;
/** Line-noise ratio at which the line component brings the score to 0. */
const LINE_BAD = 0.5;
/** Clipped fraction at which the clipping component reaches 0. */
const CLIP_BAD = 0.1;
/** Ceiling on the score as soon as a window has an artifact or any clipping: "fair" at best. */
const FLAGGED_CEILING = 0.6;
/** Minimum run of identical extreme samples treated as clipping when `range` is unknown. */
const CLIP_RUN = 3;

const clamp01 = (x: number): number => Math.min(Math.max(x, 0), 1);

/**
 * Rates the signal quality of one channel over a window (1–4 s recommended),
 * e.g. to drive a per-electrode contact indicator.
 *
 * The window is first detrended (least-squares line removed) so slow
 * electrode drift doesn't count as an artifact. Then four checks run, in
 * two allocation-free passes over the data:
 *
 * - **Flatline**: detrended standard deviation below `flatlineThreshold`.
 * - **Clipping**: fraction of samples at the amplifier rails (see `range`).
 * - **Amplitude artifact**: detrended peak-to-peak above `amplitudeLimit`.
 * - **Line noise**: power at the mains frequency (Hann-windowed
 *   {@link goertzelPower}) divided by the total detrended variance. Mains
 *   pickup grows with electrode impedance, so a high ratio usually means a
 *   loose or dry electrode. Measure on data that has not been notch filtered.
 *
 * **Score**: the product of one factor per check, each in [0, 1]:
 *
 * - flatline → 0, otherwise 1;
 * - amplitude: 1 up to the limit; above it `0.6 · (2 − p2p/limit)`, i.e. at
 *   most "fair" and 0 at twice the limit;
 * - clipping: 1 with no clipped samples; otherwise `0.6 · (1 − clipped/0.1)`,
 *   i.e. at most "fair" and 0 at 10 % clipped;
 * - line noise: 1 up to a ratio of 0.1, falling linearly to 0 at 0.5.
 *
 * A product (rather than a mean) makes any single fatal problem fatal,
 * which is what a user adjusting an electrode needs to see. `level` is
 * "good" for score ≥ 0.8, "fair" for ≥ 0.4, "poor" below.
 *
 * Non-finite samples (dropped packets often arrive as `NaN`) make the
 * window "poor" with score 0, `artifact: true` and `NaN` measurements.
 *
 * @param signal - One channel, in µV.
 * @throws RangeError for fewer than 2 samples or invalid options.
 *
 * @example
 * ```ts
 * const q = assessQuality(window[0], { sampleRate: 256, lineFrequency: 60 });
 * if (q.level === "poor") showWarning("Adjust electrode 1");
 * ```
 */
export function assessQuality(signal: Signal, options: QualityOptions): ChannelQuality {
  const { sampleRate } = options;
  const lineFrequency = options.lineFrequency ?? 50;
  const amplitudeLimit = options.amplitudeLimit ?? 150;
  const flatlineThreshold = options.flatlineThreshold ?? 0.5;
  const range = options.range;
  if (!(sampleRate > 0) || !Number.isFinite(sampleRate)) {
    throw new RangeError(`sampleRate must be a positive number in Hz, got ${sampleRate}`);
  }
  if (!(amplitudeLimit > 0)) {
    throw new RangeError(`amplitudeLimit must be a positive number of µV, got ${amplitudeLimit}`);
  }
  if (!(flatlineThreshold >= 0)) {
    throw new RangeError(
      `flatlineThreshold must be a non-negative number of µV, got ${flatlineThreshold}`,
    );
  }
  if (range !== undefined && !(range[0] < range[1])) {
    throw new RangeError(`range must be [min, max] with min < max, got [${range[0]}, ${range[1]}]`);
  }
  const n = signal.length;
  if (n < 2) throw new RangeError(`assessQuality needs at least 2 samples, got ${n}`);

  // Pass 1: sums for the mean and the trend, raw extremes, rail hits.
  const tMean = (n - 1) / 2;
  let sum = 0;
  let sumTx = 0;
  let rawMin = Number.POSITIVE_INFINITY;
  let rawMax = Number.NEGATIVE_INFINITY;
  let railHits = 0;
  let nonFinite = false;
  // Small tolerance: rails converted from ADC counts to µV can be off by rounding.
  const railTol = range === undefined ? 0 : (range[1] - range[0]) * 1e-9;
  for (let i = 0; i < n; i++) {
    const x = signal[i] as number;
    if (!Number.isFinite(x)) {
      nonFinite = true;
      continue;
    }
    sum += x;
    // Σ(t − t̄)·x equals Σ(t − t̄)(x − x̄) because Σ(t − t̄) = 0, so the trend
    // needs no separate pass for the mean.
    sumTx += (i - tMean) * x;
    if (x < rawMin) rawMin = x;
    if (x > rawMax) rawMax = x;
    if (range !== undefined && (x <= range[0] + railTol || x >= range[1] - railTol)) railHits++;
  }
  if (nonFinite) {
    return {
      score: 0,
      level: "poor",
      flatline: false,
      clipped: range === undefined ? 0 : railHits / n,
      peakToPeak: Number.NaN,
      lineNoiseRatio: Number.NaN,
      artifact: true,
    };
  }
  const mean = sum / n;
  const slope = sumTx / ((n * (n * n - 1)) / 12);

  // Pass 2: detrended residual statistics, Goertzel at the mains frequency,
  // and clipping runs when the rails are unknown.
  const measureLine = lineFrequency < sampleRate / 2;
  const coeff = 2 * Math.cos((2 * Math.PI * lineFrequency) / sampleRate);
  const hannStep = (2 * Math.PI) / n;
  let s1 = 0;
  let s2 = 0;
  let sumSq = 0;
  let resMin = Number.POSITIVE_INFINITY;
  let resMax = Number.NEGATIVE_INFINITY;
  let runValue = Number.NaN;
  let runLength = 0;
  let runClipped = 0;
  for (let i = 0; i < n; i++) {
    const x = signal[i] as number;
    const r = x - (mean + slope * (i - tMean));
    sumSq += r * r;
    if (r < resMin) resMin = r;
    if (r > resMax) resMax = r;
    if (measureLine) {
      const w = 0.5 - 0.5 * Math.cos(hannStep * i);
      const s0 = w * r + coeff * s1 - s2;
      s2 = s1;
      s1 = s0;
    }
    if (range === undefined) {
      if (x === runValue) {
        runLength++;
      } else {
        if (runLength >= CLIP_RUN) runClipped += runLength;
        runLength = x === rawMax || x === rawMin ? 1 : 0;
        runValue = runLength === 1 ? x : Number.NaN;
      }
    }
  }
  if (runLength >= CLIP_RUN) runClipped += runLength;

  const variance = sumSq / n;
  const std = Math.sqrt(variance);
  const flatline = std < flatlineThreshold || variance === 0;
  // A constant signal is one long run at its extreme; without known rails
  // that is a flatline, not clipping.
  const clipped = range !== undefined ? railHits / n : flatline ? 0 : runClipped / n;
  const peakToPeak = resMax - resMin;

  let lineNoiseRatio = Number.NaN;
  if (measureLine) {
    if (variance > 0) {
      const mag2 = s1 * s1 + s2 * s2 - coeff * s1 * s2;
      // Periodic Hann sums to n/2; see goertzelPower for the normalisation.
      const linePower = (2 * mag2) / ((n / 2) * (n / 2));
      lineNoiseRatio = clamp01(linePower / variance);
    } else {
      lineNoiseRatio = 0;
    }
  }
  const artifact = peakToPeak > amplitudeLimit;

  const fFlat = flatline ? 0 : 1;
  const fAmp = artifact ? FLAGGED_CEILING * clamp01(2 - peakToPeak / amplitudeLimit) : 1;
  const fClip = clipped > 0 ? FLAGGED_CEILING * clamp01(1 - clipped / CLIP_BAD) : 1;
  const fLine = Number.isNaN(lineNoiseRatio)
    ? 1
    : clamp01(1 - (lineNoiseRatio - LINE_OK) / (LINE_BAD - LINE_OK));
  const score = fFlat * fAmp * fClip * fLine;
  const level = score >= GOOD ? "good" : score >= FAIR ? "fair" : "poor";
  return { score, level, flatline, clipped, peakToPeak, lineNoiseRatio, artifact };
}
