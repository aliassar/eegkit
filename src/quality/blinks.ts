import type { Signal } from "../types";

/** Options for {@link detectBlinks}. */
export interface BlinkOptions {
  /** Sample rate in Hz. */
  readonly sampleRate: number;
  /**
   * Detection threshold on the filtered signal, in µV. Default 70 µV.
   *
   * Blinks measure roughly 100–200 µV at Fp1/Fp2 (less at AF7/AF8 and much
   * less further back), while background EEG after the 0.5–10 Hz smoothing
   * used here stays within about ±40 µV. 70 µV sits between the two; lower it
   * for small blinks or electrodes further from the eyes.
   */
  readonly threshold?: number;
  /**
   * Minimum time between two blinks, in seconds. Default 0.5 s: a blink
   * lasts 0.1–0.4 s and spontaneous blinks are rarely closer than that, so
   * this also stops one blink (and its overshoot) from counting twice. When
   * two peaks fall closer together, the larger one is kept.
   */
  readonly refractory?: number;
  /**
   * Sign of the blink deflection. Default `"positive"`: with the usual
   * mastoid/earlobe reference, the eyeball's positive cornea rotating upward
   * under the closing lid makes blinks positive at frontopolar sites. Use
   * `"negative"` if your montage inverts it, `"both"` if unsure.
   */
  readonly polarity?: "positive" | "negative" | "both";
}

/** Low-pass smoothing window (s): a boxcar of 0.1 s has its first zero at 10 Hz. */
const SMOOTH_SECONDS = 0.1;
/** Baseline window (s) subtracted to remove drift; long compared to a blink. */
const BASELINE_SECONDS = 1;

/**
 * Centered moving average from a prefix-sum array, with the window truncated
 * at the edges so there is no padding bias and no delay.
 * @internal
 */
function movingAverage(prefix: Float64Array, n: number, half: number, out: Float64Array): void {
  for (let i = 0; i < n; i++) {
    const lo = Math.max(i - half, 0);
    const hi = Math.min(i + half + 1, n);
    out[i] = ((prefix[hi] as number) - (prefix[lo] as number)) / (hi - lo);
  }
}

/**
 * Finds eye blinks in a frontal channel (Fp1, Fp2, AF7, AF8…) and returns the
 * sample index of each blink peak.
 *
 * Method: the signal is band-limited with a difference of two centred
 * moving averages, a 0.1 s one (low-pass; its first zero at 10 Hz removes
 * alpha) minus a 1 s one (removes drift and DC). That keeps the slow,
 * large blink waveform while suppressing ongoing EEG, at O(n) cost and with
 * zero phase, so indices line up with the input. Peaks above `threshold`
 * are then picked, one per excursion, and peaks closer than `refractory`
 * are merged (the larger one wins).
 *
 * This is a simple amplitude detector: it also fires on other large slow
 * deflections (e.g. vertical eye movements). Pair it with
 * {@link assessQuality} for general artifact flagging.
 *
 * @param signal - One frontal channel in µV, ideally a few seconds or more.
 * @returns Indices into `signal` of detected blink peaks, ascending.
 * @throws RangeError for an invalid sample rate, threshold or refractory period.
 *
 * @example
 * ```ts
 * const blinks = detectBlinks(fp1, { sampleRate: 256 });
 * const blinksPerMinute = blinks.length / (fp1.length / 256 / 60);
 * ```
 */
export function detectBlinks(signal: Signal, options: BlinkOptions): number[] {
  const { sampleRate } = options;
  const threshold = options.threshold ?? 70;
  const refractory = options.refractory ?? 0.5;
  const polarity = options.polarity ?? "positive";
  if (!(sampleRate > 0) || !Number.isFinite(sampleRate)) {
    throw new RangeError(`sampleRate must be a positive number in Hz, got ${sampleRate}`);
  }
  if (!(threshold > 0)) {
    throw new RangeError(`threshold must be a positive number of µV, got ${threshold}`);
  }
  if (!(refractory >= 0)) {
    throw new RangeError(`refractory must be a non-negative number of seconds, got ${refractory}`);
  }
  const n = signal.length;
  const peaks: number[] = [];
  if (n === 0) return peaks;

  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const x = signal[i] as number;
    // Treat dropped (non-finite) samples as 0 rather than poisoning the sums.
    prefix[i + 1] = (prefix[i] as number) + (Number.isFinite(x) ? x : 0);
  }
  const smooth = new Float64Array(n);
  const baseline = new Float64Array(n);
  movingAverage(prefix, n, Math.max(0, Math.round((SMOOTH_SECONDS * sampleRate) / 2)), smooth);
  movingAverage(prefix, n, Math.max(1, Math.round((BASELINE_SECONDS * sampleRate) / 2)), baseline);

  const sign = polarity === "negative" ? -1 : 1;
  const minGap = Math.round(refractory * sampleRate);
  let inPeak = false;
  let bestIndex = 0;
  let bestValue = 0;
  let lastValue = 0;
  const commit = (): void => {
    const last = peaks.length - 1;
    const lastIndex = peaks[last];
    if (lastIndex !== undefined && bestIndex - lastIndex < minGap) {
      // Too close to the previous blink: keep whichever is larger.
      if (bestValue > lastValue) {
        peaks[last] = bestIndex;
        lastValue = bestValue;
      }
      return;
    }
    peaks.push(bestIndex);
    lastValue = bestValue;
  };
  for (let i = 0; i < n; i++) {
    const d = (smooth[i] as number) - (baseline[i] as number);
    const v = polarity === "both" ? Math.abs(d) : sign * d;
    if (v > threshold) {
      if (!inPeak || v > bestValue) {
        bestIndex = i;
        bestValue = v;
      }
      inPeak = true;
    } else if (inPeak) {
      commit();
      inPeak = false;
    }
  }
  if (inPeak) commit();
  return peaks;
}
