/** Options for {@link Baseline}. */
export interface BaselineOptions {
  /**
   * Number of values needed before {@link Baseline.ready} is true and
   * {@link Baseline.zScore} returns numbers. Default 30: the standard
   * deviation of 30 Gaussian values has a relative standard error of about
   * `1 / sqrt(2 (n - 1))` ≈ 13 %, a reasonable floor for a calibration.
   * Must be at least 2 (a standard deviation needs two values).
   */
  readonly minSamples?: number;
}

/**
 * Running mean and standard deviation of a metric during a calibration
 * period, used to express later values as z-scores relative to that person's
 * own resting state. Neurofeedback metrics vary several-fold between people
 * and sessions, so personal baselines matter more than absolute thresholds.
 *
 * Uses Welford's online algorithm (Welford, 1962, Technometrics 4(3):419-420)
 * rather than accumulating `Σx` and `Σx²`: the naive formula subtracts two
 * large, nearly equal numbers and loses most significant digits when the mean
 * is large relative to the spread (e.g. band powers in µV²), and can even
 * yield a negative variance. Welford is numerically stable, O(1) per value,
 * and needs no stored history.
 *
 * Non-finite values (`NaN`, `±Infinity`) are ignored.
 *
 * @example
 * ```ts
 * const baseline = new Baseline({ minSamples: 60 });
 * // During a 60 s eyes-open calibration, one metric per second:
 * baseline.add(engagementIndex(powers));
 * // Afterwards:
 * if (baseline.ready) feedback(baseline.zScore(engagementIndex(powers)));
 * ```
 */
export class Baseline {
  /** Values required before the baseline is considered ready. */
  readonly minSamples: number;
  private n = 0;
  private m = 0;
  /** Sum of squared deviations from the running mean (Welford's M2). */
  private m2 = 0;

  constructor(options: BaselineOptions = {}) {
    const minSamples = options.minSamples ?? 30;
    if (!Number.isInteger(minSamples) || minSamples < 2) {
      throw new RangeError(`Baseline minSamples must be an integer >= 2, got ${minSamples}`);
    }
    this.minSamples = minSamples;
  }

  /** Adds one calibration value. Non-finite values are ignored. Never allocates. */
  add(x: number): void {
    if (!Number.isFinite(x)) return;
    this.n++;
    const delta = x - this.m;
    this.m += delta / this.n;
    this.m2 += delta * (x - this.m);
  }

  /** Number of values added. */
  get count(): number {
    return this.n;
  }

  /** True once `count >= minSamples`. */
  get ready(): boolean {
    return this.n >= this.minSamples;
  }

  /** Mean of the values added, or `NaN` if none. */
  get mean(): number {
    return this.n > 0 ? this.m : Number.NaN;
  }

  /**
   * Sample variance with Bessel's correction (divides by `count - 1`, like
   * NumPy's `var(ddof=1)`), or `NaN` with fewer than two values.
   */
  get variance(): number {
    return this.n > 1 ? this.m2 / (this.n - 1) : Number.NaN;
  }

  /** Sample standard deviation (`sqrt(variance)`), or `NaN` with fewer than two values. */
  get std(): number {
    return Math.sqrt(this.variance);
  }

  /**
   * How many standard deviations `x` lies from the baseline mean.
   *
   * @returns `(x - mean) / std`, or `NaN` until {@link ready} or when the
   *   baseline has zero spread (all calibration values equal).
   */
  zScore(x: number): number {
    if (this.n < this.minSamples) return Number.NaN;
    const sd = this.std;
    if (!(sd > 0)) return Number.NaN;
    return (x - this.m) / sd;
  }

  /** Discards all values to start a new calibration. */
  reset(): void {
    this.n = 0;
    this.m = 0;
    this.m2 = 0;
  }
}
