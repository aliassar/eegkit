/**
 * Options for {@link ExponentialSmoother}: either the smoothing factor
 * directly, or a time constant and the rate at which `update` is called.
 */
export type ExponentialSmootherOptions =
  | {
      /** Weight of each new value, in (0, 1]. 1 disables smoothing. */
      readonly alpha: number;
    }
  | {
      /** Time constant τ in seconds: after τ, a step change is 63.2 % reflected. */
      readonly timeConstant: number;
      /** How often `update` is called, in Hz (e.g. 4 for windows every 0.25 s). */
      readonly rate: number;
    };

/**
 * Exponential moving average (first-order IIR low-pass) for metric streams,
 * such as a band ratio recomputed for every window.
 *
 * `y[n] = y[n-1] + alpha * (x[n] - y[n-1])`.
 *
 * When built from a time constant, `alpha = 1 - exp(-1 / (timeConstant * rate))`.
 * That is the exact discretisation of a continuous RC low-pass, so the
 * response to a step reaches `1 - 1/e` (63.2 %) of the change after
 * `timeConstant` seconds regardless of the update rate; the common
 * approximation `alpha = 1 / (τ·rate)` is only accurate when τ·rate is large.
 *
 * The first value initialises the output (no ramp-up from zero). Non-finite
 * inputs (`NaN`, `±Infinity`, e.g. a ratio over a flat channel) are ignored
 * and leave the output unchanged, so one bad window can't poison the
 * average forever.
 *
 * @example
 * ```ts
 * const smooth = new ExponentialSmoother({ timeConstant: 2, rate: 4 });
 * win.onWindow((w) => show(smooth.update(engagementIndex(powersOf(w)))));
 * ```
 */
export class ExponentialSmoother {
  /** Smoothing factor in (0, 1]: the weight given to each new value. */
  readonly alpha: number;
  private current = Number.NaN;

  constructor(options: ExponentialSmootherOptions) {
    let alpha: number;
    if ("alpha" in options) {
      alpha = options.alpha;
      if (!(alpha > 0 && alpha <= 1)) {
        throw new RangeError(`ExponentialSmoother alpha must be in (0, 1], got ${alpha}`);
      }
    } else {
      const { timeConstant, rate } = options;
      if (!(timeConstant > 0) || !Number.isFinite(timeConstant)) {
        throw new RangeError(
          `ExponentialSmoother timeConstant must be a positive number of seconds, got ${timeConstant}`,
        );
      }
      if (!(rate > 0) || !Number.isFinite(rate)) {
        throw new RangeError(
          `ExponentialSmoother rate must be a positive number in Hz, got ${rate}`,
        );
      }
      alpha = 1 - Math.exp(-1 / (timeConstant * rate));
    }
    this.alpha = alpha;
  }

  /** The smoothed value, or `NaN` before the first finite input. */
  get value(): number {
    return this.current;
  }

  /**
   * Feeds one value and returns the new smoothed value. Non-finite inputs
   * are ignored (the current value is returned). Never allocates.
   */
  update(x: number): number {
    if (!Number.isFinite(x)) return this.current;
    const y = this.current;
    this.current = Number.isNaN(y) ? x : y + this.alpha * (x - y);
    return this.current;
  }

  /** Forgets the history; the next finite input initialises the output again. */
  reset(): void {
    this.current = Number.NaN;
  }
}
