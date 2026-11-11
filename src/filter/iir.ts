// biome-ignore-all lint/style/noNonNullAssertion: per-sample hot loops index typed arrays within bounds fixed at construction.
import type { Signal } from "../types";
import { packSOS, type SOS, sosfiltZi } from "./sos";

/**
 * A stateful IIR filter for streaming: a cascade of biquads in Direct Form II
 * Transposed, which needs only two state variables per section and has the
 * best round-off behaviour of the direct forms in floating point.
 *
 * The state persists across calls, so feeding a signal one sample at a time,
 * in blocks of any size, or all at once gives identical output (and the same
 * output as {@link sosfilt}).
 *
 * @example
 * ```ts
 * const filter = new IIRFilter(butterworthBandpass(4, 1, 40, 256));
 * device.onSample((uV) => plot(filter.process(uV)));
 * ```
 */
export class IIRFilter {
  // TypeScript `private` rather than `#private`: native private fields are
  // down-levelled to WeakMap lookups for ES2020 targets and are missing from
  // older Hermes builds, which would cost dearly in the per-sample path.
  private readonly _sos: SOS;
  private readonly coeffs: Float64Array;
  private readonly state: Float64Array;
  private readonly sections: number;
  private zi: Float64Array | undefined;

  /** @param sos - The filter, e.g. from {@link butterworthLowpass}. */
  constructor(sos: SOS) {
    this.coeffs = packSOS(sos);
    this._sos = sos;
    this.sections = sos.length;
    this.state = new Float64Array(sos.length * 2);
  }

  /** The second-order sections this filter applies. */
  get sos(): SOS {
    return this._sos;
  }

  /** Filters one sample and returns the output sample. Does not allocate. */
  process(x: number): number {
    const c = this.coeffs;
    const s = this.state;
    let v = x;
    for (let i = 0, j = 0, k = 0; i < this.sections; i++, j += 5, k += 2) {
      const y = c[j]! * v + s[k]!;
      s[k] = c[j + 1]! * v - c[j + 3]! * y + s[k + 1]!;
      s[k + 1] = c[j + 2]! * v - c[j + 4]! * y;
      v = y;
    }
    return v;
  }

  /**
   * Filters a block of samples, continuing from the current state.
   *
   * @param input - Samples to filter.
   * @param out - Optional output buffer with at least `input.length` elements;
   *   may be the same array as `input` for in-place filtering.
   * @returns `out` (or a new array) holding the filtered samples.
   */
  processBlock(input: Signal, out?: Float64Array): Float64Array {
    const n = input.length;
    if (out !== undefined && out.length < n) {
      throw new RangeError(`out has length ${out.length}, needs at least ${n}`);
    }
    const y = out ?? new Float64Array(n);
    if (n === 0) return y;
    // Copy first so every section runs in place on one Float64Array: the inner
    // loop then sees a single array type and stays monomorphic.
    if (y !== input) y.set(input);
    const c = this.coeffs;
    const s = this.state;
    // Section-outer loop: each section's state and coefficients stay in
    // registers for the whole block. The arithmetic per sample is identical to
    // `process`, so the output is bit-for-bit the same.
    for (let i = 0, j = 0, k = 0; i < this.sections; i++, j += 5, k += 2) {
      const b0 = c[j]!;
      const b1 = c[j + 1]!;
      const b2 = c[j + 2]!;
      const a1 = c[j + 3]!;
      const a2 = c[j + 4]!;
      let z0 = s[k]!;
      let z1 = s[k + 1]!;
      for (let t = 0; t < n; t++) {
        const v = y[t]!;
        const o = b0 * v + z0;
        z0 = b1 * v - a1 * o + z1;
        z1 = b2 * v - a2 * o;
        y[t] = o;
      }
      s[k] = z0;
      s[k + 1] = z1;
    }
    return y;
  }

  /** Clears the state, as if no samples had been processed. */
  reset(): void {
    this.state.fill(0);
  }

  /**
   * Sets the state to the steady state for a constant input `x0`
   * (`scipy.signal.sosfilt_zi(sos) * x0`), so a signal starting at `x0`
   * produces no startup transient. Call it with the first sample before
   * processing, which matters for high-pass filters on signals with a large
   * DC offset (common with dry electrodes).
   *
   * @param x0 - The input value to settle at, in the input's units.
   */
  prime(x0: number): void {
    this.zi ??= sosfiltZi(this._sos);
    const zi = this.zi;
    for (let i = 0; i < zi.length; i++) this.state[i] = zi[i]! * x0;
  }

  /** Returns a new filter with the same coefficients and fresh (zero) state. */
  clone(): IIRFilter {
    return new IIRFilter(this._sos);
  }
}
