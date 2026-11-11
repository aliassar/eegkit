import type { Signal } from "../types";
import { IIRFilter } from "./iir";
import type { SOS } from "./sos";

/**
 * Filters a whole signal causally from zero initial state, matching
 * `scipy.signal.sosfilt(sos, signal)`.
 *
 * The output is delayed by the filter's group delay; use {@link filtfilt}
 * for offline analysis where phase matters, or {@link IIRFilter} for streams.
 *
 * @example
 * ```ts
 * const clean = sosfilt(butterworthHighpass(2, 1, 256), raw);
 * ```
 */
export function sosfilt(sos: SOS, signal: Signal): Float64Array {
  return new IIRFilter(sos).processBlock(signal);
}

/**
 * The padding `filtfilt` adds at each end, in samples: SciPy's default
 * `3 * ntaps`, where `ntaps = 2 * sections + 1` minus one for each
 * first-order section.
 */
export function filtfiltPadLength(sos: SOS): number {
  let zeroB2 = 0;
  let zeroA2 = 0;
  for (const s of sos) {
    if (s.b2 === 0) zeroB2++;
    if (s.a2 === 0) zeroA2++;
  }
  return 3 * (2 * sos.length + 1 - Math.min(zeroB2, zeroA2));
}

/**
 * Zero-phase filtering: runs the filter forward, then backward, so the output
 * has no phase shift (peaks stay where they were) and the magnitude response
 * is squared. Matches `scipy.signal.sosfiltfilt(sos, signal)` with its
 * defaults: odd extension of {@link filtfiltPadLength} samples at each end
 * and steady-state initial conditions for both passes, which together
 * suppress edge transients.
 *
 * Offline only: the backward pass needs the whole signal.
 *
 * @throws RangeError if `signal.length <= filtfiltPadLength(sos)`.
 * @example
 * ```ts
 * const alpha = filtfilt(butterworthBandpass(4, 8, 13, 256), epoch);
 * ```
 */
export function filtfilt(sos: SOS, signal: Signal): Float64Array {
  const n = signal.length;
  const pad = filtfiltPadLength(sos);
  if (n <= pad) {
    throw new RangeError(
      `filtfilt needs a signal longer than its padding of ${pad} samples, got ${n}; ` +
        "use a shorter filter or a longer signal",
    );
  }

  // Odd extension: reflect the signal through its end points, which keeps the
  // value and slope continuous at the edges.
  const x = Float64Array.from(signal);
  const first = x[0] ?? 0;
  const last = x[n - 1] ?? 0;
  const ext = new Float64Array(n + 2 * pad);
  ext.set(
    x.slice(1, pad + 1).map((v) => 2 * first - v),
    0,
  );
  ext.subarray(0, pad).reverse();
  ext.set(x, pad);
  ext.set(
    x.slice(n - 1 - pad, n - 1).map((v) => 2 * last - v),
    pad + n,
  );
  ext.subarray(pad + n).reverse();

  // Forward then backward pass, each started from the steady state for its
  // first input sample.
  const filter = new IIRFilter(sos);
  filter.prime(ext[0] ?? 0);
  filter.processBlock(ext, ext);
  ext.reverse();
  filter.prime(ext[0] ?? 0);
  filter.processBlock(ext, ext);
  ext.reverse();

  return ext.slice(pad, pad + n);
}
