/**
 * Window functions, matching `scipy.signal.windows`.
 *
 * By default windows are *periodic* (`sym = false`): the length-n window is
 * the first n points of a symmetric window of length n + 1. That is what
 * spectral analysis wants (and what `scipy.signal.welch` uses), because the
 * window then tiles exactly under 50 % overlap and its DFT has no leakage
 * from the duplicated end point. Use `sym = true` for FIR filter design.
 */

/** Names of the built-in windows accepted by {@link welch} and {@link periodogram}. */
export type WindowName = "hann" | "hamming" | "blackman" | "rectangular";

function assertLength(n: number): void {
  if (!Number.isInteger(n) || n < 0) {
    throw new RangeError(`window length must be a non-negative integer, got ${n}`);
  }
}

/** Sum of cosines `Σ a_k·cos(k·x)` with x evenly spaced on [−π, π], like SciPy's `general_cosine`. */
function generalCosine(n: number, sym: boolean, coeffs: readonly number[]): Float64Array {
  assertLength(n);
  if (n <= 1) return new Float64Array(n).fill(1);
  const m = sym ? n : n + 1;
  const step = (2 * Math.PI) / (m - 1);
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i === m - 1 ? Math.PI : -Math.PI + i * step;
    w[i] = coeffs.reduce((sum, a, k) => sum + a * Math.cos(k * x), 0);
  }
  return w;
}

/**
 * Hann window (raised cosine), the default for EEG spectral estimation:
 * good leakage suppression with a main lobe of ±2 bins.
 *
 * @param n - Length in samples.
 * @param sym - `false` (default) for the periodic window used in spectral analysis.
 */
export function hann(n: number, sym = false): Float64Array {
  return generalCosine(n, sym, [0.5, 0.5]);
}

/**
 * Hamming window: a slightly narrower main lobe than Hann but a higher
 * far-side-lobe floor.
 *
 * @param n - Length in samples.
 * @param sym - `false` (default) for the periodic window used in spectral analysis.
 */
export function hamming(n: number, sym = false): Float64Array {
  return generalCosine(n, sym, [0.54, 0.46]);
}

/**
 * Blackman window: very low side lobes (−58 dB) at the cost of a wider
 * main lobe (±3 bins).
 *
 * @param n - Length in samples.
 * @param sym - `false` (default) for the periodic window used in spectral analysis.
 */
export function blackman(n: number, sym = false): Float64Array {
  return generalCosine(n, sym, [0.42, 0.5, 0.08]);
}

/**
 * Rectangular (boxcar) window: all ones, i.e. no tapering. Best frequency
 * resolution but strong leakage.
 *
 * @param n - Length in samples.
 */
export function rectangular(n: number): Float64Array {
  assertLength(n);
  return new Float64Array(n).fill(1);
}

/** Builds a named periodic window of length `n`. */
export function getWindow(name: WindowName, n: number): Float64Array {
  switch (name) {
    case "hann":
      return hann(n);
    case "hamming":
      return hamming(n);
    case "blackman":
      return blackman(n);
    case "rectangular":
      return rectangular(n);
    default:
      throw new TypeError(`unknown window "${String(name)}"`);
  }
}
