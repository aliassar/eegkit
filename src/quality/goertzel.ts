import type { Signal } from "../types";

/** @internal */
function checkArgs(signal: Signal, frequency: number, sampleRate: number): void {
  if (!(sampleRate > 0) || !Number.isFinite(sampleRate)) {
    throw new RangeError(`sampleRate must be a positive number in Hz, got ${sampleRate}`);
  }
  if (!(frequency >= 0 && frequency <= sampleRate / 2)) {
    throw new RangeError(
      `frequency must be in [0, ${sampleRate / 2}] Hz (Nyquist) for sampleRate ${sampleRate}, got ${frequency}`,
    );
  }
  if (signal.length === 0) throw new RangeError("signal must not be empty");
}

/**
 * Squared magnitude of the discrete-time Fourier transform of `signal` at one
 * frequency, computed with the Goertzel algorithm (Goertzel, 1958, Amer.
 * Math. Monthly 65(1):34-35).
 *
 * Returns `|Σ x[n]·e^(-iωn)|²` with `ω = 2π·frequency/sampleRate`, i.e.
 * exactly the squared magnitude of DFT bin `k` when
 * `frequency = k·sampleRate/N`, but for any frequency, not just bin centres.
 *
 * Why Goertzel: to watch one frequency (mains at 50/60 Hz) it costs O(N)
 * with two state variables and no allocation, versus O(N log N) plus buffers
 * for a full FFT, and it has no dependency on the spectral module.
 *
 * Units: µV² · samples² (unnormalised). See {@link goertzelPower} for a
 * normalised sinusoid power.
 *
 * @param signal - Samples in µV.
 * @param frequency - Frequency in Hz, in `[0, sampleRate / 2]`.
 * @param sampleRate - Sample rate in Hz.
 */
export function goertzel(signal: Signal, frequency: number, sampleRate: number): number {
  checkArgs(signal, frequency, sampleRate);
  const coeff = 2 * Math.cos((2 * Math.PI * frequency) / sampleRate);
  let s1 = 0;
  let s2 = 0;
  const n = signal.length;
  for (let i = 0; i < n; i++) {
    const s0 = (signal[i] as number) + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  // |X(ω)|² = s1² + s2² − 2cos(ω)·s1·s2 (the phase factor drops out).
  return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

/** Options for {@link goertzelPower}. */
export interface GoertzelPowerOptions {
  /**
   * Apply a periodic Hann window before measuring (default `true`). Hann
   * stops strong low-frequency content (drift, blinks, alpha) from leaking
   * into the measured frequency, at the cost of a slightly wider main lobe.
   */
  readonly hann?: boolean;
  /** Subtract the mean first (default `true`), so a DC offset can't leak in either. */
  readonly removeMean?: boolean;
}

/**
 * Power of the sinusoidal component of `signal` at `frequency`, in µV².
 *
 * Normalised so that a sinusoid `A·sin(2π·f·t)` gives `A²/2` (its mean
 * square) when `f` falls on a DFT bin: `2·|X(f)|² / (Σw)²`, where `w` is the
 * window (`Σw = N/2` for Hann, `N` for rectangular). Off-bin frequencies read
 * slightly low (scalloping: at most 1.4 dB with Hann, 3.9 dB without), so use
 * windows that are a whole number of cycles long when accuracy matters,
 * e.g. 1 s or 2 s at any integer line frequency.
 *
 * Allocation-free: the window and mean are applied on the fly.
 *
 * @param signal - Samples in µV.
 * @param frequency - Frequency in Hz, in `[0, sampleRate / 2]`.
 * @param sampleRate - Sample rate in Hz.
 * @returns Power in µV².
 *
 * @example
 * ```ts
 * const mains = goertzelPower(channel, 50, 256); // µV² of 50 Hz hum
 * ```
 */
export function goertzelPower(
  signal: Signal,
  frequency: number,
  sampleRate: number,
  options: GoertzelPowerOptions = {},
): number {
  checkArgs(signal, frequency, sampleRate);
  const hann = options.hann ?? true;
  const n = signal.length;
  let mean = 0;
  if (options.removeMean ?? true) {
    for (let i = 0; i < n; i++) mean += signal[i] as number;
    mean /= n;
  }
  const coeff = 2 * Math.cos((2 * Math.PI * frequency) / sampleRate);
  const step = (2 * Math.PI) / n;
  let s1 = 0;
  let s2 = 0;
  let wsum = 0;
  for (let i = 0; i < n; i++) {
    // Periodic Hann (as scipy.signal.get_window("hann", n)), which sums to
    // exactly n/2 and so keeps the A²/2 normalisation exact on bin centres.
    const w = hann ? 0.5 - 0.5 * Math.cos(step * i) : 1;
    wsum += w;
    const s0 = w * ((signal[i] as number) - mean) + coeff * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  const mag2 = s1 * s1 + s2 * s2 - coeff * s1 * s2;
  // DC and Nyquist have no conjugate twin in the spectrum, so no factor 2.
  const oneSided = frequency === 0 || frequency === sampleRate / 2 ? 1 : 2;
  return wsum > 0 ? (oneSided * mag2) / (wsum * wsum) : 0;
}
