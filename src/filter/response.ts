import type { Signal } from "../types";
import { type SOS, validateSOS } from "./sos";

/** Complex frequency response split into magnitude and phase. */
export interface FrequencyResponse {
  /** Linear gain |H(f)| (1 = unchanged; 1/√2 ≈ −3 dB). */
  readonly magnitude: Float64Array;
  /** Phase in radians, wrapped to (−π, π]. */
  readonly phase: Float64Array;
}

/**
 * Evaluates a filter's frequency response at the given frequencies, like
 * `scipy.signal.sosfreqz(sos, worN=frequencies, fs=sampleRate)`.
 *
 * @param frequencies - Frequencies in Hz (any values; usually 0 to `sampleRate / 2`).
 * @param sampleRate - Sampling rate in Hz.
 * @example
 * ```ts
 * const { magnitude } = frequencyResponse(notch(50, 256), [49, 50, 51], 256);
 * const dB = Array.from(magnitude, (m) => 20 * Math.log10(m));
 * ```
 */
export function frequencyResponse(
  sos: SOS,
  frequencies: Signal,
  sampleRate: number,
): FrequencyResponse {
  validateSOS(sos);
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`sampleRate must be a positive finite number (Hz), got ${sampleRate}`);
  }
  const n = frequencies.length;
  const magnitude = new Float64Array(n);
  const phase = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = (2 * Math.PI * (frequencies[i] ?? Number.NaN)) / sampleRate;
    // z⁻¹ = e^{-jw}, z⁻² = e^{-2jw}
    const c1 = Math.cos(w);
    const s1 = -Math.sin(w);
    const c2 = Math.cos(2 * w);
    const s2 = -Math.sin(2 * w);
    let hr = 1;
    let hi = 0;
    for (const s of sos) {
      const nr = s.b0 + s.b1 * c1 + s.b2 * c2;
      const ni = s.b1 * s1 + s.b2 * s2;
      const dr = 1 + s.a1 * c1 + s.a2 * c2;
      const di = s.a1 * s1 + s.a2 * s2;
      const d = dr * dr + di * di;
      const qr = (nr * dr + ni * di) / d;
      const qi = (ni * dr - nr * di) / d;
      const r = hr * qr - hi * qi;
      hi = hr * qi + hi * qr;
      hr = r;
    }
    magnitude[i] = Math.hypot(hr, hi);
    phase[i] = Math.atan2(hi, hr);
  }
  return { magnitude, phase };
}
