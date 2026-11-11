/**
 * Second-order sections (SOS): the representation every eegkit filter uses.
 *
 * A high-order IIR filter written as a single transfer function `B(z)/A(z)`
 * is numerically fragile: polynomial coefficients lose precision quickly as
 * the order grows and poles crowd near `z = 1`, which is exactly where EEG
 * high-pass and narrow band-pass filters put them. Factoring the filter into
 * a cascade of biquads keeps each section well conditioned, which is why
 * SciPy recommends `output="sos"` and why eegkit uses nothing else.
 */

/**
 * One biquad section with `a0` normalised to 1:
 *
 * `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`
 *
 * A first-order section has `b2 = a2 = 0`.
 */
export interface Biquad {
  readonly b0: number;
  readonly b1: number;
  readonly b2: number;
  readonly a1: number;
  readonly a2: number;
}

/** A cascade of biquads, applied in order. */
export type SOS = readonly Biquad[];

/**
 * Cascades several filters into one: the result applies each SOS in turn.
 *
 * @example
 * ```ts
 * const sos = cascade(butterworthHighpass(2, 1, 256), notch(50, 256));
 * const filter = new IIRFilter(sos);
 * ```
 */
export function cascade(...filters: readonly SOS[]): SOS {
  const out: Biquad[] = [];
  for (const sos of filters) {
    for (const section of sos) out.push(section);
  }
  return Object.freeze(out);
}

/** Throws unless `sos` is a non-empty array of finite biquads. */
export function validateSOS(sos: SOS): void {
  if (!Array.isArray(sos) || sos.length === 0) {
    throw new TypeError("sos must be a non-empty array of biquad sections");
  }
  for (let i = 0; i < sos.length; i++) {
    const s = sos[i];
    if (
      s === undefined ||
      !Number.isFinite(s.b0) ||
      !Number.isFinite(s.b1) ||
      !Number.isFinite(s.b2) ||
      !Number.isFinite(s.a1) ||
      !Number.isFinite(s.a2)
    ) {
      throw new TypeError(`sos[${i}] must have finite b0, b1, b2, a1, a2`);
    }
  }
}

/**
 * Packs an SOS into a flat `[b0, b1, b2, a1, a2, …]` array for the hot loops.
 * @internal
 */
export function packSOS(sos: SOS): Float64Array {
  validateSOS(sos);
  const c = new Float64Array(sos.length * 5);
  sos.forEach((s, i) => {
    c.set([s.b0, s.b1, s.b2, s.a1, s.a2], i * 5);
  });
  return c;
}

/**
 * Steady-state initial conditions for a unit-step input, matching
 * `scipy.signal.sosfilt_zi`.
 *
 * Returns `[z0, z1]` per section (length `2 * sos.length`) for the Direct
 * Form II Transposed state. Multiply by the first sample to start a filter as
 * if the input had been constant at that value forever, which removes the
 * startup transient a large DC offset would otherwise cause.
 *
 * Throws a `RangeError` if a section has a pole at `z = 1` (no steady state).
 */
export function sosfiltZi(sos: SOS): Float64Array {
  validateSOS(sos);
  const zi = new Float64Array(sos.length * 2);
  let scale = 1;
  sos.forEach((s, i) => {
    // Solve (I - Aᵀ) z = b[1:] - a[1:]·b0 for the companion matrix of this section;
    // the 2×2 system has the closed form below.
    const denom = 1 + s.a1 + s.a2;
    if (denom === 0) {
      throw new RangeError(`sos[${i}] has a pole at z = 1, so it has no steady state`);
    }
    const B0 = s.b1 - s.a1 * s.b0;
    const B1 = s.b2 - s.a2 * s.b0;
    const z0 = (B0 + B1) / denom;
    zi[2 * i] = scale * z0;
    zi[2 * i + 1] = scale * (B1 - s.a2 * z0);
    // Each later section sees the DC output of the ones before it.
    scale *= (s.b0 + s.b1 + s.b2) / denom;
  });
  return zi;
}
