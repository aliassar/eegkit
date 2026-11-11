import type { BandPowers } from "../types";

/**
 * Looks up a band, failing loudly on a typo rather than silently producing NaN.
 * @internal
 */
function getBand(powers: BandPowers, name: string): number {
  const value = powers[name];
  if (value === undefined) {
    const known = Object.keys(powers).join(", ") || "none";
    throw new TypeError(`Band "${name}" not found in band powers (available: ${known})`);
  }
  return value;
}

/**
 * Divides two powers with the library-wide divide-by-zero convention: the
 * result is `NaN` whenever the denominator is not a positive finite number
 * (or the numerator is not finite).
 *
 * `NaN` rather than `Infinity` because a zero denominator means "no signal"
 * (a flat or disconnected channel), not "infinitely strong ratio", and
 * because {@link ExponentialSmoother} and {@link Baseline} skip `NaN` inputs
 * instead of being poisoned forever by an `Infinity`.
 * @internal
 */
function safeRatio(numerator: number, denominator: number): number {
  if (!(denominator > 0) || !Number.isFinite(denominator) || !Number.isFinite(numerator)) {
    return Number.NaN;
  }
  return numerator / denominator;
}

/**
 * Ratio of two bands' power, e.g. `bandRatio(p, "theta", "beta")`.
 *
 * Both values must be in the same units (µV² or µV²/Hz, or relative power);
 * the result is dimensionless.
 *
 * @returns `numerator / denominator`, or `NaN` if the denominator is zero,
 *   negative or not finite (see the module notes on why not `Infinity`).
 * @throws TypeError if either band name is missing from `powers`.
 *
 * @example
 * ```ts
 * bandRatio({ theta: 12, beta: 4 }, "theta", "beta"); // 3
 * ```
 */
export function bandRatio(powers: BandPowers, numerator: string, denominator: string): number {
  return safeRatio(getBand(powers, numerator), getBand(powers, denominator));
}

/**
 * Theta/beta power ratio (TBR).
 *
 * Studied as an attention marker and the classic ADHD neurofeedback target,
 * usually at Cz/Fz (Monastra et al., 1999, Neuropsychology 13(3):424-433).
 * Its diagnostic value is disputed (Arns et al., 2013, J Atten Disord
 * 17(5):374-383), so treat it as a within-person trend, not a diagnosis.
 *
 * @returns theta / beta, or `NaN` if beta power is not positive.
 */
export function thetaBetaRatio(powers: BandPowers): number {
  return bandRatio(powers, "theta", "beta");
}

/**
 * Alpha/theta power ratio.
 *
 * The quantity followed in alpha-theta neurofeedback for relaxation
 * (Peniston & Kulkosky, 1989, Alcohol Clin Exp Res 13(2):271-279), usually at
 * posterior sites such as Pz or O1/O2.
 *
 * @returns alpha / theta, or `NaN` if theta power is not positive.
 */
export function alphaThetaRatio(powers: BandPowers): number {
  return bandRatio(powers, "alpha", "theta");
}

/**
 * Engagement index `beta / (alpha + theta)`.
 *
 * From Pope, Bogart & Bartolome (1995), "Biocybernetic system evaluates
 * indices of operator engagement in automated task", Biological Psychology
 * 40(1-2):187-195, where it best tracked task engagement among the indices
 * compared. Originally averaged over Cz, Pz, P3 and O2.
 *
 * @returns beta / (alpha + theta), or `NaN` if alpha + theta is not positive.
 */
export function engagementIndex(powers: BandPowers): number {
  return safeRatio(getBand(powers, "beta"), getBand(powers, "alpha") + getBand(powers, "theta"));
}
