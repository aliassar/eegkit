import type { BandPowers } from "../types";

/** @internal */
function alphaOf(value: BandPowers | number, side: string): number {
  if (typeof value === "number") return value;
  const alpha = value.alpha;
  if (alpha === undefined) {
    throw new TypeError(`frontalAlphaAsymmetry: ${side} band powers have no "alpha" entry`);
  }
  return alpha;
}

/**
 * Frontal alpha asymmetry (FAA): `ln(right alpha) - ln(left alpha)`.
 *
 * Convention (Davidson, 1992, Brain and Cognition 20(1):125-151; reviewed by
 * Coan & Allen, 2004, Biological Psychology 67(1-2):7-50): alpha power is
 * inversely related to cortical activity, so a **positive** score means
 * relatively more alpha on the right, i.e. relatively greater **left**
 * frontal activity, which is associated with approach motivation and
 * positive affect. Negative scores indicate relatively greater right
 * frontal activity (withdrawal).
 *
 * Use homologous electrode pairs: F3/F4 (most common), F7/F8, Fp1/Fp2 or
 * AF7/AF8 (left first). Pass absolute alpha power from both sides computed the
 * same way (same window, reference and band); the log makes the score
 * insensitive to an overall gain shared by both channels. Natural log is the
 * field convention.
 *
 * @param left - Left-hemisphere band powers (its `alpha` is used) or alpha power in µV².
 * @param right - Right-hemisphere band powers or alpha power in µV².
 * @returns The asymmetry score (dimensionless), or `NaN` if either alpha
 *   power is zero, negative or not finite.
 * @throws TypeError if band powers are passed without an `alpha` entry.
 *
 * @example
 * ```ts
 * frontalAlphaAsymmetry(powersF3, powersF4); // > 0: more left-frontal activity
 * frontalAlphaAsymmetry(4, 6); // ln(6) - ln(4) ≈ 0.405
 * ```
 */
export function frontalAlphaAsymmetry(
  left: BandPowers | number,
  right: BandPowers | number,
): number {
  const l = alphaOf(left, "left");
  const r = alphaOf(right, "right");
  if (!(l > 0) || !(r > 0) || !Number.isFinite(l) || !Number.isFinite(r)) return Number.NaN;
  return Math.log(r) - Math.log(l);
}
