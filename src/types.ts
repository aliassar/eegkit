/**
 * Shared types.
 *
 * Data layout conventions used across the library:
 *
 * - A **signal** is one channel: an array of samples in microvolts.
 * - A **frame** is one sample per channel, at a single instant. Devices and
 *   network streams usually deliver data this way.
 * - A **chunk** is channel-major: `chunk[channel][sample]`. Files and batch
 *   APIs usually deliver data this way.
 *
 * Every function accepts plain arrays or typed arrays, and returns
 * `Float64Array`, so results can be handed to charts or workers without
 * copying.
 */

/** One channel of samples. Plain arrays and typed arrays are both accepted. */
export type Signal = ArrayLike<number>;

/** One sample per channel, at one instant. */
export type Frame = ArrayLike<number>;

/** Channel-major block of samples: `chunk[channel][sample]`. */
export type Chunk = readonly ArrayLike<number>[];

/** A frequency band, in hertz. `low` is inclusive and `high` exclusive. */
export interface Band {
  readonly name: string;
  readonly low: number;
  readonly high: number;
}

/** The five conventional EEG bands. */
export type StandardBandName = "delta" | "theta" | "alpha" | "beta" | "gamma";

/**
 * Conventional EEG band edges. Definitions vary between labs by a hertz or
 * so; pass your own `Band[]` wherever bands are accepted if you need others.
 */
export const STANDARD_BANDS: readonly Band[] = Object.freeze([
  { name: "delta", low: 1, high: 4 },
  { name: "theta", low: 4, high: 8 },
  { name: "alpha", low: 8, high: 13 },
  { name: "beta", low: 13, high: 30 },
  { name: "gamma", low: 30, high: 45 },
]);

/** Power per band for one channel, keyed by band name. */
export type BandPowers = Record<string, number>;
