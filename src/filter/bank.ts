// biome-ignore-all lint/style/noNonNullAssertion: per-sample hot loops index typed arrays within bounds fixed at construction.
import type { Chunk, Frame } from "../types";
import { packSOS, type SOS, sosfiltZi } from "./sos";

/**
 * The same filter applied independently to every channel of a multi-channel
 * stream, with one state per channel in a single flat buffer.
 *
 * To apply several filters (say a band-pass and a notch), combine them with
 * {@link cascade} first: one bank with all the sections is cheaper than a
 * chain of banks.
 *
 * @example
 * ```ts
 * const sos = cascade(butterworthBandpass(4, 1, 40, 256), notch(50, 256));
 * const bank = new FilterBank(sos, 8);
 * const out = new Float64Array(8);
 * device.onFrame((frame) => draw(bank.processFrame(frame, out)));
 * ```
 */
export class FilterBank {
  private readonly _sos: SOS;
  private readonly _channels: number;
  private readonly coeffs: Float64Array;
  private readonly state: Float64Array;
  private readonly sections: number;

  /**
   * @param sos - The filter to apply to each channel.
   * @param channels - Number of channels (≥ 1).
   */
  constructor(sos: SOS, channels: number) {
    if (!Number.isInteger(channels) || channels < 1) {
      throw new RangeError(`channels must be an integer >= 1, got ${channels}`);
    }
    this.coeffs = packSOS(sos);
    this._sos = sos;
    this._channels = channels;
    this.sections = sos.length;
    this.state = new Float64Array(channels * sos.length * 2);
  }

  /** The filter applied to each channel. */
  get sos(): SOS {
    return this._sos;
  }

  /** Number of channels. */
  get channels(): number {
    return this._channels;
  }

  /**
   * Filters one frame (one sample per channel). Does not allocate when `out`
   * is given.
   *
   * @param frame - One sample per channel, in µV.
   * @param out - Optional output buffer of length ≥ `channels`; may be `frame` itself.
   */
  processFrame(frame: Frame, out?: Float64Array): Float64Array {
    const channels = this._channels;
    if (frame.length !== channels) {
      throw new RangeError(`frame has ${frame.length} channels, expected ${channels}`);
    }
    if (out !== undefined && out.length < channels) {
      throw new RangeError(`out has length ${out.length}, needs at least ${channels}`);
    }
    const y = out ?? new Float64Array(channels);
    const c = this.coeffs;
    const s = this.state;
    const sections = this.sections;
    for (let ch = 0, k = 0; ch < channels; ch++) {
      let v = frame[ch]!;
      for (let i = 0, j = 0; i < sections; i++, j += 5, k += 2) {
        const o = c[j]! * v + s[k]!;
        s[k] = c[j + 1]! * v - c[j + 3]! * o + s[k + 1]!;
        s[k + 1] = c[j + 2]! * v - c[j + 4]! * o;
        v = o;
      }
      y[ch] = v;
    }
    return y;
  }

  /**
   * Filters a channel-major block `chunk[channel][sample]`, continuing from
   * the current state. Gives the same result as calling
   * {@link FilterBank.processFrame} on each frame in turn.
   *
   * @returns A new array per channel.
   */
  processChunk(chunk: Chunk): Float64Array[] {
    const channels = this._channels;
    if (chunk.length !== channels) {
      throw new RangeError(`chunk has ${chunk.length} channels, expected ${channels}`);
    }
    const c = this.coeffs;
    const s = this.state;
    const out: Float64Array[] = [];
    for (let ch = 0; ch < channels; ch++) {
      const input = chunk[ch]!;
      // Copy first so every section runs in place on one Float64Array (monomorphic loop).
      const y = Float64Array.from(input);
      const n = y.length;
      for (let i = 0, j = 0, k = ch * this.sections * 2; i < this.sections; i++, j += 5, k += 2) {
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
      out.push(y);
    }
    return out;
  }

  /**
   * Sets every channel to its steady state for a constant input, so a stream
   * that starts with large DC offsets has no startup transient. Pass the
   * first frame.
   */
  prime(frame: Frame): void {
    if (frame.length !== this._channels) {
      throw new RangeError(`frame has ${frame.length} channels, expected ${this._channels}`);
    }
    const zi = sosfiltZi(this._sos);
    for (let ch = 0; ch < this._channels; ch++) {
      const x0 = frame[ch]!;
      const base = ch * zi.length;
      for (let i = 0; i < zi.length; i++) this.state[base + i] = zi[i]! * x0;
    }
  }

  /** Clears every channel's state. */
  reset(): void {
    this.state.fill(0);
  }
}
