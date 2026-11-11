import type { Chunk, Frame } from "../types";
import { RingBuffer } from "./ring-buffer";

/** Options for {@link SlidingWindow}. */
export interface SlidingWindowOptions {
  /** Number of channels in every frame. */
  readonly channels: number;
  /** Window length, in samples (e.g. `2 * sampleRate` for 2 s windows). */
  readonly size: number;
  /**
   * Hop between consecutive windows, in samples. `step < size` gives
   * overlapping windows; `step > size` skips samples between windows.
   */
  readonly step: number;
}

/** Callback receiving a ready window: one array per channel, oldest to newest. */
export type WindowListener = (window: readonly Float64Array[]) => void;

/**
 * Cuts a multi-channel stream into fixed-size, optionally overlapping windows.
 *
 * Frames (one sample per channel) are pushed as they arrive; the first window
 * is ready after `size` frames and then every `step` frames. For `n` frames
 * pushed in total, exactly `floor((n - size) / step) + 1` windows become
 * ready (none while `n < size`), however the frames were split across
 * `pushFrame` and `pushChunk` calls.
 *
 * Pushing never allocates: samples go into one preallocated
 * {@link RingBuffer} per channel. Window contents are copied into
 * preallocated arrays that are **reused for every window**, so copy them
 * (`Float64Array.from(w)`) if you need to keep a window after the next one
 * is produced.
 *
 * @example
 * ```ts
 * // 2 s windows, updated every 0.25 s, for 4 channels at 256 Hz.
 * const win = new SlidingWindow({ channels: 4, size: 512, step: 64 });
 * win.onWindow((w) => {
 *   const alpha = bandPowers(welch(w[0], { sampleRate: 256 })).alpha;
 * });
 * stream.on("frame", (frame) => win.pushFrame(frame));
 * ```
 */
export class SlidingWindow {
  /** Number of channels. */
  readonly channels: number;
  /** Window length, in samples. */
  readonly size: number;
  /** Hop between windows, in samples. */
  readonly step: number;

  private readonly rings: RingBuffer[];
  private readonly output: Float64Array[];
  private readonly listeners = new Set<WindowListener>();
  /** Frames left until the next window is ready. */
  private untilNext: number;
  private frameCount = 0;

  constructor(options: SlidingWindowOptions) {
    const { channels, size, step } = options;
    if (!Number.isInteger(channels) || channels < 1) {
      throw new RangeError(`SlidingWindow channels must be a positive integer, got ${channels}`);
    }
    if (!Number.isInteger(size) || size < 1) {
      throw new RangeError(`SlidingWindow size must be a positive integer, got ${size}`);
    }
    if (!Number.isInteger(step) || step < 1) {
      throw new RangeError(`SlidingWindow step must be a positive integer, got ${step}`);
    }
    this.channels = channels;
    this.size = size;
    this.step = step;
    this.rings = [];
    this.output = [];
    for (let c = 0; c < channels; c++) {
      this.rings.push(new RingBuffer(size));
      this.output.push(new Float64Array(size));
    }
    this.untilNext = size;
  }

  /** Total frames pushed since construction or the last `reset`. */
  get frames(): number {
    return this.frameCount;
  }

  /** True once at least `size` frames have been pushed. */
  get isFull(): boolean {
    return this.frameCount >= this.size;
  }

  /**
   * Pushes one frame (one sample per channel). Never allocates.
   *
   * @returns `true` when this frame completed a new window. Listeners
   *   registered with {@link onWindow} have already been called by then.
   * @throws RangeError if the frame length differs from `channels`.
   */
  pushFrame(frame: Frame): boolean {
    const channels = this.channels;
    if (frame.length !== channels) {
      throw new RangeError(`Expected a frame of ${channels} channels, got ${frame.length}`);
    }
    const rings = this.rings;
    for (let c = 0; c < channels; c++) (rings[c] as RingBuffer).push(frame[c] as number);
    return this.tick();
  }

  /**
   * Pushes a channel-major block of samples (`chunk[channel][sample]`).
   * Equivalent to pushing its frames one by one: listeners are called once
   * per window that becomes ready inside the chunk.
   *
   * @returns The number of windows that became ready.
   * @throws RangeError if the chunk does not have `channels` rows of equal length.
   */
  pushChunk(chunk: Chunk): number {
    const channels = this.channels;
    if (chunk.length !== channels) {
      throw new RangeError(`Expected a chunk of ${channels} channels, got ${chunk.length}`);
    }
    const n = (chunk[0] as ArrayLike<number>).length;
    for (let c = 1; c < channels; c++) {
      const len = (chunk[c] as ArrayLike<number>).length;
      if (len !== n) {
        throw new RangeError(
          `Chunk channels differ in length: channel 0 has ${n}, ${c} has ${len}`,
        );
      }
    }
    const rings = this.rings;
    let ready = 0;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < channels; c++) {
        (rings[c] as RingBuffer).push((chunk[c] as ArrayLike<number>)[i] as number);
      }
      if (this.tick()) ready++;
    }
    return ready;
  }

  /**
   * The most recent `size` samples of every channel, oldest to newest.
   *
   * The returned arrays are owned by the window and overwritten by the next
   * call (and before each `onWindow` callback). Before `size` frames have
   * been pushed, each array is shorter and holds only the samples so far.
   */
  current(): Float64Array[] {
    const out = this.output;
    if (this.frameCount < this.size) {
      // Rare warm-up path; allocating views here is fine.
      return this.rings.map((ring, c) =>
        ring.toArray((out[c] as Float64Array).subarray(0, ring.length)),
      );
    }
    for (let c = 0; c < this.channels; c++) {
      (this.rings[c] as RingBuffer).toArray(out[c]);
    }
    return out;
  }

  /**
   * Calls `listener` with every new window, synchronously, from inside
   * `pushFrame` / `pushChunk`. The arrays passed are reused between calls.
   *
   * @returns A function that removes the listener.
   */
  onWindow(listener: WindowListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** Discards buffered samples and restarts window counting. Listeners are kept. */
  reset(): void {
    for (const ring of this.rings) ring.clear();
    this.frameCount = 0;
    this.untilNext = this.size;
  }

  /** Advances the frame counter; emits when a window boundary is reached. */
  private tick(): boolean {
    this.frameCount++;
    if (--this.untilNext > 0) return false;
    this.untilNext = this.step;
    if (this.listeners.size > 0) {
      const window = this.current();
      for (const listener of this.listeners) listener(window);
    }
    return true;
  }
}
