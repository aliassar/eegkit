import type { Signal } from "../types";

/**
 * Fixed-capacity circular buffer of samples, backed by one `Float64Array`.
 *
 * Writing never allocates: once the buffer is full, each `push` overwrites the
 * oldest sample. This is the building block for sliding windows over a live
 * stream, where a growing array would put pressure on the garbage collector
 * at every sample.
 *
 * Reads (`toArray`, `latest`) return samples in chronological order, oldest
 * first, and accept an optional output array so callers can reuse memory.
 *
 * @example
 * ```ts
 * const ring = new RingBuffer(4);
 * ring.pushMany([1, 2, 3, 4, 5]);
 * ring.toArray(); // Float64Array [2, 3, 4, 5]
 * ring.latest(2); // Float64Array [4, 5]
 * ```
 */
export class RingBuffer {
  private readonly data: Float64Array;
  /** Index where the next sample will be written. */
  private head = 0;
  private size = 0;

  /**
   * @param capacity - Maximum number of samples held (a positive integer).
   */
  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError(`RingBuffer capacity must be a positive integer, got ${capacity}`);
    }
    this.data = new Float64Array(capacity);
  }

  /** Maximum number of samples the buffer holds. */
  get capacity(): number {
    return this.data.length;
  }

  /** Number of samples currently held (at most `capacity`). */
  get length(): number {
    return this.size;
  }

  /** True once `capacity` samples have been written since the last `clear`. */
  get isFull(): boolean {
    return this.size === this.data.length;
  }

  /** Appends one sample, overwriting the oldest one when full. Never allocates. */
  push(x: number): void {
    const data = this.data;
    data[this.head] = x;
    this.head = this.head + 1 === data.length ? 0 : this.head + 1;
    if (this.size < data.length) this.size++;
  }

  /**
   * Appends samples in order. If `xs` is longer than the capacity, only its
   * last `capacity` samples are kept, exactly as if pushed one by one.
   */
  pushMany(xs: Signal): void {
    const n = xs.length;
    const cap = this.data.length;
    // Skip samples that would be overwritten anyway.
    const start = n > cap ? n - cap : 0;
    for (let i = start; i < n; i++) this.push(xs[i] as number);
  }

  /**
   * Sample at a chronological position: `0` is the oldest, `length - 1` the
   * newest, and negative indices count back from the newest (`-1` is newest).
   *
   * @throws RangeError if the index is outside the held samples.
   */
  at(index: number): number {
    const i = index < 0 ? this.size + index : index;
    if (!Number.isInteger(i) || i < 0 || i >= this.size) {
      throw new RangeError(`RingBuffer index ${index} out of range for length ${this.size}`);
    }
    return this.data[this.physical(i)] as number;
  }

  /**
   * Copies all held samples, oldest to newest.
   *
   * @param out - Optional destination with at least `length` elements. Only
   *   the first `length` elements are written; it is returned as is.
   * @returns `out` if given, otherwise a new `Float64Array` of `length`.
   */
  toArray(out?: Float64Array): Float64Array {
    return this.copyNewest(this.size, out);
  }

  /**
   * Copies the newest `n` samples, oldest to newest.
   *
   * @param n - Number of samples, `0 <= n <= length`.
   * @param out - Optional destination with at least `n` elements; only the
   *   first `n` are written and `out` is returned as is.
   * @throws RangeError if `n` exceeds the number of held samples.
   */
  latest(n: number, out?: Float64Array): Float64Array {
    if (!Number.isInteger(n) || n < 0 || n > this.size) {
      throw new RangeError(`RingBuffer.latest(${n}): must be an integer in [0, ${this.size}]`);
    }
    return this.copyNewest(n, out);
  }

  /** Forgets all samples. The capacity and backing memory are kept. */
  clear(): void {
    this.head = 0;
    this.size = 0;
  }

  /** Physical index of chronological position `i` (0 = oldest). */
  private physical(i: number): number {
    const cap = this.data.length;
    let p = this.head - this.size + i;
    if (p < 0) p += cap;
    return p;
  }

  private copyNewest(n: number, out?: Float64Array): Float64Array {
    if (out !== undefined && out.length < n) {
      throw new RangeError(`Output array has length ${out.length}, needs at least ${n}`);
    }
    const dest = out ?? new Float64Array(n);
    if (n === 0) return dest;
    const data = this.data;
    const cap = data.length;
    let start = this.head - n;
    if (start < 0) start += cap;
    // At most two contiguous runs: [start, end-of-storage) then [0, head).
    // Plain loops rather than `set(subarray(...))`, which would allocate two
    // view objects per call.
    const firstRun = Math.min(n, cap - start);
    for (let i = 0; i < firstRun; i++) dest[i] = data[start + i] as number;
    for (let i = firstRun; i < n; i++) dest[i] = data[i - firstRun] as number;
    return dest;
  }
}
