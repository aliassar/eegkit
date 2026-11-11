import { describe, expect, it, vi } from "vitest";
import { RingBuffer } from "../src/stream/ring-buffer";
import { SlidingWindow } from "../src/stream/window";

const range = (start: number, end: number) =>
  Array.from({ length: end - start }, (_, i) => start + i);

describe("RingBuffer", () => {
  it("rejects invalid capacities", () => {
    expect(() => new RingBuffer(0)).toThrow(RangeError);
    expect(() => new RingBuffer(2.5)).toThrow(RangeError);
    expect(() => new RingBuffer(-1)).toThrow(RangeError);
  });

  it("tracks length and isFull while filling", () => {
    const ring = new RingBuffer(3);
    expect(ring.capacity).toBe(3);
    expect(ring.length).toBe(0);
    expect(ring.isFull).toBe(false);
    ring.push(1);
    ring.push(2);
    expect(ring.length).toBe(2);
    expect(Array.from(ring.toArray())).toEqual([1, 2]);
    ring.push(3);
    expect(ring.isFull).toBe(true);
    ring.push(4);
    expect(ring.length).toBe(3);
  });

  it("returns samples oldest to newest across many wraparounds", () => {
    const cap = 7;
    const ring = new RingBuffer(cap);
    for (let i = 0; i < 100; i++) {
      ring.push(i);
      const expected = range(Math.max(0, i + 1 - cap), i + 1);
      expect(Array.from(ring.toArray())).toEqual(expected);
      expect(ring.at(0)).toBe(expected[0]);
      expect(ring.at(-1)).toBe(i);
    }
  });

  it("latest(n) returns the newest n in order, into a reusable buffer", () => {
    const ring = new RingBuffer(5);
    ring.pushMany(range(0, 13)); // holds 8..12, head mid-array
    expect(Array.from(ring.latest(3))).toEqual([10, 11, 12]);
    expect(Array.from(ring.latest(5))).toEqual([8, 9, 10, 11, 12]);
    expect(ring.latest(0).length).toBe(0);
    const out = new Float64Array(6).fill(-1);
    expect(ring.latest(2, out)).toBe(out);
    expect(Array.from(out)).toEqual([11, 12, -1, -1, -1, -1]);
    expect(() => ring.latest(6)).toThrow(RangeError);
    expect(() => ring.latest(3, new Float64Array(2))).toThrow(RangeError);
  });

  it("pushMany equals repeated push, including inputs longer than capacity", () => {
    for (const n of [0, 3, 4, 5, 11]) {
      const a = new RingBuffer(4);
      const b = new RingBuffer(4);
      a.push(-5);
      b.push(-5);
      a.pushMany(range(0, n));
      for (const x of range(0, n)) b.push(x);
      expect(Array.from(a.toArray())).toEqual(Array.from(b.toArray()));
      expect(a.length).toBe(b.length);
    }
  });

  it("at() rejects out-of-range indices", () => {
    const ring = new RingBuffer(4);
    ring.pushMany([1, 2]);
    expect(ring.at(1)).toBe(2);
    expect(ring.at(-2)).toBe(1);
    expect(() => ring.at(2)).toThrow(RangeError);
    expect(() => ring.at(-3)).toThrow(RangeError);
  });

  it("clear() empties but keeps capacity", () => {
    const ring = new RingBuffer(3);
    ring.pushMany([1, 2, 3, 4]);
    ring.clear();
    expect(ring.length).toBe(0);
    expect(ring.toArray().length).toBe(0);
    ring.push(9);
    expect(Array.from(ring.toArray())).toEqual([9]);
  });
});

/** Frame i of a test stream: channel c holds 1000 * c + i, so contents are checkable. */
const frameAt = (i: number, channels: number) =>
  Float64Array.from({ length: channels }, (_, c) => 1000 * c + i);

describe("SlidingWindow", () => {
  it("validates options and frame shapes", () => {
    expect(() => new SlidingWindow({ channels: 0, size: 4, step: 1 })).toThrow(RangeError);
    expect(() => new SlidingWindow({ channels: 1, size: 0, step: 1 })).toThrow(RangeError);
    expect(() => new SlidingWindow({ channels: 1, size: 4, step: 0 })).toThrow(RangeError);
    const win = new SlidingWindow({ channels: 2, size: 4, step: 1 });
    expect(() => win.pushFrame([1])).toThrow(RangeError);
    expect(() => win.pushChunk([[1, 2]])).toThrow(RangeError);
    expect(() => win.pushChunk([[1, 2], [3]])).toThrow(RangeError);
  });

  it("emits exactly floor((n - size) / step) + 1 windows for n frames", () => {
    for (const [size, step] of [
      [4, 1],
      [4, 2],
      [5, 3],
      [8, 8],
      [3, 5], // step > size: gaps between windows
      [1, 1],
    ] as const) {
      for (let n = 0; n <= 30; n++) {
        const win = new SlidingWindow({ channels: 2, size, step });
        let count = 0;
        for (let i = 0; i < n; i++) if (win.pushFrame(frameAt(i, 2))) count++;
        const expected = n < size ? 0 : Math.floor((n - size) / step) + 1;
        expect(count, `size ${size} step ${step} n ${n}`).toBe(expected);
        expect(win.frames).toBe(n);
      }
    }
  });

  it("window contents are the last `size` frames, per channel, oldest first", () => {
    const size = 6;
    const step = 4;
    const win = new SlidingWindow({ channels: 3, size, step });
    const seen: number[][][] = [];
    win.onWindow((w) => seen.push(w.map((ch) => Array.from(ch))));
    for (let i = 0; i < 30; i++) win.pushFrame(frameAt(i, 3));
    expect(seen.length).toBe(Math.floor((30 - size) / step) + 1);
    seen.forEach((w, k) => {
      const end = size + k * step; // exclusive index of the newest frame
      for (let c = 0; c < 3; c++) {
        expect(w[c]).toEqual(range(end - size, end).map((i) => 1000 * c + i));
      }
    });
  });

  it("pushChunk across arbitrary chunk boundaries matches pushFrame", () => {
    const channels = 2;
    const total = 57;
    const opts = { channels, size: 10, step: 3 };
    const reference = new SlidingWindow(opts);
    const refWindows: number[][][] = [];
    reference.onWindow((w) => refWindows.push(w.map((ch) => Array.from(ch))));
    for (let i = 0; i < total; i++) reference.pushFrame(frameAt(i, channels));

    const chunked = new SlidingWindow(opts);
    const windows: number[][][] = [];
    chunked.onWindow((w) => windows.push(w.map((ch) => Array.from(ch))));
    const sizes = [1, 4, 0, 9, 2, 13, 7, 21]; // sums to 57
    let start = 0;
    let readyCount = 0;
    for (const len of sizes) {
      const chunk = Array.from({ length: channels }, (_, c) =>
        range(start, start + len).map((i) => 1000 * c + i),
      );
      readyCount += chunked.pushChunk(chunk);
      start += len;
    }
    expect(start).toBe(total);
    expect(readyCount).toBe(refWindows.length);
    expect(windows).toEqual(refWindows);
    expect(chunked.current().map((ch) => Array.from(ch))).toEqual(
      reference.current().map((ch) => Array.from(ch)),
    );
  });

  it("current() is partial before the first window and reuses arrays after", () => {
    const win = new SlidingWindow({ channels: 1, size: 4, step: 2 });
    win.pushFrame([1]);
    win.pushFrame([2]);
    expect(win.isFull).toBe(false);
    expect(Array.from(win.current()[0] ?? [])).toEqual([1, 2]);
    win.pushChunk([[3, 4, 5]]);
    const a = win.current();
    expect(Array.from(a[0] ?? [])).toEqual([2, 3, 4, 5]);
    win.pushFrame([6]);
    const b = win.current();
    expect(b[0]).toBe(a[0]); // same storage, documented reuse
    expect(Array.from(b[0] ?? [])).toEqual([3, 4, 5, 6]);
  });

  it("unsubscribe stops notifications; reset restarts counting", () => {
    const win = new SlidingWindow({ channels: 1, size: 2, step: 1 });
    const listener = vi.fn();
    const off = win.onWindow(listener);
    win.pushChunk([[1, 2, 3]]);
    expect(listener).toHaveBeenCalledTimes(2);
    off();
    win.pushFrame([4]);
    expect(listener).toHaveBeenCalledTimes(2);
    win.reset();
    expect(win.frames).toBe(0);
    expect(win.pushFrame([5])).toBe(false);
    expect(win.pushFrame([6])).toBe(true);
    expect(Array.from(win.current()[0] ?? [])).toEqual([5, 6]);
  });
});
