import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  connectWebSocket,
  parseSampleMessage,
  type WebSocketConstructor,
} from "../src/io/websocket";

/** Test double recording every instance; the test drives its events. */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  binaryType = "blob";
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  closeCalls: Array<[number | undefined, string | undefined]> = [];
  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }
  close(code?: number, reason?: string): void {
    this.closeCalls.push([code, reason]);
    // Like a real socket, closing fires onclose asynchronously-ish; sync is fine here.
    this.onclose?.({ code: code ?? 1005, reason: reason ?? "" });
  }
  // Test helpers
  open() {
    this.onopen?.({});
  }
  message(data: unknown) {
    this.onmessage?.({ data });
  }
  drop(code = 1006) {
    this.onclose?.({ code, reason: "" });
  }
  static get last(): FakeWebSocket {
    const ws = FakeWebSocket.instances.at(-1);
    if (ws === undefined) throw new Error("no socket");
    return ws;
  }
}

const arr = (x: Float64Array[] | Float64Array | null) =>
  x === null ? null : x instanceof Float64Array ? Array.from(x) : x.map((f) => Array.from(f));

describe("parseSampleMessage", () => {
  it("parses a JSON array of numbers as one frame", () => {
    const r = parseSampleMessage("[1.5, -2, 3e2]");
    expect(r).toBeInstanceOf(Float64Array);
    expect(arr(r)).toEqual([1.5, -2, 300]);
  });

  it("parses a JSON array of arrays as frames", () => {
    expect(arr(parseSampleMessage("[[1,2],[3,4],[5,6]]"))).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
  });

  it("parses objects with a data or samples field of either shape", () => {
    expect(arr(parseSampleMessage('{"data":[1,2,3],"t":17}'))).toEqual([1, 2, 3]);
    expect(arr(parseSampleMessage('{"samples":[[1,2],[3,4]]}'))).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(arr(parseSampleMessage({ data: [7, 8] }))).toEqual([7, 8]); // pre-decoded
  });

  it("parses CSV-like text lines, skipping a header", () => {
    expect(arr(parseSampleMessage("1,2,3"))).toEqual([1, 2, 3]);
    expect(arr(parseSampleMessage("Fp1,Fp2\n1.5,-2\r\n3;4\n\n5\t6\n"))).toEqual([
      [1.5, -2],
      [3, 4],
      [5, 6],
    ]);
    expect(arr(parseSampleMessage("  -1e-3 2.  .5 "))).toEqual([-0.001, 2, 0.5]);
  });

  it("decodes binary payloads as text", () => {
    const bytes = new TextEncoder().encode("[10,20]");
    expect(arr(parseSampleMessage(bytes))).toEqual([10, 20]);
    expect(arr(parseSampleMessage(bytes.buffer))).toEqual([10, 20]);
    const offset = new Uint8Array(12);
    offset.set(new TextEncoder().encode("3,4"), 5);
    expect(arr(parseSampleMessage(offset.subarray(5, 8)))).toEqual([3, 4]);
  });

  it("returns null (never throws) for anything else", () => {
    for (const bad of [
      "",
      "hello",
      "[1, 2",
      "[]",
      "[[]]",
      '[1, "2"]',
      "[1, null]",
      "[[1,2],[3]]",
      "[[1,2],3]",
      '{"value": [1]}',
      '{"data": {"data": [1]}}',
      '{"data": "1,2"}',
      "1,2\n3",
      "1,,2",
      "NaN,1",
      "0x10",
      null,
      undefined,
      42,
      true,
      { data: [Number.NaN] },
    ]) {
      expect(parseSampleMessage(bad), String(bad)).toBeNull();
    }
  });
});

describe("connectWebSocket", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("connects, sets binaryType and dispatches frames and chunks", () => {
    const frames: number[][] = [];
    const chunks: number[][][] = [];
    const onOpen = vi.fn();
    connectWebSocket(
      "ws://localhost:1234",
      {
        onFrame: (f) => frames.push(Array.from(f)),
        onChunk: (c) => chunks.push(c.map((ch) => Array.from(ch))),
        onOpen,
      },
      { WebSocket: FakeWebSocket },
    );
    const ws = FakeWebSocket.last;
    expect(ws.url).toBe("ws://localhost:1234");
    expect(ws.binaryType).toBe("arraybuffer");
    ws.open();
    expect(onOpen).toHaveBeenCalledOnce();
    ws.message("[1,2,3]");
    ws.message("[[4,5,6],[7,8,9]]");
    expect(frames).toEqual([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    // Chunks are channel-major.
    expect(chunks).toEqual([
      [[1], [2], [3]],
      [
        [4, 7],
        [5, 8],
        [6, 9],
      ],
    ]);
  });

  it("ignores invalid messages without throwing", () => {
    const onFrame = vi.fn();
    const onError = vi.fn();
    connectWebSocket("ws://x", { onFrame, onError }, { WebSocket: FakeWebSocket });
    const ws = FakeWebSocket.last;
    ws.open();
    expect(() => {
      ws.message("not json");
      ws.message("{");
      ws.message({ nope: true });
      ws.message(null);
    }).not.toThrow();
    expect(onFrame).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
    ws.message("[1]");
    expect(onFrame).toHaveBeenCalledOnce();
  });

  it("uses a custom parser and reports its exceptions via onError", () => {
    const onFrame = vi.fn();
    const onError = vi.fn();
    const parse = (data: unknown) => {
      if (data === "boom") throw new Error("bad packet");
      return data instanceof ArrayBuffer ? new Float64Array(new Float32Array(data)) : null;
    };
    connectWebSocket("ws://x", { onFrame, onError }, { WebSocket: FakeWebSocket, parse });
    const ws = FakeWebSocket.last;
    ws.message(new Float32Array([0.5, 1.5]).buffer);
    expect(Array.from(onFrame.mock.calls[0]?.[0] as Float64Array)).toEqual([0.5, 1.5]);
    ws.message("boom");
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: "bad packet" }));
  });

  it("forwards socket errors and close events", () => {
    const onError = vi.fn();
    const onClose = vi.fn();
    connectWebSocket("ws://x", { onError, onClose }, { WebSocket: FakeWebSocket });
    const ws = FakeWebSocket.last;
    ws.onerror?.({ type: "error" });
    expect(onError).toHaveBeenCalledWith({ type: "error" });
    ws.drop(1011);
    expect(onClose).toHaveBeenCalledWith({ code: 1011, reason: "" });
  });

  it("does not reconnect by default", () => {
    connectWebSocket("ws://x", {}, { WebSocket: FakeWebSocket });
    FakeWebSocket.last.drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances.length).toBe(1);
  });

  it("reconnects with exponential backoff, capped, and resets after a successful open", () => {
    const onOpen = vi.fn();
    connectWebSocket(
      "ws://x",
      { onOpen },
      { WebSocket: FakeWebSocket, reconnect: { delayMs: 100, maxDelayMs: 500 } },
    );
    const count = () => FakeWebSocket.instances.length;
    FakeWebSocket.last.drop(); // attempt 1 after 100 ms
    vi.advanceTimersByTime(99);
    expect(count()).toBe(1);
    vi.advanceTimersByTime(1);
    expect(count()).toBe(2);
    FakeWebSocket.last.drop(); // attempt 2 after 200 ms
    vi.advanceTimersByTime(199);
    expect(count()).toBe(2);
    vi.advanceTimersByTime(1);
    expect(count()).toBe(3);
    FakeWebSocket.last.drop(); // 400 ms
    vi.advanceTimersByTime(400);
    expect(count()).toBe(4);
    FakeWebSocket.last.drop(); // capped at 500 ms (not 800)
    vi.advanceTimersByTime(499);
    expect(count()).toBe(4);
    vi.advanceTimersByTime(1);
    expect(count()).toBe(5);
    // A successful connection resets the delay.
    FakeWebSocket.last.open();
    expect(onOpen).toHaveBeenCalledOnce();
    FakeWebSocket.last.drop();
    vi.advanceTimersByTime(100);
    expect(count()).toBe(6);
  });

  it("reconnect: true uses a 500 ms initial delay", () => {
    connectWebSocket("ws://x", {}, { WebSocket: FakeWebSocket, reconnect: true });
    FakeWebSocket.last.drop();
    vi.advanceTimersByTime(499);
    expect(FakeWebSocket.instances.length).toBe(1);
    vi.advanceTimersByTime(1);
    expect(FakeWebSocket.instances.length).toBe(2);
  });

  it("close() closes the socket and cancels pending and future reconnects", () => {
    const onClose = vi.fn();
    const conn = connectWebSocket(
      "ws://x",
      { onClose },
      { WebSocket: FakeWebSocket, reconnect: { delayMs: 100 } },
    );
    const first = FakeWebSocket.last;
    first.drop(); // reconnect scheduled
    conn.close(); // ... and cancelled
    vi.advanceTimersByTime(10_000);
    expect(FakeWebSocket.instances.length).toBe(1);

    const conn2 = connectWebSocket(
      "ws://y",
      { onClose },
      { WebSocket: FakeWebSocket, reconnect: { delayMs: 100 } },
    );
    const second = FakeWebSocket.last;
    second.open();
    conn2.close();
    expect(second.closeCalls).toEqual([[1000, "closed by client"]]);
    vi.advanceTimersByTime(10_000);
    expect(FakeWebSocket.instances.length).toBe(2);
    conn2.close(); // idempotent
    expect(second.closeCalls.length).toBe(1);
  });

  it("ignores messages that arrive after close()", () => {
    const onFrame = vi.fn();
    const conn = connectWebSocket("ws://x", { onFrame }, { WebSocket: FakeWebSocket });
    const ws = FakeWebSocket.last;
    ws.close = () => {}; // a socket whose close event never arrives
    conn.close();
    ws.message("[1,2]");
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("reports constructor failures during reconnection and keeps trying", () => {
    let fail = false;
    class Flaky extends FakeWebSocket {
      constructor(url: string) {
        if (fail) throw new Error("ECONNREFUSED");
        super(url);
      }
    }
    const onError = vi.fn();
    connectWebSocket("ws://x", { onError }, { WebSocket: Flaky, reconnect: { delayMs: 10 } });
    fail = true;
    FakeWebSocket.last.drop();
    vi.advanceTimersByTime(10); // throws -> reported, retry in 20
    expect(onError).toHaveBeenCalledTimes(1);
    fail = false;
    vi.advanceTimersByTime(20);
    expect(FakeWebSocket.instances.length).toBe(2);
  });

  it("accepts the platform WebSocket class at the type level", () => {
    // Compile-time check that the DOM/Node global fits WebSocketConstructor.
    const ctor: WebSocketConstructor | undefined = globalThis.WebSocket;
    expect(ctor === undefined || typeof ctor === "function").toBe(true);
  });

  it("throws a TypeError when no WebSocket implementation exists", () => {
    const g = globalThis as { WebSocket?: unknown };
    const saved = g.WebSocket;
    try {
      g.WebSocket = undefined;
      expect(() => connectWebSocket("ws://x", {})).toThrow(TypeError);
    } finally {
      g.WebSocket = saved;
    }
  });

  it("validates reconnect delays", () => {
    expect(() =>
      connectWebSocket("ws://x", {}, { WebSocket: FakeWebSocket, reconnect: { delayMs: 0 } }),
    ).toThrow(RangeError);
    expect(() =>
      connectWebSocket(
        "ws://x",
        {},
        { WebSocket: FakeWebSocket, reconnect: { delayMs: 100, maxDelayMs: 50 } },
      ),
    ).toThrow(RangeError);
  });
});
