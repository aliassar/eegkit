/**
 * The subset of the WebSocket API used by {@link connectWebSocket}.
 *
 * Handler parameters are typed `never` on purpose: that makes the DOM
 * `WebSocket`, React Native's, the `ws` package's and simple test doubles
 * all assignable, whatever event types they declare.
 */
export interface WebSocketLike {
  binaryType: string;
  onopen: ((event: never) => unknown) | null;
  onmessage: ((event: never) => unknown) | null;
  onerror: ((event: never) => unknown) | null;
  onclose: ((event: never) => unknown) | null;
  close(code?: number, reason?: string): void;
}

/** A WebSocket constructor, e.g. the global `WebSocket` or `ws`'s default export. */
export type WebSocketConstructor = new (url: string) => WebSocketLike;

/**
 * Turns one message payload into samples: a `Float64Array` is one frame (one
 * value per channel), an array of them is several frames in time order, and
 * `null` means "not a sample message, ignore it".
 */
export type MessageParser = (data: unknown) => Float64Array[] | Float64Array | null;

/** Callbacks for {@link connectWebSocket}. */
export interface WebSocketHandlers {
  /** Called for every frame (one sample per channel, µV), in order. */
  readonly onFrame?: (frame: Float64Array) => void;
  /**
   * Called once per message with all its frames as a channel-major chunk
   * (`chunk[channel][sample]`), ready for `SlidingWindow.pushChunk`.
   */
  readonly onChunk?: (chunk: Float64Array[]) => void;
  /** Called each time a connection opens (including after a reconnect). */
  readonly onOpen?: () => void;
  /** Called each time a connection closes, with the close code and reason if known. */
  readonly onClose?: (info: { readonly code: number; readonly reason: string }) => void;
  /** Called with socket errors, and with exceptions thrown by a custom `parse`. */
  readonly onError?: (error: unknown) => void;
}

/** Reconnection settings for {@link connectWebSocket}. */
export interface ReconnectOptions {
  /** Delay before the first reconnection attempt, in ms. Default 500. */
  readonly delayMs?: number;
  /** Upper bound for the doubling delay, in ms. Default 30 000. */
  readonly maxDelayMs?: number;
}

/** Options for {@link connectWebSocket}. */
export interface WebSocketOptions {
  /** Custom message parser; defaults to {@link parseSampleMessage}. */
  readonly parse?: MessageParser;
  /** WebSocket implementation; defaults to `globalThis.WebSocket`. */
  readonly WebSocket?: WebSocketConstructor;
  /**
   * Reconnect after the connection drops, with exponential backoff (the delay
   * doubles after each failed attempt and resets once a connection opens).
   * `true` uses the defaults of {@link ReconnectOptions}. Default `false`.
   */
  readonly reconnect?: boolean | ReconnectOptions;
}

/** Handle returned by {@link connectWebSocket}. */
export interface WebSocketConnection {
  /** Closes the connection and cancels any pending reconnection. Idempotent. */
  close(): void;
}

const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
/** Comma/semicolon-separated (empty fields are invalid), else whitespace-separated. */
const DELIMITED = /\s*[,;]\s*/;
const WHITESPACE = /\s+/;

/** Decodes bytes as Latin-1; sample messages are ASCII, so this is lossless for them. @internal */
function bytesToText(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i] as number);
  return s;
}

/** One frame from a JSON array of finite numbers, else null. @internal */
function toFrame(value: unknown): Float64Array | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const frame = new Float64Array(value.length);
  for (let i = 0; i < value.length; i++) {
    const v: unknown = value[i];
    if (typeof v !== "number" || !Number.isFinite(v)) return null;
    frame[i] = v;
  }
  return frame;
}

/** A single frame or a list of equal-length frames from parsed JSON, else null. @internal */
function fromJson(value: unknown): Float64Array[] | Float64Array | null {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const inner = record.data ?? record.samples;
    if (
      inner === undefined ||
      (inner !== null && typeof inner === "object" && !Array.isArray(inner))
    ) {
      return null; // no nested objects: `{ data: { data: ... } }` is not a shape we accept
    }
    return fromJson(inner);
  }
  if (!Array.isArray(value) || value.length === 0) return null;
  if (!Array.isArray(value[0])) return toFrame(value);
  const frames: Float64Array[] = [];
  for (const row of value) {
    const frame = toFrame(row);
    if (frame === null || frame.length !== (frames[0]?.length ?? frame.length)) return null;
    frames.push(frame);
  }
  return frames;
}

/** Frames from CSV-like text: one frame per line of numbers. @internal */
function fromText(text: string): Float64Array[] | Float64Array | null {
  const frames: Float64Array[] = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line === "") continue;
    const tokens = line.split(/[,;]/.test(line) ? DELIMITED : WHITESPACE);
    if (!tokens.every((t) => NUMBER.test(t))) continue; // e.g. a header row
    const frame = Float64Array.from(tokens, Number);
    if (frames.length > 0 && frame.length !== (frames[0] as Float64Array).length) return null;
    frames.push(frame);
  }
  if (frames.length === 0) return null;
  return frames.length === 1 ? (frames[0] as Float64Array) : frames;
}

/**
 * The default message parser of {@link connectWebSocket}. Never throws:
 * anything it doesn't recognise yields `null` and is ignored.
 *
 * Accepted shapes (values in µV):
 * - JSON array of numbers: one frame, `[1.5, -2, 3]`;
 * - JSON array of arrays: several frames in time order, `[[1, 2], [3, 4]]`;
 * - JSON object whose `data` or `samples` field has either shape above,
 *   `{"data": [1, 2, 3], "t": 1700000000}` (other fields are ignored);
 * - text with one frame per line, values separated by commas, semicolons,
 *   tabs or spaces, `"1,2,3\n4,5,6"`; non-numeric lines such as a CSV header
 *   are skipped.
 *
 * Binary payloads (`ArrayBuffer`, typed arrays, Node `Buffer`) are decoded
 * as text and parsed the same way. Frames within one message must all have
 * the same length. For binary sample encodings, pass your own `parse`.
 *
 * @returns One frame, several frames, or `null`.
 */
export function parseSampleMessage(data: unknown): Float64Array[] | Float64Array | null {
  let text: string;
  if (typeof data === "string") {
    text = data;
  } else if (data instanceof ArrayBuffer) {
    text = bytesToText(new Uint8Array(data));
  } else if (ArrayBuffer.isView(data)) {
    text = bytesToText(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  } else if (data !== null && typeof data === "object") {
    return fromJson(data); // already-decoded JSON, e.g. from a custom transport
  } else {
    return null;
  }
  const trimmed = text.trim();
  const first = trimmed.charAt(0);
  if (first === "[" || first === "{") {
    try {
      return fromJson(JSON.parse(trimmed));
    } catch {
      return null;
    }
  }
  return fromText(trimmed);
}

/** Channel-major chunk from frames (all the same length). @internal */
function framesToChunk(frames: readonly Float64Array[]): Float64Array[] {
  const channels = (frames[0] as Float64Array).length;
  const chunk: Float64Array[] = [];
  for (let c = 0; c < channels; c++) {
    const row = new Float64Array(frames.length);
    for (let i = 0; i < frames.length; i++) row[i] = (frames[i] as Float64Array)[c] as number;
    chunk.push(row);
  }
  return chunk;
}

/**
 * Receives EEG samples from any WebSocket server that sends numbers as JSON
 * or text (see {@link parseSampleMessage} for the accepted shapes), e.g. a
 * small bridge script in front of an amplifier, or a recording replayer.
 * This is deliberately generic: it implements no device-specific protocol.
 *
 * Uses the global `WebSocket` (browsers, React Native, Node ≥ 22) unless an
 * implementation is passed as `options.WebSocket` (e.g. the `ws` package on
 * older Node). Messages that don't parse are ignored, never thrown.
 *
 * @param url - WebSocket URL (`ws://` or `wss://`).
 * @throws TypeError if no WebSocket implementation is available; any error
 *   the WebSocket constructor throws for the first connection (e.g. a
 *   malformed URL) is rethrown.
 *
 * @example
 * ```ts
 * const win = new SlidingWindow({ channels: 8, size: 512, step: 64 });
 * const conn = connectWebSocket("ws://localhost:8080", {
 *   onChunk: (chunk) => win.pushChunk(chunk),
 *   onClose: ({ code }) => console.log("closed", code),
 * }, { reconnect: true });
 * // later: conn.close();
 * ```
 */
export function connectWebSocket(
  url: string,
  handlers: WebSocketHandlers,
  options: WebSocketOptions = {},
): WebSocketConnection {
  const Impl = options.WebSocket ?? (globalThis as { WebSocket?: WebSocketConstructor }).WebSocket;
  if (Impl === undefined) {
    throw new TypeError(
      "No WebSocket implementation found: pass one as options.WebSocket (e.g. the 'ws' package)",
    );
  }
  const parse = options.parse ?? parseSampleMessage;
  const reconnect = options.reconnect ?? false;
  const baseDelay = typeof reconnect === "object" ? (reconnect.delayMs ?? 500) : 500;
  const maxDelay = typeof reconnect === "object" ? (reconnect.maxDelayMs ?? 30_000) : 30_000;
  if (!(baseDelay > 0) || !(maxDelay >= baseDelay)) {
    throw new RangeError(
      `reconnect needs delayMs > 0 and maxDelayMs >= delayMs, got ${baseDelay} and ${maxDelay}`,
    );
  }

  let socket: WebSocketLike | null = null;
  let closed = false;
  let delay = baseDelay;
  let timer: ReturnType<typeof setTimeout> | undefined;

  const dispatch = (data: unknown): void => {
    let parsed: Float64Array[] | Float64Array | null;
    try {
      parsed = parse(data);
    } catch (error) {
      handlers.onError?.(error);
      return;
    }
    if (parsed === null) return;
    const frames = parsed instanceof Float64Array ? [parsed] : parsed;
    if (frames.length === 0) return;
    if (handlers.onFrame !== undefined) for (const frame of frames) handlers.onFrame(frame);
    if (handlers.onChunk !== undefined) handlers.onChunk(framesToChunk(frames));
  };

  const scheduleReconnect = (): void => {
    if (closed || reconnect === false || timer !== undefined) return;
    const wait = delay;
    delay = Math.min(delay * 2, maxDelay);
    timer = setTimeout(() => {
      timer = undefined;
      if (closed) return;
      try {
        open();
      } catch (error) {
        handlers.onError?.(error);
        scheduleReconnect();
      }
    }, wait);
  };

  const open = (): void => {
    const ws = new Impl(url);
    socket = ws;
    // Binary messages as ArrayBuffer rather than Blob (which is async to read).
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      if (socket !== ws) return;
      delay = baseDelay;
      handlers.onOpen?.();
    };
    ws.onmessage = (event: { readonly data: unknown }) => {
      if (socket === ws && !closed) dispatch(event.data);
    };
    ws.onerror = (event: unknown) => {
      if (socket === ws) handlers.onError?.(event);
    };
    ws.onclose = (event: { readonly code?: number; readonly reason?: string } | undefined) => {
      if (socket !== ws) return;
      socket = null;
      handlers.onClose?.({ code: event?.code ?? 1006, reason: event?.reason ?? "" });
      scheduleReconnect();
    };
  };

  open();

  return {
    close(): void {
      if (closed) return;
      closed = true;
      if (timer !== undefined) {
        clearTimeout(timer);
        timer = undefined;
      }
      socket?.close(1000, "closed by client");
    },
  };
}
