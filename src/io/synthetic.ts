import { STANDARD_BANDS, type StandardBandName } from "../types";

/** Options for {@link createSyntheticEEG}. */
export interface SyntheticEEGOptions {
  /** Sample rate in Hz. */
  readonly sampleRate: number;
  /** Number of channels to generate. */
  readonly channels: number;
  /** Seed for the pseudo-random generator (default 1). Same seed, same data. */
  readonly seed?: number;
  /**
   * Peak amplitude in µV of the oscillation placed at the centre of each
   * standard band, merged with the defaults `{ delta: 6, theta: 5, alpha: 10,
   * beta: 4, gamma: 1.5 }` (a relaxed, eyes-closed adult). Set a band to 0
   * to remove it, e.g. `{ alpha: 30 }` for a strong alpha rhythm.
   */
  readonly bands?: Partial<Record<StandardBandName, number>>;
  /** RMS amplitude in µV of the 1/f ("pink") background activity. Default 5. */
  readonly noise?: number;
  /** Mains interference added to every channel (peak amplitude in µV). Default none. */
  readonly lineNoise?: { readonly frequency: 50 | 60; readonly amplitude: number };
  /** Average number of eye blinks per minute (Poisson). Default 0; people blink ~15–20/min. */
  readonly blinkRate?: number;
  /** Peak blink amplitude in µV, as on a frontopolar channel. Default 150. */
  readonly blinkAmplitude?: number;
  /** Constant offset added to every channel, in µV (as from a DC-coupled amplifier). Default 0. */
  readonly dcOffset?: number;
}

/** A deterministic synthetic EEG source returned by {@link createSyntheticEEG}. */
export interface SyntheticEEG {
  /** Sample rate in Hz. */
  readonly sampleRate: number;
  /** Number of channels. */
  readonly channels: number;
  /**
   * Generates the next `samples` samples of every channel, as a new
   * channel-major chunk (`chunk[channel][sample]`, µV). The output does not
   * depend on how the stream is split into calls.
   */
  next(samples: number): Float64Array[];
  /** Rewinds to time zero: the same sequence will be generated again. */
  reset(): void;
}

const DEFAULT_BANDS: Readonly<Record<StandardBandName, number>> = {
  delta: 6,
  theta: 5,
  alpha: 10,
  beta: 4,
  gamma: 1.5,
};

/** Blink duration in seconds; real blinks last about 0.1–0.4 s. */
const BLINK_SECONDS = 0.3;
/** Depth of the slow amplitude modulation of each rhythm (fraction of its amplitude). */
const AM_DEPTH = 0.25;

/**
 * mulberry32: a tiny, fast 32-bit PRNG with good statistical quality for
 * simulation (not cryptography). Chosen because it is a few lines, has no
 * dependencies and gives identical sequences on every JS engine.
 * @internal
 */
class Mulberry32 {
  private state: number;
  constructor(seed: number) {
    this.state = seed | 0;
  }
  /** Uniform in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

/** Standard normal deviates by Box–Muller, caching the second value. @internal */
class Gaussian {
  private spare = Number.NaN;
  constructor(private readonly rng: Mulberry32) {}
  next(): number {
    if (!Number.isNaN(this.spare)) {
      const s = this.spare;
      this.spare = Number.NaN;
      return s;
    }
    // 1 - u keeps the log argument in (0, 1].
    const r = Math.sqrt(-2 * Math.log(1 - this.rng.next()));
    const theta = 2 * Math.PI * this.rng.next();
    this.spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  }
}

// Paul Kellet's "refined" pink-noise filter: six first-order sections plus
// two direct taps approximating a 1/f spectrum within ±0.05 dB over about
// nine octaves. At EEG sample rates its 1/f range spans roughly 0.05 Hz to
// fs/6, which covers the EEG bands; the shape scales with the sample rate.
const PINK_A = [0.99886, 0.99332, 0.969, 0.8665, 0.55, -0.7616] as const;
const PINK_G = [0.0555179, 0.0750759, 0.153852, 0.3104856, 0.5329522, -0.016898] as const;
const PINK_D0 = 0.5362;
const PINK_D1 = 0.115926;

/**
 * Stationary standard deviation of the pink filter's output for unit-variance
 * white input, from its impulse response h[0] = Σg + d0, h[1] = Σg·a + d1,
 * h[k] = Σg·a^k: Σh² = Σᵢⱼ gᵢgⱼ/(1 − aᵢaⱼ) + d0² + d1² + 2d0·Σg + 2d1·Σg·a.
 * Used to scale the background to an exact RMS.
 */
const PINK_STD = (() => {
  let v = PINK_D0 * PINK_D0 + PINK_D1 * PINK_D1;
  for (let i = 0; i < 6; i++) {
    const gi = PINK_G[i] as number;
    const ai = PINK_A[i] as number;
    v += 2 * PINK_D0 * gi + 2 * PINK_D1 * gi * ai;
    for (let j = 0; j < 6; j++)
      v += (gi * (PINK_G[j] as number)) / (1 - ai * (PINK_A[j] as number));
  }
  return Math.sqrt(v);
})();

/**
 * Creates a deterministic synthetic EEG generator, so the library can be
 * tried, tested and demoed without hardware.
 *
 * Each channel is the sum of:
 * - 1/f background activity (Kellet pink-noise filter on Gaussian white noise),
 *   scaled to `noise` µV RMS;
 * - one sinusoid per standard band at the band centre (e.g. 10.5 Hz alpha),
 *   with a per-channel frequency offset of up to ±15 % of the half-band and a
 *   slow (0.05–0.2 Hz) ±25 % amplitude drift, as real rhythms wax and wane;
 * - optional mains interference, identical on all channels (common mode);
 * - optional blinks: 0.3 s raised-cosine bumps of `blinkAmplitude` µV at
 *   Poisson-distributed times, added to every channel;
 * - an optional DC offset.
 *
 * Everything is driven by seeded PRNGs (independent streams for setup,
 * background and blinks, so enabling blinks doesn't change the background),
 * and the output is identical however the samples are requested in chunks.
 *
 * @example
 * ```ts
 * const eeg = createSyntheticEEG({ sampleRate: 256, channels: 4, bands: { alpha: 25 } });
 * const chunk = eeg.next(512); // 2 s, chunk[channel][sample] in µV
 * ```
 */
export function createSyntheticEEG(options: SyntheticEEGOptions): SyntheticEEG {
  const { sampleRate, channels } = options;
  if (!(sampleRate > 0) || !Number.isFinite(sampleRate)) {
    throw new RangeError(`sampleRate must be a positive number in Hz, got ${sampleRate}`);
  }
  if (!Number.isInteger(channels) || channels < 1) {
    throw new RangeError(`channels must be a positive integer, got ${channels}`);
  }
  const seed = options.seed ?? 1;
  const noise = options.noise ?? 5;
  const blinkRate = options.blinkRate ?? 0;
  const blinkAmplitude = options.blinkAmplitude ?? 150;
  const dcOffset = options.dcOffset ?? 0;
  const line = options.lineNoise;
  if (!(noise >= 0)) throw new RangeError(`noise must be a non-negative RMS in µV, got ${noise}`);
  if (!(blinkRate >= 0))
    throw new RangeError(`blinkRate must be >= 0 per minute, got ${blinkRate}`);
  if (line !== undefined && line.frequency >= sampleRate / 2) {
    throw new RangeError(
      `lineNoise frequency ${line.frequency} Hz is at or above Nyquist (${sampleRate / 2} Hz)`,
    );
  }
  const amplitudes = { ...DEFAULT_BANDS, ...options.bands };
  const bands = STANDARD_BANDS.filter((b) => b.name in amplitudes);
  for (const b of bands) {
    const a = amplitudes[b.name as StandardBandName];
    if (!(a >= 0)) throw new RangeError(`band amplitude for ${b.name} must be >= 0 µV, got ${a}`);
  }
  const nb = bands.length;

  // Per (channel, band) oscillator parameters, flattened [c * nb + b].
  const freq = new Float64Array(channels * nb);
  const phase = new Float64Array(channels * nb);
  const amp = new Float64Array(channels * nb);
  const amFreq = new Float64Array(channels * nb);
  const amPhase = new Float64Array(channels * nb);
  const pink = new Float64Array(channels * 7);

  let t = 0;
  let noiseRng = new Gaussian(new Mulberry32(seed ^ 0x9e3779b9));
  let blinkRng = new Mulberry32(seed ^ 0x85ebca6b);
  let nextBlink = 0;
  let blinkStart = Number.NEGATIVE_INFINITY;
  const blinkLength = Math.max(1, Math.round(BLINK_SECONDS * sampleRate));
  const meanInterval = blinkRate > 0 ? (60 / blinkRate) * sampleRate : 0;
  // A short dead time stops blinks from overlapping; the exponential part
  // is shortened by the same amount so the average rate stays `blinkRate`.
  const deadTime = Math.min(meanInterval / 2, 0.5 * sampleRate);

  const drawBlinkInterval = (): number =>
    deadTime + -Math.log(1 - blinkRng.next()) * (meanInterval - deadTime);

  const init = (): void => {
    const setup = new Mulberry32(seed);
    for (let c = 0; c < channels; c++) {
      for (let b = 0; b < nb; b++) {
        const band = bands[b] as (typeof bands)[number];
        const k = c * nb + b;
        const centre = (band.low + band.high) / 2;
        const halfWidth = (band.high - band.low) / 2;
        freq[k] = centre + (setup.next() * 2 - 1) * 0.15 * halfWidth;
        phase[k] = setup.next() * 2 * Math.PI;
        amp[k] = amplitudes[band.name as StandardBandName];
        amFreq[k] = 0.05 + setup.next() * 0.15;
        amPhase[k] = setup.next() * 2 * Math.PI;
      }
    }
    t = 0;
    noiseRng = new Gaussian(new Mulberry32(seed ^ 0x9e3779b9));
    blinkRng = new Mulberry32(seed ^ 0x85ebca6b);
    blinkStart = Number.NEGATIVE_INFINITY;
    nextBlink = blinkRate > 0 ? drawBlinkInterval() : Number.POSITIVE_INFINITY;
    // Start each pink-filter section from its own stationary distribution so
    // the slow sections don't need minutes to warm up.
    for (let c = 0; c < channels; c++) {
      for (let s = 0; s < 6; s++) {
        const a = PINK_A[s] as number;
        const g = PINK_G[s] as number;
        pink[c * 7 + s] = noiseRng.next() * Math.abs(g) * Math.sqrt(1 / (1 - a * a));
      }
      pink[c * 7 + 6] = 0;
    }
  };
  init();

  const twoPiOverFs = (2 * Math.PI) / sampleRate;
  const noiseScale = noise / PINK_STD;

  const next = (samples: number): Float64Array[] => {
    if (!Number.isInteger(samples) || samples < 0) {
      throw new RangeError(`samples must be a non-negative integer, got ${samples}`);
    }
    const out: Float64Array[] = [];
    for (let c = 0; c < channels; c++) out.push(new Float64Array(samples));
    for (let i = 0; i < samples; i++, t++) {
      // Shared (all-channel) components for this instant.
      let common = dcOffset;
      if (line !== undefined) common += line.amplitude * Math.sin(twoPiOverFs * line.frequency * t);
      if (t >= nextBlink) {
        blinkStart = t;
        nextBlink = t + Math.max(1, Math.round(drawBlinkInterval()));
      }
      const sinceBlink = t - blinkStart;
      if (sinceBlink < blinkLength) {
        common += blinkAmplitude * 0.5 * (1 - Math.cos((2 * Math.PI * sinceBlink) / blinkLength));
      }
      for (let c = 0; c < channels; c++) {
        // Pink background (Kellet), sections updated in place.
        const w = noiseRng.next();
        const p = c * 7;
        let sum = 0;
        for (let s = 0; s < 6; s++) {
          const v = (PINK_A[s] as number) * (pink[p + s] as number) + (PINK_G[s] as number) * w;
          pink[p + s] = v;
          sum += v;
        }
        sum += (pink[p + 6] as number) + PINK_D0 * w;
        pink[p + 6] = PINK_D1 * w;
        let x = common + noiseScale * sum;
        for (let b = 0; b < nb; b++) {
          const k = c * nb + b;
          const env =
            1 +
            AM_DEPTH * Math.sin(twoPiOverFs * (amFreq[k] as number) * t + (amPhase[k] as number));
          x +=
            (amp[k] as number) *
            env *
            Math.sin(twoPiOverFs * (freq[k] as number) * t + (phase[k] as number));
        }
        (out[c] as Float64Array)[i] = x;
      }
    }
    return out;
  };

  return { sampleRate, channels, next, reset: init };
}

/** Options for {@link startSyntheticStream}. */
export interface SyntheticStreamOptions extends SyntheticEEGOptions {
  /** How often frames are delivered, in milliseconds (default 40, i.e. 25 times per second). */
  readonly chunkMs?: number;
}

/**
 * Streams synthetic EEG in real time, one frame (one sample per channel, µV)
 * at a time, the way a live device would.
 *
 * A timer fires every `chunkMs` and delivers however many frames are due
 * according to the wall clock (`Date.now()`), so the long-run rate matches
 * `sampleRate` even though timers drift. If the timer was paused for more
 * than a second (e.g. a background browser tab), the backlog is dropped
 * instead of delivered in one burst.
 *
 * Each frame is a fresh `Float64Array`, safe to keep.
 *
 * @returns A function that stops the stream.
 *
 * @example
 * ```ts
 * const stop = startSyntheticStream((frame) => win.pushFrame(frame), {
 *   sampleRate: 256,
 *   channels: 4,
 *   blinkRate: 15,
 * });
 * // later: stop();
 * ```
 */
export function startSyntheticStream(
  onFrame: (frame: Float64Array) => void,
  options: SyntheticStreamOptions,
): () => void {
  const chunkMs = options.chunkMs ?? 40;
  if (!(chunkMs > 0) || !Number.isFinite(chunkMs)) {
    throw new RangeError(`chunkMs must be a positive number of milliseconds, got ${chunkMs}`);
  }
  const eeg = createSyntheticEEG(options);
  const { sampleRate, channels } = eeg;
  const maxBurst = Math.max(1, Math.ceil(sampleRate));
  const start = Date.now();
  let delivered = 0;
  let stopped = false;

  const timer: ReturnType<typeof setInterval> = setInterval(() => {
    const due = Math.floor(((Date.now() - start) * sampleRate) / 1000);
    let count = due - delivered;
    if (count <= 0) return;
    if (count > maxBurst) count = maxBurst;
    // Skip any dropped backlog so the next tick is on schedule again.
    delivered = due;
    const chunk = eeg.next(count);
    for (let i = 0; i < count && !stopped; i++) {
      const frame = new Float64Array(channels);
      for (let c = 0; c < channels; c++) frame[c] = (chunk[c] as Float64Array)[i] as number;
      onFrame(frame);
    }
  }, chunkMs);

  return () => {
    stopped = true;
    clearInterval(timer);
  };
}
