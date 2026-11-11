import { FilterBank } from "./filter/bank";
import { butterworthBandpass, notch as notchDesign } from "./filter/design";
import { cascade, type SOS } from "./filter/sos";
import { assessQuality, type ChannelQuality, type QualityOptions } from "./quality/quality";
import { bandPowers, peakFrequency, relativeBandPowers } from "./spectral/bandpower";
import { type PowerSpectrum, welch } from "./spectral/welch";
import { SlidingWindow } from "./stream/window";
import { type Band, type BandPowers, type Chunk, type Frame, STANDARD_BANDS } from "./types";

/** Options for {@link createPipeline}. */
export interface PipelineOptions {
  /** Sampling rate of the incoming data, in Hz. */
  readonly sampleRate: number;
  /** Number of channels, or their labels (e.g. `["AF7", "AF8"]`). */
  readonly channels: number | readonly string[];
  /** Band-pass edges in Hz, or `false` for none. Default `[1, 45]`. */
  readonly bandpass?: readonly [number, number] | false;
  /** Mains frequency to notch out, or `false` for none. Default `false`. */
  readonly notch?: 50 | 60 | false;
  /** Butterworth order of the band-pass. Default 4. */
  readonly filterOrder?: number;
  /** Analysis window length in seconds. Default 2. */
  readonly windowSeconds?: number;
  /** Time between windows in seconds. Default 0.25 (4 results per second). */
  readonly stepSeconds?: number;
  /** Bands to measure. Default {@link STANDARD_BANDS}. */
  readonly bands?: readonly Band[];
  /**
   * Welch segment length in samples. Default: one second of data, which
   * gives 1 Hz frequency resolution and averages several segments per window.
   */
  readonly segmentLength?: number;
  /** Range searched for the individual alpha peak, in Hz. Default `[7, 14]`. */
  readonly alphaRange?: readonly [number, number];
  /**
   * Signal-quality assessment on the raw (unfiltered) window, or `false` to
   * skip it. Default `true` with the defaults of {@link assessQuality}.
   */
  readonly quality?: boolean | Omit<QualityOptions, "sampleRate">;
}

/** Analysis of one channel over one window. */
export interface ChannelResult {
  readonly label: string;
  /** Absolute band power, µV². */
  readonly bandPower: BandPowers;
  /** Band power divided by total power across the bands. */
  readonly relativeBandPower: BandPowers;
  /** Individual alpha peak frequency in Hz, or NaN if none was found. */
  readonly peakAlpha: number;
  /** Welch spectrum of the filtered window. */
  readonly spectrum: PowerSpectrum;
  /** Quality of the raw window, when quality assessment is enabled. */
  readonly quality?: ChannelQuality;
}

/** One analysis window, across all channels. */
export interface PipelineResult {
  /** Zero-based window number since the last reset. */
  readonly index: number;
  /** Seconds of data received when this window completed. */
  readonly time: number;
  readonly channels: readonly ChannelResult[];
}

/** A running analysis pipeline. See {@link createPipeline}. */
export interface Pipeline {
  /** Push one sample per channel. */
  push(frame: Frame): void;
  /** Push a channel-major block, `chunk[channel][sample]`. */
  pushChunk(chunk: Chunk): void;
  /** Receive every result. Returns a function that unsubscribes. */
  subscribe(listener: (result: PipelineResult) => void): () => void;
  /** The most recent result, if any window has completed. */
  readonly latest: PipelineResult | undefined;
  /** Channel labels, in order. */
  readonly labels: readonly string[];
  /** Clear filter state and buffered data. Subscriptions are kept. */
  reset(): void;
}

/**
 * The common real-time chain in one call: filter → window → spectrum → band
 * power, peak alpha and signal quality, delivered a few times a second.
 *
 * Quality is assessed on the raw signal on purpose. The notch filter would
 * otherwise remove the very mains noise that reveals a loose electrode.
 *
 * @example
 * ```ts
 * const pipeline = createPipeline({ sampleRate: 256, channels: ["AF7", "AF8"], notch: 50 });
 * pipeline.subscribe(({ channels }) => {
 *   const [left, right] = channels;
 *   console.log(left.relativeBandPower.alpha, right.quality?.level);
 * });
 * socket.onmessage = (event) => pipeline.push(JSON.parse(event.data));
 * ```
 */
export function createPipeline(options: PipelineOptions): Pipeline {
  const { sampleRate } = options;
  if (!(Number.isFinite(sampleRate) && sampleRate > 0)) {
    throw new RangeError(`sampleRate must be a positive number, got ${sampleRate}`);
  }
  const labels = resolveLabels(options.channels);
  const channelCount = labels.length;

  const windowSize = Math.round((options.windowSeconds ?? 2) * sampleRate);
  const step = Math.max(1, Math.round((options.stepSeconds ?? 0.25) * sampleRate));
  if (windowSize < 2) throw new RangeError("windowSeconds is too short for this sampleRate");
  const segmentLength = Math.min(options.segmentLength ?? Math.round(sampleRate), windowSize);
  const bands = options.bands ?? STANDARD_BANDS;
  const alphaRange = options.alphaRange ?? [7, 14];
  const qualityOptions =
    options.quality === false
      ? undefined
      : {
          ...(options.notch ? { lineFrequency: options.notch } : {}),
          ...(typeof options.quality === "object" ? options.quality : {}),
          sampleRate,
        };

  const sos = designFilters(options, sampleRate);
  const bank = sos ? new FilterBank(sos, channelCount) : undefined;
  const filtered = new SlidingWindow({ channels: channelCount, size: windowSize, step });
  const raw = qualityOptions
    ? new SlidingWindow({ channels: channelCount, size: windowSize, step })
    : undefined;
  const filteredFrame = new Float64Array(channelCount);
  const listeners = new Set<(result: PipelineResult) => void>();

  let primed = false;
  let frames = 0;
  let index = 0;
  let latest: PipelineResult | undefined;

  const analyse = (): void => {
    const filteredWindow = filtered.current();
    const rawWindow = raw?.current();
    const channels: ChannelResult[] = [];
    for (let c = 0; c < channelCount; c++) {
      const signal = filteredWindow[c] as Float64Array;
      const spectrum = welch(signal, { sampleRate, segmentLength });
      const result: ChannelResult = {
        label: labels[c] as string,
        bandPower: bandPowers(spectrum, bands),
        relativeBandPower: relativeBandPowers(spectrum, bands),
        peakAlpha: peakFrequency(spectrum, [alphaRange[0], alphaRange[1]]),
        spectrum,
        ...(rawWindow && qualityOptions
          ? { quality: assessQuality(rawWindow[c] as Float64Array, qualityOptions) }
          : {}),
      };
      channels.push(result);
    }
    latest = { index: index++, time: frames / sampleRate, channels };
    for (const listener of listeners) listener(latest);
  };

  const push = (frame: Frame): void => {
    if (frame.length !== channelCount) {
      throw new RangeError(`expected ${channelCount} channels per frame, got ${frame.length}`);
    }
    frames++;
    raw?.pushFrame(frame);
    if (bank) {
      // Start the filters in steady state for the first value, so a DC offset
      // doesn't ring through the first windows as a large artifact.
      if (!primed) {
        bank.prime(frame);
        primed = true;
      }
      bank.processFrame(frame, filteredFrame);
      if (filtered.pushFrame(filteredFrame)) analyse();
    } else if (filtered.pushFrame(frame)) {
      analyse();
    }
  };

  const frame = new Float64Array(channelCount);
  return {
    push,
    pushChunk(chunk: Chunk): void {
      if (chunk.length !== channelCount) {
        throw new RangeError(`expected ${channelCount} channels in chunk, got ${chunk.length}`);
      }
      const length = chunk[0]?.length ?? 0;
      for (let i = 0; i < length; i++) {
        for (let c = 0; c < channelCount; c++)
          frame[c] = (chunk[c] as ArrayLike<number>)[i] as number;
        push(frame);
      }
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    get latest() {
      return latest;
    },
    labels,
    reset(): void {
      bank?.reset();
      filtered.reset();
      raw?.reset();
      primed = false;
      frames = 0;
      index = 0;
      latest = undefined;
    },
  };
}

function resolveLabels(channels: number | readonly string[]): string[] {
  if (typeof channels === "number") {
    if (!(Number.isInteger(channels) && channels > 0)) {
      throw new RangeError(`channels must be a positive integer, got ${channels}`);
    }
    return Array.from({ length: channels }, (_, i) => `ch${i + 1}`);
  }
  if (channels.length === 0) throw new RangeError("channels must not be empty");
  return [...channels];
}

function designFilters(options: PipelineOptions, sampleRate: number): SOS | undefined {
  const parts: SOS[] = [];
  const bandpass = options.bandpass ?? [1, 45];
  if (bandpass) {
    const high = Math.min(bandpass[1], sampleRate / 2 - 1);
    parts.push(butterworthBandpass(options.filterOrder ?? 4, bandpass[0], high, sampleRate));
  }
  if (options.notch) parts.push(notchDesign(options.notch, sampleRate));
  return parts.length > 0 ? cascade(...parts) : undefined;
}
