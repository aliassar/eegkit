# eegkit

[![ci](https://github.com/aliassar/eegkit/actions/workflows/ci.yml/badge.svg)](https://github.com/aliassar/eegkit/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/aliassar/eegkit)](LICENSE)

Real-time EEG signal processing for JavaScript and TypeScript. Filters, spectra, band power,
neurofeedback metrics and signal quality, checked against SciPy, with zero dependencies.

It runs anywhere JavaScript does: browsers, Node, React Native (Hermes) and Web Workers. Use it
to build neurofeedback apps, BCI prototypes, research dashboards, or to analyse EDF recordings
without leaving JS.

Install it straight from GitHub, pinned to a release tag:

```sh
npm install github:aliassar/eegkit#v0.1.0
```

npm builds the package on install. It isn't published to the npm registry.

## Quick start

```ts
import { createPipeline, connectWebSocket } from "eegkit";

const pipeline = createPipeline({
  sampleRate: 256,
  channels: ["TP9", "AF7", "AF8", "TP10"],
  notch: 50, // or 60 in the Americas
});

pipeline.subscribe(({ time, channels }) => {
  for (const ch of channels) {
    console.log(time, ch.label, ch.relativeBandPower.alpha, ch.peakAlpha, ch.quality?.level);
  }
});

// Any source that gives you one sample per channel works.
connectWebSocket("ws://localhost:8765", { onFrame: pipeline.push }, { reconnect: true });
```

`createPipeline` wires up the usual chain for you: a causal Butterworth band-pass and notch
(primed so a DC offset doesn't ring), a sliding 2 s window every 0.25 s, a Welch spectrum, band powers, the individual alpha
peak, and a quality score per channel. Every stage is also exported on its own, for when you
need a different chain.

No hardware? `startSyntheticStream(pipeline.push, { sampleRate: 256, channels: 4 })` produces
realistic EEG with tunable bands, mains hum, blinks and drift.

## What's inside

| Area          | Exports                                                                                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Filters**   | `butterworthLowpass` / `Highpass` / `Bandpass` / `Bandstop`, `notch`, `cascade`, `IIRFilter`, `FilterBank` (multi-channel, streaming), `sosfilt`, `filtfilt` (zero-phase), `frequencyResponse` |
| **Spectral**  | `fft`, `rfft` (any length), `welch`, `periodogram`, `hann` / `hamming` / `blackman`, `bandPower`, `bandPowers`, `relativeBandPowers`, `peakFrequency`                                          |
| **Streaming** | `RingBuffer`, `SlidingWindow` (frame or chunk input, window callbacks)                                                                                                                         |
| **Metrics**   | `thetaBetaRatio`, `alphaThetaRatio`, `engagementIndex`, `bandRatio`, `frontalAlphaAsymmetry`, `ExponentialSmoother`, `Baseline` (running z-scores)                                             |
| **Quality**   | `assessQuality` (flatline, clipping, amplitude, mains noise → good / fair / poor), `detectBlinks`, `goertzel`                                                                                  |
| **I/O**       | `parseEDF` (EDF and EDF+ with annotations), `connectWebSocket` (JSON/CSV/text samples, reconnects), `createSyntheticEEG`, `startSyntheticStream`                                               |
| **Pipeline**  | `createPipeline`                                                                                                                                                                               |

The conventions are the same everywhere:

- Inputs are anything array-like, such as `number[]` or any typed array.
- Outputs are `Float64Array`, so you can pass them to charts or workers without copying.
- Units are µV, Hz and µV².
- Bad arguments throw `RangeError`. Undefined results come back as `NaN`.

## Recipes

**Clean up a recording offline (zero phase, SciPy's `sosfiltfilt`):**

```ts
import { butterworthBandpass, cascade, filtfilt, notch } from "eegkit";

const sos = cascade(butterworthBandpass(4, 1, 40, 256), notch(50, 256));
const clean = filtfilt(sos, rawSamples);
```

**Filter a live multi-channel stream sample by sample:**

```ts
import { butterworthHighpass, FilterBank } from "eegkit";

const bank = new FilterBank(butterworthHighpass(2, 0.5, 250), 8);
socket.onFrame = (frame) => draw(bank.processFrame(frame));
```

**Turn band powers into a neurofeedback score:**

```ts
import { Baseline, ExponentialSmoother, thetaBetaRatio } from "eegkit";

const smooth = new ExponentialSmoother({ timeConstant: 2, rate: 4 }); // 4 results/s
const baseline = new Baseline(); // collect a resting baseline first

pipeline.subscribe(({ channels }) => {
  const tbr = smooth.update(thetaBetaRatio(channels[0].bandPower));
  baseline.add(tbr);
  if (baseline.ready) reward(-baseline.zScore(tbr)); // lower TBR → more reward
});
```

**Frontal alpha asymmetry:**

```ts
import { frontalAlphaAsymmetry } from "eegkit";

const [af7, af8] = result.channels;
const faa = frontalAlphaAsymmetry(af7.bandPower, af8.bandPower); // ln(R) − ln(L)
```

**Read an EDF file in the browser:**

```ts
import { parseEDF, welch, bandPowers } from "eegkit";

const { signals, annotations } = parseEDF(await file.arrayBuffer());
const powers = bandPowers(welch(signals[0].samples, { sampleRate: signals[0].sampleRate }));
```

To run the two scripts in [`examples/`](examples), clone the repo:

- `npx tsx examples/live-alpha.ts`: a neurofeedback bar in the terminal.
- `npx tsx examples/edf-band-power.ts file.edf`: a band-power table per channel.

## Data sources

eegkit doesn't depend on any particular device. It takes numbers in µV, wherever they come from:

- **WebSocket**: `connectWebSocket` accepts JSON arrays, `{ data: [...] }` objects, or CSV/TSV
  lines. Run a few lines of Python or Node next to your amplifier's SDK or
  [Lab Streaming Layer](https://labstreaminglayer.org) and forward the samples.
- **Web Bluetooth / Web Serial / native modules**: decode the packets yourself, then call
  `pipeline.push(frame)`.
- **Files**: `parseEDF` for EDF/EDF+. For anything else, decode it to arrays and use
  `pipeline.pushChunk` or the functions directly.

For React and React Native hooks, see [eegkit-react](https://github.com/aliassar/eegkit-react).
For a complete app built on it, see
[eegkit-neurofeedback](https://github.com/aliassar/eegkit-neurofeedback).

## Accuracy

The filter designs, `sosfilt`, `filtfilt`, `welch`, `periodogram` and the FFT are tested against
reference outputs from SciPy 1.18 and NumPy 2.5, generated by
[`scripts/make_fixtures.py`](scripts/make_fixtures.py). The largest relative error across the
fixtures is about 6e-12.

Everything else is checked against invariants instead: Parseval's theorem, a naive-DFT reference, the
known density of white noise, and round trips through EDF files the tests write.

## Performance

On an Apple M5 with Node 22:

| Workload                                                          | Time  | Real-time factor |
| ----------------------------------------------------------------- | ----- | ---------------- |
| `createPipeline`, 8 channels at 256 Hz, 10 s of data (40 windows) | 16 ms | ~600×            |
| `createPipeline`, 32 channels at 256 Hz, 10 s of data             | 70 ms | ~140×            |
| `welch` on a 2 s window, 1 channel                                | 29 µs |                  |
| `assessQuality` on a 2 s window, 1 channel                        | 15 µs |                  |

Run `npm run bench` to reproduce these. The streaming filters and windows don't allocate per sample,
and FFT plans are cached.

## Not medical software

eegkit is for research, education, wellness and hobby projects. It isn't a medical device and
must not be used for diagnosis or treatment.

## Development

```sh
npm install
npm test            # vitest
npm run typecheck
npm run lint        # biome
npm run bench
npm run fixtures    # regenerate SciPy fixtures (needs uv)
```

## Versions

Releases are git tags (`v0.1.0`, …) with notes in [CHANGELOG.md](CHANGELOG.md). Pin a tag in your
`package.json` so `main` can move on without changing what you install.

## License

[MIT](LICENSE) © Ali Assar
