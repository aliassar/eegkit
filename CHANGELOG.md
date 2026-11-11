# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-11

The first release.

### Added

- **Filters.** Butterworth low-pass, high-pass, band-pass and band-stop designs; notch filters;
  `IIRFilter` and the multi-channel `FilterBank` for streaming; `sosfilt`; zero-phase `filtfilt`;
  `frequencyResponse`. Outputs match SciPy.
- **Spectral analysis.** `fft` and `rfft` for any length (radix-2 or Bluestein), Welch and
  periodogram estimates, window functions, absolute and relative band power, and peak frequency.
- **Streaming.** `RingBuffer` and `SlidingWindow`.
- **Metrics.** Theta/beta and alpha/theta ratios, the engagement index, frontal alpha asymmetry,
  exponential smoothing, and a running baseline with z-scores.
- **Signal quality.** `assessQuality`, `detectBlinks`, and Goertzel single-bin power.
- **I/O.** EDF/EDF+ parsing, a generic WebSocket client, and a synthetic EEG generator and stream.
- **`createPipeline`.** The whole real-time chain behind one function.

[0.1.0]: https://github.com/aliassar/eegkit/releases/tag/v0.1.0
