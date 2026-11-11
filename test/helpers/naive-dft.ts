/**
 * Reference DFT for tests, computed directly from the definition in O(N) per
 * bin. Deliberately independent of src/spectral so the I/O and quality tests
 * don't depend on another module's correctness.
 */
export function dftBin(signal: ArrayLike<number>, frequency: number, sampleRate: number) {
  let re = 0;
  let im = 0;
  const w = (2 * Math.PI * frequency) / sampleRate;
  for (let n = 0; n < signal.length; n++) {
    const x = signal[n] as number;
    re += x * Math.cos(w * n);
    im -= x * Math.sin(w * n);
  }
  return { re, im, power: re * re + im * im };
}

/**
 * Mean-square amplitude of the sinusoid at `frequency`, `2|X|²/N²` (A²/2 for
 * a sine of amplitude A on a bin centre), without windowing.
 */
export function sinePower(signal: ArrayLike<number>, frequency: number, sampleRate: number) {
  const n = signal.length;
  return (2 * dftBin(signal, frequency, sampleRate).power) / (n * n);
}

/** Total power of the mean-removed signal in [low, high) Hz, from DFT bins (Parseval). */
export function bandPowerDft(
  signal: ArrayLike<number>,
  low: number,
  high: number,
  sampleRate: number,
) {
  const n = signal.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += signal[i] as number;
  mean /= n;
  const centred = Float64Array.from(signal, (x) => x - mean);
  let total = 0;
  for (let k = 1; k < n / 2; k++) {
    const f = (k * sampleRate) / n;
    if (f >= low && f < high) total += (2 * dftBin(centred, f, sampleRate).power) / (n * n);
  }
  return total;
}
