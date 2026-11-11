/**
 * Band power per channel for a whole EDF/EDF+ recording.
 *
 *   npx tsx examples/edf-band-power.ts recording.edf
 *
 * Free sample recordings: https://physionet.org/content/eegmmidb/
 */
import { readFileSync } from "node:fs";
import { bandPowers, filtfilt, notch, parseEDF, relativeBandPowers, welch } from "../src/index";

const path = process.argv[2];
if (!path) {
  console.error("usage: npx tsx examples/edf-band-power.ts <file.edf>");
  process.exit(1);
}

const recording = parseEDF(readFileSync(path));
console.log(
  `${recording.header.startDate.toISOString().slice(0, 10)} · ${recording.signals.length} signals`,
);

const rows = recording.signals
  .filter((s) => s.physicalDimension.toLowerCase().includes("v") && s.sampleRate >= 100)
  .map((signal) => {
    // Zero-phase notch, so offline results line up in time with the raw trace.
    const clean = filtfilt(notch(50, signal.sampleRate), signal.samples);
    const spectrum = welch(clean, {
      sampleRate: signal.sampleRate,
      segmentLength: 2 * signal.sampleRate,
    });
    const absolute = bandPowers(spectrum);
    const relative = relativeBandPowers(spectrum);
    return {
      channel: signal.label,
      seconds: Math.round(signal.samples.length / signal.sampleRate),
      ...Object.fromEntries(
        Object.keys(absolute).map((band) => [
          band,
          `${absolute[band]?.toFixed(1)} µV² (${((relative[band] ?? 0) * 100).toFixed(0)}%)`,
        ]),
      ),
    };
  });

console.table(rows);
for (const a of recording.annotations.slice(0, 10))
  console.log(`${a.onset.toFixed(1)}s  ${a.text}`);
