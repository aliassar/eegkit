import { describe, expect, it } from "vitest";
import { EDFParseError, parseEDF } from "../src/io/edf";
import { type WriterOptions, type WriterSignal, writeEDF } from "./helpers/edf-writer";

/** Physical value of a digital sample, per the EDF calibration formula. */
const physical = (d: number, s: WriterSignal) =>
  ((d - s.digitalMin) * (s.physicalMax - s.physicalMin)) / (s.digitalMax - s.digitalMin) +
  s.physicalMin;

function makeRecording(): { options: WriterOptions; eeg: WriterSignal; ecg: WriterSignal } {
  const recordCount = 3;
  // EEG: 256 Hz with 0.5 s records = 128 samples/record, asymmetric scaling.
  const eeg: WriterSignal = {
    label: "EEG Fp1-A1",
    transducer: "AgAgCl electrode",
    physicalDimension: "uV",
    physicalMin: -3276.8,
    physicalMax: 3276.7,
    digitalMin: -32768,
    digitalMax: 32767,
    prefilter: "HP:0.1Hz LP:75Hz",
    samplesPerRecord: 128,
    digital: Array.from({ length: 128 * recordCount }, (_, i) =>
      Math.round(20000 * Math.sin(i / 7) - (i % 13) * 300),
    ),
  };
  // A second signal at a different rate (100 Hz = 50/record), offset and
  // inverted, non-standard digital range including negative physical values.
  const ecg: WriterSignal = {
    label: "ECG",
    physicalDimension: "mV",
    physicalMin: 5,
    physicalMax: -5,
    digitalMin: -2048,
    digitalMax: 2047,
    samplesPerRecord: 50,
    digital: Array.from({ length: 50 * recordCount }, (_, i) => ((i * 97) % 4096) - 2048),
  };
  return {
    options: {
      patient: "MCH-0234567 F 02-MAY-1951 Haagse_Harry",
      recording: "Startdate 05-NOV-2026 EMR7 Tech BrainLab",
      startDate: "05.11.26",
      startTime: "13.45.30",
      recordCount,
      recordDuration: 0.5,
      signals: [eeg, ecg],
    },
    eeg,
    ecg,
  };
}

describe("parseEDF", () => {
  it("round-trips header fields", () => {
    const { options } = makeRecording();
    const rec = parseEDF(writeEDF(options));
    expect(rec.header).toEqual({
      version: "0",
      patient: "MCH-0234567 F 02-MAY-1951 Haagse_Harry",
      recording: "Startdate 05-NOV-2026 EMR7 Tech BrainLab",
      startDate: new Date(Date.UTC(2026, 10, 5, 13, 45, 30)),
      recordCount: 3,
      recordDuration: 0.5,
      signalCount: 2,
      reserved: "",
    });
    expect(rec.annotations).toEqual([]);
  });

  it("round-trips multiple signals with different rates and scalings", () => {
    const { options, eeg, ecg } = makeRecording();
    const rec = parseEDF(writeEDF(options));
    expect(rec.signals.length).toBe(2);
    const [s0, s1] = rec.signals;
    expect(s0).toMatchObject({
      label: "EEG Fp1-A1",
      transducer: "AgAgCl electrode",
      physicalDimension: "uV",
      physicalMin: -3276.8,
      physicalMax: 3276.7,
      digitalMin: -32768,
      digitalMax: 32767,
      prefilter: "HP:0.1Hz LP:75Hz",
      sampleRate: 256,
    });
    expect(s1).toMatchObject({ label: "ECG", physicalDimension: "mV", sampleRate: 100 });
    for (const [parsed, written] of [
      [s0, eeg],
      [s1, ecg],
    ] as const) {
      const expected = Array.from(written.digital ?? [], (d) => physical(d, written));
      expect(parsed?.samples.length).toBe(expected.length);
      parsed?.samples.forEach((v, i) => {
        expect(v).toBeCloseTo(expected[i] as number, 9);
      });
    }
    // Spot-check the scaling by hand: 0.1 µV per bit, so digital 1234 -> 123.4 µV,
    // and the inverted ECG maps digitalMin to +5 mV.
    expect(physical(1234, eeg)).toBeCloseTo(123.4, 9);
    expect(physical(-2048, ecg)).toBe(5);
    expect(s0?.samples.some((v) => v < 0)).toBe(true);
  });

  it("accepts a Uint8Array view at a non-zero offset", () => {
    const file = writeEDF(makeRecording().options);
    const padded = new Uint8Array(file.length + 7);
    padded.set(file, 3);
    const rec = parseEDF(padded.subarray(3, 3 + file.length));
    expect(rec.signals[0]?.samples).toEqual(parseEDF(file).signals[0]?.samples);
    expect(parseEDF(file.slice().buffer).signals.length).toBe(2);
  });

  it("reads EDF+ annotations, including durations and several per TAL", () => {
    const { options } = makeRecording();
    const ann: WriterSignal = {
      label: "EDF Annotations",
      physicalMin: -1,
      physicalMax: 1,
      digitalMin: -32768,
      digitalMax: 32767,
      samplesPerRecord: 40, // 80 bytes per record
      tals: [
        // record 0: timekeeping TAL, then an annotation with duration
        "+0\x14\x14\x00+0.25\x1510.5\x14Eyes closed\x14\x00",
        // record 1: timekeeping, two annotations sharing one onset, a UTF-8 text
        "+0.5\x14\x14\x00+0.75\x14Blink\x14Artifact\x14\x00+0.8\x14Δ-wave µV\x14\x00",
        // record 2: timekeeping only
        "+1\x14\x14\x00",
      ],
    };
    const rec = parseEDF(
      writeEDF({ ...options, reserved: "EDF+C", signals: [...options.signals, ann] }),
    );
    expect(rec.header.reserved).toBe("EDF+C");
    expect(rec.header.signalCount).toBe(3);
    expect(rec.signals.map((s) => s.label)).toEqual(["EEG Fp1-A1", "ECG"]);
    expect(rec.annotations).toEqual([
      { onset: 0.25, duration: 10.5, text: "Eyes closed" },
      { onset: 0.75, text: "Blink" },
      { onset: 0.75, text: "Artifact" },
      { onset: 0.8, text: "Δ-wave µV" },
    ]);
    // The data signals are unaffected by the interleaved annotation bytes.
    expect(rec.signals[1]?.samples).toEqual(parseEDF(writeEDF(options)).signals[1]?.samples);
  });

  it("supports annotation-only EDF+ files with zero record duration and negative onsets", () => {
    const rec = parseEDF(
      writeEDF({
        reserved: "EDF+C",
        recordCount: 1,
        recordDuration: 0,
        signals: [
          {
            label: "EDF Annotations",
            physicalMin: -1,
            physicalMax: 1,
            digitalMin: -32768,
            digitalMax: 32767,
            samplesPerRecord: 30,
            tals: ["+0\x14\x14\x00-2.5\x151\x14Pre-start\x14\x00"],
          },
        ],
      }),
    );
    expect(rec.signals).toEqual([]);
    expect(rec.annotations).toEqual([{ onset: -2.5, duration: 1, text: "Pre-start" }]);
  });

  it("uses the EDF+ four-digit year and the 1985-2084 window otherwise", () => {
    const base = makeRecording().options;
    const plain = { ...base, recording: "plain EDF" };
    expect(
      parseEDF(writeEDF({ ...plain, startDate: "01.02.85" })).header.startDate.getUTCFullYear(),
    ).toBe(1985);
    expect(
      parseEDF(writeEDF({ ...plain, startDate: "01.02.84" })).header.startDate.getUTCFullYear(),
    ).toBe(2084);
    // After 2084 EDF+ writes a literal "yy"; the recording field carries the year.
    const late = { ...base, startDate: "01.02.yy", recording: "Startdate 01-FEB-2091 X X X" };
    expect(parseEDF(writeEDF(late)).header.startDate).toEqual(
      new Date(Date.UTC(2091, 1, 1, 13, 45, 30)),
    );
    expect(() => parseEDF(writeEDF({ ...late, recording: "X" }))).toThrow(/no Startdate year/);
    expect(
      parseEDF(
        writeEDF({ ...base, startDate: "01.02.91", recording: "Startdate 01-FEB-2091 X" }),
      ).header.startDate.getUTCFullYear(),
    ).toBe(2091);
  });

  it("computes the record count when the header says -1", () => {
    const { options } = makeRecording();
    const rec = parseEDF(writeEDF({ ...options, recordCountField: "-1" }));
    expect(rec.header.recordCount).toBe(3);
    expect(rec.signals[0]?.samples.length).toBe(384);
  });

  describe("malformed files throw clear EDFParseErrors", () => {
    const good = () => writeEDF(makeRecording().options);
    const patch = (bytes: Uint8Array, offset: number, text: string) => {
      for (let i = 0; i < text.length; i++) bytes[offset + i] = text.charCodeAt(i);
      return bytes;
    };
    const cases: Array<[string, () => Uint8Array, RegExp]> = [
      ["too short", () => new Uint8Array(100), /shorter than the 256-byte header/],
      ["wrong version", () => patch(good(), 0, "1       "), /version field must be "0"/],
      [
        "BDF",
        () => {
          const b = good();
          b[0] = 0xff;
          return patch(b, 1, "BIOSEMI");
        },
        /BDF/,
      ],
      ["bad date", () => patch(good(), 168, "5.11.26 "), /start date must be dd.mm.yy/],
      ["impossible date", () => patch(good(), 168, "35.13.26"), /invalid start date/],
      ["bad time", () => patch(good(), 176, "13:45:30"), /start time must be hh.mm.ss/],
      [
        "header size mismatch",
        () => patch(good(), 184, "512     "),
        /header byte count is 512, expected 768/,
      ],
      [
        "non-numeric record count",
        () => patch(good(), 236, "abc     "),
        /number of data records is not a number/,
      ],
      [
        "zero signals",
        () => patch(good(), 252, "0   "),
        /number of signals must be an integer >= 1/,
      ],
      [
        "truncated data",
        () => good().subarray(0, good().length - 10),
        /declares 3 data records but the file holds only 2/,
      ],
      [
        "more records than present",
        () => patch(good(), 236, "4       "),
        /declares 4 data records/,
      ],
      [
        "bad digital range",
        () => patch(good(), 256 + 2 * (16 + 80 + 8 + 8 + 8), "40000   "), // digitalMin of signal 1
        /signal 1 .*digital minimum must be an integer >= -32768|digital range/,
      ],
      [
        "equal physical min/max",
        () => patch(good(), 256 + 2 * (16 + 80 + 8 + 8), "-3276.8 "), // physicalMax of signal 1
        /physical minimum and maximum are both/,
      ],
      [
        "zero samples per record",
        () => patch(good(), 256 + 2 * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80), "0       "),
        /samples per record must be an integer >= 1/,
      ],
      ["negative duration", () => patch(good(), 244, "-1      "), /record duration must be >= 0/],
    ];
    for (const [name, make, message] of cases) {
      it(name, () => {
        const bytes = make();
        expect(() => parseEDF(bytes)).toThrow(EDFParseError);
        expect(() => parseEDF(bytes)).toThrow(message);
      });
    }

    it("malformed annotation onsets", () => {
      const { options } = makeRecording();
      const bad: WriterSignal = {
        label: "EDF Annotations",
        physicalMin: -1,
        physicalMax: 1,
        digitalMin: -32768,
        digitalMax: 32767,
        samplesPerRecord: 20,
        tals: ["0.5\x14x\x14\x00", "", ""],
      };
      expect(() => parseEDF(writeEDF({ ...options, signals: [bad] }))).toThrow(
        /annotation onset must start with \+ or -/,
      );
    });
  });
});
