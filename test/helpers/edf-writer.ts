/**
 * Minimal EDF/EDF+ writer for round-trip tests. Hand-built from the spec
 * (https://www.edfplus.info/specs/edf.html) rather than taken from a library,
 * so the parser is checked against an independent implementation of the layout.
 */

export interface WriterSignal {
  label: string;
  transducer?: string;
  physicalDimension?: string;
  physicalMin: number;
  physicalMax: number;
  digitalMin: number;
  digitalMax: number;
  prefilter?: string;
  samplesPerRecord: number;
  /** Digital values, `samplesPerRecord * recordCount` of them. */
  digital?: ArrayLike<number>;
  /** For "EDF Annotations": one TAL string per record (UTF-8, padded with 0). */
  tals?: string[];
}

export interface WriterOptions {
  patient?: string;
  recording?: string;
  startDate?: string; // dd.mm.yy
  startTime?: string; // hh.mm.ss
  reserved?: string;
  recordCount: number;
  /** Written as-is into the header (to test "-1" or lies); defaults to recordCount. */
  recordCountField?: string;
  recordDuration: number;
  signals: WriterSignal[];
}

const encoder = new TextEncoder();

function field(value: string | number, width: number): string {
  const s = String(value);
  if (s.length > width) throw new Error(`"${s}" does not fit in ${width} chars`);
  return s.padEnd(width, " ");
}

export function writeEDF(options: WriterOptions): Uint8Array {
  const { signals, recordCount, recordDuration } = options;
  const ns = signals.length;
  const headerBytes = 256 * (ns + 1);
  let header = "";
  header += field("0", 8);
  header += field(options.patient ?? "X X X X", 80);
  header += field(options.recording ?? "Startdate X X X X", 80);
  header += field(options.startDate ?? "05.11.26", 8);
  header += field(options.startTime ?? "13.45.30", 8);
  header += field(headerBytes, 8);
  header += field(options.reserved ?? "", 44);
  header += field(options.recordCountField ?? recordCount, 8);
  header += field(recordDuration, 8);
  header += field(ns, 4);
  const per = <T>(f: (s: WriterSignal) => T, width: number) => {
    for (const s of signals) header += field(f(s) as string | number, width);
  };
  per((s) => s.label, 16);
  per((s) => s.transducer ?? "", 80);
  per((s) => s.physicalDimension ?? "uV", 8);
  per((s) => s.physicalMin, 8);
  per((s) => s.physicalMax, 8);
  per((s) => s.digitalMin, 8);
  per((s) => s.digitalMax, 8);
  per((s) => s.prefilter ?? "", 80);
  per((s) => s.samplesPerRecord, 8);
  per(() => "", 32);
  if (header.length !== headerBytes) throw new Error("header length mismatch");

  const recordBytes = signals.reduce((sum, s) => sum + 2 * s.samplesPerRecord, 0);
  const out = new Uint8Array(headerBytes + recordCount * recordBytes);
  for (let i = 0; i < header.length; i++) out[i] = header.charCodeAt(i);
  const view = new DataView(out.buffer);
  let offset = headerBytes;
  for (let r = 0; r < recordCount; r++) {
    for (const s of signals) {
      if (s.tals !== undefined) {
        const bytes = encoder.encode(s.tals[r] ?? "");
        if (bytes.length > 2 * s.samplesPerRecord) throw new Error("TALs overflow the record");
        out.set(bytes, offset);
      } else {
        for (let i = 0; i < s.samplesPerRecord; i++) {
          const v = s.digital?.[r * s.samplesPerRecord + i] ?? 0;
          view.setInt16(offset + 2 * i, v, true);
        }
      }
      offset += 2 * s.samplesPerRecord;
    }
  }
  return out;
}
