/** Thrown by {@link parseEDF} when the bytes are not a valid EDF/EDF+ file. */
export class EDFParseError extends Error {
  constructor(message: string) {
    super(`Invalid EDF: ${message}`);
    this.name = "EDFParseError";
  }
}

/** File-level header fields of an EDF/EDF+ recording. */
export interface EDFHeader {
  /** Format version field; always `"0"` for EDF and EDF+. */
  readonly version: string;
  /** Local patient identification (EDF+ structures it as code, sex, birthdate, name). */
  readonly patient: string;
  /** Local recording identification (EDF+: `Startdate dd-MMM-yyyy ...`). */
  readonly recording: string;
  /**
   * Start of the recording. EDF stores a local clock time without time zone,
   * so the returned `Date`'s **UTC** fields equal the header's fields; read
   * it with `getUTCHours()` etc. or attach the right zone yourself.
   */
  readonly startDate: Date;
  /** Number of data records. */
  readonly recordCount: number;
  /** Duration of one data record, in seconds. */
  readonly recordDuration: number;
  /** Number of signals in the file, **including** any "EDF Annotations" signals. */
  readonly signalCount: number;
  /** Reserved field: `"EDF+C"` (continuous) or `"EDF+D"` (discontinuous) for EDF+, else usually empty. */
  readonly reserved: string;
}

/** One data signal (channel) of an EDF recording. */
export interface EDFSignal {
  /** Signal label, e.g. `"EEG Fp1-A1"`. */
  readonly label: string;
  /** Transducer type, e.g. `"AgAgCl electrode"`. */
  readonly transducer: string;
  /** Physical unit, e.g. `"uV"`. Samples are in this unit. */
  readonly physicalDimension: string;
  /** Physical value of `digitalMin`. */
  readonly physicalMin: number;
  /** Physical value of `digitalMax`. */
  readonly physicalMax: number;
  /** Lowest digital (ADC) value, typically -32768. */
  readonly digitalMin: number;
  /** Highest digital (ADC) value, typically 32767. */
  readonly digitalMax: number;
  /** Prefiltering applied before digitisation, e.g. `"HP:0.1Hz LP:75Hz"`. */
  readonly prefilter: string;
  /** Samples per second (samples per record ÷ record duration), in Hz. */
  readonly sampleRate: number;
  /** All samples, converted to physical units. */
  readonly samples: Float64Array;
}

/** One EDF+ annotation (from a Time-stamped Annotations List). */
export interface EDFAnnotation {
  /** Onset in seconds relative to the recording start (may be negative). */
  readonly onset: number;
  /** Duration in seconds, if given. */
  readonly duration?: number;
  /** Annotation text (UTF-8 decoded). */
  readonly text: string;
}

/** A parsed EDF/EDF+ file. */
export interface EDFRecording {
  readonly header: EDFHeader;
  /** Data signals, in file order, excluding "EDF Annotations" signals. */
  readonly signals: EDFSignal[];
  /** EDF+ annotations in file order (empty for plain EDF). */
  readonly annotations: EDFAnnotation[];
}

const ANNOTATION_LABEL = "EDF Annotations";

/**
 * Header fields are ASCII by specification. Decoded by hand rather than with
 * `TextDecoder`, which older React Native engines lack.
 * @internal
 */
function ascii(bytes: Uint8Array, offset: number, length: number): string {
  let s = "";
  for (let i = offset; i < offset + length; i++) s += String.fromCharCode(bytes[i] as number);
  return s.trim();
}

/**
 * Minimal UTF-8 decoder for annotation text (EDF+ specifies UTF-8); invalid
 * sequences become U+FFFD instead of throwing.
 * @internal
 */
function utf8(bytes: Uint8Array, start: number, end: number): string {
  let s = "";
  let i = start;
  while (i < end) {
    const b0 = bytes[i] as number;
    const extra =
      b0 < 0x80
        ? 0
        : b0 >= 0xc2 && b0 < 0xe0
          ? 1
          : b0 >= 0xe0 && b0 < 0xf0
            ? 2
            : b0 >= 0xf0 && b0 < 0xf5
              ? 3
              : -1;
    if (extra < 0 || i + extra >= end) {
      s += "\ufffd";
      i++;
      continue;
    }
    let cp = extra === 0 ? b0 : b0 & (0x3f >> extra);
    let ok = true;
    for (let k = 1; k <= extra; k++) {
      const b = bytes[i + k] as number;
      if ((b & 0xc0) !== 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (b & 0x3f);
    }
    if (!ok) {
      s += "\ufffd";
      i++;
      continue;
    }
    s += String.fromCodePoint(cp);
    i += extra + 1;
  }
  return s;
}

/** Parses a numeric header field, with a clear error naming the field. @internal */
function num(text: string, field: string): number {
  const value = Number(text);
  if (text === "" || !Number.isFinite(value)) {
    throw new EDFParseError(`${field} is not a number: "${text}"`);
  }
  return value;
}

/** Positive integer header field. @internal */
function int(text: string, field: string, min: number): number {
  const value = num(text, field);
  if (!Number.isInteger(value) || value < min) {
    throw new EDFParseError(`${field} must be an integer >= ${min}, got "${text}"`);
  }
  return value;
}

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

/** Start date/time from `dd.mm.yy` and `hh.mm.ss`, with EDF's 1985–2084 year window. @internal */
function parseStart(date: string, time: string, recording: string): Date {
  const d = /^(\d\d)\.(\d\d)\.(\d\d|yy)$/.exec(date);
  const t = /^(\d\d)\.(\d\d)\.(\d\d)$/.exec(time);
  if (d === null) throw new EDFParseError(`start date must be dd.mm.yy, got "${date}"`);
  if (t === null) throw new EDFParseError(`start time must be hh.mm.ss, got "${time}"`);
  // EDF+ also stores a four-digit year in the recording field, which removes
  // the two-digit ambiguity; after 2084 the date field holds a literal "yy".
  const plus = /^Startdate \d\d-([A-Z]{3})-(\d{4})/.exec(recording);
  let year: number;
  if (plus !== null && MONTHS.includes(plus[1] as string)) {
    year = Number(plus[2]);
  } else if (d[3] === "yy") {
    throw new EDFParseError(
      `start date year is "yy" but the recording field has no Startdate year`,
    );
  } else {
    const yy = Number(d[3]);
    year = yy >= 85 ? 1900 + yy : 2000 + yy;
  }
  const day = Number(d[1]);
  const month = Number(d[2]);
  const [hour, minute, second] = [Number(t[1]), Number(t[2]), Number(t[3])];
  if (day < 1 || day > 31 || month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59) {
    throw new EDFParseError(`invalid start date/time "${date} ${time}"`);
  }
  return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

/**
 * Parses the Time-stamped Annotations Lists (TALs) in one record's bytes of
 * an "EDF Annotations" signal. A TAL is
 * `+onset[\x15duration]\x14text\x14[text\x14...]\x00`; the first TAL of each
 * record has empty text and only timestamps the record, so it is skipped.
 * @internal
 */
function parseTALs(bytes: Uint8Array, start: number, end: number, out: EDFAnnotation[]): void {
  let i = start;
  while (i < end) {
    if (bytes[i] === 0) {
      i++; // padding between/after TALs
      continue;
    }
    // Find the end of this TAL.
    let talEnd = i;
    while (talEnd < end && bytes[talEnd] !== 0) talEnd++;
    // Header: onset and optional duration, terminated by the first 0x14.
    let headerEnd = i;
    while (headerEnd < talEnd && bytes[headerEnd] !== 0x14) headerEnd++;
    if (headerEnd === talEnd) throw new EDFParseError("annotation TAL without a 0x14 separator");
    const header = utf8(bytes, i, headerEnd);
    const [onsetText = "", durationText] = header.split("\x15");
    if (!/^[+-]\d+(\.\d*)?$/.test(onsetText)) {
      throw new EDFParseError(`annotation onset must start with + or -, got "${onsetText}"`);
    }
    const onset = Number(onsetText);
    let duration: number | undefined;
    if (durationText !== undefined && durationText !== "") {
      duration = Number(durationText);
      if (!/^\d+(\.\d*)?$/.test(durationText)) {
        throw new EDFParseError(`annotation duration is not a number: "${durationText}"`);
      }
    }
    // Texts are 0x14-terminated; empty texts (the record timestamp) are skipped.
    let textStart = headerEnd + 1;
    for (let k = textStart; k < talEnd; k++) {
      if (bytes[k] === 0x14) {
        if (k > textStart) {
          const text = utf8(bytes, textStart, k);
          out.push(duration === undefined ? { onset, text } : { onset, duration, text });
        }
        textStart = k + 1;
      }
    }
    i = talEnd + 1;
  }
}

/**
 * Parses an EDF or EDF+ file (European Data Format; Kemp et al., 1992,
 * Electroenceph. Clin. Neurophysiol. 82:391-393; EDF+: Kemp & Olivan, 2003,
 * Clin. Neurophysiol. 114:1755-1761).
 *
 * Digital samples (16-bit little-endian integers) are converted to physical
 * units with each signal's linear calibration:
 * `physical = (digital − digitalMin) · (physicalMax − physicalMin) / (digitalMax − digitalMin) + physicalMin`.
 *
 * EDF+ "EDF Annotations" signals are decoded into `annotations` and left out
 * of `signals`. Discontinuous EDF+ files (`"EDF+D"`) are read with their
 * records concatenated; check `header.reserved` if gaps matter. BDF (24-bit)
 * files are rejected.
 *
 * Pure byte parsing: works in browsers, React Native, workers and Node.
 *
 * @param buffer - The whole file, e.g. from `await file.arrayBuffer()` or `fs.readFileSync`.
 * @throws EDFParseError with a specific message if the file is malformed or truncated.
 *
 * @example
 * ```ts
 * const rec = parseEDF(await file.arrayBuffer());
 * const fp1 = rec.signals.find((s) => s.label.includes("Fp1"));
 * console.log(fp1?.sampleRate, rec.annotations);
 * ```
 */
export function parseEDF(buffer: ArrayBuffer | Uint8Array): EDFRecording {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length < 256) {
    throw new EDFParseError(`file is ${bytes.length} bytes, shorter than the 256-byte header`);
  }
  if (bytes[0] === 0xff && ascii(bytes, 1, 7) === "BIOSEMI") {
    throw new EDFParseError("this is a BDF (24-bit) file; only 16-bit EDF/EDF+ is supported");
  }
  const version = ascii(bytes, 0, 8);
  if (version !== "0") throw new EDFParseError(`version field must be "0", got "${version}"`);
  const patient = ascii(bytes, 8, 80);
  const recording = ascii(bytes, 88, 80);
  const startDate = parseStart(ascii(bytes, 168, 8), ascii(bytes, 176, 8), recording);
  const headerBytes = int(ascii(bytes, 184, 8), "header byte count", 256);
  const reserved = ascii(bytes, 192, 44);
  let recordCount = num(ascii(bytes, 236, 8), "number of data records");
  const recordDuration = num(ascii(bytes, 244, 8), "data record duration");
  const signalCount = int(ascii(bytes, 252, 4), "number of signals", 1);
  if (recordDuration < 0) {
    throw new EDFParseError(`data record duration must be >= 0 s, got ${recordDuration}`);
  }
  if (headerBytes !== 256 * (signalCount + 1)) {
    throw new EDFParseError(
      `header byte count is ${headerBytes}, expected ${256 * (signalCount + 1)} for ${signalCount} signals`,
    );
  }
  if (bytes.length < headerBytes) {
    throw new EDFParseError(
      `file is ${bytes.length} bytes, shorter than its ${headerBytes}-byte header`,
    );
  }

  // Signal header fields are stored field by field: all labels, then all transducers, etc.
  let offset = 256;
  const field = (width: number): string[] => {
    const values: string[] = [];
    for (let s = 0; s < signalCount; s++) values.push(ascii(bytes, offset + s * width, width));
    offset += width * signalCount;
    return values;
  };
  const labels = field(16);
  const transducers = field(80);
  const dimensions = field(8);
  const physMins = field(8);
  const physMaxs = field(8);
  const digMins = field(8);
  const digMaxs = field(8);
  const prefilters = field(80);
  const samplesPerRecordText = field(8);

  interface Layout {
    readonly index: number;
    readonly annotation: boolean;
    readonly samplesPerRecord: number;
    readonly byteOffset: number; // within a record
    readonly scale: number;
    readonly physMin: number;
    readonly digMin: number;
    readonly physMax: number;
    readonly digMax: number;
  }
  const layouts: Layout[] = [];
  let recordBytes = 0;
  for (let s = 0; s < signalCount; s++) {
    const name = `signal ${s + 1} (${labels[s] ?? ""})`;
    const samplesPerRecord = int(samplesPerRecordText[s] ?? "", `${name} samples per record`, 1);
    const physMin = num(physMins[s] ?? "", `${name} physical minimum`);
    const physMax = num(physMaxs[s] ?? "", `${name} physical maximum`);
    const digMin = int(digMins[s] ?? "", `${name} digital minimum`, -32768);
    const digMax = int(digMaxs[s] ?? "", `${name} digital maximum`, -32768);
    const annotation = labels[s] === ANNOTATION_LABEL;
    if (!annotation) {
      if (digMax <= digMin || digMax > 32767) {
        throw new EDFParseError(
          `${name} digital range [${digMin}, ${digMax}] must be increasing and within 16 bits`,
        );
      }
      if (physMax === physMin) {
        throw new EDFParseError(`${name} physical minimum and maximum are both ${physMin}`);
      }
      if (recordDuration === 0) {
        throw new EDFParseError(`record duration is 0 but ${name} is not an annotation signal`);
      }
    }
    layouts.push({
      index: s,
      annotation,
      samplesPerRecord,
      byteOffset: recordBytes,
      scale: annotation ? 0 : (physMax - physMin) / (digMax - digMin),
      physMin,
      digMin,
      physMax,
      digMax,
    });
    recordBytes += 2 * samplesPerRecord;
  }

  const available = Math.floor((bytes.length - headerBytes) / recordBytes);
  if (recordCount === -1) {
    // -1 means "unknown" (file still being written): use what is there.
    recordCount = available;
  } else if (!Number.isInteger(recordCount) || recordCount < 0) {
    throw new EDFParseError(
      `number of data records must be a non-negative integer or -1, got ${recordCount}`,
    );
  } else if (recordCount > available) {
    throw new EDFParseError(
      `header declares ${recordCount} data records but the file holds only ${available} (truncated?)`,
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const signals: EDFSignal[] = [];
  const annotations: EDFAnnotation[] = [];
  for (const layout of layouts) {
    const s = layout.index;
    if (layout.annotation) {
      for (let r = 0; r < recordCount; r++) {
        const start = headerBytes + r * recordBytes + layout.byteOffset;
        parseTALs(bytes, start, start + 2 * layout.samplesPerRecord, annotations);
      }
      continue;
    }
    const perRecord = layout.samplesPerRecord;
    const samples = new Float64Array(perRecord * recordCount);
    const { scale, physMin, digMin } = layout;
    for (let r = 0; r < recordCount; r++) {
      const base = headerBytes + r * recordBytes + layout.byteOffset;
      for (let i = 0; i < perRecord; i++) {
        const digital = view.getInt16(base + 2 * i, true);
        samples[r * perRecord + i] = (digital - digMin) * scale + physMin;
      }
    }
    signals.push({
      label: labels[s] ?? "",
      transducer: transducers[s] ?? "",
      physicalDimension: dimensions[s] ?? "",
      physicalMin: layout.physMin,
      physicalMax: layout.physMax,
      digitalMin: layout.digMin,
      digitalMax: layout.digMax,
      prefilter: prefilters[s] ?? "",
      sampleRate: perRecord / recordDuration,
      samples,
    });
  }

  return {
    header: {
      version,
      patient,
      recording,
      startDate,
      recordCount,
      recordDuration,
      signalCount,
      reserved,
    },
    signals,
    annotations,
  };
}
