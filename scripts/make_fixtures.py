"""Generate SciPy/NumPy reference fixtures for eegkit's filter and spectral tests.

Run from the repository root:

    uv run --with numpy --with scipy python scripts/make_fixtures.py

Writes test/fixtures/filter.json and test/fixtures/spectral.json. The output is
deterministic (seeded RNG). Every number is rounded to 15 significant digits to
keep the JSON compact; inputs are rounded *before* the reference outputs are
computed, so the TypeScript tests feed SciPy's exact inputs.

Filters are compared by their outputs, not by SOS coefficients: section pairing
and ordering may legitimately differ from scipy.signal.zpk2sos.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test" / "fixtures"


def r15(x) -> np.ndarray:
    """Round to 15 significant digits (what is stored in the JSON)."""
    arr = np.asarray(x, dtype=float)
    return np.array([float(f"{v:.15g}") for v in arr.ravel()]).reshape(arr.shape)


def lst(x) -> list[float]:
    return [float(v) for v in r15(x).ravel()]


def num(x) -> float:
    return float(f"{float(x):.15g}")


def test_signal(n: int, fs: float, seed: int) -> np.ndarray:
    """DC offset + sines at known frequencies + white noise, in microvolts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = (
        25.0
        + 10.0 * np.sin(2 * np.pi * 10.0 * t)
        + 6.0 * np.sin(2 * np.pi * 3.0 * t + 1.1)
        + 3.0 * np.sin(2 * np.pi * 22.0 * t)
        + 4.0 * np.sin(2 * np.pi * 50.0 * t + 0.3)
        + 2.0 * rng.standard_normal(n)
    )
    return r15(x)


def sos_freqz(sos, freqs, fs):
    fn = getattr(signal, "freqz_sos", None) or signal.sosfreqz
    _, h = fn(sos, worN=np.asarray(freqs, dtype=float), fs=fs)
    return h


# --------------------------------------------------------------------------- filters

FILTER_FS = 250.0
FILTER_N = 400

FILTERS = [
    {"name": "lowpass1_10", "type": "lowpass", "order": 1, "cutoff": 10.0},
    {"name": "lowpass4_30", "type": "lowpass", "order": 4, "cutoff": 30.0},
    {"name": "lowpass5_40", "type": "lowpass", "order": 5, "cutoff": 40.0},
    {"name": "lowpass8_5", "type": "lowpass", "order": 8, "cutoff": 5.0},
    {"name": "highpass2_1", "type": "highpass", "order": 2, "cutoff": 1.0},
    {"name": "highpass3_0.5", "type": "highpass", "order": 3, "cutoff": 0.5},
    {"name": "highpass4_100", "type": "highpass", "order": 4, "cutoff": 100.0},
    {"name": "bandpass1_0.5_100", "type": "bandpass", "order": 1, "low": 0.5, "high": 100.0},
    {"name": "bandpass3_0.5_100", "type": "bandpass", "order": 3, "low": 0.5, "high": 100.0},
    {"name": "bandpass4_1_40", "type": "bandpass", "order": 4, "low": 1.0, "high": 40.0},
    {"name": "bandpass3_8_13", "type": "bandpass", "order": 3, "low": 8.0, "high": 13.0},
    {"name": "bandstop2_48_52", "type": "bandstop", "order": 2, "low": 48.0, "high": 52.0},
    {"name": "bandstop3_45_55", "type": "bandstop", "order": 3, "low": 45.0, "high": 55.0},
    {"name": "notch50_q30", "type": "notch", "frequency": 50.0, "quality": 30.0},
    {"name": "notch60_q10", "type": "notch", "frequency": 60.0, "quality": 10.0},
]

FREQZ_POINTS = [0.0, 0.25, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0, 13.0, 20.0, 30.0, 40.0, 45.0, 48.0,
                50.0, 52.0, 55.0, 60.0, 80.0, 100.0, 110.0, 124.0, 125.0]


def design(spec: dict, fs: float):
    t = spec["type"]
    if t == "notch":
        b, a = signal.iirnotch(spec["frequency"], spec["quality"], fs=fs)
        return np.array([[b[0], b[1], b[2], 1.0, a[1] / a[0], a[2] / a[0]]])
    if t in ("lowpass", "highpass"):
        return signal.butter(spec["order"], spec["cutoff"], btype=t, output="sos", fs=fs)
    return signal.butter(spec["order"], [spec["low"], spec["high"]], btype=t, output="sos", fs=fs)


def make_filter_fixtures() -> dict:
    x = test_signal(FILTER_N, FILTER_FS, seed=1)
    cases = []
    for spec in FILTERS:
        sos = design(spec, FILTER_FS)
        zi = signal.sosfilt_zi(sos)
        h = sos_freqz(sos, FREQZ_POINTS, FILTER_FS)
        ntaps = 2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
        padlen = 3 * ntaps
        short = x[: padlen + 1]
        cases.append({
            **spec,
            "padlen": padlen,
            "sosfilt": lst(signal.sosfilt(sos, x)),
            "sosfiltPrimed": lst(signal.sosfilt(sos, x, zi=zi * x[0])[0]),
            "sosfiltfilt": lst(signal.sosfiltfilt(sos, x)),
            "sosfiltfiltShort": lst(signal.sosfiltfilt(sos, short)),
            "freqzMagnitude": lst(np.abs(h)),
            "freqzPhase": lst(np.angle(h)),
            "dcGain": num(np.prod(sos[:, :3].sum(axis=1) / sos[:, 3:].sum(axis=1))),
        })
    return {
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "sampleRate": FILTER_FS,
        "signal": lst(x),
        "freqzPoints": FREQZ_POINTS,
        "cases": cases,
    }


# --------------------------------------------------------------------------- spectral

SPECTRAL_FS = 250.0
SPECTRAL_N = 1500


def band_power_ref(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoid integral of the linearly interpolated PSD over exactly [lo, hi]."""
    lo = max(lo, f[0])
    hi = min(hi, f[-1])
    if hi <= lo:
        return 0.0
    inner = f[(f > lo) & (f < hi)]
    xs = np.concatenate(([lo], inner, [hi]))
    ys = np.interp(xs, f, p)
    return float(np.trapezoid(ys, xs))


STANDARD_BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30),
                  ("gamma", 30, 45)]


def make_spectral_fixtures() -> dict:
    x = test_signal(SPECTRAL_N, SPECTRAL_FS, seed=2)
    tukey = r15(signal.windows.tukey(200, 0.25))

    welch_cases = [
        ("default", {}, {}),
        ("seg250_median", {"nperseg": 250, "average": "median"}, {"segmentLength": 250, "average": "median"}),
        ("seg375_median_even", {"nperseg": 375, "noverlap": 0, "average": "median"},
         {"segmentLength": 375, "overlap": 0, "average": "median"}),
        ("seg255_hamming_ov100", {"window": "hamming", "nperseg": 255, "noverlap": 100},
         {"window": "hamming", "segmentLength": 255, "overlap": 100}),
        ("seg128_blackman_linear_spectrum",
         {"window": "blackman", "nperseg": 128, "detrend": "linear", "scaling": "spectrum"},
         {"window": "blackman", "segmentLength": 128, "detrend": "linear", "scaling": "spectrum"}),
        ("seg500_boxcar_nodetrend", {"window": "boxcar", "nperseg": 500, "detrend": False},
         {"window": "rectangular", "segmentLength": 500, "detrend": False}),
        ("seg97_odd_prime", {"nperseg": 97}, {"segmentLength": 97}),
        ("seg3000_clamped", {"nperseg": 3000}, {"segmentLength": 3000}),
        ("tukey_array", {"window": tukey}, {"window": "tukey200"}),
    ]
    welch_out = []
    for name, kw, ts in welch_cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, p = signal.welch(x, fs=SPECTRAL_FS, **kw)
        welch_out.append({"name": name, "options": ts, "frequencies": lst(f), "power": lst(p)})

    periodogram_cases = [
        ("default", x, {}, {}),
        ("hann", x, {"window": "hann"}, {"window": "hann"}),
        ("spectrum_linear", x, {"scaling": "spectrum", "detrend": "linear"},
         {"scaling": "spectrum", "detrend": "linear"}),
        ("odd_length", x[:999], {"window": "hamming"}, {"window": "hamming"}),
    ]
    pg_out = []
    for name, sig, kw, ts in periodogram_cases:
        f, p = signal.periodogram(sig, fs=SPECTRAL_FS, **kw)
        # Frequencies are k * fs / n; store only the spacing and the count.
        pg_out.append({"name": name, "length": len(sig), "options": ts, "df": num(f[1] - f[0]),
                       "bins": len(f), "power": lst(p)})

    # Band powers on the default Welch spectrum.
    f, p = signal.welch(x, fs=SPECTRAL_FS)
    bands = STANDARD_BANDS + [("custom", 7.3, 12.9), ("edgeClamped", 100.0, 200.0)]
    bp = {name: num(band_power_ref(f, p, lo, hi)) for name, lo, hi in bands}
    total = band_power_ref(f, p, 1.0, 45.0)
    rel = {name: num(band_power_ref(f, p, lo, hi) / total) for name, lo, hi in STANDARD_BANDS}

    # FFTs (complex input) and real FFTs, including prime and non-power-of-two sizes.
    rng = np.random.default_rng(3)
    ffts = []
    for n in (1, 2, 3, 5, 16, 64, 97, 100, 250):
        re = r15(rng.standard_normal(n))
        im = r15(rng.standard_normal(n))
        y = np.fft.fft(re + 1j * im)
        ffts.append({"n": n, "re": lst(re), "im": lst(im), "outRe": lst(y.real), "outIm": lst(y.imag)})
    rffts = []
    for n in (1, 2, 7, 64, 99, 250, 256):
        sig = r15(rng.standard_normal(n) * 10 + 3)
        y = np.fft.rfft(sig)
        rffts.append({"n": n, "signal": lst(sig), "outRe": lst(y.real), "outIm": lst(y.imag)})

    windows = []
    for n in (1, 2, 7, 8, 16, 33):
        for kind in ("hann", "hamming", "blackman"):
            for sym in (False, True):
                w = signal.get_window(kind, n, fftbins=not sym)
                windows.append({"type": kind, "n": n, "sym": sym, "values": lst(w)})

    return {
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "sampleRate": SPECTRAL_FS,
        "signal": lst(x),
        "tukey200": lst(tukey),
        "welch": welch_out,
        "periodogram": pg_out,
        "bandPower": {"bands": [[n, lo, hi] for n, lo, hi in bands], "absolute": bp, "relative": rel,
                      "relativeRange": [1.0, 45.0]},
        "fft": ffts,
        "rfft": rffts,
        "windows": windows,
    }


def write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    write("filter.json", make_filter_fixtures())
    write("spectral.json", make_spectral_fixtures())
