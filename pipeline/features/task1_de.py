"""Task 1: Differential Entropy per segment (4s window or 60s trial) x channel x band, from Welch band power."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bcc.features import (
    BAND_DEFS, BAND_NAMES, EEG_CHANNELS, SANITY_DIR, band_powers, run_domain, window_psd, window_subject,
)

POWER_FLOOR = 1e-12             # uV^2; only guards log(0) on a degenerate all-zero window
POSTERIOR = ["O1", "O2", "Oz", "PO3", "PO4"]
FIT_RANGE_HZ = (4, 45)          # DEAP preprocessed data is already band-passed 4-45Hz
ALPHA_EXCLUDE_HZ = (7, 14)      # left out of the 1/f fit so the alpha bump shows as a residual
ALPHA_PEAK_MIN_NATS = 0.05
LINE_HZ = 22.0                  # non-physiological narrowband line seen in 30/32 subjects
LINE_FLANK_HZ = (1.0, 3.0)
COLUMNS = [f"{ch}_{band}_de" for ch in EEG_CHANNELS for band in BAND_NAMES]


def _de(power: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * np.e * np.maximum(power, POWER_FLOOR))


def _alpha_residual(freqs, spectrum) -> tuple[float, float]:
    """Peak (nats, Hz) of the spectrum above a log-log 1/f fit, within the alpha band."""
    fit = (freqs >= FIT_RANGE_HZ[0]) & (freqs <= FIT_RANGE_HZ[1])
    fit &= ~((freqs >= ALPHA_EXCLUDE_HZ[0]) & (freqs <= ALPHA_EXCLUDE_HZ[1]))
    coef = np.polyfit(np.log10(freqs[fit]), spectrum[fit], 1)
    alpha = (freqs >= BAND_DEFS["alpha"][0]) & (freqs <= BAND_DEFS["alpha"][1])
    resid = spectrum[alpha] - np.polyval(coef, np.log10(freqs[alpha]))
    return float(resid.max()), float(freqs[alpha][resid.argmax()])


def _line_prominence(freqs, spectrum) -> float:
    dist = np.abs(freqs - LINE_HZ)
    flank = (dist >= LINE_FLANK_HZ[0]) & (dist <= LINE_FLANK_HZ[1])
    return float(spectrum[dist.argmin()] - spectrum[flank].mean())


def run(seg="4s"):
    spectra, freqs = {}, None

    def compute(sid, eeg):
        nonlocal freqs
        freqs, psd = window_psd(window_subject(eeg, seg))                 # (600 or 40, 32, n_freqs)
        spectra[sid] = _de(psd).mean(axis=0)                              # per-frequency DE, for checks
        return _de(band_powers(freqs, psd)).reshape(len(psd), -1)

    df = run_domain("de", compute, COLUMNS, seg)
    if seg != "4s":                                                       # spectrum checks are 4s-only
        return df, {}

    post = [EEG_CHANNELS.index(c) for c in POSTERIOR]
    rows = []
    for sid, spec in spectra.items():
        resid, peak = _alpha_residual(freqs, spec[post].mean(axis=0))
        rows.append({"subject_id": sid, "posterior_alpha_residual_nats": resid, "posterior_alpha_peak_hz": peak,
                     "line_22hz_prominence_nats": _line_prominence(freqs, spec.mean(axis=0))})
    checks = pd.DataFrame(rows)
    checks.to_csv(SANITY_DIR / "de" / "subject_checks.csv", index=False)
    has_alpha = (checks["posterior_alpha_residual_nats"] > ALPHA_PEAK_MIN_NATS) & \
        checks["posterior_alpha_peak_hz"].between(*BAND_DEFS["alpha"], inclusive="neither")

    pooled = np.mean(list(spectra.values()), axis=0).mean(axis=0)
    plt.figure(figsize=(9, 5))
    plt.plot(freqs, pooled, color="black")
    for band, (lo, hi) in BAND_DEFS.items():
        plt.axvspan(lo, hi, alpha=0.08, color=f"C{BAND_NAMES.index(band)}", label=band)
    plt.axvline(LINE_HZ, color="gray", linestyle=":", label=f"{LINE_HZ:g}Hz line")
    plt.xlim(1, 50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("0.5*ln(2*pi*e*PSD) (nats)")
    plt.title("Pooled DE spectrum, all subjects and channels")
    plt.legend()
    plt.savefig(SANITY_DIR / "de" / "de_spectrum_pooled.png", dpi=150)
    plt.close()

    summary = {"n_subjects_with_alpha_peak": int(has_alpha.sum()),
               "n_subjects_with_22hz_line": int((checks["line_22hz_prominence_nats"] > 0.05).sum())}
    print(f"  checks: {summary}")
    return df, summary


if __name__ == "__main__":
    run()
