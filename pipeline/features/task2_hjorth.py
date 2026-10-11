"""Task 2: Hjorth Activity/Mobility/Complexity per 4s window x channel (broadband, time domain)."""
import numpy as np
import pandas as pd

from bcc.features import EEG_CHANNELS, SANITY_DIR, run_domain, window_subject

PARAMS = ["activity", "mobility", "complexity"]
COLUMNS = [f"{ch}_{p}" for ch in EEG_CHANNELS for p in PARAMS]


def _hjorth(windows: np.ndarray) -> np.ndarray:
    """(..., n_samples) -> (..., 3); zero-variance windows give 0, not NaN."""
    dx = np.diff(windows, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x, var_dx, var_ddx = windows.var(axis=-1), dx.var(axis=-1), ddx.var(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mobility = np.where(var_x > 0, np.sqrt(var_dx / var_x), 0.0)
        mobility_dx = np.where(var_dx > 0, np.sqrt(var_ddx / var_dx), 0.0)
        complexity = np.where(mobility > 0, mobility_dx / mobility, 0.0)
    return np.stack([var_x, mobility, complexity], axis=-1)


def run():
    df = run_domain("hjorth", lambda sid, eeg: _hjorth(window_subject(eeg)).reshape(-1, len(COLUMNS)), COLUMNS)
    ranges = pd.DataFrame([
        {"parameter": p, "n_nonfinite": int((~np.isfinite(v)).sum()), "n_zero": int((v == 0).sum()),
         "min": np.nanmin(v), "median": np.nanmedian(v), "max": np.nanmax(v)}
        for p in PARAMS for v in [df[[f"{ch}_{p}" for ch in EEG_CHANNELS]].to_numpy()]
    ])
    ranges.to_csv(SANITY_DIR / "hjorth" / "hjorth_ranges.csv", index=False)
    summary = {"n_nonfinite": int(ranges["n_nonfinite"].sum()), "n_zero": int(ranges["n_zero"].sum())}
    print(f"  checks: {summary}")
    return df, summary


if __name__ == "__main__":
    run()
