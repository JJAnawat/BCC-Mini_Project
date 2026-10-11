"""Windowing/band-power helpers, feature-file I/O and the per-subject loop for the feature domains."""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from bcc.config import CONFIG, FEATURES_DIR, METADATA_DIR, MODELS_DIR, SANITY_DIR  # noqa: F401 - re-exported
from bcc.data import (  # noqa: F401 - re-exported for the feature scripts
    BAND_DEFS, EEG_CHANNELS, FS, N_EEG_CHANNELS, N_SUBJECTS, N_TRIALS, SYMMETRIC_PAIRS,
    THRESHOLD_DEFAULT, iter_subjects, iter_subjects_corrected, load_subject,
)

WINDOW_SAMPLES = CONFIG["windowing"]["window_samples"]           # 512
N_WINDOWS = CONFIG["windowing"]["n_windows_per_trial"]           # 15
BASELINE_SAMPLES = CONFIG["windowing"]["drop_baseline_sec"] * FS  # 384
BAND_NAMES = list(BAND_DEFS)
ID_COLS = ["subject_id", "trial_idx", "window_idx"]
DOMAINS = ["de", "hjorth", "asymmetry", "mse", "microstates"]

for _d in [MODELS_DIR / "microstates", *(FEATURES_DIR / d for d in DOMAINS), *(SANITY_DIR / d for d in DOMAINS)]:
    _d.mkdir(parents=True, exist_ok=True)


def feature_path(domain: str, subject_id: int) -> Path:
    return FEATURES_DIR / domain / f"s{subject_id:02d}_{domain}.csv"


def load_domain(domain: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(feature_path(domain, s)) for s in range(1, N_SUBJECTS + 1)], ignore_index=True)
    assert df["subject_id"].nunique() == N_SUBJECTS
    return df


def window_ids(subject_id: int) -> pd.DataFrame:
    trial_idx, window_idx = np.divmod(np.arange(N_TRIALS * N_WINDOWS), N_WINDOWS)
    return pd.DataFrame({"subject_id": subject_id, "trial_idx": trial_idx, "window_idx": window_idx})


def run_domain(domain: str, compute, columns: list[str], cache: bool = False) -> pd.DataFrame:
    """compute(sid, eeg) -> (600, n_features) array; saves s{NN}_{domain}.csv, returns all subjects."""
    for sid in range(1, N_SUBJECTS + 1):
        out = feature_path(domain, sid)
        if cache and out.exists():
            continue
        values = compute(sid, load_subject(sid)[0][:, :N_EEG_CHANNELS, :])
        pd.concat([window_ids(sid), pd.DataFrame(values, columns=columns)], axis=1).to_csv(out, index=False)
        print(f"  s{sid:02d}: saved")
    return load_domain(domain)


def stimulus(eeg_trial: np.ndarray) -> np.ndarray:
    """(32, 8064) trial -> (32, 7680) with the 3s pre-stimulus baseline dropped."""
    return eeg_trial[:, BASELINE_SAMPLES:]


def window_subject(eeg: np.ndarray) -> np.ndarray:
    """(40, 32, 8064) -> (600, 32, 512) non-overlapping 4s windows, rows ordered like window_ids()."""
    windows = []
    for trial in eeg:
        stim = stimulus(trial)
        windows.append(stim.reshape(stim.shape[0], N_WINDOWS, WINDOW_SAMPLES).transpose(1, 0, 2))
    return np.concatenate(windows, axis=0)


def window_psd(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Week 1's welch idiom (nperseg=min(512, n)), vectorized over the last axis."""
    return welch(windows, fs=FS, nperseg=min(512, windows.shape[-1]), axis=-1)


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """PSD (..., n_freqs) -> (..., n_bands), Week 1's inclusive-mask trapezoid integration."""
    masks = [(freqs >= BAND_DEFS[b][0]) & (freqs <= BAND_DEFS[b][1]) for b in BAND_NAMES]
    return np.stack([np.trapezoid(psd[..., m], freqs[m], axis=-1) for m in masks], axis=-1)


def quadrant(valence: float, arousal: float) -> str:
    """Same convention as pipeline/eda/task2b_quadrant.py."""
    return f"{'H' if valence >= THRESHOLD_DEFAULT else 'L'}V{'H' if arousal >= THRESHOLD_DEFAULT else 'L'}A"


def montage_info() -> mne.Info:
    info = mne.create_info(EEG_CHANNELS, sfreq=FS, ch_types="eeg")
    info.set_montage("colin27_1020")
    return info
