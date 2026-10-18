"""Windowing/band-power helpers, feature-file I/O and the per-subject loop for the feature domains."""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from bcc.config import (  # noqa: F401 - re-exported
    CONFIG, FEATURES_60S_DIR, FEATURES_DIR, METADATA_DIR, MODELS_DIR, SANITY_DIR,
)
from bcc.data import (  # noqa: F401 - re-exported for the feature scripts
    BAND_DEFS, EEG_CHANNELS, FS, N_EEG_CHANNELS, N_SUBJECTS, N_TRIALS, SYMMETRIC_PAIRS,
    THRESHOLD_DEFAULT, iter_subjects, iter_subjects_corrected, load_subject,
)

# {seg: (samples per segment, segments per trial)}: "4s" = 512 x 15 windows, "60s" = 7680 x 1 (whole stimulus)
SEGMENTS = {name: (s["window_samples"], s["n_windows_per_trial"])
            for name, s in CONFIG["windowing"]["segmentations"].items()}
SEGMENTATIONS = list(SEGMENTS)
WINDOW_SAMPLES, N_WINDOWS = SEGMENTS["4s"]
FEATURES_DIRS = {"4s": FEATURES_DIR, "60s": FEATURES_60S_DIR}
BASELINE_SAMPLES = CONFIG["windowing"]["drop_baseline_sec"] * FS  # 384
BAND_NAMES = list(BAND_DEFS)
ID_COLS = ["subject_id", "trial_idx", "window_idx"]
DOMAINS = ["de", "hjorth", "asymmetry", "mse", "microstates"]

for _d in [MODELS_DIR / "microstates", *(FEATURES_DIRS[s] / d for s in SEGMENTATIONS for d in DOMAINS),
           *(SANITY_DIR / d for d in DOMAINS)]:
    _d.mkdir(parents=True, exist_ok=True)


def feature_path(domain: str, subject_id: int, seg: str = "4s") -> Path:
    return FEATURES_DIRS[seg] / domain / f"s{subject_id:02d}_{domain}.csv"


def sanity_path(domain: str, filename: str, seg: str = "4s") -> Path:
    """sanity_checks/<domain>/<filename>; the 60s set gets a _60s suffix in the same folder."""
    path = SANITY_DIR / domain / filename
    return path if seg == "4s" else path.with_stem(f"{path.stem}_{seg}")


def load_domain(domain: str, seg: str = "4s") -> pd.DataFrame:
    df = pd.concat([pd.read_csv(feature_path(domain, s, seg)) for s in range(1, N_SUBJECTS + 1)], ignore_index=True)
    assert df["subject_id"].nunique() == N_SUBJECTS
    return df


def window_ids(subject_id: int, seg: str = "4s") -> pd.DataFrame:
    """ID columns; 60s rows have window_idx 0, so they join to the window_idx == 0 metadata rows."""
    n_windows = SEGMENTS[seg][1]
    trial_idx, window_idx = np.divmod(np.arange(N_TRIALS * n_windows), n_windows)
    return pd.DataFrame({"subject_id": subject_id, "trial_idx": trial_idx, "window_idx": window_idx})


def run_domain(domain: str, compute, columns: list[str], seg: str = "4s", cache: bool = False) -> pd.DataFrame:
    """compute(sid, eeg) -> (n_rows, n_features) array (600 for 4s, 40 for 60s); saves
    s{NN}_{domain}.csv for that segmentation, returns all subjects."""
    for sid in range(1, N_SUBJECTS + 1):
        out = feature_path(domain, sid, seg)
        if cache and out.exists():
            continue
        values = compute(sid, load_subject(sid)[0][:, :N_EEG_CHANNELS, :])
        pd.concat([window_ids(sid, seg), pd.DataFrame(values, columns=columns)], axis=1).to_csv(out, index=False)
        print(f"  s{sid:02d}: saved")
    return load_domain(domain, seg)


def stimulus(eeg_trial: np.ndarray) -> np.ndarray:
    """(32, 8064) trial -> (32, 7680) with the 3s pre-stimulus baseline dropped."""
    return eeg_trial[:, BASELINE_SAMPLES:]


def window_subject(eeg: np.ndarray, seg: str = "4s") -> np.ndarray:
    """(40, 32, 8064) -> (600, 32, 512) non-overlapping 4s windows, or (40, 32, 7680) for 60s,
    rows ordered like window_ids()."""
    samples, n_windows = SEGMENTS[seg]
    windows = []
    for trial in eeg:
        stim = stimulus(trial)
        windows.append(stim.reshape(stim.shape[0], n_windows, samples).transpose(1, 0, 2))
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
