"""Shared constants and helpers for the DEAP EDA scripts.

Channels/bands/threshold/paths are sourced from the locked root config.yaml
(via config.py) rather than redefined here, so nothing can silently drift
from the Week 1 EDA decisions.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / CONFIG["paths"]["data_dir"]
RATINGS_XLS_PATH = BASE_DIR / CONFIG["paths"]["ratings_xls"]
OUTPUT_DIR = BASE_DIR / CONFIG["paths"]["eda_outputs"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SUBJECTS = CONFIG["dataset"]["n_subjects"]
N_TRIALS = CONFIG["dataset"]["n_trials"]
N_CHANNELS_TOTAL = 40
N_EEG_CHANNELS = CONFIG["dataset"]["n_eeg_channels"]
N_SAMPLES = CONFIG["dataset"]["n_samples_per_trial"]
FS = CONFIG["dataset"]["fs_hz"]
BASELINE_SEC = CONFIG["dataset"]["baseline_sec"]
TRIAL_SEC = 63

# NOTE: The DEAP .dat files do not embed channel names/metadata (only
# 'data' and 'labels' keys). No montage/channel-list file ships with this
# download either (checked Metadata/ and metadata_xls/). This list is the
# standard 32-channel Biosemi "Geneva order" as documented on the DEAP
# dataset website/paper -- it is an ASSUMED mapping onto data[:, 0:32, :],
# not one verified against in-file metadata.
EEG_CHANNELS = CONFIG["channels"]["order"]
assert len(EEG_CHANNELS) == N_EEG_CHANNELS

# Symmetric left/right electrode pairs for later DASM/RASM features. Not in
# config.yaml: these are *produced* here into symmetric_pairs.json (see
# task1_structural.py), so sourcing them from config would be circular.
# Midline channels (Fz, Cz, Pz, Oz) have no symmetric partner.
SYMMETRIC_PAIRS = [
    ("Fp1", "Fp2"), ("AF3", "AF4"), ("F3", "F4"), ("F7", "F8"),
    ("FC5", "FC6"), ("FC1", "FC2"), ("C3", "C4"), ("T7", "T8"),
    ("CP5", "CP6"), ("CP1", "CP2"), ("P3", "P4"), ("P7", "P8"),
    ("PO3", "PO4"), ("O1", "O2"),
]
MIDLINE_CHANNELS = ["Fz", "Cz", "Pz", "Oz"]
for a, b in SYMMETRIC_PAIRS:
    assert a in EEG_CHANNELS and b in EEG_CHANNELS

BAND_DEFS = {name: tuple(bounds) for name, bounds in CONFIG["windowing"]["bands_hz"].items()}

FACE_VIDEO_GAP_SUBJECTS = CONFIG["subjects"]["face_video_gap_ids"]

LABEL_NAMES = ["valence", "arousal", "dominance", "liking"]

# Binarization convention used throughout the EDA: rating >= THRESHOLD -> "high".
THRESHOLD_DEFAULT = CONFIG["labels"]["binarize_threshold"]


def subject_path(subject_id: int) -> Path:
    return DATA_DIR / f"s{subject_id:02d}.dat"


def load_subject(subject_id: int):
    """Load one subject's (data, labels) arrays -- RAW as shipped in the .dat file."""
    with open(subject_path(subject_id), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return np.asarray(d["data"]), np.asarray(d["labels"])


def iter_subjects(subject_ids=None):
    """Yield (subject_id, data, labels, err) with RAW .dat labels, skipping load errors.

    Use this only for structural/file-integrity checks. For anything that
    consumes valence/arousal/dominance/liking values, use
    iter_subjects_corrected() instead -- see the module docstring below for why.
    """
    ids = subject_ids if subject_ids is not None else range(1, N_SUBJECTS + 1)
    for sid in ids:
        try:
            data, labels = load_subject(sid)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to caller
            yield sid, None, None, exc
            continue
        yield sid, data, labels, None


# ---------------------------------------------------------------------------
# Label correction
#
# The .dat files' `labels[:, 0:2]` (valence, arousal) are corrupted for ~34%
# of trials: for those trials the stored value is `9 - true_rating` instead
# of the true 1-9 SAM rating (confirmed by cross-checking against
# Metadata/participant_ratings.xls, where Dominance and Liking match the
# .dat file exactly on every trial, but Valence/Arousal diverge on 439/1280
# trials -- and every divergent pair satisfies dat_value == 9 - xls_value to
# 2 decimal places). The .dat file's valence/arousal also drop below the
# documented 1-9 range (down to 0.0), while the xls file never does --
# independent confirmation that the xls values are the reliable ones.
#
# get_corrected_labels() / iter_subjects_corrected() replace valence/arousal
# (and, for a single consistent source, dominance/liking too, since they are
# identical to the .dat values anyway) with the participant_ratings.xls
# values, row-matched by Participant_id + Experiment_id.
# ---------------------------------------------------------------------------

_ratings_xls_cache = None


def _load_ratings_xls() -> pd.DataFrame:
    global _ratings_xls_cache
    if _ratings_xls_cache is None:
        _ratings_xls_cache = pd.read_excel(RATINGS_XLS_PATH)
    return _ratings_xls_cache


def get_corrected_labels(subject_id: int) -> np.ndarray:
    """Return (40, 4) [valence, arousal, dominance, liking] from participant_ratings.xls,
    ordered by Experiment_id to match the trial order used in the .dat data array."""
    df = _load_ratings_xls()
    sub = df[df["Participant_id"] == subject_id].sort_values("Experiment_id")
    if len(sub) != N_TRIALS:
        raise ValueError(
            f"Expected {N_TRIALS} rating rows for subject {subject_id}, found {len(sub)}"
        )
    return sub[["Valence", "Arousal", "Dominance", "Liking"]].to_numpy(dtype=float)


def iter_subjects_corrected(subject_ids=None):
    """Yield (subject_id, data, corrected_labels, err).

    `data` is the raw EEG/peripheral array from the .dat file (unaffected by
    the label bug). `corrected_labels` comes from participant_ratings.xls --
    see the note above load_subject_corrected for why.
    """
    ids = subject_ids if subject_ids is not None else range(1, N_SUBJECTS + 1)
    for sid in ids:
        try:
            data, _raw_labels = load_subject(sid)
            corrected_labels = get_corrected_labels(sid)
        except Exception as exc:  # noqa: BLE001 - surface any load/parse failure to caller
            yield sid, None, None, exc
            continue
        yield sid, data, corrected_labels, None
