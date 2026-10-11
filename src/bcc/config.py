"""Typed loader for the locked project config.yaml.

Every downstream feature-extraction / classification script should do
`from bcc.config import CONFIG` (or call load_config() directly) instead of
redefining thresholds, bands, windowing, or exclusion lists -- that's the
whole point of freezing these decisions after Week 1's EDA.

All output paths live here too, so every stage reads and writes the same places.
"""
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    fs = cfg["dataset"]["fs_hz"]
    window_sec = cfg["windowing"]["window_sec"]
    window_samples = cfg["windowing"]["window_samples"]
    assert window_samples == fs * window_sec, (
        f"window_samples ({window_samples}) != fs_hz*window_sec "
        f"({fs}*{window_sec}={fs * window_sec}) -- config.yaml is inconsistent"
    )

    n_samples = cfg["dataset"]["n_samples_per_trial"]
    baseline_samples = cfg["dataset"]["baseline_sec"] * fs
    stim_samples = n_samples - baseline_samples
    n_windows = cfg["windowing"]["n_windows_per_trial"]
    assert stim_samples % window_samples == 0, (
        f"{stim_samples} post-baseline samples not evenly divisible by "
        f"window_samples ({window_samples}) -- windowing would need a "
        "partial/dropped window; update n_windows_per_trial or window_sec"
    )
    assert stim_samples // window_samples == n_windows, (
        f"n_windows_per_trial ({n_windows}) != computed value "
        f"({stim_samples // window_samples})"
    )

    channels = cfg["channels"]["order"]
    assert len(channels) == cfg["dataset"]["n_eeg_channels"]


CONFIG = load_config()

DATA_DIR = BASE_DIR / CONFIG["paths"]["data_dir"]
RATINGS_XLS_PATH = BASE_DIR / CONFIG["paths"]["ratings_xls"]

OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
FEATURES_DIR = OUTPUT_DIR / "features"      # features/<domain>/s{NN}_<domain>.csv
METADATA_DIR = OUTPUT_DIR / "metadata"
MODELS_DIR = OUTPUT_DIR / "models"
SANITY_DIR = OUTPUT_DIR / "sanity_checks"   # Week 2 feature sanity checks, per domain
VIZ_DIR = OUTPUT_DIR / "viz"
for _d in [EDA_DIR, FEATURES_DIR, METADATA_DIR, MODELS_DIR, SANITY_DIR, VIZ_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import json
    print(json.dumps(CONFIG, indent=2, default=str))
