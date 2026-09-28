"""Typed loader for the locked project config.yaml.

Every downstream feature-extraction / classification script should do
`from config import CONFIG` (or call load_config() directly) instead of
redefining thresholds, bands, windowing, or exclusion lists -- that's the
whole point of freezing these decisions after Week 1's EDA.
"""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


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


if __name__ == "__main__":
    import json
    print(json.dumps(CONFIG, indent=2, default=str))
