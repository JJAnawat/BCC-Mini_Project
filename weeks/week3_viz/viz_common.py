"""Shared loading, normalization and plot style for the Week 3 visualization scripts."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "week2_features"))
from features_common import (  # noqa: E402,F401 - re-exported for the task modules
    BAND_NAMES, DOMAINS, EEG_CHANNELS, ID_COLS, METADATA_DIR, SYMMETRIC_PAIRS, load_domain, montage_info,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

QUADRANTS = ["HVHA", "HVLA", "LVHA", "LVLA"]
# Categorical slots validated all-pairs for CVD separation (dataviz skill's validate_palette.js).
QUADRANT_COLORS = dict(zip(QUADRANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#9e2a2a"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9,
})


def load_all() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Window-level labels + all 5 domains, and a {domain: feature columns} map."""
    frames = {d: load_domain(d).set_index(ID_COLS) for d in DOMAINS}
    # MSE inf (no template match at coarse scales) -> that subject's column median. Week 3 only.
    mse = frames["mse"].replace([np.inf, -np.inf], np.nan)
    frames["mse"] = mse.fillna(mse.groupby(level="subject_id").transform("median"))
    cols = {d: list(f.columns) for d, f in frames.items()}
    feats = pd.concat(frames.values(), axis=1, verify_integrity=True).copy()  # .copy() consolidates blocks
    df = pd.read_csv(METADATA_DIR / "metadata_window_level.csv").merge(feats.reset_index(), on=ID_COLS, validate="1:1")
    return df, cols


def trial_means(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Window rows -> one row per trial (labels are per trial)."""
    return df.groupby(["subject_id", "trial_idx", "quadrant"], as_index=False)[feature_cols].mean()


def zscore_within_subject(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-subject z-score; columns constant within a subject (degenerate microstates) become 0."""
    grouped = df.groupby("subject_id")[feature_cols]
    std = grouped.transform("std")
    out = df.copy()
    out[feature_cols] = ((df[feature_cols] - grouped.transform("mean")) / std.where(std > 1e-12)).fillna(0.0)
    return out
