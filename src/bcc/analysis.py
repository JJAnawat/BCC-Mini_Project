"""Load all feature domains with labels, and the normalization helpers shared by analysis stages."""
import numpy as np
import pandas as pd

from bcc.features import DOMAINS, ID_COLS, METADATA_DIR, load_domain

QUADRANTS = ["HVHA", "HVLA", "LVHA", "LVLA"]


def load_all() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Window-level labels + all 5 domains, and a {domain: feature columns} map."""
    frames = {d: load_domain(d).set_index(ID_COLS) for d in DOMAINS}
    # MSE inf (no template match at coarse scales) -> that subject's column median. In memory only.
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
