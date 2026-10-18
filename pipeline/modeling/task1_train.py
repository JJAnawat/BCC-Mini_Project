"""Task 1: Nested-LOSO AutoGluon per target and feature domain (60s set), out-of-fold valence/arousal predictions."""
import time

import pandas as pd

from bcc.analysis import zscore_within_subject
from bcc.config import MODELING_DIR as OUTPUT_DIR
from bcc.modeling import MODEL_DOMAINS, TARGETS, binary_labels, loso_autogluon

CACHE_DIR = OUTPUT_DIR / "predictions"   # one file per (target, domain), so an interrupted run resumes


def run(df, cols):
    CACHE_DIR.mkdir(exist_ok=True)
    groups = df["subject_id"].to_numpy()
    for target in TARGETS:
        y = binary_labels(df["quadrant"], target)
        for domain in MODEL_DOMAINS:
            pred_path, board_path = CACHE_DIR / f"{target}_{domain}.csv", CACHE_DIR / f"{target}_{domain}_leaderboard.csv"
            if pred_path.exists() and board_path.exists():
                print(f"  {target:8s} {domain:12s} cached")
                continue
            start = time.time()
            x = zscore_within_subject(df, cols[domain])[cols[domain]]
            y_pred, boards = loso_autogluon(x, y, groups)
            boards.assign(target=target, domain=domain).to_csv(board_path, index=False)
            df[["subject_id", "trial_idx"]].assign(target=target, domain=domain, y_true=y, y_pred=y_pred).to_csv(
                pred_path, index=False)
            print(f"  {target:8s} {domain:12s} {time.time() - start:.0f}s")
    keys = [f"{t}_{d}" for t in TARGETS for d in MODEL_DOMAINS]
    pd.concat(pd.read_csv(CACHE_DIR / f"{k}.csv") for k in keys).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.concat(pd.read_csv(CACHE_DIR / f"{k}_leaderboard.csv") for k in keys).to_csv(OUTPUT_DIR / "leaderboards.csv", index=False)


if __name__ == "__main__":
    from bcc.analysis import load_all
    run(*load_all("60s"))
