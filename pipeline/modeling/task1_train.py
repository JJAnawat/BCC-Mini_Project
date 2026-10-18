"""Task 1: Nested-LOSO SVM and RF per feature domain (60s set), out-of-fold quadrant predictions."""
import time

import pandas as pd

from bcc.analysis import zscore_within_subject
from bcc.config import MODELING_DIR as OUTPUT_DIR
from bcc.features import DOMAINS
from bcc.modeling import MODELS, nested_loso


def run(df, cols):
    y, groups = df["quadrant"].to_numpy(), df["subject_id"].to_numpy()
    preds, params = [], []
    for domain in DOMAINS:
        x = zscore_within_subject(df, cols[domain])[cols[domain]].to_numpy()
        for model in MODELS:
            start = time.time()
            y_pred, best = nested_loso(x, y, groups, model)
            preds.append(df[["subject_id", "trial_idx"]].assign(domain=domain, model=model, y_true=y, y_pred=y_pred))
            params.append(best.assign(domain=domain, model=model))
            print(f"  {domain:12s} {model}: {time.time() - start:.0f}s")
    pd.concat(preds).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.concat(params).to_csv(OUTPUT_DIR / "best_params.csv", index=False)


if __name__ == "__main__":
    from bcc.analysis import load_all
    run(*load_all("60s"))
