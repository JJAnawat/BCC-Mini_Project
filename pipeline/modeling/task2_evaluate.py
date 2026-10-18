"""Task 2: Fold metrics, results table, confusion matrices and Wilcoxon/Holm significance from the predictions."""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

from bcc.analysis import QUADRANTS
from bcc.config import MODELING_DIR as OUTPUT_DIR
from bcc.features import DOMAINS, SANITY_DIR
from bcc.modeling import MODELS, fold_metrics
from bcc.style import SEQUENTIAL

METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "valence_accuracy", "arousal_accuracy"]


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _significance(folds: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon on fold macro-F1 for every domain pair, Holm-corrected within each model."""
    tables = []
    for model in MODELS:
        f1 = folds[folds["model"] == model].pivot(index="subject_id", columns="domain", values="macro_f1")
        rows = []
        for a, b in combinations(DOMAINS, 2):
            test = wilcoxon(f1[a], f1[b])
            rows.append({"model": model, "domain_a": a, "domain_b": b, "median_diff": (f1[a] - f1[b]).median(),
                         "W": test.statistic, "p": test.pvalue})
        table = pd.DataFrame(rows)
        table["p_holm"] = _holm(table["p"].to_numpy())
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _plot_confusion(pred: pd.DataFrame):
    """Pooled over the 32 held-out subjects; rows = true quadrant, normalized to sum to 1."""
    fig, axes = plt.subplots(len(MODELS), len(DOMAINS), figsize=(17, 7.2), constrained_layout=True)
    for r, model in enumerate(MODELS):
        for c, domain in enumerate(DOMAINS):
            sel = pred[(pred["model"] == model) & (pred["domain"] == domain)]
            counts = confusion_matrix(sel["y_true"], sel["y_pred"], labels=QUADRANTS)
            rates = counts / counts.sum(axis=1, keepdims=True)
            ax = axes[r, c]
            im = ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)
            for i in range(len(QUADRANTS)):
                for j in range(len(QUADRANTS)):
                    ax.text(j, i, f"{rates[i, j]:.2f}\n({counts[i, j]})", ha="center", va="center", fontsize=7.5,
                            color="white" if rates[i, j] > 0.55 else "#1f1f1d")
            ax.set_xticks(range(4), QUADRANTS)
            ax.set_yticks(range(4), QUADRANTS)
            ax.set_title(f"{domain} · {model.upper()}")
            ax.spines[:].set_visible(False)
            if r == len(MODELS) - 1:
                ax.set_xlabel("Predicted")
            if c == 0:
                ax.set_ylabel("True")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Share of true-class trials")
    fig.suptitle("Quadrant confusion, 60s features, pooled over 32 LOSO folds (row-normalized; counts in brackets)")
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png")
    plt.close(fig)


def run():
    pred = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    folds = pd.DataFrame([
        {"domain": d, "model": m, "subject_id": s, **fold_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())}
        for (d, m, s), g in pred.groupby(["domain", "model", "subject_id"], sort=False)
    ])
    folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    results = folds.groupby(["model", "domain"], sort=False)[METRICS].agg(["mean", "std"])
    results.columns = [f"{metric}_{stat}" for metric, stat in results.columns]
    results.round(4).to_csv(OUTPUT_DIR / "results.csv")
    print(results.round(3).to_string())

    sig = _significance(folds)
    sig.round(4).to_csv(OUTPUT_DIR / "significance.csv", index=False)
    print(sig.round(4).to_string(index=False))

    _plot_confusion(pred)

    flagged = pd.read_csv(SANITY_DIR / "microstates" / "subject_quality.csv").set_index("subject_id")["flag_degenerate"]
    micro = folds[folds["domain"] == "microstates"].assign(flagged=lambda f: f["subject_id"].map(flagged))
    print("\nmicrostates macro-F1, flagged vs clean subjects:")
    print(micro.groupby(["model", "flagged"])["macro_f1"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    run()
