"""Task 2: Fold metrics, results table, confusion matrices and Wilcoxon/Holm significance from the predictions."""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix

from bcc.config import MODELING_DIR as OUTPUT_DIR
from bcc.features import SANITY_DIR
from bcc.modeling import MODEL_DOMAINS, TARGETS, fold_metrics
from bcc.style import SEQUENTIAL

METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]


def _holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adjusted = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _vs_chance(folds: pd.DataFrame) -> pd.Series:
    """One-sided Wilcoxon of fold balanced accuracy against 0.5, Holm-corrected over all target x domain tests."""
    p = folds.groupby(["target", "domain"], sort=False)["balanced_accuracy"].apply(
        lambda b: wilcoxon(b - 0.5, alternative="greater").pvalue)
    return pd.Series(_holm(p.to_numpy()), index=p.index, name="p_vs_chance_holm")


def _significance(folds: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon on fold macro-F1 for every domain pair, Holm-corrected within each target."""
    tables = []
    for target in TARGETS:
        f1 = folds[folds["target"] == target].pivot(index="subject_id", columns="domain", values="macro_f1")
        rows = []
        for a, b in combinations(MODEL_DOMAINS, 2):
            test = wilcoxon(f1[a], f1[b])
            rows.append({"target": target, "domain_a": a, "domain_b": b, "median_diff": (f1[a] - f1[b]).median(),
                         "W": test.statistic, "p": test.pvalue})
        table = pd.DataFrame(rows)
        table["p_holm"] = _holm(table["p"].to_numpy())
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _plot_confusion(pred: pd.DataFrame):
    """Pooled over the 32 held-out subjects; rows = true class, normalized to sum to 1."""
    fig, axes = plt.subplots(len(TARGETS), len(MODEL_DOMAINS), figsize=(10.5, 7), constrained_layout=True)
    for r, (target, labels) in enumerate(TARGETS.items()):
        for c, domain in enumerate(MODEL_DOMAINS):
            sel = pred[(pred["target"] == target) & (pred["domain"] == domain)]
            counts = confusion_matrix(sel["y_true"], sel["y_pred"], labels=labels)
            rates = counts / counts.sum(axis=1, keepdims=True)
            ax = axes[r, c]
            im = ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{rates[i, j]:.2f}\n({counts[i, j]})", ha="center", va="center", fontsize=9,
                            color="white" if rates[i, j] > 0.55 else "#1f1f1d")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_title(f"{domain} · {target}")
            ax.spines[:].set_visible(False)
            if r == len(TARGETS) - 1:
                ax.set_xlabel("Predicted")
            if c == 0:
                ax.set_ylabel("True")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Share of true-class trials")
    fig.suptitle("AutoGluon valence/arousal confusion, 60s features, pooled over 32 LOSO folds\n"
                 "(row-normalized; counts in brackets)")
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png")
    plt.close(fig)


def run():
    pred = pd.read_csv(OUTPUT_DIR / "predictions.csv")
    folds = pd.DataFrame([
        {"target": t, "domain": d, "subject_id": s, **fold_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())}
        for (t, d, s), g in pred.groupby(["target", "domain", "subject_id"], sort=False)
    ])
    folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    results = folds.groupby(["target", "domain"], sort=False)[METRICS].agg(["mean", "std"])
    results.columns = [f"{metric}_{stat}" for metric, stat in results.columns]
    results = results.join(_vs_chance(folds))
    results.round(4).to_csv(OUTPUT_DIR / "results.csv")
    print(results.round(3).to_string())

    sig = _significance(folds)
    sig.round(4).to_csv(OUTPUT_DIR / "significance.csv", index=False)
    print(sig.round(4).to_string(index=False))

    _plot_confusion(pred)

    flagged = pd.read_csv(SANITY_DIR / "microstates" / "subject_quality.csv").set_index("subject_id")["flag_degenerate"]
    micro = folds[folds["domain"] == "microstates"].assign(flagged=lambda f: f["subject_id"].map(flagged))
    print("\nmicrostates macro-F1, flagged vs clean subjects:")
    print(micro.groupby(["target", "flagged"])["macro_f1"].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    run()
