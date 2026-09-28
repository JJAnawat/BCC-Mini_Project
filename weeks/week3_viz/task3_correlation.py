"""Task 3: Cross-domain Spearman correlation (all 680 features) and a 5x5 domain redundancy summary."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from viz_common import DIVERGING, DOMAINS, OUTPUT_DIR, zscore_within_subject


def _full_heatmap(corr, sizes):
    edges = np.cumsum([0, *sizes])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="#0b0b0b", lw=0.8)
        ax.axvline(e - 0.5, color="#0b0b0b", lw=0.8)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(centers, DOMAINS)
    ax.set_yticks(centers, DOMAINS)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    ax.set_title("Feature correlation across all 5 domains (19,200 windows, per-subject z-scored)")
    fig.savefig(OUTPUT_DIR / "correlation_all_features.png")
    plt.close(fig)


def _summary_heatmap(summary):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(summary.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(summary.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="#fcfcfb" if v > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(DOMAINS)), DOMAINS)
    ax.set_yticks(range(len(DOMAINS)), DOMAINS)
    ax.set_xlabel("best match searched in")
    ax.set_ylabel("feature from")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean best-match |r|")
    ax.set_title("Domain redundancy: for each feature, its strongest |r|\nwith any feature of the other domain, averaged")
    fig.savefig(OUTPUT_DIR / "correlation_domain_summary.png")
    plt.close(fig)


def run(df, cols):
    features = [c for d in DOMAINS for c in cols[d]]
    x = zscore_within_subject(df, features)[features].to_numpy()
    corr = np.nan_to_num(spearmanr(x).statistic)
    _full_heatmap(corr, [len(cols[d]) for d in DOMAINS])

    idx = np.split(np.arange(len(features)), np.cumsum([len(cols[d]) for d in DOMAINS])[:-1])
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, np.nan)  # within-domain: best match excludes the feature itself
    summary = pd.DataFrame([[np.nanmax(abs_corr[np.ix_(a, b)], axis=1).mean() for b in idx] for a in idx],
                           index=DOMAINS, columns=DOMAINS)
    summary.round(3).to_csv(OUTPUT_DIR / "correlation_domain_summary.csv")
    _summary_heatmap(summary)
    print(summary.round(2).to_string())


if __name__ == "__main__":
    from viz_common import load_all
    run(*load_all())
