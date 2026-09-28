"""Task 2: PCA / t-SNE per domain, colored by quadrant vs subject, raw vs per-subject z-scored."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from viz_common import DOMAINS, OUTPUT_DIR, QUADRANT_COLORS, QUADRANTS, trial_means, zscore_within_subject

SUBJECT_COLORS = np.vstack([plt.cm.tab20(np.arange(20)), plt.cm.tab20b(np.arange(12))])  # 32 subjects


def _embed(x):
    return {"PCA": PCA(2).fit_transform(x),
            "t-SNE": TSNE(2, perplexity=30, init="pca", random_state=42).fit_transform(x)}


def run(df, cols):
    rows = []
    for domain in DOMAINS:
        trials = trial_means(df, cols[domain])
        versions = {
            "raw (pooled z-score)": StandardScaler().fit_transform(trials[cols[domain]]),
            "per-subject z-score": zscore_within_subject(trials, cols[domain])[cols[domain]].to_numpy(),
        }
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for r, (norm, x) in enumerate(versions.items()):
            rows.append({"domain": domain, "normalization": norm,
                         "silhouette_subject": silhouette_score(x, trials["subject_id"]),
                         "silhouette_quadrant": silhouette_score(x, trials["quadrant"])})
            for m, (method, xy) in enumerate(_embed(x).items()):
                ax_q, ax_s = axes[r, 2 * m], axes[r, 2 * m + 1]
                for q in QUADRANTS:
                    sel = (trials["quadrant"] == q).to_numpy()
                    ax_q.scatter(*xy[sel].T, s=8, color=QUADRANT_COLORS[q], alpha=0.7, lw=0, label=q)
                ax_s.scatter(*xy.T, s=8, c=SUBJECT_COLORS[trials["subject_id"] - 1], alpha=0.7, lw=0)
                ax_q.set_title(f"{method}, {norm}\nby quadrant")
                ax_s.set_title(f"{method}, {norm}\nby subject (one color each)")
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 0].legend(markerscale=2, frameon=False, fontsize=8)
        fig.suptitle(f"{domain}: 1,280 trial means ({len(cols[domain])} features)")
        fig.savefig(OUTPUT_DIR / f"embedding_{domain}.png")
        plt.close(fig)

    sil = pd.DataFrame(rows).round(3)
    sil.to_csv(OUTPUT_DIR / "silhouette.csv", index=False)
    print(sil.to_string(index=False))


if __name__ == "__main__":
    from viz_common import load_all
    run(*load_all())
