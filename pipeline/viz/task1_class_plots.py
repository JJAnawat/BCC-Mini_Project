"""Task 1: Quadrant boxplots of a priori features, and DE / DASM topomaps by quadrant."""
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import kruskal

from bcc.analysis import QUADRANTS, trial_means, zscore_within_subject
from bcc.config import VIZ_DIR as OUTPUT_DIR
from bcc.features import BAND_NAMES, EEG_CHANNELS, SYMMETRIC_PAIRS, montage_info
from bcc.style import DIVERGING, QUADRANT_COLORS

# Chosen before looking at results: classic valence/arousal markers plus one pair per remaining domain.
BOX_FEATURES = {
    "de": ["Pz_alpha_de", "T7_gamma_de"],
    "hjorth": ["Fz_mobility", "Fz_complexity"],
    "asymmetry": ["F3-F4_alpha_dasm", "T7-T8_gamma_dasm"],
    "mse": ["Fz_scale3_mse", "Fz_scale8_mse"],
    "microstates": ["map0_coverage", "map1_coverage"],
}


def _boxplots(trials):
    fig, axes = plt.subplots(2, len(BOX_FEATURES), figsize=(15, 7), sharey=True, layout="constrained")
    for col, (domain, feats) in enumerate(BOX_FEATURES.items()):
        for row, feat in enumerate(feats):
            ax = axes[row, col]
            groups = [trials.loc[trials["quadrant"] == q, feat] for q in QUADRANTS]
            h = kruskal(*groups).statistic
            eta2 = (h - len(QUADRANTS) + 1) / (len(trials) - len(QUADRANTS))
            box = ax.boxplot(groups, tick_labels=QUADRANTS, patch_artist=True, widths=0.6,
                             medianprops={"color": "#0b0b0b"}, flierprops={"markersize": 2, "alpha": 0.4})
            for patch, q in zip(box["boxes"], QUADRANTS):
                patch.set(facecolor=QUADRANT_COLORS[q], alpha=0.75, edgecolor="#fcfcfb")
            ax.set_title(f"{domain}: {feat}\nKruskal-Wallis η² = {eta2:.3f}", fontsize=9)
            ax.set_ylim(-4, 4)
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score within subject")
    fig.suptitle("Trial-mean features by quadrant (1,280 trials; y-axis clipped at ±4 z)")
    fig.savefig(OUTPUT_DIR / "boxplots_by_quadrant.png")
    plt.close(fig)


def _topomap_grid(trials, info, columns_for_band, title, fname):
    """Rows = bands, columns = quadrants; value = quadrant mean of within-subject z-scores."""
    fig, axes = plt.subplots(len(BAND_NAMES), len(QUADRANTS), figsize=(9, 9.5))
    for r, band in enumerate(BAND_NAMES):
        means = trials.groupby("quadrant")[columns_for_band(band)].mean()
        vmax = np.abs(means.to_numpy()).max()
        for c, q in enumerate(QUADRANTS):
            im, _ = mne.viz.plot_topomap(means.loc[q].to_numpy(), info, axes=axes[r, c], cmap=DIVERGING,
                                         vlim=(-vmax, vmax), contours=0, extrapolate="local", show=False)
            if r == 0:
                axes[r, c].set_title(q)
        axes[r, 0].set_ylabel(band, rotation=0, labelpad=25)
        fig.colorbar(im, ax=axes[r, :], shrink=0.8, label="mean z")
    fig.suptitle(title)
    fig.savefig(OUTPUT_DIR / fname)
    plt.close(fig)


def run(df, cols):
    feats = list(dict.fromkeys([f for fs in BOX_FEATURES.values() for f in fs] + cols["de"] + cols["asymmetry"]))
    trials = zscore_within_subject(trial_means(df, feats), feats)
    _boxplots(trials)

    info = montage_info()
    _topomap_grid(trials, info, lambda b: [f"{ch}_{b}_de" for ch in EEG_CHANNELS],
                  "DE by quadrant (deviation from each subject's own mean)", "topomap_de.png")
    left_info = mne.pick_info(info, [EEG_CHANNELS.index(left) for left, _ in SYMMETRIC_PAIRS])
    _topomap_grid(trials, left_info, lambda b: [f"{l}-{r}_{b}_dasm" for l, r in SYMMETRIC_PAIRS],
                  "DASM (left − right DE) by quadrant, drawn at the left electrode", "topomap_dasm.png")


if __name__ == "__main__":
    from bcc.analysis import load_all
    run(*load_all())
