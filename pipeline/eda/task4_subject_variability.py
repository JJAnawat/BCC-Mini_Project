"""Task 4: Subject-level variability in band power vs. label skew."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

from bcc.config import EDA_DIR as OUTPUT_DIR
from bcc.data import BAND_DEFS, EEG_CHANNELS, FS, N_EEG_CHANNELS, iter_subjects

POWER_OUTLIER_SD_MULT = 2.5  # flag subject if mean alpha power beyond this many SDs from group mean


def _band_power(psd_freqs, psd_vals, band):
    lo, hi = band
    mask = (psd_freqs >= lo) & (psd_freqs <= hi)
    return np.trapezoid(psd_vals[mask], psd_freqs[mask])


def _compute_band_powers():
    """Return long-form df: subject_id, channel, band, power (averaged across trials)."""
    rows = []
    for sid, data, labels, err in iter_subjects():
        if err is not None:
            continue
        eeg = data[:, :N_EEG_CHANNELS, :]  # (trials, 32, samples)
        n_trials = eeg.shape[0]
        for ci, ch_name in enumerate(EEG_CHANNELS):
            band_powers = {b: [] for b in BAND_DEFS}
            for t in range(n_trials):
                f, pxx = welch(eeg[t, ci, :], fs=FS, nperseg=min(512, eeg.shape[2]))
                for band_name, band_range in BAND_DEFS.items():
                    band_powers[band_name].append(_band_power(f, pxx, band_range))
            for band_name, vals in band_powers.items():
                rows.append({
                    "subject_id": sid,
                    "channel": ch_name,
                    "band": band_name,
                    "mean_power": float(np.mean(vals)),
                })
        print(f"  Computed band power for s{sid:02d}")
    return pd.DataFrame(rows)


def run():
    print("Computing per-subject/channel band power (this may take a while)...")
    bp_df = _compute_band_powers()
    bp_df.to_csv(OUTPUT_DIR / "task4_band_power.csv", index=False)

    # Per-subject mean alpha power (averaged across channels) for the boxplot / outlier check
    alpha_df = bp_df[bp_df["band"] == "alpha"]
    subj_alpha_mean = alpha_df.groupby("subject_id")["mean_power"].mean()

    plt.figure(figsize=(12, 5))
    order = sorted(alpha_df["subject_id"].unique())
    sns.boxplot(data=alpha_df, x="subject_id", y="mean_power", order=order, color="lightsteelblue")
    plt.yscale("log")
    plt.xlabel("Subject")
    plt.ylabel("Alpha power (log scale, per channel)")
    plt.title("Per-subject alpha power distribution across channels")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task4_alpha_boxplot.png", dpi=150)
    plt.close()

    grp_mean, grp_std = subj_alpha_mean.mean(), subj_alpha_mean.std()
    power_outliers = subj_alpha_mean[
        (subj_alpha_mean - grp_mean).abs() > POWER_OUTLIER_SD_MULT * grp_std
    ]
    print(f"\nGroup mean alpha power: {grp_mean:.2f} (SD {grp_std:.2f})")
    if len(power_outliers):
        print(f"Subjects with alpha power beyond {POWER_OUTLIER_SD_MULT} SD of group mean:")
        print(power_outliers.to_string())
    else:
        print(f"No subjects exceed the {POWER_OUTLIER_SD_MULT} SD alpha-power outlier threshold.")

    # --- Cross-reference with Task 2 per-subject label skew ---
    label_balance_path = OUTPUT_DIR / "task2_per_subject_balance.csv"
    cross_df = None
    if label_balance_path.exists():
        label_df = pd.read_csv(label_balance_path).set_index("subject_id")
        cross_df = label_df.copy()
        cross_df["mean_alpha_power"] = subj_alpha_mean
        cross_df["alpha_power_outlier"] = cross_df.index.isin(power_outliers.index)
        # empty flagged_reason cells round-trip through CSV as NaN, and NaN != "" is
        # always True -- fillna first or every subject reads as "skew flagged".
        cross_df["label_skew_flagged"] = cross_df["flagged_reason"].fillna("") != ""
        cross_df.to_csv(OUTPUT_DIR / "task4_cross_reference.csv")

        both_flagged = cross_df[cross_df["alpha_power_outlier"] & cross_df["label_skew_flagged"]]
        print(f"\nSubjects flagged on BOTH signal (alpha power) and label skew: "
              f"{list(both_flagged.index) if len(both_flagged) else 'none'}")
    else:
        print("\ntask2_per_subject_balance.csv not found -- run task2 first for full cross-reference.")

    summary = {
        "n_alpha_power_outlier_subjects": len(power_outliers),
        "alpha_power_outlier_subjects": list(power_outliers.index),
    }
    return bp_df, cross_df, summary


if __name__ == "__main__":
    run()
