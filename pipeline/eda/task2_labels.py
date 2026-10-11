"""Task 2: Label distribution checks (valence/arousal/liking)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from bcc.config import EDA_DIR as OUTPUT_DIR
from bcc.data import LABEL_NAMES, THRESHOLD_DEFAULT, iter_subjects_corrected

# Binarization convention (stated explicitly, per task spec):
#   rating >= THRESHOLD -> "high"
#   rating <  THRESHOLD -> "low"
THRESHOLD = THRESHOLD_DEFAULT
SKEW_FLAG_PCT = 90.0  # flag a subject if either class is >= this % for a label


def _collect_all_labels():
    """Uses corrected labels (from participant_ratings.xls) -- see common.py's note
    on why the .dat files' valence/arousal cannot be used directly."""
    per_subject = {}
    rows = []
    for sid, data, labels, err in iter_subjects_corrected():
        if err is not None:
            continue
        per_subject[sid] = labels
        for trial_idx in range(labels.shape[0]):
            rows.append({
                "subject_id": sid,
                "trial": trial_idx,
                "valence": labels[trial_idx, 0],
                "arousal": labels[trial_idx, 1],
                "dominance": labels[trial_idx, 2],
                "liking": labels[trial_idx, 3],
            })
    return per_subject, pd.DataFrame(rows)


def _plot_hist(series, title, fname, color):
    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=np.arange(0.5, 9.6, 1), color=color, edgecolor="black")
    plt.axvline(THRESHOLD, color="red", linestyle="--", label=f"threshold={THRESHOLD}")
    plt.title(title)
    plt.xlabel("Rating (1-9)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close()


def run():
    print("NOTE: valence/arousal/dominance/liking are loaded from "
          "Metadata/participant_ratings.xls, not the .dat files' 'labels' array -- "
          "the .dat labels have a confirmed scale-flip bug on ~34% of trials for "
          "valence/arousal (dat_value == 9 - true_value). See common.py for details.\n")
    per_subject, df = _collect_all_labels()
    print(f"Loaded labels for {len(per_subject)} subjects, {len(df)} trials total.\n")

    # --- Raw distribution histograms (pooled) ---
    _plot_hist(df["valence"], "Valence distribution (pooled)", "task2_valence_hist.png", "steelblue")
    _plot_hist(df["arousal"], "Arousal distribution (pooled)", "task2_arousal_hist.png", "darkorange")
    _plot_hist(df["liking"], "Liking distribution (pooled, for reference)", "task2_liking_hist.png", "seagreen")

    # --- Binarize + pooled class balance ---
    df["valence_high"] = df["valence"] >= THRESHOLD
    df["arousal_high"] = df["arousal"] >= THRESHOLD

    pooled_balance = {
        "valence_high_pct": 100 * df["valence_high"].mean(),
        "valence_low_pct": 100 * (~df["valence_high"]).mean(),
        "arousal_high_pct": 100 * df["arousal_high"].mean(),
        "arousal_low_pct": 100 * (~df["arousal_high"]).mean(),
    }
    print("Pooled binarized class balance (threshold=%.1f, >=high):" % THRESHOLD)
    for k, v in pooled_balance.items():
        print(f"  {k}: {v:.1f}%")

    liking_low_pct = 100 * (df["liking"] < THRESHOLD).mean()
    print(f"\nLiking (reference only): {liking_low_pct:.1f}% below {THRESHOLD} "
          f"(DEAP paper reports ~1/3 below 5, ~2/3 above)")

    # --- Per-subject class balance table ---
    per_subj_rows = []
    for sid, g in df.groupby("subject_id"):
        v_high_pct = 100 * g["valence_high"].mean()
        a_high_pct = 100 * g["arousal_high"].mean()
        flags = []
        if v_high_pct >= SKEW_FLAG_PCT or v_high_pct <= (100 - SKEW_FLAG_PCT):
            flags.append("valence_skew")
        if a_high_pct >= SKEW_FLAG_PCT or a_high_pct <= (100 - SKEW_FLAG_PCT):
            flags.append("arousal_skew")
        pear_r, pear_p = stats.pearsonr(g["valence"], g["arousal"])
        spear_r, spear_p = stats.spearmanr(g["valence"], g["arousal"])
        per_subj_rows.append({
            "subject_id": sid,
            "pct_high_valence": round(v_high_pct, 1),
            "pct_high_arousal": round(a_high_pct, 1),
            "valence_arousal_pearson_r": round(pear_r, 3),
            "valence_arousal_spearman_r": round(spear_r, 3),
            "flagged_reason": ";".join(flags),
        })
    per_subj_df = pd.DataFrame(per_subj_rows)
    per_subj_df.to_csv(OUTPUT_DIR / "task2_per_subject_balance.csv", index=False)

    print(f"\nPer-subject class balance saved to task2_per_subject_balance.csv")
    flagged = per_subj_df[per_subj_df["flagged_reason"] != ""]
    if len(flagged):
        print(f"Subjects with extreme (>{SKEW_FLAG_PCT:.0f}/{100-SKEW_FLAG_PCT:.0f}) skew:")
        print(flagged[["subject_id", "pct_high_valence", "pct_high_arousal", "flagged_reason"]]
              .to_string(index=False))
    else:
        print(f"No subjects exceed the {SKEW_FLAG_PCT:.0f}/{100-SKEW_FLAG_PCT:.0f} skew threshold.")

    # --- Pooled valence-arousal correlation ---
    pooled_pear_r, pooled_pear_p = stats.pearsonr(df["valence"], df["arousal"])
    pooled_spear_r, pooled_spear_p = stats.spearmanr(df["valence"], df["arousal"])
    print(f"\nPooled valence-arousal correlation: "
          f"Pearson r={pooled_pear_r:.3f} (p={pooled_pear_p:.2e}), "
          f"Spearman r={pooled_spear_r:.3f} (p={pooled_spear_p:.2e})")

    # Scatter for visual sanity check
    plt.figure(figsize=(5, 5))
    plt.scatter(df["valence"], df["arousal"], alpha=0.15, s=15)
    plt.axvline(THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    plt.axhline(THRESHOLD, color="red", linestyle="--", linewidth=0.8)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title(f"Valence vs Arousal (pooled)\nPearson r={pooled_pear_r:.3f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_valence_arousal_scatter.png", dpi=150)
    plt.close()

    summary = {
        "pooled_valence_high_pct": pooled_balance["valence_high_pct"],
        "pooled_arousal_high_pct": pooled_balance["arousal_high_pct"],
        "pooled_pearson_r": pooled_pear_r,
        "pooled_spearman_r": pooled_spear_r,
        "n_subjects_flagged_skew": len(flagged),
    }
    return per_subj_df, summary


if __name__ == "__main__":
    run()
