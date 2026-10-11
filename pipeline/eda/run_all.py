"""Run the full DEAP EDA pipeline (Tasks 1-4) and build eda_summary.csv."""
import pandas as pd

from bcc.config import EDA_DIR as OUTPUT_DIR
import task1_structural
import task2_labels
import task2b_quadrant
import task3_signal_quality
import task4_subject_variability


def main():
    print("=" * 70)
    print("TASK 1: Structural sanity checks")
    print("=" * 70)
    t1_df, t1_flagged = task1_structural.run()

    print("\n" + "=" * 70)
    print("TASK 2: Label distribution")
    print("=" * 70)
    t2_per_subj, t2_summary = task2_labels.run()

    print("\n" + "=" * 70)
    print("TASK 2b: Valence x Arousal quadrant counts")
    print("=" * 70)
    task2b_quadrant.run()

    print("\n" + "=" * 70)
    print("TASK 3: Signal quality")
    print("=" * 70)
    t3_amp_df, t3_summary = task3_signal_quality.run()

    print("\n" + "=" * 70)
    print("TASK 4: Subject-level variability")
    print("=" * 70)
    t4_bp_df, t4_cross_df, t4_summary = task4_subject_variability.run()

    print("\n" + "=" * 70)
    print("Building combined eda_summary.csv")
    print("=" * 70)

    summary = t1_df[["subject_id", "shape_ok", "has_nan", "n_allzero_trials", "flagged_reason"]].copy()
    summary = summary.rename(columns={"flagged_reason": "task1_flag"})

    summary = summary.merge(
        t2_per_subj[["subject_id", "pct_high_valence", "pct_high_arousal",
                     "valence_arousal_pearson_r", "flagged_reason"]]
        .rename(columns={"flagged_reason": "task2_flag"}),
        on="subject_id", how="left",
    )

    t3_per_subj = t3_amp_df.groupby("subject_id").agg(
        n_flat_channels=("flagged_flat", "sum"),
        n_outlier_channels=("flagged_outlier", "sum"),
    ).reset_index()
    summary = summary.merge(t3_per_subj, on="subject_id", how="left")

    if t4_cross_df is not None:
        t4_join = t4_cross_df.reset_index()[["subject_id", "mean_alpha_power", "alpha_power_outlier"]]
        summary = summary.merge(t4_join, on="subject_id", how="left")

    def combine_flags(row):
        flags = []
        if row.get("task1_flag"):
            flags.append(f"T1:{row['task1_flag']}")
        if row.get("task2_flag"):
            flags.append(f"T2:{row['task2_flag']}")
        if row.get("n_flat_channels", 0) and row["n_flat_channels"] > 0:
            flags.append(f"T3:flat_ch={int(row['n_flat_channels'])}")
        if row.get("n_outlier_channels", 0) and row["n_outlier_channels"] > 0:
            flags.append(f"T3:outlier_ch={int(row['n_outlier_channels'])}")
        if row.get("alpha_power_outlier"):
            flags.append("T4:alpha_power_outlier")
        return "; ".join(flags)

    summary["flagged_reason"] = summary.apply(combine_flags, axis=1)
    summary.to_csv(OUTPUT_DIR / "eda_summary.csv", index=False)

    print(f"\nSaved combined summary to {OUTPUT_DIR / 'eda_summary.csv'}")
    flagged_any = summary[summary["flagged_reason"] != ""]
    print(f"\n{len(flagged_any)}/{len(summary)} subjects flagged for at least one reason:")
    if len(flagged_any):
        print(flagged_any[["subject_id", "flagged_reason"]].to_string(index=False))
    else:
        print("None.")

    print("\nDone. See outputs/eda/ for all CSVs and plots.")


if __name__ == "__main__":
    main()
