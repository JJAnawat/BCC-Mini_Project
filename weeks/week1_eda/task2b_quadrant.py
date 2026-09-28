"""Task 2b: Valence x Arousal quadrant class counts (HVHA/HVLA/LVHA/LVLA).

This is the Week 1 "quadrant cross-tab" deliverable. Answers the actual
question that decides whether quadrant classification needs imbalance
handling -- not an estimate derived from the two marginal (valence, arousal)
binarizations separately.
"""
import pandas as pd

from common import OUTPUT_DIR, THRESHOLD_DEFAULT, iter_subjects_corrected

THRESHOLD = THRESHOLD_DEFAULT  # >=5 = high, <5 = low (same convention as Task 2)


def _quadrant(v, a):
    return f"{'H' if v >= THRESHOLD else 'L'}V{'H' if a >= THRESHOLD else 'L'}A"


def run():
    rows = []
    for sid, data, labels, err in iter_subjects_corrected():
        if err is not None:
            continue
        for t in range(labels.shape[0]):
            v, a = labels[t, 0], labels[t, 1]
            rows.append({"subject_id": sid, "trial": t, "valence": v, "arousal": a,
                         "quadrant": _quadrant(v, a)})
    df = pd.DataFrame(rows)

    pooled_counts = df["quadrant"].value_counts().reindex(
        ["HVHA", "HVLA", "LVHA", "LVLA"], fill_value=0
    )
    pooled_pct = (pooled_counts / len(df) * 100).round(1)
    pooled_summary = pd.DataFrame({"count": pooled_counts, "pct": pooled_pct})
    pooled_summary.to_csv(OUTPUT_DIR / "task2b_quadrant_pooled.csv")

    print(f"Pooled quadrant counts (threshold={THRESHOLD}, n={len(df)} trials):")
    print(pooled_summary.to_string())
    ratio = pooled_counts.max() / pooled_counts.min()
    print(f"\nMax/min quadrant ratio: {ratio:.2f}x "
          f"({pooled_counts.idxmax()}={pooled_counts.max()} vs "
          f"{pooled_counts.idxmin()}={pooled_counts.min()})")

    # Per-subject quadrant counts -- catches degenerate folds a pooled view can't see.
    per_subj = (
        df.groupby(["subject_id", "quadrant"]).size().unstack(fill_value=0)
        .reindex(columns=["HVHA", "HVLA", "LVHA", "LVLA"], fill_value=0)
    )
    per_subj.to_csv(OUTPUT_DIR / "task2b_quadrant_per_subject.csv")

    empty_quadrant_subjects = per_subj[(per_subj == 0).any(axis=1)]
    print(f"\nSubjects with at least one empty quadrant (0 trials): "
          f"{list(empty_quadrant_subjects.index) if len(empty_quadrant_subjects) else 'none'}")
    if len(empty_quadrant_subjects):
        print(empty_quadrant_subjects.to_string())

    min_nonzero_cell = per_subj[per_subj > 0].min().min()
    print(f"\nSmallest nonzero per-subject quadrant cell: {min_nonzero_cell} trials")

    return pooled_summary, per_subj


if __name__ == "__main__":
    run()
