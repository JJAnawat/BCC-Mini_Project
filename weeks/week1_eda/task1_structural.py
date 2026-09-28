"""Task 1: Structural sanity checks for the DEAP dataset."""
import json

import numpy as np
import pandas as pd

from common import (
    EEG_CHANNELS,
    FACE_VIDEO_GAP_SUBJECTS,
    N_CHANNELS_TOTAL,
    N_SAMPLES,
    N_SUBJECTS,
    N_TRIALS,
    OUTPUT_DIR,
    SYMMETRIC_PAIRS,
    get_corrected_labels,
    iter_subjects,
)

EXPECTED_DATA_SHAPE = (N_TRIALS, N_CHANNELS_TOTAL, N_SAMPLES)
EXPECTED_LABELS_SHAPE = (N_TRIALS, 4)


def run():
    rows = []
    for sid, data, labels, err in iter_subjects():
        row = {
            "subject_id": sid,
            "load_ok": err is None,
            "data_shape": None,
            "labels_shape": None,
            "shape_ok": False,
            "has_nan": None,
            "n_allzero_trials": None,
            "n_trials_present": None,
            "n_labels_out_of_range_1_9": None,
            "n_valence_arousal_corrected": None,
            "flagged_reason": "",
        }
        if err is not None:
            row["flagged_reason"] = f"load error: {err}"
            rows.append(row)
            print(f"[s{sid:02d}] FAILED TO LOAD: {err}")
            continue

        data_shape_ok = data.shape == EXPECTED_DATA_SHAPE
        labels_shape_ok = labels.shape == EXPECTED_LABELS_SHAPE
        row["data_shape"] = str(data.shape)
        row["labels_shape"] = str(labels.shape)
        row["shape_ok"] = bool(data_shape_ok and labels_shape_ok)
        row["n_trials_present"] = int(data.shape[0]) if data.ndim >= 1 else 0

        has_nan = bool(np.isnan(data).any()) or bool(np.isnan(labels).any())
        row["has_nan"] = has_nan

        # all-zero trial = a trial where every channel/sample is exactly 0
        if data_shape_ok:
            allzero_mask = np.all(data == 0, axis=(1, 2))
            n_allzero = int(allzero_mask.sum())
        else:
            n_allzero = int(np.all(data == 0, axis=tuple(range(1, data.ndim))).sum()) if data.ndim >= 2 else 0
        row["n_allzero_trials"] = n_allzero

        # Known bug: .dat labels[:, 0:2] (valence, arousal) fall outside the
        # documented 1-9 range and are mirrored (9 - true_value) on ~34% of
        # trials for ~1/3 of trials; see common.py's note above
        # get_corrected_labels(). Flag it here rather than only downstream.
        n_out_of_range = int(((labels < 1) | (labels > 9)).sum()) if labels_shape_ok else 0
        row["n_labels_out_of_range_1_9"] = n_out_of_range
        try:
            corrected = get_corrected_labels(sid)
            n_corrected = int(
                (np.abs(labels[:, 0] - corrected[:, 0]) > 0.01).sum()
                + (np.abs(labels[:, 1] - corrected[:, 1]) > 0.01).sum()
            )
        except Exception:  # noqa: BLE001 - ratings xls issues shouldn't break the structural check
            n_corrected = None
        row["n_valence_arousal_corrected"] = n_corrected

        reasons = []
        if not data_shape_ok:
            reasons.append(f"data shape {data.shape} != {EXPECTED_DATA_SHAPE}")
        if not labels_shape_ok:
            reasons.append(f"labels shape {labels.shape} != {EXPECTED_LABELS_SHAPE}")
        if has_nan:
            reasons.append("contains NaN")
        if n_allzero > 0:
            reasons.append(f"{n_allzero} all-zero trial(s)")
        # NOTE: n_out_of_range (raw .dat valence/arousal outside 1-9) is NOT
        # added to `reasons` -- it's the known, already-corrected scale-flip
        # bug (see common.py), not a per-subject anomaly worth flagging in
        # eda_summary.csv. It's still recorded in n_labels_out_of_range_1_9
        # and reported in aggregate below.
        if sid in FACE_VIDEO_GAP_SUBJECTS and row["n_trials_present"] != N_TRIALS:
            reasons.append(
                f"EEG trial count {row['n_trials_present']} != {N_TRIALS} "
                "(face-video gap subject)"
            )
        row["flagged_reason"] = "; ".join(reasons)

        status = "OK" if not reasons else "FLAGGED"
        print(f"[s{sid:02d}] {status} data={data.shape} labels={labels.shape} "
              f"nan={has_nan} allzero_trials={n_allzero}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "task1_structural.csv", index=False)

    total_subjects = len(df)
    ok_subjects = int(df["shape_ok"].sum()) if "shape_ok" in df else 0
    total_trials_present = int(df["n_trials_present"].fillna(0).sum())
    print(f"\nSubjects checked: {total_subjects} | shape OK: {ok_subjects}")
    print(f"Total trials present: {total_trials_present} (expected {N_SUBJECTS * N_TRIALS})")

    total_corrected = int(df["n_valence_arousal_corrected"].fillna(0).sum())
    total_out_of_range = int(df["n_labels_out_of_range_1_9"].fillna(0).sum())
    print(f"\nRaw .dat label values outside 1-9 range: {total_out_of_range} "
          f"(valence/arousal scale-flip bug)")
    print(f"Valence/arousal values corrected via participant_ratings.xls: "
          f"{total_corrected} trial-values across all subjects "
          f"(see task2_labels.py, which uses the corrected values throughout)")

    print("\nFace-video gap subjects (3, 5, 11, 14) EEG trial-count check:")
    gap_df = df[df["subject_id"].isin(FACE_VIDEO_GAP_SUBJECTS)]
    for _, r in gap_df.iterrows():
        ok = r["n_trials_present"] == N_TRIALS
        print(f"  s{int(r['subject_id']):02d}: n_trials={r['n_trials_present']} "
              f"-> {'EEG unaffected (40/40 trials present)' if ok else 'MISMATCH'}")

    # Save the symmetric electrode pair list for reuse in feature extraction.
    pairs_out = {
        "eeg_channel_order": EEG_CHANNELS,
        "symmetric_pairs": [{"left": a, "right": b} for a, b in SYMMETRIC_PAIRS],
        "note": (
            "Channel order is the documented DEAP/Biosemi 32-channel Geneva "
            "order, assumed (not verified against in-file metadata, since "
            "none ships with this dataset copy)."
        ),
    }
    with open(OUTPUT_DIR / "symmetric_pairs.json", "w") as f:
        json.dump(pairs_out, f, indent=2)
    print(f"\nSaved {len(SYMMETRIC_PAIRS)} symmetric electrode pairs to "
          f"{OUTPUT_DIR / 'symmetric_pairs.json'}")

    flagged = df[df["flagged_reason"] != ""]
    return df, flagged


if __name__ == "__main__":
    run()
