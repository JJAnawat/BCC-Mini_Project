"""Task 0: Window-level label/quadrant metadata, the join table for every feature domain."""
import numpy as np
import pandas as pd

from bcc.features import METADATA_DIR, N_WINDOWS, iter_subjects_corrected, quadrant, window_ids


def run():
    frames = []
    for sid, _data, labels, err in iter_subjects_corrected():
        if err is not None:
            raise err
        df = window_ids(sid)
        df[["valence", "arousal", "dominance", "liking"]] = np.repeat(labels, N_WINDOWS, axis=0)
        df["quadrant"] = [quadrant(v, a) for v, a in zip(df["valence"], df["arousal"])]
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta.to_csv(METADATA_DIR / "metadata_window_level.csv", index=False)

    pct = meta.drop_duplicates(["subject_id", "trial_idx"])["quadrant"].value_counts(normalize=True).mul(100).round(1)
    print(f"  {len(meta)} rows; trial-level quadrant %: {pct.to_dict()}")
    return meta, {"n_rows": len(meta), "quadrant_pct": pct.to_dict()}


if __name__ == "__main__":
    run()
