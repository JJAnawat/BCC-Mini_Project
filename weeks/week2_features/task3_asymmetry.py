"""Task 3: DASM/RASM hemispheric asymmetry from Task 1's saved DE features (not recomputed)."""
import numpy as np
import pandas as pd

from features_common import BAND_NAMES, ID_COLS, N_SUBJECTS, SANITY_DIR, SYMMETRIC_PAIRS, feature_path, load_domain


def _asymmetry(de: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for left, right in SYMMETRIC_PAIRS:
        for band in BAND_NAMES:
            de_l, de_r = de[f"{left}_{band}_de"], de[f"{right}_{band}_de"]
            out[f"{left}-{right}_{band}_dasm"] = de_l - de_r
            out[f"{left}-{right}_{band}_rasm"] = de_l / de_r
    return pd.concat([de[ID_COLS], pd.DataFrame(out)], axis=1)


def run():
    for sid in range(1, N_SUBJECTS + 1):
        _asymmetry(pd.read_csv(feature_path("de", sid))).to_csv(feature_path("asymmetry", sid), index=False)
    df = load_domain("asymmetry")

    rasm = df[[c for c in df.columns if c.endswith("_rasm")]].to_numpy()
    summary = {"n_nonfinite_rasm": int((~np.isfinite(rasm)).sum()),
               **dict(zip(["rasm_p0.1", "rasm_p50", "rasm_p99.9"], np.percentile(rasm, [0.1, 50, 99.9]).round(3).tolist()))}
    pd.DataFrame([summary]).to_csv(SANITY_DIR / "asymmetry" / "rasm_range.csv", index=False)
    print(f"  checks: {summary}")
    return df, summary


if __name__ == "__main__":
    run()
