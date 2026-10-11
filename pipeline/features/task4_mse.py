"""Task 4: Multiscale Sample Entropy (scales 1-10) per 4s window x channel."""
import contextlib
import io
import warnings

import antropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bcc.features import EEG_CHANNELS, N_EEG_CHANNELS, SANITY_DIR, load_subject, run_domain, window_subject

MSE_M = 2
MSE_R_FRAC = 0.15               # r = 0.15 * std of the scale-1 window, fixed across scales (Costa 2002)
MSE_SCALES = list(range(1, 11))
CROSSCHECK_SUBJECTS, CROSSCHECK_PER_SUBJECT, CROSSCHECK_SEED = [1, 11], 15, 0
COLUMNS = [f"{ch}_scale{s}_mse" for ch in EEG_CHANNELS for s in MSE_SCALES]


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    n = len(x) // scale
    return x[: n * scale].reshape(n, scale).mean(axis=1)


def _mse(x: np.ndarray) -> list[float]:
    r = MSE_R_FRAC * np.std(x)
    if r == 0:
        return [np.nan] * len(MSE_SCALES)
    return [antropy.sample_entropy(_coarse_grain(x, s), order=MSE_M, tolerance=r) for s in MSE_SCALES]


def _compute(sid, eeg):
    windows = window_subject(eeg)
    return np.array([[_mse(w[c]) for c in range(w.shape[0])] for w in windows]).reshape(len(windows), -1)


def _entropyhub_max_diff() -> float:
    """antropy vs EntropyHub's reference MSEn (which keeps r fixed when RadNew=0) on random windows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        import EntropyHub as EH
    rng = np.random.default_rng(CROSSCHECK_SEED)
    diffs = []
    for sid in CROSSCHECK_SUBJECTS:
        windows = window_subject(load_subject(sid)[0][:, :N_EEG_CHANNELS, :])
        for _ in range(CROSSCHECK_PER_SUBJECT):
            x = windows[rng.integers(len(windows)), rng.integers(windows.shape[1])]
            with contextlib.redirect_stdout(io.StringIO()):     # MSEn prints progress dots
                ref, _ = EH.MSEn(x, EH.MSobject("SampEn", m=MSE_M, r=MSE_R_FRAC * np.std(x)), Scales=len(MSE_SCALES))
            ours = np.array(_mse(x))
            same_pattern = np.array_equal(np.isfinite(ref), np.isfinite(ours))
            both = np.isfinite(ref) & np.isfinite(ours)
            diffs.append(np.abs(ref[both] - ours[both]).max(initial=0.0) if same_pattern else np.inf)
    return float(np.max(diffs))


def run():
    df = run_domain("mse", _compute, COLUMNS, cache=True)
    by_scale = {s: df[[f"{ch}_scale{s}_mse" for ch in EEG_CHANNELS]].to_numpy() for s in MSE_SCALES}
    checks = pd.DataFrame([{"scale": s, "n_nonfinite": int((~np.isfinite(v)).sum()),
                            "mean_finite": v[np.isfinite(v)].mean()} for s, v in by_scale.items()])
    checks.to_csv(SANITY_DIR / "mse" / "mse_nonfinite_per_scale.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(checks["scale"], checks["mean_finite"], marker="o")
    plt.xticks(MSE_SCALES)
    plt.xlabel("Scale factor")
    plt.ylabel(f"SampEn (m={MSE_M}, r={MSE_R_FRAC}*std)")
    plt.title("Pooled MSE curve, all subjects/windows/channels")
    plt.savefig(SANITY_DIR / "mse" / "mse_curve.png", dpi=150)
    plt.close()

    summary = {"n_nonfinite": int(checks["n_nonfinite"].sum()), "entropyhub_max_abs_diff": _entropyhub_max_diff()}
    print(f"  checks: {summary}")
    return df, summary


if __name__ == "__main__":
    run()
