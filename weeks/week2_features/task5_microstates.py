"""Task 5: EEG microstates -- K=4 group maps from pooled GFP peaks, backfit and summarized per 4s window."""
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData, read_cluster
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.segmentation import RawSegmentation

from features_common import (
    MODELS_DIR, N_EEG_CHANNELS, N_SUBJECTS, N_WINDOWS, SANITY_DIR, WINDOW_SAMPLES,
    iter_subjects, load_subject, montage_info, run_domain, stimulus,
)

N_MAPS = 4
MAP_NAMES = [f"map{i}" for i in range(N_MAPS)]   # arbitrary k-means order, NOT canonical A/B/C/D
UV_TO_V = 1e-6
GFP_MIN_PEAK_DISTANCE = 2        # samples (~16ms at 128Hz)
MAX_PEAKS_PER_SUBJECT = 1000     # caps the pooled fit and gives every subject equal weight
PEAK_SEED, N_INIT, FIT_SEED = 0, 100, 42
SMOOTH_FACTOR = 10               # Pascual-Marqui 1995 smoothing; without it mean durations are ~12ms
SMOOTH_HALF_WINDOW = 4           # samples, i.e. +-31ms
PC1_FLAG = 0.9                   # one fixed topography (artifact) explaining >90% of variance
DOMINATED_COVERAGE, DOMINATED_PCT_FLAG = 0.9, 50.0
MODEL_PATH = MODELS_DIR / "microstates" / "group_model.fif"
TRANSITIONS = [(i, j) for i in range(N_MAPS) for j in range(N_MAPS) if i != j]  # self-transitions are always 0
COLUMNS = ([f"{m}_{s}" for m in MAP_NAMES for s in ("coverage", "duration_s", "occurrence_per_s")]
           + [f"trans_{MAP_NAMES[i]}_to_{MAP_NAMES[j]}" for i, j in TRANSITIONS])


def _fit_group_model(info) -> ModKMeans:
    if MODEL_PATH.exists():
        return read_cluster(MODEL_PATH)
    rng = np.random.default_rng(PEAK_SEED)
    pooled = []
    for _sid, data, _labels, err in iter_subjects():
        if err is not None:
            raise err
        stim = np.stack([stimulus(trial) for trial in data[:, :N_EEG_CHANNELS, :]]) * UV_TO_V
        peaks = extract_gfp_peaks(mne.EpochsArray(stim, info, verbose="ERROR"),
                                  min_peak_distance=GFP_MIN_PEAK_DISTANCE, verbose="ERROR").get_data()
        if peaks.shape[1] > MAX_PEAKS_PER_SUBJECT:
            peaks = peaks[:, np.sort(rng.choice(peaks.shape[1], MAX_PEAKS_PER_SUBJECT, replace=False))]
        pooled.append(peaks / np.median(peaks.std(axis=0)))   # unit median GFP: stops s22 claiming whole maps
    model = ModKMeans(n_clusters=N_MAPS, n_init=N_INIT, random_state=FIT_SEED)
    model.fit(ChData(np.concatenate(pooled, axis=1), info), n_jobs=-1, verbose="ERROR")
    model.rename_clusters(new_names=MAP_NAMES)
    model.save(MODEL_PATH)
    return model


def _backfit(model, eeg, info) -> list[list[float]]:
    """Smoothed backfit over each whole 60s trial (no per-window edge effects), stats per window."""
    rows = []
    for trial in eeg:
        stim = stimulus(trial) * UV_TO_V
        labels = model.predict(mne.io.RawArray(stim, info, verbose="ERROR"), factor=SMOOTH_FACTOR,
                               half_window_size=SMOOTH_HALF_WINDOW, reject_edges=False, verbose="ERROR").labels
        for w in range(N_WINDOWS):
            sl = slice(w * WINDOW_SAMPLES, (w + 1) * WINDOW_SAMPLES)
            seg = RawSegmentation(labels=labels[sl], inst=mne.io.RawArray(stim[:, sl], info, verbose="ERROR"),
                                  cluster_centers_=model.cluster_centers_, cluster_names=model.cluster_names)
            p = seg.compute_parameters(norm_gfp=True)
            trans = seg.compute_transition_matrix(stat="probability", ignore_repetitions=True)
            rows.append([p[f"{m}_{k}"] for m in MAP_NAMES for k in ("timecov", "meandurs", "occurrences")]
                        + [trans[i, j] for i, j in TRANSITIONS])
    return rows


def _pc1_share(eeg) -> float:
    """Variance share of the first spatial PC of the average-referenced stimulus EEG."""
    x = np.concatenate([stimulus(trial) for trial in eeg], axis=1)
    ev = np.linalg.eigvalsh(np.cov((x - x.mean(axis=0))[:, ::8]))
    return float(ev[-1] / ev.sum())


def run():
    info = montage_info()
    model = _fit_group_model(info)
    fig = model.plot(show=False)
    fig.suptitle(f"Group microstate maps (GEV={model.GEV_:.3f}); order is arbitrary")
    fig.savefig(SANITY_DIR / "microstates" / "group_maps.png", dpi=150)
    plt.close(fig)

    df = run_domain("microstates", lambda sid, eeg: _backfit(model, eeg, info), COLUMNS, cache=True)
    cov = df[[f"{m}_coverage" for m in MAP_NAMES]]
    assert np.allclose(cov.sum(axis=1), 1.0), "coverage must sum to 1 per window"
    for i in range(N_MAPS):
        row = df[[f"trans_{MAP_NAMES[i]}_to_{MAP_NAMES[j]}" for j in range(N_MAPS) if j != i]].sum(axis=1)
        assert (np.isclose(row, 1.0) | np.isclose(row, 0.0)).all(), "transition rows must sum to 1 (or 0)"

    quality = pd.DataFrame({
        "subject_id": range(1, N_SUBJECTS + 1),
        "pc1_variance_share": [_pc1_share(load_subject(s)[0][:, :N_EEG_CHANNELS, :]) for s in range(1, N_SUBJECTS + 1)],
        "pct_windows_one_map_dominated": ((cov.max(axis=1) > DOMINATED_COVERAGE).groupby(df["subject_id"]).mean() * 100).values,
    })
    quality["flag_degenerate"] = ((quality["pc1_variance_share"] > PC1_FLAG)
                                  | (quality["pct_windows_one_map_dominated"] > DOMINATED_PCT_FLAG))
    quality.to_csv(SANITY_DIR / "microstates" / "subject_quality.csv", index=False)

    summary = {"gev": round(float(model.GEV_), 3),
               "flagged_subjects": quality.loc[quality["flag_degenerate"], "subject_id"].tolist()}
    print(f"  checks: {summary}")
    return df, summary


if __name__ == "__main__":
    run()
