"""Task 3: Signal quality checks (amplitude stats, PSD, raw waveform spot-checks)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from common import EEG_CHANNELS, FS, N_EEG_CHANNELS, OUTPUT_DIR, iter_subjects, load_subject

FLAT_STD_THRESHOLD = 1.0  # uV; below this, a channel is considered near-flat
OUTLIER_SD_MULT = 3.0     # flag channel std beyond mean +/- this many SDs (within-subject)

# subject 11 included because it's an amplitude/alpha-power outlier flagged
# by Tasks 3/4 -- config.yaml's subject_11_decision cites these exact plots.
PSD_SUBJECTS = [1, 2, 3, 11]              # subjects to plot PSD for
WAVEFORM_SUBJECTS = [1, 2, 11]            # subjects for raw waveform spot-check
WAVEFORM_TRIALS = [0, 1]                  # trial indices to inspect
# Trial 2 is extra evidence for subject 11 specifically (F7 spike check).
EXTRA_WAVEFORM_TRIALS = {11: [2]}
WAVEFORM_CHANNELS = ["F7", "Fp1", "Fp2", "Oz", "Cz"]  # F7 included for the subject-11 bad-channel check

# Ad-hoc comparison: subject 1 (baseline) vs 11 and 22 (flagged outliers),
# plotted on a shared y-axis so amplitude/PSD differences are directly
# comparable across subjects (task3's per-subject plots above use
# independent, auto-scaled axes per figure).
COMPARISON_SUBJECTS = [1, 11, 22]
COMPARISON_TRIAL = 0
COMPARISON_STYLES = {
    1: dict(color="tab:blue", linestyle="-"),
    11: dict(color="tab:red", linestyle="-"),
    22: dict(color="tab:green", linestyle="-"),
}


def _amplitude_stats():
    rows = []
    for sid, data, labels, err in iter_subjects():
        if err is not None:
            continue
        eeg = data[:, :N_EEG_CHANNELS, :]  # (trials, 32, samples)
        # Stats per channel over the full recording (all trials concatenated)
        flat_ch = np.moveaxis(eeg, 1, 0).reshape(N_EEG_CHANNELS, -1)  # (32, trials*samples)
        ch_min = flat_ch.min(axis=1)
        ch_max = flat_ch.max(axis=1)
        ch_std = flat_ch.std(axis=1)

        std_mean, std_sd = ch_std.mean(), ch_std.std()
        for ci, ch_name in enumerate(EEG_CHANNELS):
            is_flat = ch_std[ci] < FLAT_STD_THRESHOLD
            is_outlier = abs(ch_std[ci] - std_mean) > OUTLIER_SD_MULT * std_sd if std_sd > 0 else False
            rows.append({
                "subject_id": sid,
                "channel": ch_name,
                "min": ch_min[ci],
                "max": ch_max[ci],
                "std": ch_std[ci],
                "flagged_flat": bool(is_flat),
                "flagged_outlier": bool(is_outlier),
            })
    return pd.DataFrame(rows)


def _plot_psd(subject_ids):
    for sid, data, labels, err in iter_subjects(subject_ids):
        if err is not None:
            continue
        eeg = data[:, :N_EEG_CHANNELS, :]  # (trials, 32, samples)
        plt.figure(figsize=(9, 6))
        for ci, ch_name in enumerate(EEG_CHANNELS):
            # average PSD across trials for this channel
            psds = []
            for t in range(eeg.shape[0]):
                f, pxx = welch(eeg[t, ci, :], fs=FS, nperseg=min(512, eeg.shape[2]))
                psds.append(pxx)
            mean_psd = np.mean(psds, axis=0)
            plt.plot(f, mean_psd, linewidth=0.6, alpha=0.6)
        plt.axvspan(8, 13, color="green", alpha=0.1, label="alpha (8-13Hz)")
        plt.axvline(50, color="red", linestyle="--", linewidth=1, label="50Hz line noise")
        plt.xlim(0, 60)
        plt.yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (log scale)")
        plt.title(f"Subject s{sid:02d}: PSD per EEG channel (avg across trials)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"task3_psd_s{sid:02d}.png", dpi=150)
        plt.close()
        print(f"  Saved PSD plot for s{sid:02d}")


def _plot_waveforms(subject_ids, trial_idxs, channel_names):
    ch_idx = [EEG_CHANNELS.index(c) for c in channel_names]
    t_axis = None
    for sid, data, labels, err in iter_subjects(subject_ids):
        if err is not None:
            continue
        trials_for_subject = list(trial_idxs) + EXTRA_WAVEFORM_TRIALS.get(sid, [])
        for trial in trials_for_subject:
            fig, axes = plt.subplots(len(ch_idx), 1, figsize=(10, 2.2 * len(ch_idx)), sharex=True)
            if t_axis is None:
                t_axis = np.arange(data.shape[2]) / FS
            for ax, ci, cname in zip(axes, ch_idx, channel_names):
                ax.plot(t_axis, data[trial, ci, :], linewidth=0.6)
                ax.set_ylabel(cname)
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Subject s{sid:02d}, trial {trial}: raw waveform spot-check")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / f"task3_waveform_s{sid:02d}_trial{trial}.png", dpi=150)
            plt.close(fig)
            print(f"  Saved waveform spot-check for s{sid:02d} trial {trial}")


def _plot_comparison_waveforms():
    ch_idx = [EEG_CHANNELS.index(c) for c in WAVEFORM_CHANNELS]
    data_by_subj = {sid: load_subject(sid)[0] for sid in COMPARISON_SUBJECTS}
    t_axis = np.arange(data_by_subj[COMPARISON_SUBJECTS[0]].shape[2]) / FS

    # Shared y-limit per channel row, computed across all subjects, so
    # amplitude differences are directly comparable (not auto-scaled away).
    row_max = {}
    for ci, cname in zip(ch_idx, WAVEFORM_CHANNELS):
        vals = [data_by_subj[sid][COMPARISON_TRIAL, ci, :] for sid in COMPARISON_SUBJECTS]
        row_max[cname] = max(np.max(np.abs(v)) for v in vals)

    fig, axes = plt.subplots(len(WAVEFORM_CHANNELS), 1, figsize=(11, 2.3 * len(WAVEFORM_CHANNELS)), sharex=True)
    for ax, ci, cname in zip(axes, ch_idx, WAVEFORM_CHANNELS):
        ylim = row_max[cname] * 1.05
        for sid in COMPARISON_SUBJECTS:
            ax.plot(t_axis, data_by_subj[sid][COMPARISON_TRIAL, ci, :], linewidth=0.6,
                     label=f"s{sid:02d}", **COMPARISON_STYLES[sid])
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(cname)
        ax.axhline(0, color="gray", linewidth=0.4)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Subjects {COMPARISON_SUBJECTS[0]} (baseline) vs "
                 f"{' vs '.join(str(s) for s in COMPARISON_SUBJECTS[1:])} -- "
                 f"trial {COMPARISON_TRIAL}, per-channel y-axis shared across subjects")
    fig.tight_layout()
    out = OUTPUT_DIR / "compare_s01_s11_s22_waveform.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def _plot_comparison_psd():
    fig, axes = plt.subplots(1, len(COMPARISON_SUBJECTS), figsize=(15, 5), sharey=True, sharex=True)
    global_max = 0
    subj_psds = {}
    for sid in COMPARISON_SUBJECTS:
        data, _ = load_subject(sid)
        eeg = data[:, :N_EEG_CHANNELS, :]
        chan_psds = []
        for ci in range(N_EEG_CHANNELS):
            trial_psds = []
            for t in range(eeg.shape[0]):
                f, pxx = welch(eeg[t, ci, :], fs=FS, nperseg=min(512, eeg.shape[2]))
                trial_psds.append(pxx)
            chan_psds.append(np.mean(trial_psds, axis=0))
        subj_psds[sid] = (f, chan_psds)
        global_max = max(global_max, np.max(chan_psds))

    for ax, sid in zip(axes, COMPARISON_SUBJECTS):
        f, chan_psds = subj_psds[sid]
        for ci, psd in enumerate(chan_psds):
            ax.plot(f, psd, linewidth=0.6, alpha=0.6)
        ax.axvspan(8, 13, color="green", alpha=0.1)
        ax.axvline(50, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 60)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, global_max * 2)
        ax.set_title(f"s{sid:02d}" + (" (baseline)" if sid == COMPARISON_SUBJECTS[0] else " (flagged outlier)"))
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (log scale, shared y-axis)")
    fig.suptitle(f"Per-channel PSD (avg across trials): subject comparison "
                 f"{' vs '.join(str(s) for s in COMPARISON_SUBJECTS)}, shared y-axis")
    fig.tight_layout()
    out = OUTPUT_DIR / "compare_s01_s11_s22_psd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")


def run():
    print("Computing per-subject/channel amplitude stats...")
    amp_df = _amplitude_stats()
    amp_df.to_csv(OUTPUT_DIR / "task3_amplitude_stats.csv", index=False)

    n_flat = int(amp_df["flagged_flat"].sum())
    n_outlier = int(amp_df["flagged_outlier"].sum())
    print(f"Flagged {n_flat} near-flat (std<{FLAT_STD_THRESHOLD}) channel-subject rows.")
    print(f"Flagged {n_outlier} outlier-amplitude channel-subject rows "
          f"(std beyond {OUTLIER_SD_MULT} SD of within-subject channel-std distribution).")

    if n_flat:
        print("\nNear-flat channels:")
        print(amp_df[amp_df["flagged_flat"]][["subject_id", "channel", "std"]].to_string(index=False))
    if n_outlier:
        print("\nOutlier-amplitude channels:")
        print(amp_df[amp_df["flagged_outlier"]][["subject_id", "channel", "std"]].to_string(index=False))

    print(f"\nPlotting PSD for subjects {PSD_SUBJECTS}...")
    _plot_psd(PSD_SUBJECTS)

    print(f"\nPlotting raw waveform spot-checks for subjects {WAVEFORM_SUBJECTS}, "
          f"trials {WAVEFORM_TRIALS}, channels {WAVEFORM_CHANNELS}...")
    _plot_waveforms(WAVEFORM_SUBJECTS, WAVEFORM_TRIALS, WAVEFORM_CHANNELS)

    print(f"\nPlotting shared-axis comparison for subjects {COMPARISON_SUBJECTS}...")
    _plot_comparison_waveforms()
    _plot_comparison_psd()

    summary = {
        "n_flat_channel_rows": n_flat,
        "n_outlier_channel_rows": n_outlier,
    }
    return amp_df, summary


if __name__ == "__main__":
    run()
