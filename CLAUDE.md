# BCC — DEAP EEG Emotion Recognition

Valence/arousal (and quadrant: HVHA/HVLA/LVHA/LVLA) emotion classification from EEG, using the
DEAP dataset (32 subjects, 40 trials, 32 EEG channels, 128Hz).

## Layout

Organized by function, not by week. Weekly milestones are git tags (`week3-done`, ...) and
sections in `reports/`.

```
config.yaml            locked, shared preprocessing config -- see below
requirements.txt       pinned deps (Python 3.12, .venv managed with uv); also installs src/bcc (-e .)
pyproject.toml         makes src/bcc importable as `bcc`
deap-dataset/          raw DEAP data (do not modify)
src/bcc/               shared library, imported by every stage
  config.py              CONFIG loader + all input/output paths
  data.py                .dat loading, corrected labels, channels, symmetric pairs
  features.py            segmentations (4s/60s), PSD/band power, feature-file I/O (load_domain), per-subject loop
  analysis.py            load_all(seg) (all domains + labels), per-subject z-score, trial means
  modeling.py            nested-LOSO AutoGluon harness (binary valence/arousal), per-fold metrics
  style.py               shared plot colors/style
pipeline/              one folder per stage, each with a run_all.py entry point
  eda/                   Week 1: exploratory checks
  features/              Week 2: one task per feature domain
  viz/                   Week 3: boxplots/topomaps, PCA/t-SNE, correlation
  modeling/              Week 4: nested-LOSO AutoGluon training, evaluation
outputs/               eda/, viz/, features/<domain>/ (4s), features_60s/<domain>/, metadata/, models/,
                       sanity_checks/, modeling/
reports/               EDA_SUMMARY.md, FEATURES_SUMMARY.md, VIZ_SUMMARY.md, MODELING_SUMMARY.md
```

**Conventions:** code used by more than one stage goes in `src/bcc`; stage-specific scripts stay
in `pipeline/<stage>/`. A new stage gets `pipeline/<stage>/run_all.py`, writes to a path defined
in `bcc/config.py`, and adds a report in `reports/`. Changing `src/bcc` means rerunning that
stage and every stage downstream of it. Setup: `uv pip install -r requirements.txt`.

## Week 1 status: done

Key findings (full detail in `reports/EDA_SUMMARY.md`, frozen values in
`config.yaml`):
- The `.dat` files' valence/arousal labels are corrupted on ~34% of trials; corrected labels are
  read from `deap-dataset/Metadata/participant_ratings.xls` instead.
- Quadrant class balance: HVHA 35.8%, LVHA 23.1%, HVLA 20.8%, LVLA 20.3% (moderate imbalance,
  max/min ratio 1.76x — use class-weighted loss / macro-F1, not necessarily resampling).
- Subject 11: keep the subject and all channels, including F7 (electrode-level artifact noted,
  not excluded).
- Locked preprocessing: 4s non-overlapping windows (15/trial), theta/alpha/beta/gamma bands,
  32-channel order + 14 symmetric pairs. No channels are excluded — all 32 channels are used for
  every subject.

Reproduce with: `python pipeline/eda/run_all.py`

## Week 2 status: done

`outputs/features/<domain>/s{NN}_<domain>.csv`, 600 rows per subject (40 trials × 15 windows).
Each file = 3 ID columns (`subject_id, trial_idx, window_idx`) + features. Labels:
`outputs/metadata/metadata_window_level.csv`, joined on the ID columns. Full detail in
`reports/FEATURES_SUMMARY.md`.
- `de`: 32 channels × 4 bands = 128 · `hjorth`: 32 × 3 = 96 · `asymmetry`: 14 pairs × 4 bands ×
  DASM/RASM = 112 · `mse`: 32 × 10 scales = 320 · `microstates`: 4 maps × 3 stats + 12
  transitions = 24
- **60 s set (added Week 4, user decision):** the same domains computed once on the whole 60 s
  stimulus. It lives in `outputs/features_60s/<domain>/`, 40 rows per subject with `window_idx = 0`.
  Load it with `load_domain(d, "60s")` / `load_all("60s")`. It has no MSE `inf`.
  - Don't average 4 s window features into trial rows; use this set when you need trial-level
    features.
  - The 4 s files are unchanged.

Reproduce with: `python pipeline/features/run_all.py` (both sets: ~1 h from scratch; MSE and
microstates cached after that)

**Before modeling:**
- Labels are per trial: never split one trial's windows across train/test; use LOSO for
  cross-subject results.
- MSE (4s set) has 538 `inf` values; impute or drop.
- Microstates are unreliable for 18/32 subjects (`outputs/sanity_checks/microstates/subject_quality.csv`,
  `flag_degenerate`); compare with and without them.
- A 22 Hz noise line (29/32 subjects) biases beta-band features; no notch applied.
- Between-subject offsets dominate every feature; expect to need per-subject normalization.
- Keep preprocessing locked: the user decided not to change it mid-project; ask first.

## Week 3 status: done

Full detail in `reports/VIZ_SUMMARY.md`.
- Raw features cluster by subject (silhouette 0.24-0.71). Per-subject z-scoring removes that
  entirely, and no domain separates quadrants in 2-D. Class effects are tiny (η² ≤ 0.012).
- Cross-domain redundancy (mean best-match |Spearman r|): DE↔Hjorth 0.54-0.61, Hjorth↔MSE 0.72,
  DE↔asymmetry 0.37 (DASM cancels the shared L/R power), microstates ≤ 0.18 with everything.
  Within-domain redundancy is high (asymmetry 0.92, Hjorth 0.84, DE 0.75).
- `bcc.analysis.load_all` imputes MSE `inf` with the subject's column median, in memory only;
  the feature CSVs are untouched.

Reproduce with: `python pipeline/viz/run_all.py` (~2 min)

## Week 4 status: done

Full detail in `reports/MODELING_SUMMARY.md`. The SVM/RF 4-class run (balanced acc 0.30, chance 0.25)
is at git tag `week4-svm-rf`. The current stage replaced it:
- **Setup:**
  - AutoGluon (`bcc.modeling.loso_autogluon`), nested LOSO on the 60 s set with per-subject z-scoring.
  - Two binary targets, valence HV/LV and arousal HA/LA, taken from the quadrant labels.
  - Domains: DE, Hjorth and microstates only (`MODEL_DOMAINS`). Asymmetry and MSE were dropped from
    modeling by the user's decision; their feature files are kept.
  - Tree models only, bagged over 5 group-disjoint inner folds, combined in a weighted ensemble.
- **Valence:** the only target above chance. DE reaches balanced accuracy 0.566 (Holm p < 0.001) and
  Hjorth 0.543. Plain accuracy still doesn't beat the majority class (0.566).
- **Arousal:** no domain beats chance.
- **Against Week 4:** AutoGluon doesn't beat the SVM/RF 4-class predictions collapsed to binary
  (all differences within ±0.04). The ceiling is the features.
- **Fine-tuning:** not done. The top families differ by < 0.01 on inner CV.
- **Environment:** AutoGluon needs `pandas<2.4`, so pandas is pinned to 2.3.3. `load_all` output was
  verified identical to pandas 3.0.5.
- **4s set:** not modeled.

Reproduce with: `python pipeline/modeling/run_all.py` (~3.1 h; resumable via `outputs/modeling/predictions/`)

## Rules for downstream code

`config.yaml` is the single source of truth for all preprocessing decisions above. New scripts
must load it via `bcc.config.CONFIG` rather than redefining channels, bands,
thresholds, or windowing — that's the whole point of freezing these after Week 1. Likewise, the
feature CSVs in `outputs/features/` and `outputs/features_60s/` are the frozen modeling inputs.
