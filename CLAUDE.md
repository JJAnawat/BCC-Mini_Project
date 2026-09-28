# BCC — DEAP EEG Emotion Recognition

Valence/arousal (and quadrant: HVHA/HVLA/LVHA/LVLA) emotion classification from EEG, using the
DEAP dataset (32 subjects, 40 trials, 32 EEG channels, 128Hz).

## Layout

```
config.yaml            locked, shared preprocessing config -- see below
requirements.txt       pinned deps (Python 3.12, .venv managed with uv)
deap-dataset/           raw DEAP data (do not modify)
weeks/
  week1_eda/            Week 1: exploratory data analysis (done)
    common.py, config.py   data loading + label correction; config.yaml loader
    run_all.py, task*.py    entry point + individual EDA checks
    outputs/EDA_SUMMARY.md  full findings
  week2_features/       Week 2: feature extraction (done)
    features_common.py      paths, windowing, shared per-subject loop
    run_all.py, task*.py    entry point + one task per feature domain
    outputs/FEATURES_SUMMARY.md  columns, methods, sanity checks
  week3_viz/            Week 3: visualization + cross-domain correlation (done)
    viz_common.py           load all domains + labels, per-subject z-score, trial means
    run_all.py, task*.py    boxplots/topomaps, PCA/t-SNE, correlation
    outputs/VIZ_SUMMARY.md  findings
```

**Convention for new weeks:** add a sibling `weeks/week<N>_<topic>/` with its own `outputs/`,
one `run_all.py` entry point, everything importing the one root `config.yaml`. Week folders
aren't packages: reach earlier weeks via `sys.path` (see `features_common.py`), and never name
a module `common.py`, `config.py`, `features_common.py` or `viz_common.py`.

## Week 1 status: done

Key findings (full detail in `weeks/week1_eda/outputs/EDA_SUMMARY.md`, frozen values in
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

Reproduce with: `python weeks/week1_eda/run_all.py`

## Week 2 status: done

`outputs/features/<domain>/s{NN}_<domain>.csv`, 600 rows per subject (40 trials × 15 windows).
Each file = 3 ID columns (`subject_id, trial_idx, window_idx`) + features. Labels:
`outputs/metadata/metadata_window_level.csv`, joined on the ID columns. Full detail in
`weeks/week2_features/outputs/FEATURES_SUMMARY.md`.
- `de`: 32 channels × 4 bands = 128 · `hjorth`: 32 × 3 = 96 · `asymmetry`: 14 pairs × 4 bands ×
  DASM/RASM = 112 · `mse`: 32 × 10 scales = 320 · `microstates`: 4 maps × 3 stats + 12
  transitions = 24

Reproduce with: `python weeks/week2_features/run_all.py` (~15 min from scratch; MSE and
microstates cached after that)

**Before modeling:**
- Labels are per trial: never split one trial's windows across train/test; use LOSO for
  cross-subject results.
- MSE has 538 `inf` values; impute or drop.
- Microstates are unreliable for 18/32 subjects (`sanity_checks/microstates/subject_quality.csv`,
  `flag_degenerate`); compare with and without them.
- A 22 Hz noise line (29/32 subjects) biases beta-band features; no notch applied.
- Between-subject offsets dominate every feature; expect to need per-subject normalization.
- Keep preprocessing locked: the user decided not to change it mid-project; ask first.

## Week 3 status: done

Full detail in `weeks/week3_viz/outputs/VIZ_SUMMARY.md`.
- Raw features cluster by subject (silhouette 0.24-0.71). Per-subject z-scoring removes that
  entirely, and no domain separates quadrants in 2-D. Class effects are tiny (η² ≤ 0.012).
- Cross-domain redundancy (mean best-match |Spearman r|): DE↔Hjorth 0.54-0.61, Hjorth↔MSE 0.72,
  DE↔asymmetry 0.37 (DASM cancels the shared L/R power), microstates ≤ 0.18 with everything.
  Within-domain redundancy is high (asymmetry 0.92, Hjorth 0.84, DE 0.75).
- Week 3 only imputes MSE `inf` with the subject's column median. The Week 2 CSVs are untouched.

Reproduce with: `python weeks/week3_viz/run_all.py` (~2 min)

## Rules for downstream code

`config.yaml` is the single source of truth for all preprocessing decisions above. New scripts
must load it via `weeks/week1_eda/config.py`'s `CONFIG` rather than redefining channels, bands,
thresholds, or windowing — that's the whole point of freezing these after Week 1. Likewise, the
Week 2 feature CSVs are the frozen modeling inputs.
