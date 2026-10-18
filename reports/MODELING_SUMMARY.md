# Week 4 — Nested-LOSO AutoGluon, binary valence / arousal, per domain

Files named below are in `outputs/modeling/`. Code: `pipeline/modeling/`, with the harness in `src/bcc/modeling.py`.
The first Week 4 run (SVM/RF on 4-class quadrants) is at git tag `week4-svm-rf`; it is the baseline in §4.

## Setup

- **Input:** the 60 s feature set, one row per trial, 1,280 trials. See `FEATURES_SUMMARY.md`, "60 s set".
  Features are z-scored per subject, including the held-out subject; this uses only its unlabeled features.
- **Targets:** two binary tasks, trained separately. Valence is HV/LV and arousal is HA/LA. Both come
  from the quadrant labels, so the thresholds are the same.
- **Domains:** DE, Hjorth, microstates.
  - Asymmetry was dropped because it was the weakest domain for both SVM and RF.
  - MSE was dropped because it is redundant with Hjorth (Week 3 |r| 0.72), ranked mid-table, and is
    the largest domain (320 features).
  - Caveat: both were judged on the 4-class task. Frontal asymmetry is a classic valence feature,
    and binary valence is untested for it.
  - Their feature files are kept; they are just not modeled.
- **Outer loop:** 32 leave-one-subject-out folds.
- **Inner loop:** AutoGluon 1.6.3 bagging over 5 group-disjoint folds of the 31 training subjects
  (`GroupKFold(5)`). Model choice and the weighted-ensemble weights are therefore scored cross-subject.
- **Same protocol for every domain and target:**
  - Metric: `f1_macro`, with `sample_weight="balance_weight"`.
  - No stacking and no time limit, so results do not depend on the hardware.
  - Seed 42.
- **Models:** AutoGluon's default configs for LightGBM (default and ExtraTrees variants), XGBoost,
  RandomForest (Gini/entropy) and ExtraTrees (Gini/entropy), combined in a weighted ensemble.
  - Neural nets were left out for speed.
  - CatBoost and LightGBMLarge were dropped after a 4-fold pilot (DE, valence). They took about half
    the fit time and never scored best on inner CV.
  - That choice used inner scores of folds 1–4, which include subjects held out in other folds. The
    leak is small, because the gaps were small and the main reason for dropping them was speed.
- **Metrics:** computed per held-out subject; the tables show mean ± SD over the 32 folds.

**Chance levels:**

| Measure | Valence | Arousal |
|---|---|---|
| Balanced accuracy / macro-F1 | 0.5 | 0.5 |
| Accuracy, always the majority class (HV / HA) | 0.566 | 0.589 |

## 1. Results (`results.csv`, `fold_metrics.csv`)

| Target | Domain | Accuracy | Macro-F1 | Balanced acc | Holm p vs 0.5 |
|---|---|---|---|---|---|
| valence | de | 0.571 ± 0.068 | **0.555** ± 0.068 | **0.566** ± 0.068 | **< 0.001** |
| valence | hjorth | 0.556 ± 0.079 | 0.531 ± 0.084 | 0.543 ± 0.079 | **0.025** |
| valence | microstates | 0.556 ± 0.105 | 0.515 ± 0.109 | 0.535 ± 0.103 | 0.077 |
| arousal | de | 0.541 ± 0.098 | 0.502 ± 0.093 | 0.526 ± 0.098 | 0.14 |
| arousal | hjorth | 0.533 ± 0.106 | 0.486 ± 0.107 | 0.516 ± 0.110 | 0.32 |
| arousal | microstates | 0.523 ± 0.105 | 0.478 ± 0.108 | 0.524 ± 0.085 | 0.20 |

The p-values come from a one-sided Wilcoxon of fold balanced accuracy against 0.5, Holm-corrected
over the 6 tests.

- **Valence is the only target above chance.** DE and Hjorth reach balanced accuracies of 0.57 and
  0.54. The effect is small.
- **Plain accuracy does not beat always predicting the majority class.**
  - Valence DE is +0.005 over the per-subject majority baseline. It wins in 16 of 32 folds
    (one-sided p = 0.21).
  - Every arousal model is 0.05–0.07 *below* its majority baseline.
  - So the valence signal is real but weak: about 57 % balanced accuracy across subjects.
- **Arousal is not learned across subjects** by any domain.
- **The inner CV is mildly optimistic.** The ensemble's inner macro-F1 was 0.57–0.59 for valence and
  0.55–0.56 for arousal, against 0.52–0.56 and 0.48–0.50 held out. Part of the gap is the ensemble
  weights, which are fitted on the same out-of-fold predictions they are scored on.
- **Microstate subject quality** (`flag_degenerate`) shows no consistent pattern.
  - Arousal macro-F1 is lower on the 18 flagged subjects than on the 14 clean ones (0.43 vs 0.54).
  - Valence goes the other way (0.53 vs 0.50).
  - Both gaps are within the fold SD.

## 2. Confusion matrices (`confusion_matrices.png`)

Each matrix pools the 32 held-out subjects and is row-normalized.

- **Every model leans to the majority class**, even with balanced sample weights.
  - Valence: HV recall is 0.63–0.69 and LV recall 0.38–0.49.
  - Arousal: HA recall is 0.62–0.66 and LA recall 0.34–0.39.
- DE on valence comes closest to a symmetric split: HV recall 0.64, LV recall 0.49.
- On arousal, the LA row looks almost the same as the HA row. That is the pattern of a model that
  does not separate the classes.

## 3. Model screen (`leaderboards.csv`) and domain comparison (`significance.csv`)

- **The ensemble was chosen in all 192 fits.** Its base models are close to one another:

  | Model | Mean inner macro-F1, valence | Mean inner macro-F1, arousal |
  |---|---|---|
  | LightGBMXT | 0.576 | 0.550 |
  | XGBoost | 0.569 | 0.549 |
  | LightGBM | 0.569 | 0.547 |
  | RF / ExtraTrees | 0.53–0.54 | 0.50–0.51 |

  - The best single model per fold was LightGBMXT in 89 fits, XGBoost in 56 and LightGBM in 47.
  - The boosted models beat the bagged-tree models by about 0.04, but they are within 0.007 of each
    other.
- **No fine-tuning stage.** Tuning the top family (LightGBMXT) could only separate models that are
  already within 0.01 on inner CV. With a held-out signal this weak, it would mostly fit inner-CV noise.
- **No domain pair differs significantly** (paired Wilcoxon on fold macro-F1, Holm within target).
  The nearest is valence DE vs Hjorth: median difference +0.021, raw p = 0.047, Holm p = 0.14.

## 4. Against the Week 4 SVM/RF baseline

For the baseline, the Week 4 4-class predictions were collapsed to H/L on each axis, and the same
binary fold metrics were computed. Each AutoGluon domain is compared with SVM and RF by paired
Wilcoxon on fold balanced accuracy.

| Target | Domain | SVM (4-class) | RF (4-class) | AutoGluon (binary) |
|---|---|---|---|---|
| valence | de | 0.572 | 0.580 | 0.566 |
| valence | hjorth | 0.563 | 0.549 | 0.543 |
| valence | microstates | 0.508 | 0.495 | 0.535 |
| arousal | de | 0.539 | 0.519 | 0.526 |
| arousal | hjorth | 0.507 | 0.526 | 0.516 |
| arousal | microstates | 0.540 | 0.542 | 0.524 |

- **AutoGluon with binary targets does not beat SVM/RF trained on quadrants.** All 12 differences
  are within ±0.04.
  - The only raw p < 0.05 is valence microstates against RF: +0.041, p = 0.025. It does not survive
    correction for 12 tests.
- The ceiling is in the features, not the classifier or the target framing. The cross-subject
  valence signal in DE and Hjorth is about 0.57 balanced accuracy whichever way it is modeled.
  Arousal has none.

**For Week 5:**
- Valence is the target worth pursuing, with DE as the anchor domain.
- Fusion or new information has to add signal. More model search on these features has run out.
- Asymmetry deserves one binary-valence check before it is ruled out for good.

Reproduce with `python pipeline/modeling/run_all.py`: about 3.1 h on 8 cores, running 4 outer folds
in parallel with 2 cores each. The run is resumable; finished target × domain results are cached in
`outputs/modeling/predictions/`.
