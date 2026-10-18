# Week 4 — Nested-LOSO quadrant classification, per domain

Files named below are in `outputs/modeling/`. Code: `pipeline/modeling/`, with the harness in `src/bcc/modeling.py`.

## Setup

- **Input:** the 60 s feature set, one row per trial, 1,280 trials. See `FEATURES_SUMMARY.md`, "60 s set".
  Features are z-scored per subject, including the held-out subject; this uses only its unlabeled features.
- **Target:** 4-class quadrant.
- **Outer loop:** 32 leave-one-subject-out folds.
- **Inner loop:** `GroupKFold(5)` over the 31 training subjects, tuned on macro-F1.
- **Same protocol for every domain:**
  - **SVM:** RBF kernel, C ∈ {0.01, 0.1, 1, 10, 100}, γ ∈ {0.01, 0.1, 1, 10}/n_features.
  - **RF:** 150 trees, max_depth ∈ {None, 10}, min_samples_leaf ∈ {1, 5}.
  - Both use `class_weight="balanced"`.
- **Metrics:** computed per held-out subject; the tables show mean ± SD over the 32 folds.
  - Subject 23 has no LVHA trials. In that fold, macro-F1 and balanced accuracy average its 3 classes.
- **Valence and arousal accuracy:** taken from the same quadrant predictions (H/L of each axis).

**Chance levels:**

| Measure | Chance |
|---|---|
| Balanced accuracy | 0.25 |
| Accuracy, always HVHA | 0.358 |
| Valence accuracy, majority (HV) | 0.566 |
| Arousal accuracy, majority (HA) | 0.589 |

## 1. Results (`results.csv`, `fold_metrics.csv`)

| Domain | Model | Accuracy | Macro-F1 | Balanced acc | Valence acc | Arousal acc |
|---|---|---|---|---|---|---|
| de | SVM | 0.285 ± 0.095 | 0.266 ± 0.095 | **0.300** ± 0.105 | 0.562 | 0.538 |
| de | RF | 0.301 ± 0.085 | **0.271** ± 0.085 | 0.293 ± 0.089 | 0.580 | 0.530 |
| hjorth | SVM | 0.279 ± 0.091 | 0.256 ± 0.089 | 0.296 ± 0.102 | 0.552 | 0.508 |
| hjorth | RF | 0.302 ± 0.095 | 0.267 ± 0.095 | 0.294 ± 0.104 | 0.557 | 0.530 |
| asymmetry | SVM | 0.266 ± 0.081 | 0.222 ± 0.062 | 0.258 ± 0.078 | 0.494 | 0.544 |
| asymmetry | RF | 0.273 ± 0.078 | 0.229 ± 0.073 | 0.265 ± 0.099 | 0.496 | 0.538 |
| mse | SVM | 0.274 ± 0.098 | 0.249 ± 0.084 | 0.274 ± 0.102 | 0.533 | 0.529 |
| mse | RF | 0.294 ± 0.092 | 0.256 ± 0.085 | 0.278 ± 0.103 | 0.527 | 0.544 |
| microstates | SVM | 0.288 ± 0.121 | 0.256 ± 0.116 | 0.287 ± 0.126 | 0.503 | 0.547 |
| microstates | RF | 0.284 ± 0.102 | 0.240 ± 0.095 | 0.272 ± 0.093 | 0.512 | 0.548 |

- **Every domain is close to chance.** The best result is DE (balanced accuracy 0.30, macro-F1 0.27).
  - DE and Hjorth beat 0.25 balanced accuracy in 21–23 of 32 folds (one-sided Wilcoxon p ≈ 0.01 each).
    That is not significant after correcting for the 10 domain × model tests.
  - Asymmetry, MSE and microstates are not distinguishable from chance (p ≥ 0.1).
- **No model beats the majority baselines** for accuracy, valence or arousal. Class-balanced
  training gives up the majority class to predict all four.
  - Valence accuracy tops out at 0.58 (DE, RF), against a majority baseline of 0.566.
  - Arousal accuracy is at most 0.55, against 0.589.
- **SVM and RF agree.** The ranking (DE ≈ Hjorth > MSE ≈ microstates > asymmetry) is the same for both,
  and no RF–SVM gap is larger than its SD.
- **Tuning leaned towards smooth models** (`best_params.csv`):
  - SVM never chose the largest γ (10/n_features).
  - DE and microstates mostly chose 0.01–0.1/n_features. Hjorth, asymmetry and MSE mostly chose
    1/n_features.
  - For DE and Hjorth, RF mostly chose min_samples_leaf = 5.
  - Inner-CV macro-F1 (0.25–0.30) matches the outer scores, so the tuning is not overfitting. There
    is just little signal to find.

## 2. Confusion matrices (`confusion_matrices.png`)

Each matrix pools the 32 held-out subjects and is row-normalized.

- **There is no clear diagonal.** The best per-class recall is about 0.35, and HVHA recall reaches
  0.40–0.45 only for models that predict HVHA for everything.
- **Each model mostly picks a favourite class** instead of confusing neighbouring quadrants.
  - Asymmetry and MSE, and RF in general, put about 35–45% of every row into HVHA, the largest class.
  - SVM on DE and Hjorth puts about 30–38% of every row into LVHA.
- **Errors show no single-axis structure.**
  - Of all wrong predictions, about one third get only valence wrong, one third only arousal, and
    one third both. That is the split expected from guessing.
  - DE comes closest to an axis effect: arousal-only errors are 0.39–0.40 against valence-only
    0.33–0.35, which points to a slightly better valence signal. The difference is small.
- So the models are not tracking one axis well and the other poorly. Neither axis is learned across
  subjects.

## 3. Significance (`significance.csv`)

Paired Wilcoxon signed-rank on fold macro-F1 (32 folds) for all 10 domain pairs, Holm-corrected
within each model.

- **No pair is significant.** For SVM all Holm p ≥ 0.71; for RF all Holm p ≥ 0.11.
- The nearest are RF DE vs asymmetry (median difference +0.049, raw p = 0.011, Holm 0.11) and RF
  Hjorth vs asymmetry (+0.052, raw p = 0.019, Holm 0.18).
- Asymmetry is the weakest domain for both models, but the data cannot rank the domains.
- **Microstate subject quality has no effect.** SVM macro-F1 is the same for the 18 flagged
  (degenerate) subjects and the 14 clean ones (0.255 vs 0.257). RF is slightly lower on flagged
  subjects (0.221 vs 0.265), well within the fold SD.

**For Week 5:**
- No single domain carries a usable cross-subject quadrant signal. Fusion has to add information,
  not pick a winner.
- DE and Hjorth overlap heavily (Week 3), so fusing them adds little. Asymmetry and microstates are
  the independent views, but they are also the weakest here.

Reproduce with `python pipeline/modeling/run_all.py` (~55 min on 8 cores; RF about half of it).
