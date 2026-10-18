"""Nested leave-one-subject-out AutoGluon harness and fold metrics, identical for every feature domain.

Outer loop: one fold per held-out subject. Inner loop: AutoGluon's bagging over GroupKFold(5) splits of
the training subjects, so model selection and the weighted ensemble are scored cross-subject too.
Inputs are expected to be z-scored per subject already (bcc.analysis.zscore_within_subject).
"""
import tempfile
import warnings

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

INNER_FOLDS = 5
SEED = 42
# outer folds run in parallel, each fit on CPUS_PER_FIT cores with its bags trained sequentially:
# on ~1,240 rows AutoGluon's per-bag worker processes cost more than the training itself
OUTER_JOBS = 4
CPUS_PER_FIT = 2

# coarse stage: AutoGluon's default configs for the tree families only. The neural nets (NN_TORCH,
# FASTAI) are by far the slowest at this size and are not expected to help on 1,240 rows. A 4-fold
# pilot (DE, valence) also dropped CatBoost and LightGBMLarge: ~half of each fit's time, and never top
# on inner-CV macro-F1 (LightGBMXT / XGBoost were).
FAMILIES = ["GBM", "XGB", "RF", "XT"]
HYPERPARAMETERS = {k: v for k, v in get_hyperparameter_config("default").items() if k in FAMILIES}
HYPERPARAMETERS["GBM"] = [c for c in HYPERPARAMETERS["GBM"] if c.get("ag_args", {}).get("name_suffix") != "Large"]

# asymmetry was the weakest domain for SVM and RF in the 4-class run; MSE is redundant with Hjorth
# (Week 3 |r| 0.72), mid-ranked, and the largest domain (320 features). Their feature files stay.
MODEL_DOMAINS = ["de", "hjorth", "microstates"]

# binary targets taken from the quadrant label ("HVLA" -> valence "HV", arousal "LA"), so the H/L
# thresholds are the ones the quadrants already use
TARGETS = {"valence": ("HV", "LV"), "arousal": ("HA", "LA")}


def binary_labels(quadrant: pd.Series, target: str) -> np.ndarray:
    return (quadrant.str[:2] if target == "valence" else quadrant.str[2:]).to_numpy()


def _fit_fold(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, train: np.ndarray, test: np.ndarray):
    inner = np.empty(len(train), dtype=int)
    for k, (_, val) in enumerate(GroupKFold(INNER_FOLDS).split(train, groups=groups[train])):
        inner[val] = k
    data = x.iloc[train].assign(label=y[train], inner_fold=inner)
    with tempfile.TemporaryDirectory() as path:
        predictor = TabularPredictor(label="label", eval_metric="f1_macro", sample_weight="balance_weight",
                                     groups="inner_fold", path=path, verbosity=0)
        predictor.fit(data, hyperparameters=HYPERPARAMETERS, num_bag_folds=INNER_FOLDS, num_stack_levels=0,
                      num_cpus=CPUS_PER_FIT, num_gpus=0,
                      ag_args_ensemble={"model_random_seed": SEED, "fold_fitting_strategy": "sequential_local"})
        y_pred = predictor.predict(x.iloc[test]).to_numpy()
        board = predictor.leaderboard()[["model", "score_val", "fit_time"]]
        board = board.assign(chosen=board["model"] == predictor.model_best)   # the model predict() used
    return y_pred, board.assign(subject_id=groups[test][0])


def loso_autogluon(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold predictions aligned with y, and every model's inner-CV macro-F1 per held-out subject."""
    splits = list(LeaveOneGroupOut().split(x, y, groups))
    fits = Parallel(n_jobs=OUTER_JOBS)(delayed(_fit_fold)(x, y, groups, tr, te) for tr, te in splits)
    y_pred = np.empty_like(y)
    for (_, test), (pred, _) in zip(splits, fits):
        y_pred[test] = pred
    return y_pred, pd.concat([board for _, board in fits], ignore_index=True)


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary metrics for one held-out subject.

    Macro-F1 averages only the classes this subject actually has, so an absent class is not counted
    as a 0; n_classes records how many were averaged.
    """
    present = np.unique(y_true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # "y_pred contains classes not in y_true"
        balanced = balanced_accuracy_score(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "balanced_accuracy": balanced,
        "n_classes": len(present),
    }
