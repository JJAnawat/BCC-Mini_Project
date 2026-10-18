"""Nested leave-one-subject-out harness and fold metrics, identical for every feature domain.

Outer loop: one fold per held-out subject. Inner loop: GroupKFold(5) over the training subjects,
so hyperparameters are tuned on cross-subject macro-F1 too. Inputs are expected to be z-scored
per subject already (bcc.analysis.zscore_within_subject).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.svm import SVC

INNER_FOLDS = 5
SEED = 42


def _svm(n_features: int):
    # gamma scaled by 1/n_features (sklearn's "scale" for unit-variance inputs), so the grid means
    # the same thing for 24 microstate features and 320 MSE features
    grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [g / n_features for g in (0.01, 0.1, 1, 10)]}
    return SVC(kernel="rbf", class_weight="balanced"), grid


def _rf(n_features: int):
    # kept small on purpose: one 150-tree fit takes ~1 s here and nested LOSO needs 640 per domain
    grid = {"max_depth": [None, 10], "min_samples_leaf": [1, 5]}
    return RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=SEED), grid


MODELS = {"svm": _svm, "rf": _rf}


def nested_loso(x: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Out-of-fold predictions aligned with y, and the tuned params for each held-out subject."""
    estimator, grid = MODELS[model](x.shape[1])
    y_pred = np.empty_like(y)
    best = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=GroupKFold(INNER_FOLDS), n_jobs=-1)
        search.fit(x[train], y[train], groups=groups[train])
        y_pred[test] = search.predict(x[test])
        best.append({"subject_id": groups[test][0], "inner_macro_f1": search.best_score_, **search.best_params_})
    return y_pred, pd.DataFrame(best)


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Quadrant metrics for one held-out subject, plus valence/arousal accuracy from the same predictions.

    Macro-F1 and balanced accuracy average only the classes this subject actually has (subject 23 has
    no LVHA), so an absent class is not counted as a 0; n_classes records how many were averaged.
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
        "valence_accuracy": np.mean([t[0] == p[0] for t, p in zip(y_true, y_pred)]),   # H/L in "HVLA"
        "arousal_accuracy": np.mean([t[2] == p[2] for t, p in zip(y_true, y_pred)]),
    }
