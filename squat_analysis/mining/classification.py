"""
Pseudo-labelling and Random Forest classification.

Assigns biomechanical threshold-based risk labels, then evaluates
three model variants (full, honest, binary) via stratified CV.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from squat_analysis.config import (
    RF_N_ESTIMATORS, RF_CV_FOLDS, RF_N_SEEDS, RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# Features that define the pseudo-labels — excluding these gives the
# "honest" generalisation estimate (Objective 1 in the report).
LABEL_FEATURES = {
    "trunk_lean_max", "trunk_lean_mean",
    "trunk_lean_at_bottom", "trunk_lean_range",
}


def assign_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign low/medium/high risk from biomechanical thresholds.

    Thresholds (calibrated on 2,492-rep dataset):
        high_risk   : trunk_lean_max > 45°  (Schoenfeld 2010)
        low_risk    : knee_flexion < 80° AND trunk_lean < 20°
        medium_risk : everything else

    Symmetry excluded from labelling — it's NaN for ~85% of side-view
    videos and would introduce severe sampling bias.
    """
    df = df.copy()
    labels = []

    for _, row in df.iterrows():
        trunk_max = row.get("trunk_lean_max", np.nan)
        knee_bot  = row.get("knee_flexion_at_bottom", np.nan)

        if not np.isnan(trunk_max) and trunk_max > 45.0:
            labels.append("high_risk")
        elif (not np.isnan(knee_bot) and knee_bot < 80.0 and
              not np.isnan(trunk_max) and trunk_max < 20.0):
            labels.append("low_risk")
        else:
            labels.append("medium_risk")

    df["risk_label"] = labels

    counts = pd.Series(labels).value_counts()
    for label, count in counts.items():
        logger.info("  %s: %d (%.1f%%)", label, count, count / len(labels) * 100)

    return df


def run_classification(
    df: pd.DataFrame,
    feat_cols: list,
    output_dir: Path,
) -> dict:
    """Train RF on pseudo-labels with stratified CV.

    Runs three variants:
        Full   — all features (inflated — label features included)
        Honest — excludes trunk lean features used to define labels
        Binary — risky vs safe (merges high+medium → risky)

    Also computes feature importance across RF_N_SEEDS random seeds.
    """
    if "risk_label" not in df.columns:
        logger.warning("No risk_label column — skipping classification.")
        return {}

    sub = df[feat_cols + ["risk_label"]].dropna()
    sub = sub[sub["risk_label"].isin(["low_risk", "medium_risk", "high_risk"])]

    if len(sub) < 20:
        logger.warning("Only %d labelled rows — skipping.", len(sub))
        return {}

    X = sub[feat_cols].values.astype(float)
    y = sub["risk_label"].values

    counts = pd.Series(y).value_counts()
    logger.info("  Class distribution: %s", dict(counts))
    if counts.min() < 5:
        logger.warning("Class '%s' has only %d samples — skipping.",
                        counts.idxmin(), counts.min())
        return {}

    # Honest feature set (no label-defining features)
    honest_cols = [c for c in feat_cols if c not in LABEL_FEATURES]
    sub_honest  = df[honest_cols + ["risk_label"]].dropna()
    sub_honest  = sub_honest[sub_honest["risk_label"].isin(
                      ["low_risk", "medium_risk", "high_risk"])]
    X_honest = sub_honest[honest_cols].values.astype(float)
    y_honest = sub_honest["risk_label"].values

    cv = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)

    models = {
        "Dummy (baseline)":    Pipeline([("sc", RobustScaler()),
                                          ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([("sc", RobustScaler()),
                                          ("clf", LogisticRegression(
                                              max_iter=1000, class_weight="balanced",
                                              random_state=RANDOM_SEED))]),
        "Random Forest":       Pipeline([("sc", RobustScaler()),
                                          ("clf", RandomForestClassifier(
                                              n_estimators=RF_N_ESTIMATORS,
                                              class_weight="balanced",
                                              random_state=RANDOM_SEED))]),
    }

    results = {}

    # Full variant
    logger.info("  CV — Full (n=%d, f=%d):", len(sub), len(feat_cols))
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipeline, X, y, cv=cv,
                                    scoring=["accuracy", "f1_macro"])
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_macro"].mean()
        results[f"{name} (full)"] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
        logger.info("    %-22s  %.3f  %.3f", name, acc, f1)

    # Honest variant
    logger.info("  CV — Honest (n=%d, f=%d):", len(sub_honest), len(honest_cols))
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipeline, X_honest, y_honest, cv=cv,
                                    scoring=["accuracy", "f1_macro"])
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_macro"].mean()
        results[f"{name} (honest)"] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
        logger.info("    %-22s  %.3f  %.3f", name, acc, f1)

    # Binary variant (risky vs safe)
    y_bin = np.where(y_honest == "low_risk", "safe", "risky")
    logger.info("  CV — Binary (n=%d):", len(y_bin))
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipeline, X_honest, y_bin, cv=cv,
                                    scoring=["accuracy", "f1_weighted"])
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_weighted"].mean()
        results[f"{name} (binary)"] = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
        logger.info("    %-22s  %.3f  %.3f", name, acc, f1)

    # Feature importance (full feature set, averaged over N seeds)
    logger.info("  Feature importance (%d seeds):", RF_N_SEEDS)
    scaler_fi   = RobustScaler()
    X_scaled_fi = scaler_fi.fit_transform(X)
    importances = []
    for seed in range(RF_N_SEEDS):
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                    class_weight="balanced", random_state=seed)
        rf.fit(X_scaled_fi, y)
        importances.append(rf.feature_importances_)

    feat_imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": np.array(importances).mean(axis=0),
        "std":        np.array(importances).std(axis=0),
    }).sort_values("importance", ascending=False)

    for _, row in feat_imp.head(5).iterrows():
        logger.info("    %-35s  %.4f ± %.4f", row["feature"], row["importance"], row["std"])

    feat_imp.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save classification results
    clf_results = {
        "cv_results":              results,
        "top_5_features":          feat_imp.head(5)["feature"].tolist(),
        "n_samples_full":          len(sub),
        "n_samples_honest":        len(sub_honest),
        "n_features_full":         len(feat_cols),
        "n_features_honest":       len(honest_cols),
        "label_features_excluded": list(LABEL_FEATURES),
        "cv_folds":                RF_CV_FOLDS,
        "n_importance_seeds":      RF_N_SEEDS,
        "random_seed":             RANDOM_SEED,
        "class_distribution":      {k: int(v) for k, v in counts.items()},
    }
    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(clf_results, f, indent=2)

    return clf_results