"""
run_mining.py
=============
Orchestrates the full mining layer (Stage 4) on master_dataset.csv.

Runs in order:
    1. Data preparation  — load, validate, scale, add extra flags
    2. anomaly.py        — PCA risk scoring
    3. clustering.py     — DTW k-means form archetypes
    4. association.py    — Apriori co-occurring error rules
    5. sequential.py     — cross-dataset pattern mining
    6. Pseudo-labelling  — cluster + threshold → risk labels
    7. Classification    — Random Forest with CV (answers proposal KPIs)

Usage:
    python run_mining.py
    python run_mining.py --dataset data/processed/master_dataset.csv
    python run_mining.py --dataset data/processed/master_dataset.csv --skip-clustering
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from squat_analysis.config import (
    PROCESSED_DIR, OUTPUTS_DIR,
    PCA_FEATURES, DTW_FEATURES, ALL_FLAGS,
    PIPELINE_FLAGS, EXTRA_FLAG_DEFS,
    RANDOM_SEED, DTW_MAX_K, DTW_N_INIT,
    PCA_VARIANCE_THRESHOLD, PCA_ANOMALY_SIGMA,
    ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT, ARM_EXCLUDE_FLAGS,
    RF_N_ESTIMATORS, RF_CV_FOLDS, RF_N_SEEDS,
)
from squat_analysis.mining.anomaly     import run_anomaly
from squat_analysis.mining.clustering  import run_clustering
from squat_analysis.mining.association import run_association
from squat_analysis.mining.sequential  import run_sequential


# =============================================================================
# Data preparation
# =============================================================================

def load_and_validate(dataset_path: str) -> pd.DataFrame:
    """Load master_dataset.csv with sanity checks."""
    df = pd.read_csv(dataset_path)

    print(f"  Loaded: {len(df)} rows × {len(df.columns)} columns")

    # Minimum row count
    if len(df) < 20:
        raise RuntimeError(
            f"Only {len(df)} rows — need at least 20 for meaningful mining."
        )

    # Check required feature columns exist
    missing = [c for c in PCA_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing}")

    # Report NaN situation
    nan_pct = df[PCA_FEATURES].isnull().mean() * 100
    high_nan = nan_pct[nan_pct > 50]
    if len(high_nan) > 0:
        warnings.warn(
            f"Columns with >50% NaN (will be dropped from PCA/clustering):\n"
            f"{high_nan.to_string()}"
        )

    # Check variance — constant columns are useless
    zero_var = [c for c in PCA_FEATURES
                if c in df.columns and df[c].dropna().std() < 1e-6]
    if zero_var:
        warnings.warn(f"Near-zero variance columns: {zero_var}")

    print(f"  NaN cols (>50%): {len(high_nan)}")
    print(f"  Zero-var cols  : {len(zero_var)}")

    return df


def add_extra_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 extra binary flags from existing continuous features.

    Computed at mining time — no pipeline re-run needed.
    p75 threshold is data-driven (75th percentile of dataset).
    Warns if computed threshold is near-zero (skewed distribution).
    """
    df = df.copy()
    print("  Extra flag thresholds:")
    for flag_name, (col, direction, threshold) in EXTRA_FLAG_DEFS.items():
        if col not in df.columns:
            df[flag_name] = np.nan
            print(f"    {flag_name:<22} SKIPPED (column '{col}' not found)")
            continue
        if threshold == "p75":
            # Cap at 99th percentile before computing p75 to prevent
            # extreme outliers inflating the threshold to unusable values
            p99 = float(df[col].quantile(0.99))
            col_capped = df[col].clip(upper=p99)
            thr = float(col_capped.quantile(0.75))
            if thr < 1e-3:
                warnings.warn(
                    f"Flag '{flag_name}': p75 threshold={thr:.6f} is near-zero. "
                    f"Distribution may be degenerate — flag will be mostly 0."
                )
        else:
            thr = float(threshold)
        print(f"    {flag_name:<22} {col} {direction} {thr:.4f}")
        df[flag_name] = (df[col] > thr).astype(float) if direction == "gt" \
                        else (df[col] < thr).astype(float)
        df.loc[df[col].isna(), flag_name] = np.nan
    return df


def prepare_features(df: pd.DataFrame, output_dir: Path) -> tuple:
    """Select, clean, and scale features for PCA and anomaly detection.

    Fix 1 — jerk/velocity outliers clipped at p99 before scaling to
             eliminate PCA overflow warnings and stabilise the feature space.
    Fix 2 — NaN temporal features imputed with column median to recover
             rows that would otherwise be dropped.
    """
    nan_pct   = df[PCA_FEATURES].isnull().mean()
    feat_cols = [c for c in PCA_FEATURES if c in df.columns
                 and nan_pct.get(c, 1.0) < 0.50]

    expected_order = [c for c in PCA_FEATURES if c in feat_cols]
    assert feat_cols == expected_order, "Feature column order mismatch"

    X = df[feat_cols].copy()

    # Fix 1 — clip extreme outliers in jerk/velocity before scaling
    # These columns regularly hit 72,000+ which causes PCA overflow
    outlier_cols = [c for c in feat_cols if any(
        kw in c for kw in ["jerk", "vel", "ratio", "cost"]
    )]
    for col in outlier_cols:
        p99 = float(X[col].quantile(0.99))
        p01 = float(X[col].quantile(0.01))
        X[col] = X[col].clip(lower=p01, upper=p99)

    # Fix 2 — impute NaN temporal features with median
    # Recovers rows with partial NaN rather than dropping them
    impute_cols = [c for c in feat_cols if any(
        kw in c for kw in ["vel", "jerk", "time", "ratio", "cost", "end"]
    )]
    for col in impute_cols:
        if X[col].isna().any():
            median_val = float(X[col].median())
            X[col]     = X[col].fillna(median_val)

    valid_mask = X.notna().all(axis=1)
    valid_pos  = np.where(valid_mask.values)[0]
    X_clean    = X.values[valid_pos].astype(float)

    # RobustScaler uses IQR instead of mean/std — resistant to heavy tails
    # in jerk/velocity features that StandardScaler cannot handle cleanly
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)

    joblib.dump(scaler, output_dir / "scaler.pkl")

    n_imputed = sum(df[c].isna().sum() for c in impute_cols if c in df.columns)
    print(f"  Feature matrix : {X_scaled.shape[0]} rows × {X_scaled.shape[1]} features")
    print(f"  Rows dropped   : {len(df) - len(valid_pos)} (NaN after imputation)")
    print(f"  Cells imputed  : {n_imputed} (median fill on temporal features)")
    print(f"  Outlier cols   : {len(outlier_cols)} clipped at p1/p99")
    print(f"  Scaler saved   : {output_dir / 'scaler.pkl'}")

    return X_scaled, scaler, valid_pos, feat_cols


# =============================================================================
# Pseudo-labelling
# =============================================================================

def assign_pseudo_labels(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Assign low/medium/high risk labels based on biomechanical thresholds.

    Thresholds calibrated to your dataset distribution (verified on 2492 reps):
        high_risk   : trunk_lean_max > 45°
                      → form error clearly visible in trunk mechanics
        low_risk    : knee_flexion_at_bottom < 80° AND trunk_lean_max < 20°
                      → excellent depth + upright trunk
        medium_risk : everything else

    Symmetry removed from high_risk trigger — symmetry is NaN for ~85% of
    side-view videos and would create a severe sampling bias if used here.
    Symmetry is still used in association rules where NaN → 0 (absent).
    """
    df = df.copy()
    labels = []

    for _, row in df.iterrows():
        trunk_max = row.get("trunk_lean_max",         np.nan)
        knee_bot  = row.get("knee_flexion_at_bottom", np.nan)

        if not np.isnan(trunk_max) and trunk_max > 45.0:
            labels.append("high_risk")
        elif (not np.isnan(knee_bot)  and knee_bot  < 80.0) and \
             (not np.isnan(trunk_max) and trunk_max < 20.0):
            labels.append("low_risk")
        else:
            labels.append("medium_risk")

    df["risk_label"] = labels

    counts = pd.Series(labels).value_counts()
    print(f"  Pseudo-labels:")
    for label, count in counts.items():
        print(f"    {label:<15} {count:5d}  ({count/len(labels)*100:.1f}%)")

    if cluster_col in df.columns:
        for c in sorted(df[cluster_col].dropna().unique()):
            cluster_labels = df.loc[df[cluster_col] == c, "risk_label"]
            dominant_pct   = cluster_labels.value_counts(normalize=True).iloc[0] * 100
            print(f"    Cluster {int(c)}: {cluster_labels.value_counts().to_dict()} "
                  f"(dominant={dominant_pct:.0f}%)")

    return df


# =============================================================================
# Classification (answers proposal KPIs)
# =============================================================================

def run_classification(
    df: pd.DataFrame,
    feat_cols: list,
    output_dir: Path,
) -> dict:
    """Train Random Forest on pseudo-labels with cross-validation.

    Runs two variants:
        Full     — all feat_cols including trunk lean (labelling features)
        Honest   — excludes features used to define the pseudo-labels
                   (trunk_lean_max, trunk_lean_mean, trunk_lean_at_bottom,
                   trunk_lean_range). This is the primary reportable result.

    Answers:
        Objective 1 — classification accuracy (target ≥85%)
        Objective 2 — top 5 features with stable importance scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import warnings

    if "risk_label" not in df.columns:
        print("  Skipping classification — no risk_label column.")
        return {}

    sub = df[feat_cols + ["risk_label"]].dropna()
    sub = sub[sub["risk_label"].isin(["low_risk", "medium_risk", "high_risk"])]

    if len(sub) < 20:
        print(f"  Skipping classification — only {len(sub)} labelled rows.")
        return {}

    X = sub[feat_cols].values.astype(float)
    y = sub["risk_label"].values

    counts = pd.Series(y).value_counts()
    print(f"  Class distribution: {dict(counts)}")
    if counts.min() < 5:
        print(f"  Skipping — class '{counts.idxmin()}' has only "
              f"{counts.min()} samples (need ≥5).")
        return {}
    if counts.min() < 10:
        print(f"  Warning: small minority class ({counts.min()} samples) "
              f"— CV results may be noisy.")

    # Features used to define pseudo-labels — excluding these gives
    # the honest generalisation estimate
    LABEL_FEATURES = {
        "trunk_lean_max", "trunk_lean_mean",
        "trunk_lean_at_bottom", "trunk_lean_range",
    }
    honest_cols = [c for c in feat_cols if c not in LABEL_FEATURES]
    sub_honest  = df[honest_cols + ["risk_label"]].dropna()
    sub_honest  = sub_honest[sub_honest["risk_label"].isin(
                      ["low_risk", "medium_risk", "high_risk"])]
    X_honest    = sub_honest[honest_cols].values.astype(float)
    y_honest    = sub_honest["risk_label"].values

    cv = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)

    models = {
        "Dummy (baseline)":    Pipeline([("sc", RobustScaler()),
                                          ("clf", DummyClassifier(
                                              strategy="most_frequent"))]),
        "Logistic Regression": Pipeline([("sc", RobustScaler()),
                                          ("clf", LogisticRegression(
                                              max_iter=1000,
                                              class_weight="balanced",
                                              random_state=RANDOM_SEED))]),
        "Random Forest":       Pipeline([("sc", RobustScaler()),
                                          ("clf", RandomForestClassifier(
                                              n_estimators=RF_N_ESTIMATORS,
                                              class_weight="balanced",
                                              random_state=RANDOM_SEED))]),
    }

    results = {}

    # ── Full variant (all features) ───────────────────────────────────────────
    print(f"\n  {RF_CV_FOLDS}-fold CV — Full features (n={len(sub)}, "
          f"f={len(feat_cols)}):")
    print(f"  {'Model':<22}  Accuracy   Macro-F1")
    print(f"  {'-'*48}")
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipeline, X, y, cv=cv,
                                    scoring=["accuracy","f1_macro"])
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_macro"].mean()
        results[f"{name} (full)"] = {"accuracy": round(acc,4), "f1_macro": round(f1,4)}
        print(f"  {name:<22}  {acc:.3f}      {f1:.3f}")

    # ── Honest variant (no labelling features) ────────────────────────────────
    print(f"\n  {RF_CV_FOLDS}-fold CV — Honest (no label features, "
          f"n={len(sub_honest)}, f={len(honest_cols)}):")
    print(f"  Note: trunk lean features excluded — these define the labels.")
    print(f"  {'Model':<22}  Accuracy   Macro-F1")
    print(f"  {'-'*48}")
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(pipeline, X_honest, y_honest, cv=cv,
                                    scoring=["accuracy","f1_macro"])
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_macro"].mean()
        results[f"{name} (honest)"] = {"accuracy": round(acc,4), "f1_macro": round(f1,4)}
        print(f"  {name:<22}  {acc:.3f}      {f1:.3f}")

    # ── Fix 3 — Binary classification (risky vs safe) ────────────────────────
    # Addresses class imbalance — merges high+medium into "risky",
    # keeps low_risk as "safe". Gives cleaner, more reliable metrics.
    y_bin       = np.where(y_honest == "low_risk", "safe", "risky")
    counts_bin  = pd.Series(y_bin).value_counts()
    print(f"\n  {RF_CV_FOLDS}-fold CV — Binary (risky vs safe, "
          f"n={len(y_bin)}):")
    print(f"  Class distribution: {dict(counts_bin)}")
    print(f"  {'Model':<22}  Accuracy   F1-risky")
    print(f"  {'-'*48}")
    for name, pipeline in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_res = cross_validate(
                pipeline, X_honest, y_bin, cv=cv,
                scoring=["accuracy",
                         "f1_weighted"],
            )
        acc = cv_res["test_accuracy"].mean()
        f1  = cv_res["test_f1_weighted"].mean()
        results[f"{name} (binary)"] = {"accuracy": round(acc,4), "f1_weighted": round(f1,4)}
        print(f"  {name:<22}  {acc:.3f}      {f1:.3f}")

    # ── Feature importance (full feature set) ─────────────────────────────────
    print(f"\n  Feature importance (mean ± std over {RF_N_SEEDS} seeds):")
    scaler_fi   = RobustScaler()
    X_scaled_fi = scaler_fi.fit_transform(X)
    importances = []
    for seed in range(RF_N_SEEDS):
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                    class_weight="balanced",
                                    random_state=seed)
        rf.fit(X_scaled_fi, y)
        importances.append(rf.feature_importances_)

    imp_mean = np.array(importances).mean(axis=0)
    imp_std  = np.array(importances).std(axis=0)
    feat_imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": imp_mean,
        "std":        imp_std,
    }).sort_values("importance", ascending=False)

    print(f"  {'Feature':<35}  Importance  Std")
    for _, row in feat_imp.head(10).iterrows():
        print(f"  {row['feature']:<35}  {row['importance']:.4f}      {row['std']:.4f}")

    # Full-data report
    rf_final = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                      class_weight="balanced",
                                      random_state=RANDOM_SEED)
    rf_final.fit(X_scaled_fi, y)
    y_pred     = rf_final.predict(X_scaled_fi)
    report_str = classification_report(y, y_pred, zero_division=0)
    print(f"\n  Classification report (train):\n{report_str}")

    feat_imp.to_csv(output_dir / "feature_importance.csv", index=False)

    # Get honest RF result for primary reporting
    rf_acc_honest = results.get("Random Forest (honest)", {}).get("accuracy", None)
    rf_acc_full   = results.get("Random Forest (full)",   {}).get("accuracy", None)

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
        "primary_result_note":     (
            f"Honest RF accuracy {rf_acc_honest} (kinematic features only) "
            f"is the primary reportable result. Full RF {rf_acc_full} "
            f"includes trunk lean features used to define labels."
        ),
    }
    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(clf_results, f, indent=2)

    return clf_results


# =============================================================================
# Main orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run full mining layer on master_dataset.csv"
    )
    parser.add_argument("--dataset", default=str(PROCESSED_DIR / "master_dataset.csv"))
    parser.add_argument("--skip-clustering",  action="store_true")
    parser.add_argument("--skip-association", action="store_true")
    parser.add_argument("--skip-sequential",  action="store_true")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Mining layer — Stage 4  (seed={RANDOM_SEED})")
    print(f"  Dataset : {args.dataset}")
    print(f"{'='*60}\n")

    # ── 1. Load and validate ──────────────────────────────────────────────────
    print("[1/7] Loading and validating dataset...")
    df = load_and_validate(args.dataset)
    df = add_extra_flags(df)

    # ── 2. Prepare scaled features ────────────────────────────────────────────
    print("\n[2/7] Preparing feature matrix...")
    X_scaled, scaler, valid_pos, feat_cols = prepare_features(df, OUTPUTS_DIR)

    # Point 7 — build df_valid from integer positions, reset index
    df_valid = df.iloc[valid_pos].copy().reset_index(drop=True)
    assert len(df_valid) == len(X_scaled), \
        f"Row alignment error: df_valid={len(df_valid)} X_scaled={len(X_scaled)}"

    # ── 3. Anomaly detection ──────────────────────────────────────────────────
    print("\n[3/7] PCA anomaly detection...")
    assert len(df_valid) == len(X_scaled)
    df_valid = run_anomaly(
        df=df_valid,
        X_scaled=X_scaled,
        feat_cols=feat_cols,
        output_dir=OUTPUTS_DIR,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        high_risk_percentile=90.0,
    )
    assert len(df_valid) == len(X_scaled), "Row count changed after anomaly"

    # ── 4. Clustering ─────────────────────────────────────────────────────────
    if not args.skip_clustering:
        print("\n[4/7] DTW k-means clustering...")
        # Point 1 & 11 — clustering uses its own data loader (trajectories.npz)
        # X_scaled is NOT passed — clustering module loads time-series directly
        assert len(df_valid) == len(X_scaled)
        df_valid = run_clustering(
            df=df_valid,
            output_dir=OUTPUTS_DIR,
            max_k=DTW_MAX_K,
            n_init=DTW_N_INIT,
            random_seed=RANDOM_SEED,
        )
        assert len(df_valid) == len(X_scaled), "Row count changed after clustering"
    else:
        print("\n[4/7] Clustering skipped.")

    # ── 5. Pseudo-labelling ───────────────────────────────────────────────────
    print("\n[5/7] Pseudo-labelling...")
    df_valid = assign_pseudo_labels(df_valid)

    # ── 6. Association rules ──────────────────────────────────────────────────
    if not args.skip_association:
        print("\n[6/7] Association rule mining...")
        run_association(
            df=df_valid,
            flag_cols=[f for f in ALL_FLAGS if f not in ARM_EXCLUDE_FLAGS],
            output_dir=OUTPUTS_DIR,
            min_support=ARM_MIN_SUPPORT,
            min_confidence=ARM_MIN_CONFIDENCE,
            min_lift=ARM_MIN_LIFT,
        )
    else:
        print("\n[6/7] Association rules skipped.")

    # ── 7. Sequential ─────────────────────────────────────────────────────────
    if not args.skip_sequential:
        print("\n[7/7] Sequential pattern mining...")
        # Point 12 — sort by global_rep_id for deterministic ordering
        df_seq = df_valid.sort_values("global_rep_id").reset_index(drop=True) \
                 if "global_rep_id" in df_valid.columns else df_valid.copy()
        run_sequential(
            df=df_seq,
            flag_cols=ALL_FLAGS,
            output_dir=OUTPUTS_DIR,
        )
    else:
        print("\n[7/7] Sequential mining skipped.")

    # Classification always runs
    print("\n  --- Classification (proposal KPIs) ---")
    clf_results = run_classification(df_valid, feat_cols, OUTPUTS_DIR)

    # ── Save enriched dataset ─────────────────────────────────────────────────
    enriched_path = OUTPUTS_DIR / "enriched_dataset.csv"
    df_valid.to_csv(enriched_path, index=False)
    print(f"\n  Enriched dataset : {enriched_path}  ({len(df_valid)} rows)")

    # ── Save experiment config ────────────────────────────────────────────────
    mining_config = {
        "dataset":                args.dataset,
        "random_seed":            RANDOM_SEED,
        "n_rows":                 len(df_valid),
        "n_features_used":        len(feat_cols),
        "features_used":          feat_cols,
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
        "pca_anomaly_sigma":      PCA_ANOMALY_SIGMA,
        "dtw_max_k":              DTW_MAX_K,
        "dtw_n_init":             DTW_N_INIT,
        "arm_min_support":        ARM_MIN_SUPPORT,
        "arm_min_confidence":     ARM_MIN_CONFIDENCE,
        "arm_min_lift":           ARM_MIN_LIFT,
        "arm_exclude_flags":      ARM_EXCLUDE_FLAGS,
        "rf_n_estimators":        RF_N_ESTIMATORS,
        "rf_cv_folds":            RF_CV_FOLDS,
        "rf_n_seeds":             RF_N_SEEDS,
        "all_flags":              ALL_FLAGS,
        "extra_flag_defs":        {k: list(v) for k, v in EXTRA_FLAG_DEFS.items()},
    }
    with open(OUTPUTS_DIR / "mining_config.json", "w") as f:
        json.dump(mining_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Mining complete. Outputs: {OUTPUTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()