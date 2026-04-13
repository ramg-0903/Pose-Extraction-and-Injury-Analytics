"""
anomaly.py  —  Stage 4b
=======================
PCA-based anomaly detection for injury risk scoring.

Each rep is projected into PCA space and reconstructed back.
High reconstruction error = rep deviates from the normal movement pattern
= elevated injury risk signal.

Entry point:
    run_anomaly(df, X_scaled, feat_cols, output_dir, ...) -> pd.DataFrame

Outputs saved to output_dir:
    pca_model.pkl                        fitted PCA model
    anomaly_scores.csv                   per-rep risk score and flags
    anomaly_feature_contributions.csv    global feature error contributions
    anomaly_per_rep_contributions.csv    per-rep feature error breakdown
    anomaly_meta.json                    run parameters and statistics
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

MIN_COMPONENTS = 3   # never select fewer than this many PCA components


def run_anomaly(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    feat_cols: list,
    output_dir: Path,
    variance_threshold: float = 0.95,
    high_risk_percentile: float = 90.0,
) -> pd.DataFrame:
    """Compute PCA reconstruction error as a per-rep injury risk score.

    Args:
        df:                   DataFrame aligned to X_scaled rows.
        X_scaled:             (N, F) scaled feature matrix from prepare_features.
        feat_cols:            Feature column names — must match X_scaled columns.
        output_dir:           Where to save outputs.
        variance_threshold:   Retain enough components to explain this fraction
                              of variance (default 0.95).
        high_risk_percentile: Reps above this error percentile are flagged
                              high_risk (default 90 = top 10%).

    Returns:
        df with three new columns:
            risk_score      float  reconstruction error (higher = more unusual)
            risk_score_pct  float  percentile rank [0, 100]
            is_high_risk    int    1 if risk_score > high_risk_percentile
    """
    # ── Input validation ──────────────────────────────────────────────────────
    # Point 11 — strict index alignment check
    assert len(df) == len(X_scaled), \
        f"Row mismatch: df={len(df)} X_scaled={len(X_scaled)}"
    assert list(df.index) == list(range(len(df))), \
        "df index must be 0..N-1 after reset_index. Call reset_index(drop=True) first."
    assert len(feat_cols) == X_scaled.shape[1], \
        f"Feature mismatch: feat_cols={len(feat_cols)} X cols={X_scaled.shape[1]}"

    # Point 2 — ensure no binary columns entered PCA
    n_samples, n_features = X_scaled.shape
    for i, col in enumerate(feat_cols):
        unique_vals = np.unique(X_scaled[:, i])
        if len(unique_vals) <= 2:
            raise ValueError(
                f"Column '{col}' appears binary ({unique_vals}) — "
                f"binary features must not be passed to PCA."
            )

    # Point 8 — sample vs feature count check
    if n_samples < n_features * 2:
        import warnings
        warnings.warn(
            f"Low sample-to-feature ratio: {n_samples} samples / {n_features} features. "
            f"PCA may overfit. Consider reducing PCA_FEATURES in config."
        )

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(X_scaled)

    # Point 7 — enforce minimum component count
    n_components = max(int(pca.n_components_), MIN_COMPONENTS)
    if n_components > pca.n_components_:
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(X_scaled)

    X_proj  = pca.transform(X_scaled)
    X_recon = pca.inverse_transform(X_proj)

    explained_var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA components  : {n_components} "
          f"(explains {explained_var*100:.1f}% variance)")

    # ── Reconstruction error ──────────────────────────────────────────────────
    errors   = np.mean((X_scaled - X_recon) ** 2, axis=1)
    mean_err = float(errors.mean())
    std_err  = float(errors.std())

    # Point 6 — percentile-based threshold (robust to non-Gaussian errors)
    threshold = float(np.percentile(errors, high_risk_percentile))

    # Point 12 — guard for near-zero variance (degenerate case)
    if std_err < 1e-6:
        import warnings
        warnings.warn(
            "Reconstruction error variance is near-zero — all reps reconstruct "
            "similarly. Anomaly detection results may not be meaningful."
        )

    n_high = int((errors > threshold).sum())
    print(f"  Error mean      : {mean_err:.4f}  std: {std_err:.4f}")
    print(f"  High-risk thresh: {threshold:.4f}  "
          f"(p{high_risk_percentile:.0f})")
    print(f"  High-risk reps  : {n_high} / {len(errors)} "
          f"({n_high/len(errors)*100:.1f}%)")

    # ── Global feature contributions ──────────────────────────────────────────
    # Per-feature MSE across all reps — which features the model reconstructs
    # poorly overall. Identifies biomechanical dimensions driving anomalies.
    feature_errors = np.mean((X_scaled - X_recon) ** 2, axis=0)
    feat_contrib   = pd.DataFrame({
        "feature":         feat_cols,
        "mean_sq_error":   feature_errors,
        "pca_loading_sum": np.abs(pca.components_).sum(axis=0),
    }).sort_values("mean_sq_error", ascending=False)

    print(f"\n  Top 5 anomaly-driving features:")
    for _, row in feat_contrib.head(5).iterrows():
        print(f"    {row['feature']:<35}  mse={row['mean_sq_error']:.4f}")

    # ── Per-rep feature contributions (point 10) ──────────────────────────────
    # For each rep, which features contributed most to its reconstruction error.
    # Shape: (N, F) — each row sums to that rep's total squared error * F
    per_rep_errors = (X_scaled - X_recon) ** 2   # (N, F)
    id_cols        = [c for c in ["session_id", "rep_index", "global_rep_id"]
                      if c in df.columns]
    per_rep_df = pd.DataFrame(
        per_rep_errors,
        columns=[f"err_{c}" for c in feat_cols]
    )
    if id_cols:
        per_rep_df = pd.concat(
            [df[id_cols].reset_index(drop=True), per_rep_df], axis=1
        )

    # ── Attach to dataframe ───────────────────────────────────────────────────
    pct_ranks        = rankdata(errors, method="average") / len(errors) * 100
    df               = df.copy()
    df["risk_score"]     = np.round(errors, 6)
    df["risk_score_pct"] = np.round(pct_ranks, 2)
    df["is_high_risk"]   = (errors > threshold).astype(int)

    # ── Save outputs ──────────────────────────────────────────────────────────
    joblib.dump(pca, output_dir / "pca_model.pkl")

    score_cols = ["risk_score", "risk_score_pct", "is_high_risk"]
    df[id_cols + score_cols].to_csv(
        output_dir / "anomaly_scores.csv", index=False
    )
    feat_contrib.to_csv(
        output_dir / "anomaly_feature_contributions.csv", index=False
    )
    per_rep_df.to_csv(
        output_dir / "anomaly_per_rep_contributions.csv", index=False
    )

    with open(output_dir / "anomaly_meta.json", "w") as f:
        json.dump({
            "n_components":          int(n_components),
            "explained_variance":    round(explained_var, 4),
            "variance_threshold":    variance_threshold,
            "high_risk_percentile":  high_risk_percentile,
            "high_risk_threshold":   round(threshold, 6),
            "error_mean":            round(mean_err, 6),
            "error_std":             round(std_err, 6),
            "n_high_risk":           int(n_high),
            "n_total":               int(len(errors)),
            "high_risk_pct":         round(float(n_high / len(errors) * 100), 2),
            "min_components_guard":  MIN_COMPONENTS,
        }, f, indent=2)

    print(f"\n  Saved: pca_model.pkl, anomaly_scores.csv, "
          f"anomaly_feature_contributions.csv, "
          f"anomaly_per_rep_contributions.csv, anomaly_meta.json")

    return df