"""
PCA-based anomaly detection for injury risk scoring.

Each rep is projected into PCA space and reconstructed.  High
reconstruction error indicates deviation from normal movement
patterns — an elevated injury risk signal.

Outputs (saved to output_dir):
    pca_model.pkl                       fitted PCA model
    anomaly_scores.csv                  per-rep risk score and flags
    anomaly_feature_contributions.csv   global feature error contributions
    anomaly_per_rep_contributions.csv   per-rep feature error breakdown
    anomaly_meta.json                   run parameters and statistics
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

MIN_COMPONENTS = 3  # floor on PCA component count


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
        df:                   DataFrame aligned 1:1 with X_scaled rows.
        X_scaled:             (N, F) scaled feature matrix.
        feat_cols:            Feature column names matching X_scaled columns.
        output_dir:           Where to save outputs.
        variance_threshold:   Retain components explaining this fraction.
        high_risk_percentile: Reps above this error percentile → high_risk.

    Returns:
        df with new columns: risk_score, risk_score_pct, is_high_risk.
    """
    n_samples, n_features = X_scaled.shape

    # ── Validation ────────────────────────────────────────────────────────
    assert len(df) == n_samples, \
        f"Row mismatch: df={len(df)} X_scaled={n_samples}"
    assert list(df.index) == list(range(len(df))), \
        "df index must be 0..N-1 (call reset_index(drop=True) first)."
    assert len(feat_cols) == n_features, \
        f"Feature mismatch: feat_cols={len(feat_cols)} X cols={n_features}"

    # Reject binary columns — they don't belong in PCA
    for i, col in enumerate(feat_cols):
        if len(np.unique(X_scaled[:, i])) <= 2:
            raise ValueError(f"Column '{col}' is binary — must not enter PCA.")

    if n_samples < n_features * 2:
        warnings.warn(
            f"Low sample-to-feature ratio ({n_samples}/{n_features}). "
            "PCA may overfit — consider reducing PCA_FEATURES."
        )

    # ── Fit PCA ───────────────────────────────────────────────────────────
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    pca.fit(X_scaled)

    # Enforce minimum component count
    n_components = max(int(pca.n_components_), MIN_COMPONENTS)
    if n_components > pca.n_components_:
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(X_scaled)

    X_proj  = pca.transform(X_scaled)
    X_recon = pca.inverse_transform(X_proj)
    explained_var = float(pca.explained_variance_ratio_.sum())

    logger.info("  PCA: %d components (%.1f%% variance)",
                n_components, explained_var * 100)

    # ── Reconstruction error ──────────────────────────────────────────────
    errors   = np.mean((X_scaled - X_recon) ** 2, axis=1)
    mean_err = float(errors.mean())
    std_err  = float(errors.std())

    # Percentile-based threshold (robust to non-Gaussian error distributions)
    threshold = float(np.percentile(errors, high_risk_percentile))
    n_high    = int((errors > threshold).sum())

    if std_err < 1e-6:
        warnings.warn(
            "Reconstruction error variance near-zero — all reps reconstruct "
            "similarly.  Anomaly detection may not be meaningful."
        )

    logger.info("  Error: mean=%.4f std=%.4f | threshold=%.4f (p%.0f) | "
                "high-risk=%d/%d (%.1f%%)",
                mean_err, std_err, threshold, high_risk_percentile,
                n_high, len(errors), n_high / len(errors) * 100)

    # ── Feature contributions (global) ────────────────────────────────────
    # Per-feature MSE across all reps — identifies which biomechanical
    # dimensions drive anomalies
    feature_errors = np.mean((X_scaled - X_recon) ** 2, axis=0)
    feat_contrib = pd.DataFrame({
        "feature":         feat_cols,
        "mean_sq_error":   feature_errors,
        "pca_loading_sum": np.abs(pca.components_).sum(axis=0),
    }).sort_values("mean_sq_error", ascending=False)

    for _, row in feat_contrib.head(5).iterrows():
        logger.info("    %-35s mse=%.4f", row["feature"], row["mean_sq_error"])

    # ── Per-rep feature contributions ─────────────────────────────────────
    per_rep_errors = (X_scaled - X_recon) ** 2
    id_cols = [c for c in ["session_id", "rep_index", "global_rep_id"]
               if c in df.columns]
    per_rep_df = pd.DataFrame(
        per_rep_errors, columns=[f"err_{c}" for c in feat_cols],
    )
    if id_cols:
        per_rep_df = pd.concat(
            [df[id_cols].reset_index(drop=True), per_rep_df], axis=1,
        )

    # ── Attach scores to DataFrame ────────────────────────────────────────
    pct_ranks = rankdata(errors, method="average") / len(errors) * 100
    df = df.copy()
    df["risk_score"]     = np.round(errors, 6)
    df["risk_score_pct"] = np.round(pct_ranks, 2)
    df["is_high_risk"]   = (errors > threshold).astype(int)

    # ── Save ──────────────────────────────────────────────────────────────
    joblib.dump(pca, output_dir / "pca_model.pkl")

    score_cols = ["risk_score", "risk_score_pct", "is_high_risk"]
    df[id_cols + score_cols].to_csv(
        output_dir / "anomaly_scores.csv", index=False,
    )
    feat_contrib.to_csv(
        output_dir / "anomaly_feature_contributions.csv", index=False,
    )
    per_rep_df.to_csv(
        output_dir / "anomaly_per_rep_contributions.csv", index=False,
    )

    with open(output_dir / "anomaly_meta.json", "w") as f:
        json.dump({
            "n_components":         int(n_components),
            "explained_variance":   round(explained_var, 4),
            "variance_threshold":   variance_threshold,
            "high_risk_percentile": high_risk_percentile,
            "high_risk_threshold":  round(threshold, 6),
            "error_mean":           round(mean_err, 6),
            "error_std":            round(std_err, 6),
            "n_high_risk":          int(n_high),
            "n_total":              int(len(errors)),
            "high_risk_pct":        round(float(n_high / len(errors) * 100), 2),
            "min_components_guard": MIN_COMPONENTS,
        }, f, indent=2)

    logger.info("  Saved: pca_model.pkl, anomaly_scores.csv, "
                "anomaly_feature_contributions.csv")
    return df
