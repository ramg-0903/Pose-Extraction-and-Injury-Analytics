"""
Data preparation for the mining layer.

Handles: loading master_dataset.csv, validation, extra flag computation,
feature selection, outlier clipping, NaN imputation, and RobustScaler fitting.
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from squat_analysis.config import PCA_FEATURES, EXTRA_FLAG_DEFS

logger = logging.getLogger(__name__)


def load_and_validate(dataset_path: str) -> pd.DataFrame:
    """Load master_dataset.csv with sanity checks.

    Raises RuntimeError if fewer than 20 rows or required columns are missing.
    """
    df = pd.read_csv(dataset_path)
    logger.info("Loaded: %d rows × %d columns", len(df), len(df.columns))

    if len(df) < 20:
        raise RuntimeError(f"Only {len(df)} rows — need at least 20 for mining.")

    missing = [c for c in PCA_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Report high-NaN and zero-variance columns
    nan_pct = df[PCA_FEATURES].isnull().mean() * 100
    high_nan = nan_pct[nan_pct > 50]
    if len(high_nan) > 0:
        warnings.warn(f"Columns with >50% NaN:\n{high_nan.to_string()}")

    zero_var = [c for c in PCA_FEATURES
                if c in df.columns and df[c].dropna().std() < 1e-6]
    if zero_var:
        warnings.warn(f"Near-zero variance columns: {zero_var}")

    logger.info("  NaN cols (>50%%): %d | Zero-var: %d", len(high_nan), len(zero_var))
    return df


def add_extra_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 extra binary flags from continuous features.

    Flags are defined in config.EXTRA_FLAG_DEFS.  "p75" thresholds are
    data-driven (75th percentile, capped at p99 to resist outliers).
    """
    df = df.copy()
    for flag_name, (col, direction, threshold) in EXTRA_FLAG_DEFS.items():
        if col not in df.columns:
            df[flag_name] = np.nan
            logger.info("  %s — SKIPPED (column '%s' missing)", flag_name, col)
            continue

        if threshold == "p75":
            p99 = float(df[col].quantile(0.99))
            thr = float(df[col].clip(upper=p99).quantile(0.75))
            if thr < 1e-3:
                warnings.warn(f"Flag '{flag_name}': p75={thr:.6f} near-zero.")
        else:
            thr = float(threshold)

        logger.info("  %s: %s %s %.4f", flag_name, col, direction, thr)
        df[flag_name] = (df[col] > thr if direction == "gt"
                         else df[col] < thr).astype(float)
        df.loc[df[col].isna(), flag_name] = np.nan

    return df


def prepare_features(df: pd.DataFrame, output_dir: Path) -> tuple:
    """Select, clean, clip, impute, and scale features for PCA/classification.

    Returns (X_scaled, scaler, valid_positions, feature_columns).

    Design decisions:
      - Jerk/velocity outliers clipped at p1/p99 before scaling to prevent
        PCA overflow (these columns regularly exceed 72,000).
      - NaN temporal features imputed with column median to recover rows
        that would otherwise be dropped entirely.
      - RobustScaler (IQR-based) used instead of StandardScaler because
        heavy-tailed jerk distributions inflate std and compress real signal.
    """
    nan_pct   = df[PCA_FEATURES].isnull().mean()
    feat_cols = [c for c in PCA_FEATURES
                 if c in df.columns and nan_pct.get(c, 1.0) < 0.50]

    X = df[feat_cols].copy()

    # Clip extreme outliers in jerk/velocity/ratio columns
    outlier_cols = [c for c in feat_cols
                    if any(kw in c for kw in ["jerk", "vel", "ratio", "cost"])]
    for col in outlier_cols:
        X[col] = X[col].clip(lower=float(X[col].quantile(0.01)),
                              upper=float(X[col].quantile(0.99)))

    # Median-impute temporal features
    impute_cols = [c for c in feat_cols
                   if any(kw in c for kw in ["vel", "jerk", "time", "ratio", "cost", "end"])]
    for col in impute_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(float(X[col].median()))

    valid_mask = X.notna().all(axis=1)
    valid_pos  = np.where(valid_mask.values)[0]
    X_clean    = X.values[valid_pos].astype(float)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_clean)

    joblib.dump(scaler, output_dir / "scaler.pkl")

    logger.info("  Features: %d×%d | Dropped: %d | Imputed cols: %d | Clipped: %d",
                X_scaled.shape[0], X_scaled.shape[1],
                len(df) - len(valid_pos), len(impute_cols), len(outlier_cols))

    return X_scaled, scaler, valid_pos, feat_cols