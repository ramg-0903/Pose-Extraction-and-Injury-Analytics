"""
Mining orchestrator — runs Stage 4 on master_dataset.csv.

This is the offline training script.  Run once after batch_process.py
to produce trained models (scaler.pkl) and analysis outputs.

Usage:
    python run_mining.py
    python run_mining.py --dataset data/processed/master_dataset.csv
    python run_mining.py --skip-clustering --skip-sequential
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from squat_analysis.config import (
    PROCESSED_DIR, OUTPUTS_DIR,
    ALL_FLAGS, EXTRA_FLAG_DEFS, ARM_EXCLUDE_FLAGS,
    RANDOM_SEED, DTW_MAX_K, DTW_N_INIT,
    PCA_VARIANCE_THRESHOLD, PCA_ANOMALY_SIGMA,
    ARM_MIN_SUPPORT, ARM_MIN_CONFIDENCE, ARM_MIN_LIFT,
    RF_N_ESTIMATORS, RF_CV_FOLDS, RF_N_SEEDS,
)
from squat_analysis.mining.preparation import (
    load_and_validate, add_extra_flags, prepare_features,
)
from squat_analysis.mining.classification import (
    assign_pseudo_labels, run_classification,
)
from squat_analysis.mining.anomaly import run_anomaly
from squat_analysis.mining.clustering import run_clustering
from squat_analysis.mining.association import run_association
from squat_analysis.mining.sequential import run_sequential

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mining layer (Stage 4)")
    parser.add_argument("--dataset", default=str(PROCESSED_DIR / "master_dataset.csv"))
    parser.add_argument("--skip-clustering",  action="store_true")
    parser.add_argument("--skip-association", action="store_true")
    parser.add_argument("--skip-sequential",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Mining layer — seed=%d | dataset=%s", RANDOM_SEED, args.dataset)

    # 1. Load and validate
    df = load_and_validate(args.dataset)
    df = add_extra_flags(df)

    # 2. Prepare scaled features
    X_scaled, scaler, valid_pos, feat_cols = prepare_features(df, OUTPUTS_DIR)
    df_valid = df.iloc[valid_pos].copy().reset_index(drop=True)
    assert len(df_valid) == len(X_scaled)

    # 3. PCA anomaly detection
    logger.info("[3] PCA anomaly detection")
    df_valid = run_anomaly(
        df=df_valid, X_scaled=X_scaled, feat_cols=feat_cols,
        output_dir=OUTPUTS_DIR, variance_threshold=PCA_VARIANCE_THRESHOLD,
        high_risk_percentile=90.0,
    )

    # 4. Clustering
    if not args.skip_clustering:
        logger.info("[4] DTW k-means clustering")
        df_valid = run_clustering(
            df=df_valid, output_dir=OUTPUTS_DIR,
            max_k=DTW_MAX_K, n_init=DTW_N_INIT, random_seed=RANDOM_SEED,
        )
    else:
        logger.info("[4] Clustering — skipped")

    # 5. Pseudo-labelling
    logger.info("[5] Pseudo-labelling")
    df_valid = assign_pseudo_labels(df_valid)

    # 6. Association rules
    if not args.skip_association:
        logger.info("[6] Association rule mining")
        run_association(
            df=df_valid,
            flag_cols=[f for f in ALL_FLAGS if f not in ARM_EXCLUDE_FLAGS],
            output_dir=OUTPUTS_DIR,
            min_support=ARM_MIN_SUPPORT,
            min_confidence=ARM_MIN_CONFIDENCE,
            min_lift=ARM_MIN_LIFT,
        )
    else:
        logger.info("[6] Association rules — skipped")

    # 7. Sequential
    if not args.skip_sequential:
        logger.info("[7] Sequential pattern mining")
        df_seq = (df_valid.sort_values("global_rep_id").reset_index(drop=True)
                  if "global_rep_id" in df_valid.columns else df_valid.copy())
        run_sequential(df=df_seq, flag_cols=ALL_FLAGS, output_dir=OUTPUTS_DIR)
    else:
        logger.info("[7] Sequential — skipped")

    # 8. Classification
    logger.info("[8] Classification")
    run_classification(df_valid, feat_cols, OUTPUTS_DIR)

    # Save enriched dataset
    df_valid.to_csv(OUTPUTS_DIR / "enriched_dataset.csv", index=False)

    # Save experiment config
    mining_config = {
        "dataset":                args.dataset,
        "random_seed":            RANDOM_SEED,
        "n_rows":                 len(df_valid),
        "n_features":             len(feat_cols),
        "features":               feat_cols,
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
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

    logger.info("Mining complete → %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()