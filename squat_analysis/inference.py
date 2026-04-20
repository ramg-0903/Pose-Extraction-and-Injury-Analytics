"""
Real-time inference — scores a single video using pre-trained artifacts.

Loads the scaler and PCA model produced by the offline training pipeline
(run_mining.py) and applies them to features from a single video.
This is the module the FastAPI endpoint calls.

Typical usage:
    from squat_analysis.inference import SquatScorer

    scorer = SquatScorer("outputs/")          # load once at app startup
    result = scorer.score_video("my_squat.mp4")  # call per upload

The scorer chains:
    pipeline.py (Stages 1→3) → scale → PCA anomaly → pseudo-labels → result dict
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from squat_analysis.config import PCA_FEATURES, DISCRETIZATION_THRESHOLDS
from squat_analysis.pipeline import run_pipeline

logger = logging.getLogger(__name__)


class SquatScorer:
    """Stateful scorer that loads trained artifacts once and scores many videos.

    Artifacts loaded from model_dir (produced by run_mining.py):
        scaler.pkl           RobustScaler fitted on the training set
        pca_model.pkl        PCA model for anomaly detection
        anomaly_meta.json    error distribution stats for threshold calibration
    """

    def __init__(self, model_dir: str):
        """Load pre-trained artifacts.

        Args:
            model_dir: Directory containing scaler.pkl, pca_model.pkl, and
                       anomaly_meta.json from a completed run_mining.py run.

        Raises:
            FileNotFoundError: If required artifacts are missing.
        """
        model_dir = Path(model_dir)

        scaler_path = model_dir / "scaler.pkl"
        pca_path    = model_dir / "pca_model.pkl"
        meta_path   = model_dir / "anomaly_meta.json"

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scaler.pkl not found in {model_dir}. "
                "Run `python run_mining.py` first to train."
            )
        if not pca_path.exists():
            raise FileNotFoundError(
                f"pca_model.pkl not found in {model_dir}. "
                "Run `python run_mining.py` first to train."
            )

        self.scaler = joblib.load(scaler_path)
        self.pca    = joblib.load(pca_path)

        # Load training-set error distribution for calibrated thresholds
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.train_error_mean = meta.get("error_mean", 0.0)
            self.train_error_std  = meta.get("error_std", 1.0)
            self.train_threshold  = meta.get("high_risk_threshold", None)
        else:
            warnings.warn(
                "anomaly_meta.json not found — using uncalibrated thresholds."
            )
            self.train_error_mean = 0.0
            self.train_error_std  = 1.0
            self.train_threshold  = None

        # Feature columns the scaler expects (same order as training)
        self.feat_cols = PCA_FEATURES

        logger.info(
            "SquatScorer loaded — scaler=%s pca=%d components "
            "threshold=%.4f",
            type(self.scaler).__name__,
            self.pca.n_components_,
            self.train_threshold or 0.0,
        )

    def score_video(
        self,
        video_path: str,
        session_id: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        """Run the full pipeline on a video and score each rep.

        Args:
            video_path: Path to input video file.
            session_id: Optional session identifier.
            output_dir: Optional override for intermediate file storage.
            max_frames: Process only first N frames (for testing).

        Returns:
            dict with keys:
                session_id   (str)
                metadata     (dict)   — fps, resolution, detection rate
                n_reps       (int)
                duration_s   (float)  — total processing time
                reps         (list[dict]) — per-rep results, each containing:
                    rep_index, risk_label, risk_score, risk_score_pct,
                    form_errors (list[str]), all features, all flags
                summary      (dict)   — session-level aggregates
        """
        # Stage 1→3
        pipeline_result = run_pipeline(
            video_path=video_path,
            session_id=session_id,
            output_dir=output_dir,
            max_frames=max_frames,
        )

        df          = pipeline_result["df"]
        session_dir = pipeline_result["session_dir"]
        metadata    = pipeline_result["metadata"]

        # Score the features
        scored_df = self._score_features(df)

        # Build per-rep result dicts
        reps = []
        for _, row in scored_df.iterrows():
            rep_dict = row.to_dict()

            # Collect active form-error flags into a readable list
            form_errors = []
            for feat_name, (threshold, direction, flag_name) in DISCRETIZATION_THRESHOLDS.items():
                flag_val = rep_dict.get(flag_name, 0)
                if flag_val == 1:
                    form_errors.append(flag_name)

            rep_dict["form_errors"] = form_errors
            reps.append(rep_dict)

        # Session-level summary
        summary = self._build_summary(scored_df)

        result = {
            "session_id":  metadata.get("session_id", "unknown"),
            "metadata":    metadata,
            "n_reps":      len(reps),
            "duration_s":  pipeline_result["duration_s"],
            "reps":        reps,
            "summary":     summary,
        }

        logger.info(
            "Scored %d reps — %d high-risk, %d form errors total",
            len(reps),
            sum(1 for r in reps if r.get("risk_label") == "high_risk"),
            sum(len(r["form_errors"]) for r in reps),
        )

        return result

    def score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public interface — score a pre-computed feature DataFrame.

        Use this when you already have features (e.g. from batch processing)
        and don't need to re-run the video pipeline.
        """
        return self._score_features(df)

    def _score_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling, PCA anomaly detection, and pseudo-labelling.

        Handles missing PCA_FEATURES gracefully — columns absent from df
        are filled with the training median (0.0 after RobustScaler).
        """
        df = df.copy()

        # Align features to training column order, fill missing with 0
        X = pd.DataFrame(index=df.index, columns=self.feat_cols, dtype=float)
        for col in self.feat_cols:
            if col in df.columns:
                X[col] = df[col].values.astype(float)
            else:
                X[col] = 0.0

        # Clip outliers the same way prepare_features does
        outlier_cols = [c for c in self.feat_cols
                        if any(kw in c for kw in ["jerk", "vel", "ratio", "cost"])]
        for col in outlier_cols:
            p01 = float(X[col].quantile(0.01))
            p99 = float(X[col].quantile(0.99))
            X[col] = X[col].clip(lower=p01, upper=p99)

        # Median-impute NaN temporal features
        impute_cols = [c for c in self.feat_cols
                       if any(kw in c for kw in ["vel", "jerk", "time", "ratio", "cost", "end"])]
        for col in impute_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(float(X[col].median()))

        # Fill any remaining NaN with 0 (robust for single-rep videos)
        X = X.fillna(0.0)

        X_arr = X.values

        # Scale using the training-fitted scaler
        X_scaled = self.scaler.transform(X_arr)

        # PCA reconstruction error → anomaly score
        X_proj  = self.pca.transform(X_scaled)
        X_recon = self.pca.inverse_transform(X_proj)
        errors  = np.mean((X_scaled - X_recon) ** 2, axis=1)

        # Percentile rank relative to training distribution
        if self.train_error_std > 1e-9:
            z_scores = (errors - self.train_error_mean) / self.train_error_std
        else:
            z_scores = np.zeros_like(errors)

        df["risk_score"]     = np.round(errors, 6)
        df["risk_score_pct"] = np.round(
            rankdata(errors, method="average") / len(errors) * 100, 2,
        )
        df["risk_z_score"] = np.round(z_scores, 4)

        # High-risk flag using training-calibrated threshold
        if self.train_threshold is not None:
            df["is_high_risk"] = (errors > self.train_threshold).astype(int)
        else:
            # Fallback: top 10% within this session
            df["is_high_risk"] = (
                errors > np.percentile(errors, 90)
            ).astype(int)

        # Pseudo-labels (same thresholds as training)
        df["risk_label"] = df.apply(self._assign_label, axis=1)

        return df

    @staticmethod
    def _assign_label(row) -> str:
        """Assign risk label using the same thresholds as training.

        Thresholds (from classification.py / project report):
            high_risk   : trunk_lean_max > 45°
            low_risk    : knee_flexion < 80° AND trunk_lean < 20°
            medium_risk : everything else
        """
        trunk_max = row.get("trunk_lean_max", np.nan)
        knee_bot  = row.get("knee_flexion_at_bottom", np.nan)

        if not np.isnan(trunk_max) and trunk_max > 45.0:
            return "high_risk"
        if (not np.isnan(knee_bot) and knee_bot < 80.0 and
                not np.isnan(trunk_max) and trunk_max < 20.0):
            return "low_risk"
        return "medium_risk"

    @staticmethod
    def _build_summary(df: pd.DataFrame) -> dict:
        """Session-level aggregates across all reps."""
        n = len(df)
        if n == 0:
            return {"n_reps": 0}

        risk_counts = df["risk_label"].value_counts().to_dict()

        # Collect all form errors across reps
        flag_cols = [c for c in DISCRETIZATION_THRESHOLDS.values()
                     for c in [c[2]]]  # flag_name from threshold tuples
        flag_names = [v[2] for v in DISCRETIZATION_THRESHOLDS.values()]
        flag_prevalence = {}
        for flag in flag_names:
            if flag in df.columns:
                flag_prevalence[flag] = round(float(df[flag].mean()), 3)

        # Key metric averages
        metrics = {}
        for col in ["trunk_lean_max", "trunk_lean_mean",
                     "knee_flexion_at_bottom", "knee_flexion_range",
                     "descent_ascent_ratio", "normalized_jerk_cost",
                     "risk_score"]:
            if col in df.columns:
                metrics[col] = {
                    "mean": round(float(df[col].mean()), 2),
                    "std":  round(float(df[col].std()), 2),
                    "min":  round(float(df[col].min()), 2),
                    "max":  round(float(df[col].max()), 2),
                }

        return {
            "n_reps":          n,
            "risk_distribution": risk_counts,
            "flag_prevalence": flag_prevalence,
            "metrics":         metrics,
            "mean_confidence": round(float(df["rep_mean_confidence"].mean()), 3)
                               if "rep_mean_confidence" in df.columns else None,
        }
