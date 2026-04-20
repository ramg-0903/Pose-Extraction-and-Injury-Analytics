"""
DTW k-means clustering on squat rep trajectories.

Not implemented — requires per-session trajectories.npz files (produced
by running the pipeline with --save-trajectories) and sufficient reps
per session.  With 1 rep per video, cluster stability is limited.
Marked as future work in the project report.

run_clustering returns df unchanged so the orchestrator continues cleanly.
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_clustering(
    df: pd.DataFrame,
    output_dir: Path,
    max_k: int = 6,
    n_init: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Stub — returns df unchanged."""
    warnings.warn(
        "DTW clustering not implemented. Requires trajectories.npz and "
        "multiple reps per session.  Skipping."
    )
    logger.info("  DTW clustering: skipped (not implemented)")
    return df
