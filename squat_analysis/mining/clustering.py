"""
clustering.py  —  Stage 4a
===========================
DTW k-means clustering on squat rep trajectories.

NOTE: Not implemented in this version. Clustering requires per-session
trajectories.npz files which are only saved when run.py is called with
--save-trajectories. With 1 rep per video the cluster stability is also
limited. Marked as future work.

run_clustering returns df unchanged so run_mining.py continues cleanly.
"""

import warnings
from pathlib import Path
import pandas as pd


def run_clustering(
    df: pd.DataFrame,
    output_dir: Path,
    max_k: int = 6,
    n_init: int = 10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Stub — DTW clustering not implemented in this version.

    Returns df unchanged. Logs a clear message so the user knows
    clustering was skipped and why.
    """
    warnings.warn(
        "DTW clustering is not implemented in this version. "
        "Requires trajectories.npz files (run pipeline with --save-trajectories) "
        "and sufficient reps per session (>1 rep/video). "
        "Skipping — df returned unchanged."
    )
    print("  DTW clustering  : skipped (see clustering.py for details)")
    return df