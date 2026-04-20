"""
Sequential pattern mining on rep sequences.

Not implemented — meaningful sequential mining requires multiple reps
per session to detect intra-session fatigue patterns.  With 1 rep per
video and no subject grouping, sequences are not temporally meaningful.
Marked as a limitation in the project report.

run_sequential returns df unchanged so the orchestrator continues cleanly.
"""

import logging
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_sequential(
    df: pd.DataFrame,
    flag_cols: list,
    output_dir: Path,
) -> pd.DataFrame:
    """Stub — returns df unchanged."""
    warnings.warn(
        "Sequential mining not implemented. Requires multiple reps per "
        "session for temporal patterns.  Skipping."
    )
    logger.info("  Sequential mining: skipped (1 rep/video)")
    return df
