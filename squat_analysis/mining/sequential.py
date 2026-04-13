"""
sequential.py  —  Stage 4d
===========================
Sequential pattern mining on rep sequences.

NOTE: Not implemented in this version. Meaningful sequential mining
requires multiple reps per session to detect intra-session fatigue
patterns. With 1 rep per video and no subject grouping, sequences
are synthetic and results would not be statistically valid.

Marked as a limitation in the project report. run_sequential returns
cleanly so run_mining.py continues without error.
"""

import warnings
from pathlib import Path
import pandas as pd


def run_sequential(
    df: pd.DataFrame,
    flag_cols: list,
    output_dir: Path,
) -> pd.DataFrame:
    """Stub — sequential pattern mining not implemented in this version.

    Returns df unchanged. Logs a clear message explaining the limitation.
    """
    warnings.warn(
        "Sequential pattern mining is not implemented in this version. "
        "Requires multiple reps per session for meaningful intra-session "
        "fatigue pattern detection. With 1 rep per video, sequences are "
        "not temporally meaningful. Skipping — noted as project limitation."
    )
    print("  Sequential mining: skipped (1 rep/video — no intra-session sequences)")
    return df