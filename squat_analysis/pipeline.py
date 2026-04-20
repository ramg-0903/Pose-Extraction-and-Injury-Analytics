"""
Single-video pipeline — chains Stages 1→3.

This is the core entry point for both CLI and web API usage.
For a single video: extract landmarks → preprocess → extract features.

    from squat_analysis.pipeline import run_pipeline

    result = run_pipeline("path/to/video.mp4")
    # result["df"]          → DataFrame with one row per rep
    # result["session_dir"] → Path to session directory with all artifacts
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from squat_analysis.extraction import extract
from squat_analysis.preprocessing import preprocess
from squat_analysis.features import extract_features

logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str,
    session_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    save_trajectories: bool = False,
) -> dict:
    """Run the full Stages 1→3 pipeline on a single video.

    Args:
        video_path:        Path to input video (.mp4 / .mov / .avi).
        session_id:        Identifier for this recording (defaults to file stem).
        output_dir:        Override output directory for all stages.
        model_path:        Override MediaPipe model path.
        max_frames:        Process only first N frames (for quick tests).
        save_trajectories: Save per-rep angle trajectory arrays.

    Returns:
        dict with keys:
            session_dir  (Path)         — where all artifacts are saved
            df           (DataFrame)    — one row per rep, all features + flags
            n_reps       (int)          — number of reps detected
            duration_s   (float)        — total processing time
            metadata     (dict)         — Stage 1 metadata (fps, resolution, etc.)
    """
    t_start = time.time()
    video_path = Path(video_path)

    logger.info("Pipeline start — %s", video_path.name)

    # Stage 1 — Extract raw landmarks
    session_dir = extract(
        video_path=str(video_path),
        session_id=session_id,
        output_dir=output_dir,
        model_path=model_path,
        max_frames=max_frames,
    )

    # Stage 2 — Preprocess (NaN fill, scale, smooth, confidence)
    preprocess(str(session_dir))

    # Stage 3 — Feature extraction (rep segmentation + biomechanical features)
    df = extract_features(
        str(session_dir),
        save_trajectories=save_trajectories,
    )

    duration = time.time() - t_start

    # Load Stage 1 metadata for the result
    import json
    with open(session_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(
        "Pipeline complete — %d reps, %d features, %.1fs",
        len(df), len(df.columns), duration,
    )

    return {
        "session_dir": session_dir,
        "df":          df,
        "n_reps":      len(df),
        "duration_s":  round(duration, 2),
        "metadata":    metadata,
    }