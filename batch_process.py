"""
Batch processing — run all videos through Stages 1→3, combine into
master_dataset.csv.

Usage:
    python batch_process.py
    python batch_process.py --dry-run
    python batch_process.py --reprocess --limit 10

Output:
    data/processed/master_dataset.csv  — all reps combined
    data/processed/batch_report.json   — processing summary
"""

import argparse
import json
import logging
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from squat_analysis.config import RAW_VIDEO_DIR, PROCESSED_DIR
from squat_analysis.pipeline import run_pipeline

logger = logging.getLogger(__name__)


# ── Row-level quality filters ─────────────────────────────────────────────────

def _check_detection_rate(session_dir: Path, min_rate: float) -> tuple:
    """(passed, detection_rate).  Fails if too many frames were missed."""
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return False, 0.0
    with open(meta_path) as f:
        rate = float(json.load(f).get("detection_rate", 0.0))
    return rate >= min_rate, rate


def _check_motion_range(session_dir: Path, min_range: float = 0.05) -> tuple:
    """(passed, max_range).  Fails if no meaningful vertical motion detected."""
    scaled_path = session_dir / "landmarks_scaled.npy"
    if not scaled_path.exists():
        return False, 0.0

    lm = np.load(scaled_path)
    ranges = []
    for l_idx, r_idx in [(23, 24), (25, 26), (27, 28)]:  # hip, knee, ankle
        l = lm[:, l_idx, 1]
        r = lm[:, r_idx, 1]
        sig = np.where(~np.isnan(l) & ~np.isnan(r), (l + r) / 2.0,
              np.where(~np.isnan(l), l, r))
        rng = float(np.nanmax(sig) - np.nanmin(sig)) if not np.isnan(sig).all() else 0.0
        ranges.append(rng)

    max_range = max(ranges)
    return max_range >= min_range, max_range


def _is_half_rep(row: pd.Series, min_phase_ratio: float = 0.15) -> bool:
    """True if one phase is near-zero relative to total rep duration.

    Half-reps come from videos that start or end mid-squat — the
    segmentation algorithm detects a partial descent or ascent.
    """
    descent = float(row.get("descent_time_s", 0))
    ascent  = float(row.get("ascent_time_s", 0))
    total   = descent + ascent
    if total < 1e-6:
        return True
    return (descent / total) < min_phase_ratio or (ascent / total) < min_phase_ratio


def _has_valid_angles(row: pd.Series) -> bool:
    """True if knee flexion at bottom is within plausible range [30°, 170°]."""
    knee = row.get("knee_flexion_at_bottom", np.nan)
    if not np.isnan(knee) and not (30.0 <= knee <= 170.0):
        return False
    return True


# ── Per-video processor ───────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    min_detection_rate: float,
    filter_half_reps: bool,
    reprocess: bool,
) -> dict:
    """Run full pipeline on one video with quality gates.

    Returns a result dict with keys: status, reason, rows, rows_dropped,
    detection_rate, motion_range, df, duration_s.
    """
    session_id  = video_path.stem
    session_dir = PROCESSED_DIR / session_id
    t_start     = time.time()

    result = {
        "session_id": session_id, "video": str(video_path),
        "status": "failed", "reason": "", "rows": 0, "rows_dropped": 0,
        "detection_rate": 0.0, "motion_range": 0.0, "df": None, "duration_s": 0.0,
    }

    try:
        # Skip if already processed
        features_csv = session_dir / "features.csv"
        if features_csv.exists() and not reprocess:
            df = pd.read_csv(features_csv)
            result.update(status="skipped", reason="already processed",
                          rows=len(df), df=df)
            result["duration_s"] = time.time() - t_start
            return result

        # Run Stages 1→3 via pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline_result = run_pipeline(
                video_path=str(video_path),
                session_id=session_id,
            )

        session_dir = pipeline_result["session_dir"]
        df = pipeline_result["df"]

        # Quality gate: detection rate
        passed, det_rate = _check_detection_rate(session_dir, min_detection_rate)
        result["detection_rate"] = det_rate
        if not passed:
            result.update(status="filtered_out",
                          reason=f"detection_rate={det_rate:.2f} < {min_detection_rate}")
            result["duration_s"] = time.time() - t_start
            return result

        # Quality gate: motion range
        passed, motion_range = _check_motion_range(session_dir)
        result["motion_range"] = motion_range
        if not passed:
            result.update(status="filtered_out",
                          reason=f"motion_range={motion_range:.4f} < 0.05")
            result["duration_s"] = time.time() - t_start
            return result

        if len(df) == 0:
            result.update(status="no_reps", reason="no reps extracted")
            result["duration_s"] = time.time() - t_start
            return result

        # Row-level quality filters
        original_count = len(df)
        drop_mask = pd.Series([False] * len(df))

        if filter_half_reps:
            drop_mask = drop_mask | df.apply(_is_half_rep, axis=1)
        drop_mask = drop_mask | ~df.apply(_has_valid_angles, axis=1)

        df_clean     = df[~drop_mask].copy()
        rows_dropped = int(drop_mask.sum())

        if len(df_clean) == 0:
            result.update(status="filtered_out",
                          reason=f"all {original_count} rows failed quality filters",
                          rows_dropped=rows_dropped)
            result["duration_s"] = time.time() - t_start
            return result

        result.update(status="ok", reason="", rows=len(df_clean),
                      rows_dropped=rows_dropped, df=df_clean)

    except RuntimeError as e:
        result.update(status="no_reps", reason=str(e))
    except Exception as e:
        result.update(status="failed", reason=str(e),
                      _traceback=traceback.format_exc())

    result["duration_s"] = time.time() - t_start
    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    min_detection_rate: float = 0.50,
    filter_half_reps: bool = False,
    reprocess: bool = False,
    limit: int = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Process all videos → master_dataset.csv."""
    videos = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    for ext in ["*.mov", "*.avi", "*.MP4", "*.MOV"]:
        videos += sorted(RAW_VIDEO_DIR.glob(ext))

    if not videos:
        logger.warning("No videos found in %s", RAW_VIDEO_DIR)
        return pd.DataFrame()

    if limit:
        videos = videos[:limit]

    logger.info("Batch: %d videos | min_det=%.2f | half_rep_filter=%s | reprocess=%s",
                len(videos), min_detection_rate, filter_half_reps, reprocess)

    if dry_run:
        for v in videos:
            done = (PROCESSED_DIR / v.stem / "features.csv").exists()
            print(f"  {'[done]' if done else '[todo]'} {v.name}")
        return pd.DataFrame()

    all_dfs = []
    report  = {"total": len(videos), "ok": 0, "skipped": 0, "no_reps": 0,
               "filtered_out": 0, "failed": 0, "total_rows": 0,
               "total_dropped": 0, "videos": []}

    for i, video_path in enumerate(videos):
        print(f"[{i+1:3d}/{len(videos)}] {video_path.name}", end=" ... ", flush=True)

        result = process_video(video_path, min_detection_rate,
                               filter_half_reps, reprocess)
        status = result["status"]
        report[status] = report.get(status, 0) + 1

        if status in ("ok", "skipped") and result["df"] is not None:
            all_dfs.append(result["df"])
            report["total_rows"]    += result["rows"]
            report["total_dropped"] += result["rows_dropped"]
            print(f"OK  ({result['rows']} rows, {result['duration_s']:.1f}s)")
        else:
            print(f"{status.upper()} — {result['reason'][:60]}")

        report["videos"].append({
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in result.items() if k not in ("df", "_traceback")
        })

    # Combine
    if not all_dfs:
        logger.warning("No data to combine.")
        combined = pd.DataFrame()
    else:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.insert(0, "global_rep_id", range(len(combined)))

        out_csv = PROCESSED_DIR / "master_dataset.csv"
        combined.to_csv(out_csv, index=False)
        logger.info("Master dataset: %s (%d × %d)",
                     out_csv, combined.shape[0], combined.shape[1])

    # Save report
    report_path = PROCESSED_DIR / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Report: ok=%d skipped=%d no_reps=%d filtered=%d failed=%d",
                report["ok"], report["skipped"], report["no_reps"],
                report["filtered_out"], report["failed"])
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch process all squat videos → master_dataset.csv",
    )
    parser.add_argument("--dry-run",            action="store_true")
    parser.add_argument("--reprocess",          action="store_true")
    parser.add_argument("--limit",              type=int, default=None)
    parser.add_argument("--min-detection-rate", type=float, default=0.50)
    parser.add_argument("--filter-half-reps",   action="store_true",
                        help="Enable half-rep filtering (off by default)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    run_batch(
        min_detection_rate=args.min_detection_rate,
        filter_half_reps=args.filter_half_reps,
        reprocess=args.reprocess,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()