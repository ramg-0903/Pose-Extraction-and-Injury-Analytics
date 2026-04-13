"""
batch_process.py
================
Process all videos in data/raw_videos/ through the full pipeline
(Stages 1-3) and combine all per-video CSVs into one master dataset.

Usage:
    # Process all videos
    python batch_process.py

    # Dry run — show what would be processed without running
    python batch_process.py --dry-run

    # Re-process videos even if already done
    python batch_process.py --reprocess

    # Process only first N videos (for testing)
    python batch_process.py --limit 10

    # Adjust quality filters (if too many rows are dropped)
    python batch_process.py --min-detection-rate 0.5
    python batch_process.py --no-half-rep-filter

Output:
    data/processed/master_dataset.csv   — all reps combined
    data/processed/batch_report.json    — processing summary
"""

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from squat_analysis.config import RAW_VIDEO_DIR, PROCESSED_DIR
from squat_analysis.extraction import extract
from squat_analysis.preprocessing import preprocess
from squat_analysis.features import extract_features


# =============================================================================
# Quality filters
# =============================================================================

def _check_detection_rate(session_dir: Path, min_rate: float) -> tuple:
    """Return (pass, detection_rate). Fails if too many frames missed."""
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return False, 0.0
    with open(meta_path) as f:
        meta = json.load(f)
    rate = float(meta.get("detection_rate", 0.0))
    return rate >= min_rate, rate


def _check_motion_range(session_dir: Path, min_range: float = 0.05) -> tuple:
    """Return (pass, max_range). Fails if no meaningful motion detected.

    Checks the maximum range across hip, knee, and ankle signals.
    Low range = person not squatting / static video.
    """
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
    """Return True if this row looks like a half-rep artefact.

    A half-rep has one phase (descent or ascent) that is near-zero
    relative to the total rep duration. These come from videos that
    start or end mid-squat.

    min_phase_ratio: minimum fraction of rep duration a phase must have.
    e.g. 0.15 means descent must be >= 15% of total rep time.
    """
    descent = float(row.get("descent_time_s", 0))
    ascent  = float(row.get("ascent_time_s",  0))
    total   = descent + ascent
    if total < 1e-6:
        return True
    descent_ratio = descent / total
    ascent_ratio  = ascent  / total
    return descent_ratio < min_phase_ratio or ascent_ratio < min_phase_ratio


def _has_valid_angles(row: pd.Series) -> bool:
    """Return True if key angle features are physically plausible."""
    knee = row.get("knee_flexion_at_bottom", np.nan)
    # Knee flexion at bottom should be between 30° and 170°
    # Outside this = tracking failure or non-squat movement
    if not np.isnan(knee) and not (30.0 <= knee <= 170.0):
        return False
    return True


# =============================================================================
# Per-video processor
# =============================================================================

def process_video(
    video_path: Path,
    min_detection_rate: float,
    filter_half_reps: bool,
    reprocess: bool,
) -> dict:
    """Run full pipeline on one video. Returns a result dict.

    Result dict keys:
        status      : 'ok' | 'skipped' | 'failed' | 'no_reps' | 'filtered_out'
        reason      : explanation string
        rows        : int — number of rows accepted
        rows_dropped: int — rows removed by quality filters
        detection_rate: float
        motion_range  : float
        df          : pd.DataFrame | None — accepted rows
        duration_s  : float — processing time
    """
    session_id  = video_path.stem
    session_dir = PROCESSED_DIR / session_id
    t_start     = time.time()

    result = {
        "session_id":     session_id,
        "video":          str(video_path),
        "status":         "failed",
        "reason":         "",
        "rows":           0,
        "rows_dropped":   0,
        "detection_rate": 0.0,
        "motion_range":   0.0,
        "df":             None,
        "duration_s":     0.0,
    }

    try:
        # ── Skip if already processed ─────────────────────────────────────────
        features_csv = session_dir / "features.csv"
        if features_csv.exists() and not reprocess:
            df = pd.read_csv(features_csv)
            result.update({"status": "skipped", "reason": "already processed",
                           "rows": len(df), "df": df})
            result["duration_s"] = time.time() - t_start
            return result

        # ── Stage 1 ───────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session_dir_out = extract(
                video_path=str(video_path),
                session_id=session_id,
            )

        # ── Detection rate check ──────────────────────────────────────────────
        passed, det_rate = _check_detection_rate(session_dir_out, min_detection_rate)
        result["detection_rate"] = det_rate
        if not passed:
            result.update({"status": "filtered_out",
                           "reason": f"detection_rate={det_rate:.2f} < {min_detection_rate}"})
            result["duration_s"] = time.time() - t_start
            return result

        # ── Stage 2 ───────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocess(str(session_dir_out))

        # ── Motion range check ────────────────────────────────────────────────
        passed, motion_range = _check_motion_range(session_dir_out)
        result["motion_range"] = motion_range
        if not passed:
            result.update({"status": "filtered_out",
                           "reason": f"motion_range={motion_range:.4f} < 0.05 (static video)"})
            result["duration_s"] = time.time() - t_start
            return result

        # ── Stage 3 ───────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = extract_features(str(session_dir_out))

        if len(df) == 0:
            result.update({"status": "no_reps", "reason": "no reps extracted"})
            result["duration_s"] = time.time() - t_start
            return result

        # ── Row-level quality filters ─────────────────────────────────────────
        original_count = len(df)
        drop_mask      = pd.Series([False] * len(df))

        # Filter half-reps (videos starting/ending mid-squat)
        if filter_half_reps:
            half_rep_mask = df.apply(_is_half_rep, axis=1)
            drop_mask     = drop_mask | half_rep_mask

        # Filter implausible angles
        bad_angle_mask = ~df.apply(_has_valid_angles, axis=1)
        drop_mask      = drop_mask | bad_angle_mask

        df_clean      = df[~drop_mask].copy()
        rows_dropped  = int(drop_mask.sum())

        if len(df_clean) == 0:
            result.update({
                "status":       "filtered_out",
                "reason":       f"all {original_count} rows failed quality filters",
                "rows_dropped": rows_dropped,
            })
            result["duration_s"] = time.time() - t_start
            return result

        result.update({
            "status":       "ok",
            "reason":       "",
            "rows":         len(df_clean),
            "rows_dropped": rows_dropped,
            "df":           df_clean,
        })

    except RuntimeError as e:
        # RuntimeError from pipeline (e.g. no reps detected)
        result.update({"status": "no_reps", "reason": str(e)})
    except Exception as e:
        result.update({"status": "failed", "reason": str(e),
                       "_traceback": traceback.format_exc()})

    result["duration_s"] = time.time() - t_start
    return result


# =============================================================================
# Main batch runner
# =============================================================================

def run_batch(
    min_detection_rate: float = 0.50,
    filter_half_reps:   bool  = False,
    reprocess:          bool  = False,
    limit:              int   = None,
    dry_run:            bool  = False,
) -> pd.DataFrame:
    """Process all videos and combine into master_dataset.csv.

    Args:
        min_detection_rate: Minimum MediaPipe detection rate to accept video.
        filter_half_reps:   Drop rows that are clearly half-reps.
        reprocess:          Re-run even if features.csv already exists.
        limit:              Process only first N videos.
        dry_run:            Print what would run without executing.

    Returns:
        Combined pd.DataFrame of all accepted rows.
    """
    videos = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        # Also check common video extensions
        for ext in ["*.mov", "*.avi", "*.MP4", "*.MOV"]:
            videos += sorted(RAW_VIDEO_DIR.glob(ext))

    if not videos:
        print(f"No videos found in {RAW_VIDEO_DIR}")
        return pd.DataFrame()

    if limit:
        videos = videos[:limit]

    print(f"\n{'='*60}")
    print(f"  Batch processing {len(videos)} videos")
    print(f"  min_detection_rate : {min_detection_rate}")
    print(f"  filter_half_reps   : {filter_half_reps}")
    print(f"  reprocess          : {reprocess}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN — videos that would be processed:")
        for v in videos:
            session_dir = PROCESSED_DIR / v.stem
            done = (session_dir / "features.csv").exists()
            print(f"  {'[done]' if done else '[todo]'} {v.name}")
        return pd.DataFrame()

    # ── Process each video ────────────────────────────────────────────────────
    all_dfs    = []
    report     = {
        "total":        len(videos),
        "ok":           0,
        "skipped":      0,
        "no_reps":      0,
        "filtered_out": 0,
        "failed":       0,
        "total_rows":   0,
        "total_dropped":0,
        "videos":       [],
    }

    for i, video_path in enumerate(videos):
        print(f"[{i+1:3d}/{len(videos)}] {video_path.name}", end=" ... ", flush=True)
        result = process_video(
            video_path        = video_path,
            min_detection_rate= min_detection_rate,
            filter_half_reps  = filter_half_reps,
            reprocess         = reprocess,
        )

        status = result["status"]
        report[status] = report.get(status, 0) + 1

        if status == "ok" or status == "skipped":
            if result["df"] is not None:
                all_dfs.append(result["df"])
                report["total_rows"]    += result["rows"]
                report["total_dropped"] += result["rows_dropped"]
            print(f"OK  ({result['rows']} rows, dropped {result['rows_dropped']}, "
                  f"{result['duration_s']:.1f}s)")
        else:
            print(f"{status.upper()}  — {result['reason'][:60]}")

        report["videos"].append({
            "session_id":     result["session_id"],
            "status":         status,
            "reason":         result["reason"],
            "rows":           result["rows"],
            "rows_dropped":   result["rows_dropped"],
            "detection_rate": round(result["detection_rate"], 3),
            "motion_range":   round(result["motion_range"], 4),
            "duration_s":     round(result["duration_s"], 2),
        })

    # ── Combine all CSVs ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Processing complete")
    print(f"  OK           : {report['ok']}")
    print(f"  Skipped      : {report['skipped']}")
    print(f"  No reps      : {report['no_reps']}")
    print(f"  Filtered out : {report['filtered_out']}")
    print(f"  Failed       : {report['failed']}")
    print(f"  Total rows   : {report['total_rows']}")
    print(f"  Rows dropped : {report['total_dropped']}")
    print(f"{'='*60}\n")

    if not all_dfs:
        print("No data to combine. Check batch_report.json for details.")
        combined = pd.DataFrame()
    else:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Add a global rep_id for easy reference
        combined.insert(0, "global_rep_id", range(len(combined)))

        out_csv = PROCESSED_DIR / "master_dataset.csv"
        combined.to_csv(out_csv, index=False)
        print(f"Master dataset saved: {out_csv}")
        print(f"  Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")

    # ── Save batch report ─────────────────────────────────────────────────────
    report_path = PROCESSED_DIR / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Batch report saved : {report_path}")

    return combined


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch process all squat videos → master_dataset.csv"
    )
    parser.add_argument("--dry-run",            action="store_true",
                        help="Show what would run without processing")
    parser.add_argument("--reprocess",          action="store_true",
                        help="Re-process videos even if already done")
    parser.add_argument("--limit",              type=int, default=None,
                        help="Process only first N videos")
    parser.add_argument("--min-detection-rate", type=float, default=0.50,
                        help="Min MediaPipe detection rate (default 0.50)")
    parser.add_argument("--no-half-rep-filter", action="store_true", default=True,
                        help="Disable half-rep filtering (default: disabled for messy data)")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = run_batch(
        min_detection_rate = args.min_detection_rate,
        filter_half_reps   = False,   # disabled by default for messy real-world data
        reprocess          = args.reprocess,
        limit              = args.limit,
        dry_run            = args.dry_run,
    )

    if len(df) > 0:
        print(f"\nReady for mining:")
        print(f"  python mining/run_mining.py --dataset data/processed/master_dataset.csv")


if __name__ == "__main__":
    main()
