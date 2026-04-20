"""
Single-video CLI.  Thin wrapper around squat_analysis.pipeline.

Usage:
    python run.py --video data/raw_videos/session_01.mp4
    python run.py --video data/raw_videos/session_01.mp4 --session my_session
    python run.py --video data/raw_videos/session_01.mp4 --max-frames 300
"""

import argparse
import logging

from squat_analysis.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Squat analysis: video → features")
    parser.add_argument("--video",             required=True)
    parser.add_argument("--session",           default=None)
    parser.add_argument("--max-frames",        type=int, default=None)
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        session_id=args.session,
        max_frames=args.max_frames,
        save_trajectories=args.save_trajectories,
    )

    print(f"\nPipeline complete.")
    print(f"  Session : {result['session_dir']}")
    print(f"  Reps    : {result['n_reps']}")
    print(f"  Time    : {result['duration_s']:.1f}s")
    print(f"  CSV     : {result['session_dir'] / 'features.csv'}")


if __name__ == "__main__":
    main()