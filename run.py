"""
run.py
======
Single CLI entry point. Runs Stages 1–3 in sequence.

Usage:
    python run.py --video data/raw_videos/session_01.mp4
    python run.py --video data/raw_videos/session_01.mp4 --session my_session
    python run.py --video data/raw_videos/session_01.mp4 --max-frames 300
    python run.py --video data/raw_videos/session_01.mp4 --save-trajectories
"""

import argparse
from pathlib import Path

from squat_analysis.extraction import extract
from squat_analysis.preprocessing import preprocess
from squat_analysis.features import extract_features


def main():
    parser = argparse.ArgumentParser(
        description="Squat analysis pipeline: video → features"
    )
    parser.add_argument("--video",              required=True)
    parser.add_argument("--session",            default=None)
    parser.add_argument("--max-frames",         type=int, default=None)
    parser.add_argument("--save-trajectories",  action="store_true",
                        help="Save per-rep angle trajectories for debugging")
    args = parser.parse_args()

    # Stage 1 — extract raw landmarks
    session_dir = extract(
        video_path=args.video,
        session_id=args.session,
        max_frames=args.max_frames,
    )

    # Stage 2 — preprocess
    preprocess(str(session_dir))

    # Stage 3 — features
    df = extract_features(
        str(session_dir),
        save_trajectories=args.save_trajectories,
    )

    print(f"\nPipeline complete.")
    print(f"  Session : {session_dir}")
    print(f"  Reps    : {len(df)}")
    print(f"  Features: {len(df.columns)} columns")
    print(f"  CSV     : {session_dir / 'features.csv'}")


if __name__ == "__main__":
    main()
