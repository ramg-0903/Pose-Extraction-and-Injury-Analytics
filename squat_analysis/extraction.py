"""
Stage 1 — Raw Landmark Extraction

Reads an MP4 video frame-by-frame, runs MediaPipe PoseLandmarker (Heavy),
and saves per-frame (x, y, visibility) for all 33 landmarks.

Outputs (saved to ``data/processed/{session_id}/``):
    landmarks.npy          (T, 33, 3)  x, y, visibility
    timestamps.npy         (T,)        milliseconds per frame
    detection_quality.npy  (T,)        fraction of squat landmarks visible
    metadata.json          session info, fps, model version, resolution

Design decisions
    * Z-coordinate is dropped — monocular depth from a single camera is
      unreliable and actively misleading for side-view squats.
    * World landmarks are used for (x, y) while *image* landmarks provide
      visibility, which is a more reliable signal for occlusion detection.
"""

import json
import logging
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from squat_analysis.config import (
    MODEL_URL, MODEL_PATH, MODEL_NAME, PROCESSED_DIR,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE, VISIBILITY_NAN_THRESHOLD,
    MIN_FRAME_DETECTION_QUALITY, N_LANDMARKS, SQUAT_LANDMARKS,
)

logger = logging.getLogger(__name__)


# ── Model management ──────────────────────────────────────────────────────────

def _ensure_model(model_path: Optional[Path] = None) -> str:
    """Return path to the .task model file, downloading if needed."""
    target = Path(model_path) if model_path else MODEL_PATH

    if target.exists() and target.stat().st_size > 1_000_000:
        return str(target)

    logger.info("Downloading MediaPipe pose model (~30 MB) → %s", target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, target)
        logger.info("Downloaded (%.1f MB)", target.stat().st_size / 1e6)
    except Exception as exc:
        raise RuntimeError(
            f"Model download failed: {exc}\n"
            f"Download manually from:\n  {MODEL_URL}\n"
            f"Save to: {target}"
        ) from exc

    return str(target)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract(
    video_path: str,
    session_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Path:
    """Run Stage 1 on a single video file.

    Args:
        video_path:  Path to input video (.mp4 / .mov / .avi).
        session_id:  Identifier for this recording (defaults to file stem).
        output_dir:  Override output directory (defaults to PROCESSED_DIR).
        model_path:  Override path to ``pose_landmarker_heavy.task``.
        max_frames:  Process only first N frames (for quick tests).

    Returns:
        Path to the session output directory.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if session_id is None:
        session_id = video_path.stem

    out_dir = Path(output_dir) / session_id if output_dir else PROCESSED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = _ensure_model(Path(model_path) if model_path else None)

    logger.info("[Stage 1] Extracting landmarks — %s", video_path.name)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Configure PoseLandmarker (VIDEO mode) ─────────────────────────────
    # VIDEO mode gives frame-consistent tracking without callback complexity.
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_file),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False,
    )

    # ── Frame-by-frame collection ─────────────────────────────────────────
    all_landmarks:         list[np.ndarray] = []
    all_timestamps_ms:     list[float]      = []
    all_detection_quality: list[float]      = []

    frames_read     = 0
    frames_detected = 0

    with mp_vision.PoseLandmarker.create_from_options(pose_opts) as detector:
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # MediaPipe requires monotonically increasing timestamps
            ts_int = max(int(timestamp_ms), frames_read)
            result = detector.detect_for_video(mp_img, ts_int)

            if result.pose_world_landmarks and result.pose_landmarks:
                frames_detected += 1
                world_lms = result.pose_world_landmarks[0]
                image_lms = result.pose_landmarks[0]

                coords = np.array(
                    [[lm.x, lm.y] for lm in world_lms], dtype=np.float32,
                )  # (33, 2)
                vis = np.array(
                    [lm.visibility for lm in image_lms], dtype=np.float32,
                )  # (33,)

                frame_data = np.column_stack([coords, vis])   # (33, 3)

                # NaN out low-visibility landmarks; visibility value itself is kept
                low_vis = vis < VISIBILITY_NAN_THRESHOLD
                frame_data[low_vis, :2] = np.nan

                squat_vis = vis[SQUAT_LANDMARKS]
                quality   = float(np.mean(squat_vis >= VISIBILITY_NAN_THRESHOLD))
            else:
                frame_data = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
                quality    = 0.0

            all_landmarks.append(frame_data)
            all_timestamps_ms.append(float(timestamp_ms))
            all_detection_quality.append(quality)

            frames_read += 1
            if max_frames and frames_read >= max_frames:
                break

    cap.release()

    if frames_read == 0:
        raise RuntimeError(f"No frames could be read from {video_path}")

    # ── Stack into arrays ─────────────────────────────────────────────────
    landmarks         = np.stack(all_landmarks, axis=0)            # (T, 33, 3)
    timestamps_ms     = np.array(all_timestamps_ms, dtype=np.float64)
    detection_quality = np.array(all_detection_quality, dtype=np.float32)

    detection_rate     = frames_detected / frames_read
    low_quality_frames = int(np.sum(detection_quality < MIN_FRAME_DETECTION_QUALITY))

    if detection_rate < 0.5:
        warnings.warn(
            f"Low detection rate: {detection_rate:.0%} of frames. "
            "Check that the full body (hips to feet) is visible."
        )

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(out_dir / "landmarks.npy",         landmarks)
    np.save(out_dir / "timestamps.npy",        timestamps_ms)
    np.save(out_dir / "detection_quality.npy", detection_quality)

    metadata = {
        "session_id":         session_id,
        "source_video":       str(video_path),
        "fps":                fps,
        "total_frames":       frames_read,
        "width":              width,
        "height":             height,
        "detection_rate":     round(detection_rate, 4),
        "low_quality_frames": low_quality_frames,
        "mediapipe_version":  mp.__version__,
        "model_name":         MODEL_NAME,
        "camera_view":        "unknown",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "  %d/%d frames detected (%.0f%%), %d low-quality → %s",
        frames_detected, frames_read, detection_rate * 100,
        low_quality_frames, out_dir,
    )
    return out_dir


# ── Loader (used by Stage 2) ─────────────────────────────────────────────────

def load_extraction(session_dir: str) -> dict:
    """Load Stage 1 outputs into memory.

    Returns dict with keys: landmarks (T,33,3), timestamps_ms (T,),
    detection_quality (T,), metadata (dict).
    """
    d = Path(session_dir)
    with open(d / "metadata.json") as f:
        metadata = json.load(f)
    return {
        "landmarks":         np.load(d / "landmarks.npy"),
        "timestamps_ms":     np.load(d / "timestamps.npy"),
        "detection_quality": np.load(d / "detection_quality.npy"),
        "metadata":          metadata,
    }