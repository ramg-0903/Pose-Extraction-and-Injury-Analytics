"""
preprocessing.py
================
Stage 2 — Preprocessing.

Takes the raw Stage 1 output and produces clean, normalised landmark
arrays ready for Stage 3 feature engineering.

Steps in order:
    1. NaN interpolation        Fill short gaps; track with interpolation_mask
    2. Y-flip                   MediaPipe Y-down → Y-up
    3. Global scale             Median torso length → body-length units
    4. landmarks_scaled         Snapshot before origin shift (for rep detection)
    5. Per-frame origin shift   Mid-hip → (0, 0) per frame (for features)
    6. Smoothing                Savitzky-Golay; skips across long NaN gaps
    7. Jolt detection           Jerk-based; expands mask ±2 frames
    8. Torso alignment check    Deviation from standing baseline
    9. Confidence array         Single per-frame score combining all signals

Outputs saved to data/processed/{session_id}/:
    landmarks_normalised.npy    (T, 33, 2)  per-frame origin, scaled, Y-up, smoothed
    landmarks_scaled.npy        (T, 33, 2)  global scale only, Y-up, smoothed
    interpolation_mask.npy      (T, 33)     bool — True where x/y was interpolated
    jolt_mask.npy               (T,)        bool — True where jolt detected
    low_confidence_mask.npy     (T,)        bool — True where torso tilted
    confidence.npy              (T,)        float32 [0,1] — combined per-frame score
    preprocessing_meta.json     stats and parameters used
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from squat_analysis.config import (
    L_HIP, R_HIP,
    L_SHOULDER, R_SHOULDER,
    SQUAT_LANDMARKS,
    MIN_FRAME_DETECTION_QUALITY,
    SG_WINDOW, SG_POLY,
    TORSO_ALIGNMENT_THRESHOLD,
    JERK_THRESHOLD,
)
from squat_analysis.extraction import load_extraction
from squat_analysis.utils import angle_to_vertical, midpoint

logger = logging.getLogger(__name__)

# ── Timing constants (seconds) — converted to frames at runtime using FPS ────
# This makes behaviour consistent regardless of whether the video is
# 24, 30, or 60 fps.
MAX_INTERP_GAP_SEC    = 0.17   # max gap to interpolate (~5 frames @ 30fps)
JOLT_EXPAND_SEC       = 0.07   # expand jolt mask each side (~2 frames @ 30fps)
BASELINE_SEC          = 1.0    # standing baseline at video start (1 second)

def _frames(seconds: float, fps: float) -> int:
    """Convert a duration in seconds to a frame count for a given FPS."""
    return max(1, int(round(seconds * fps)))


# =============================================================================
# Step 1 — NaN interpolation
# =============================================================================

def _interpolate_nans(
    landmarks: np.ndarray,
    fps: float,
) -> tuple:
    """Fill short NaN gaps in landmark trajectories via linear interpolation.

    Only fills gaps of MAX_INTERP_GAP_SEC seconds or fewer.
    Longer gaps are left as NaN — they represent genuine occlusion and
    should not be fabricated.

    Args:
        landmarks: (T, 33, 3) raw landmarks from Stage 1.
                   Channel 2 is visibility — never interpolated.
        fps:       Video frame rate, used to convert the time-based gap
                   limit to a frame count.

    Returns:
        filled:             (T, 33, 3) with short gaps filled.
        interpolation_mask: (T, 33) bool, True where x/y was interpolated.
    """
    max_gap = _frames(MAX_INTERP_GAP_SEC, fps)
    T = landmarks.shape[0]
    filled      = landmarks.copy()
    interp_mask = np.zeros((T, 33), dtype=bool)

    for lm in range(33):
        for ax in range(2):                   # x and y only — skip visibility
            col      = filled[:, lm, ax]
            nan_mask = np.isnan(col)

            if not nan_mask.any():
                continue

            # Find contiguous NaN runs
            changes = np.diff(nan_mask.astype(int), prepend=0, append=0)
            starts  = np.where(changes ==  1)[0]
            ends    = np.where(changes == -1)[0]

            for s, e in zip(starts, ends):
                gap_len = e - s
                if gap_len > max_gap:
                    continue               # too long — leave as NaN

                has_left  = s > 0        and not np.isnan(col[s - 1])
                has_right = e < T        and not np.isnan(col[e])

                if not (has_left and has_right):
                    continue               # edge gap — no anchor

                x0, y0 = float(s - 1), col[s - 1]
                x1, y1 = float(e),     col[e]
                for i in range(s, e):
                    col[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
                    interp_mask[i, lm] = True

            filled[:, lm, ax] = col

    n_filled = int(interp_mask.sum())
    logger.info("  interpolation: %d landmark-frames filled", n_filled)
    return filled, interp_mask


# =============================================================================
# Step 2 — Y-flip
# =============================================================================

def _flip_y(landmarks: np.ndarray) -> np.ndarray:
    """Flip Y axis: MediaPipe (Y-down) → standard biomechanics (Y-up).

    Only the Y channel (index 1) is negated.
    Visibility channel (index 2) is unchanged.

    Args:
        landmarks: (T, 33, 3)

    Returns:
        (T, 33, 3)
    """
    out = landmarks.copy()
    out[:, :, 1] *= -1.0
    return out


# =============================================================================
# Step 3 — Global scale
# =============================================================================

def _compute_global_scale(
    landmarks: np.ndarray,
    detection_quality: np.ndarray,
) -> float:
    """Compute a single global scale factor from the median torso length.

    Only uses frames with detection quality above the threshold so that
    bad frames do not corrupt the scale estimate.

    Torso length = distance from mid-hip to mid-shoulder.

    Args:
        landmarks:         (T, 33, 3) Y-flipped landmarks.
        detection_quality: (T,) per-frame quality scores from Stage 1.

    Returns:
        Scalar scale factor. Divide all coordinates by this to get
        body-length units.
    """
    good = np.where(detection_quality >= MIN_FRAME_DETECTION_QUALITY)[0]

    if len(good) < 5:
        warnings.warn(
            "Fewer than 5 good-quality frames for scale estimation. "
            "Falling back to all frames."
        )
        good = np.arange(landmarks.shape[0])

    torso_lengths = []
    for t in good:
        l_hip = landmarks[t, L_HIP,     :2]
        r_hip = landmarks[t, R_HIP,     :2]
        l_sho = landmarks[t, L_SHOULDER, :2]
        r_sho = landmarks[t, R_SHOULDER, :2]

        if np.isnan(l_hip).any() or np.isnan(r_hip).any() \
                or np.isnan(l_sho).any() or np.isnan(r_sho).any():
            continue

        torso_lengths.append(
            float(np.linalg.norm(midpoint(l_sho, r_sho) - midpoint(l_hip, r_hip)))
        )

    if len(torso_lengths) < 3:
        raise RuntimeError(
            "Cannot compute torso length — too many NaN frames at hip/shoulder. "
            "Check video quality."
        )

    scale = float(np.median(torso_lengths))
    logger.info("  global scale: %.4f (from %d frames)", scale, len(torso_lengths))
    return scale


# =============================================================================
# Steps 4 + 5 — Apply scale and per-frame origin shift
# =============================================================================

def _apply_scale_and_origin(
    landmarks: np.ndarray,
    scale: float,
    original_landmarks: np.ndarray,
) -> tuple:
    """Apply global scale and per-frame origin shift.

    Returns TWO arrays with an important distinction:

    landmarks_scaled:
        Globally scaled, Y-up, NO per-frame origin shift.
        The hip still moves up and down across frames.
        Used by Stage 3 for rep segmentation (valley detection on hip-Y).

    landmarks_normalised:
        Globally scaled + per-frame hip origin.
        Hip is at (0, 0) in every frame.
        Used by Stage 3 for joint angle feature computation.

    Frames where BOTH hips were originally NaN in Stage 1 (before any
    interpolation) are explicitly set to NaN in landmarks_normalised.
    Interpolation may have filled these frames with plausible-looking values,
    but they were never genuinely observed and must not be treated as
    normalised data.

    Args:
        landmarks:          (T, 33, 3) Y-flipped, interpolated landmarks.
        scale:              Global torso-length scale factor.
        original_landmarks: (T, 33, 3) raw Stage 1 landmarks BEFORE
                            interpolation, used to identify originally-NaN
                            hip frames.

    Returns:
        landmarks_scaled:      (T, 33, 2) float32
        landmarks_normalised:  (T, 33, 2) float32
    """
    T  = landmarks.shape[0]
    xy = landmarks[:, :, :2].copy().astype(np.float32)

    if scale > 1e-9:
        xy /= scale

    landmarks_scaled = xy.copy()

    # Identify frames where BOTH hips were originally missing (pre-interpolation)
    orig_l_hip_nan = np.isnan(original_landmarks[:, L_HIP,  0])
    orig_r_hip_nan = np.isnan(original_landmarks[:, R_HIP,  0])
    both_hips_originally_nan = orig_l_hip_nan & orig_r_hip_nan

    for t in range(T):
        if both_hips_originally_nan[t]:
            # Both hips were never observed in this frame.
            # Even if interpolation filled them, we cannot trust the origin.
            # Explicitly invalidate the entire normalised frame.
            xy[t] = np.nan
            continue

        l_hip = xy[t, L_HIP]
        r_hip = xy[t, R_HIP]
        l_nan = np.isnan(l_hip).any()
        r_nan = np.isnan(r_hip).any()

        if l_nan and r_nan:
            # Both hips are NaN after interpolation too (long gap, no fill)
            xy[t] = np.nan
            continue

        if l_nan:
            origin = r_hip.copy()
        elif r_nan:
            origin = l_hip.copy()
        else:
            origin = midpoint(l_hip, r_hip)

        xy[t] -= origin

    landmarks_normalised = xy
    return landmarks_scaled, landmarks_normalised


# =============================================================================
# Step 6 — Smoothing
# =============================================================================

def _smooth(
    landmarks: np.ndarray,
    interpolation_mask: np.ndarray,
) -> np.ndarray:
    """Savitzky-Golay smoothing per landmark per axis.

    Smoothing is applied only within continuous non-NaN segments.
    It never bridges across a long NaN gap, preventing smoothed
    interpolated data from contaminating real measurements.

    Args:
        landmarks:          (T, 33, 2) scaled or normalised landmarks.
        interpolation_mask: (T, 33) bool — identifies filled gaps.

    Returns:
        (T, 33, 2) smoothed array.
    """
    T        = landmarks.shape[0]
    smoothed = landmarks.copy()

    window = SG_WINDOW
    if window >= T:
        window = T - 1 if T % 2 == 0 else T
    if window % 2 == 0:
        window -= 1
    if window < SG_POLY + 2:
        logger.warning("Signal too short for smoothing (T=%d). Skipping.", T)
        return smoothed

    for lm in range(33):
        for ax in range(2):
            col     = landmarks[:, lm, ax].copy()
            nan_col = np.isnan(col)

            if nan_col.all():
                continue

            if not nan_col.any():
                smoothed[:, lm, ax] = savgol_filter(
                    col, window_length=window,
                    polyorder=SG_POLY, mode="interp"
                )
                continue

            # Smooth each continuous non-NaN segment independently
            valid   = ~nan_col
            changes = np.diff(valid.astype(int), prepend=0, append=0)
            starts  = np.where(changes ==  1)[0]
            ends    = np.where(changes == -1)[0]

            for s, e in zip(starts, ends):
                seg_len = e - s
                if seg_len < SG_POLY + 2:
                    continue
                seg_win = min(window, seg_len if seg_len % 2 == 1 else seg_len - 1)
                if seg_win < SG_POLY + 2:
                    continue
                smoothed[s:e, lm, ax] = savgol_filter(
                    col[s:e], window_length=seg_win,
                    polyorder=SG_POLY, mode="interp"
                )

    return smoothed


# =============================================================================
# Step 7 — Jolt detection
# =============================================================================

def _detect_jolts(
    landmarks_scaled_unsmoothed: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Detect camera jolts using jerk on the UNSMOOTHED mid-hip Y trajectory.

    Runs on the unsmoothed scaled array intentionally — smoothing suppresses
    the sharp spikes that distinguish jolts from normal movement.

    Jerk (3rd derivative) distinguishes camera jolts from fast human
    movement. A squat produces high velocity but controlled jerk.
    A camera knock produces a sudden jerk spike.

    Detected frames are expanded by JOLT_EXPAND_SEC on each side to
    capture the full effect of the jolt regardless of video FPS.

    Args:
        landmarks_scaled_unsmoothed: (T, 33, 2) globally scaled, unsmoothed.
        fps:                         Video frame rate.

    Returns:
        jolt_mask: (T,) bool
    """
    expand = _frames(JOLT_EXPAND_SEC, fps)
    T      = landmarks_scaled_unsmoothed.shape[0]
    l_hip  = landmarks_scaled_unsmoothed[:, L_HIP, 1]
    r_hip  = landmarks_scaled_unsmoothed[:, R_HIP, 1]

    # Use the hip side with fewer NaNs for the jerk signal
    hip_y = l_hip if np.isnan(l_hip).sum() <= np.isnan(r_hip).sum() else r_hip

    if np.isnan(hip_y).all():
        logger.warning("  jolt detection: all hip-Y values NaN. Skipping.")
        return np.zeros(T, dtype=bool)

    # Fill NaNs for derivative computation only
    if np.isnan(hip_y).any():
        t_idx = np.arange(T, dtype=float)
        valid = ~np.isnan(hip_y)
        hip_y = np.interp(t_idx, t_idx[valid], hip_y[valid])

    jerk     = np.gradient(np.gradient(np.gradient(hip_y)))
    jolt_raw = np.abs(jerk) > JERK_THRESHOLD

    # Expand jolt regions by time-based amount
    jolt_mask = jolt_raw.copy()
    for idx in np.where(jolt_raw)[0]:
        lo = max(0, idx - expand)
        hi = min(T, idx + expand + 1)
        jolt_mask[lo:hi] = True

    logger.info("  jolts: %d frames flagged", int(jolt_mask.sum()))
    return jolt_mask


# =============================================================================
# Step 8 — Torso alignment check
# =============================================================================

def _check_torso_alignment(
    landmarks_scaled: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Flag frames where torso deviates too far from the standing baseline.

    Computes torso angle per frame, uses the first BASELINE_SEC seconds
    to establish a baseline (subject standing), and flags any frame that
    deviates more than TORSO_ALIGNMENT_THRESHOLD degrees from it.

    This detects camera movement during the session. A camera that is
    already slightly tilted at the start is absorbed into the baseline —
    only changes from that baseline are flagged.

    Args:
        landmarks_scaled: (T, 33, 2) globally scaled landmarks.
        fps:              Video frame rate for converting baseline to frames.

    Returns:
        low_confidence_mask: (T,) bool
    """
    baseline_frames = _frames(BASELINE_SEC, fps)
    T               = landmarks_scaled.shape[0]
    low_conf        = np.zeros(T, dtype=bool)
    angles          = np.full(T, np.nan)

    for t in range(T):
        l_hip = landmarks_scaled[t, L_HIP,      :2]
        r_hip = landmarks_scaled[t, R_HIP,      :2]
        l_sho = landmarks_scaled[t, L_SHOULDER,  :2]
        r_sho = landmarks_scaled[t, R_SHOULDER,  :2]

        if np.isnan(l_hip).any() or np.isnan(r_hip).any() \
                or np.isnan(l_sho).any() or np.isnan(r_sho).any():
            continue

        angles[t] = angle_to_vertical(midpoint(l_hip, r_hip), midpoint(l_sho, r_sho))

    valid_baseline = angles[:baseline_frames]
    valid_baseline = valid_baseline[~np.isnan(valid_baseline)]

    if len(valid_baseline) < 3:
        logger.warning("  alignment: insufficient baseline frames. Skipping.")
        return low_conf

    baseline = float(np.median(valid_baseline))

    for t in range(T):
        if not np.isnan(angles[t]) and abs(angles[t] - baseline) > TORSO_ALIGNMENT_THRESHOLD:
            low_conf[t] = True

    logger.info(
        "  torso baseline: %.1f deg | low-confidence: %d frames",
        baseline, int(low_conf.sum())
    )
    return low_conf


# =============================================================================
# Step 9 — Confidence array
# =============================================================================

def _build_confidence(
    detection_quality: np.ndarray,
    interpolation_mask: np.ndarray,
    jolt_mask: np.ndarray,
    low_confidence_mask: np.ndarray,
) -> np.ndarray:
    """Build a single per-frame confidence score in [0, 1].

    Formula:
        conf = detection_quality
             × (1 - interp_fraction)    penalise interpolated frames
             × 0.0  if jolt             zero out jolted frames entirely
             × 0.5  if low_confidence   halve confidence for tilted frames

    Args:
        detection_quality:   (T,)     float32 [0,1]
        interpolation_mask:  (T, 33)  bool
        jolt_mask:           (T,)     bool
        low_confidence_mask: (T,)     bool

    Returns:
        confidence: (T,) float32 [0, 1]
    """
    squat_interp = interpolation_mask[:, SQUAT_LANDMARKS]       # (T, 12)
    interp_frac  = squat_interp.mean(axis=1).astype(np.float32) # (T,)

    conf  = detection_quality.copy().astype(np.float32)
    conf *= (1.0 - interp_frac)
    conf[jolt_mask]            = 0.0
    conf[low_confidence_mask] *= 0.5

    return np.clip(conf, 0.0, 1.0)


# =============================================================================
# Smoothing validation
# =============================================================================

def _validate_smoothing(
    raw: np.ndarray,
    smoothed: np.ndarray,
    min_ratio: float = 0.80,
) -> None:
    """Warn if smoothing has over-dampened the hip-Y motion range.

    Args:
        raw:       (T, 33, 2) unsmoothed scaled landmarks.
        smoothed:  (T, 33, 2) smoothed scaled landmarks.
        min_ratio: Minimum acceptable (smoothed range / raw range).
    """
    def _range(arr: np.ndarray) -> float:
        hip_y = (arr[:, L_HIP, 1] + arr[:, R_HIP, 1]) / 2.0
        valid = hip_y[~np.isnan(hip_y)]
        return float(valid.max() - valid.min()) if len(valid) > 1 else 0.0

    raw_r = _range(raw)
    if raw_r < 1e-6:
        return

    ratio = _range(smoothed) / raw_r
    if ratio < min_ratio:
        warnings.warn(
            f"Smoothing may have over-dampened squat depth signal: "
            f"range ratio = {ratio:.2f} (min {min_ratio}). "
            f"Consider reducing SG_WINDOW in config.py."
        )


# =============================================================================
# Loader (used by Stage 3)
# =============================================================================

def load_preprocessing(session_dir: str) -> dict:
    """Load Stage 2 outputs into memory.

    Args:
        session_dir: Path produced by preprocess().

    Returns:
        dict with keys:
            landmarks_normalised  (T, 33, 2) float32
            landmarks_scaled      (T, 33, 2) float32
            interpolation_mask    (T, 33)    bool
            jolt_mask             (T,)       bool
            low_confidence_mask   (T,)       bool
            confidence            (T,)       float32
            preprocessing_meta    dict
    """
    d = Path(session_dir)
    with open(d / "preprocessing_meta.json") as f:
        meta = json.load(f)

    return {
        "landmarks_normalised": np.load(d / "landmarks_normalised.npy"),
        "landmarks_scaled":     np.load(d / "landmarks_scaled.npy"),
        "interpolation_mask":   np.load(d / "interpolation_mask.npy"),
        "jolt_mask":            np.load(d / "jolt_mask.npy"),
        "low_confidence_mask":  np.load(d / "low_confidence_mask.npy"),
        "confidence":           np.load(d / "confidence.npy"),
        "preprocessing_meta":   meta,
    }


# =============================================================================
# Main entry point
# =============================================================================

def preprocess(
    session_dir: str,
    output_dir: Optional[str] = None,
) -> Path:
    """Run Stage 2 preprocessing on a Stage 1 session directory.

    Args:
        session_dir: Path to session directory containing Stage 1 outputs.
        output_dir:  Override save location. Defaults to session_dir.

    Returns:
        Path to the output directory.
    """
    session_dir = Path(session_dir)
    out_dir     = Path(output_dir) if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stage 2] Preprocessing")
    print(f"  session : {session_dir}")

    # Load Stage 1 outputs
    data              = load_extraction(session_dir)
    landmarks_raw     = data["landmarks"]           # (T, 33, 3)
    detection_quality = data["detection_quality"]   # (T,)
    meta_s1           = data["metadata"]
    T                 = landmarks_raw.shape[0]
    fps               = float(meta_s1["fps"])
    print(f"  frames  : {T}  fps: {fps}")

    # Step 1
    print("  [1/8] NaN interpolation")
    filled, interp_mask = _interpolate_nans(landmarks_raw, fps)

    # Step 2
    print("  [2/8] Y-flip")
    yup = _flip_y(filled)

    # Step 3
    print("  [3/8] Global scale")
    scale = _compute_global_scale(yup, detection_quality)

    # Steps 4+5 — pass original raw landmarks so both-hips-NaN frames
    # are correctly identified even if interpolation filled them
    print("  [4/8] Scale + per-frame origin shift")
    lm_scaled, lm_norm = _apply_scale_and_origin(yup, scale, _flip_y(landmarks_raw))

    # Step 6 — jolt detection runs on UNSMOOTHED scaled signal
    print("  [5/8] Jolt detection (pre-smoothing)")
    jolt_mask = _detect_jolts(lm_scaled, fps)

    # Step 7 — smoothing after jolt detection
    print("  [6/8] Savitzky-Golay smoothing")
    lm_scaled_s = _smooth(lm_scaled, interp_mask)
    lm_norm_s   = _smooth(lm_norm,   interp_mask)
    _validate_smoothing(lm_scaled, lm_scaled_s)

    # Step 8
    print("  [7/8] Torso alignment check")
    low_conf_mask = _check_torso_alignment(lm_scaled_s, fps)

    # Step 9
    print("  [8/8] Confidence array")
    confidence = _build_confidence(
        detection_quality, interp_mask, jolt_mask, low_conf_mask
    )

    # Save
    np.save(out_dir / "landmarks_normalised.npy", lm_norm_s)
    np.save(out_dir / "landmarks_scaled.npy",     lm_scaled_s)
    np.save(out_dir / "interpolation_mask.npy",   interp_mask)
    np.save(out_dir / "jolt_mask.npy",            jolt_mask)
    np.save(out_dir / "low_confidence_mask.npy",  low_conf_mask)
    np.save(out_dir / "confidence.npy",           confidence)

    max_gap_frames = _frames(MAX_INTERP_GAP_SEC, fps)
    baseline_frames = _frames(BASELINE_SEC, fps)
    expand_frames   = _frames(JOLT_EXPAND_SEC, fps)

    meta_s2 = {
        "session_id":                meta_s1["session_id"],
        "source_session":            str(session_dir),
        "fps":                       fps,
        "total_frames":              T,
        "global_scale":              round(scale, 6),
        "frames_interpolated":       int(interp_mask.sum()),
        "nan_frames_remaining":      int(np.isnan(lm_norm_s).any(axis=(1, 2)).sum()),
        "jolts_detected_frames":     int(jolt_mask.sum()),
        "low_confidence_frames":     int(low_conf_mask.sum()),
        "mean_confidence":           round(float(confidence.mean()), 4),
        "sg_window":                 SG_WINDOW,
        "sg_poly":                   SG_POLY,
        "max_interp_gap_sec":        MAX_INTERP_GAP_SEC,
        "max_interp_gap_frames":     max_gap_frames,
        "jerk_threshold":            JERK_THRESHOLD,
        "jolt_expand_sec":           JOLT_EXPAND_SEC,
        "jolt_expand_frames":        expand_frames,
        "torso_baseline_sec":        BASELINE_SEC,
        "torso_baseline_frames":     baseline_frames,
        "torso_alignment_threshold": TORSO_ALIGNMENT_THRESHOLD,
    }

    with open(out_dir / "preprocessing_meta.json", "w") as f:
        json.dump(meta_s2, f, indent=2)

    print(f"  interpolated     : {int(interp_mask.sum())} landmark-frames")
    print(f"  jolts            : {int(jolt_mask.sum())} frames")
    print(f"  low confidence   : {int(low_conf_mask.sum())} frames")
    print(f"  mean confidence  : {float(confidence.mean()):.3f}")
    print(f"  normalised shape : {lm_norm_s.shape}")
    print(f"  scaled shape     : {lm_scaled_s.shape}")
    print(f"  saved to         : {out_dir}")

    return out_dir
