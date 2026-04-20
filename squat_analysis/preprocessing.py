"""
Stage 2 — Preprocessing

Transforms raw Stage 1 landmarks into clean, normalised arrays ready for
feature engineering.

Processing order:
    1. NaN interpolation   — fill short gaps; track with interpolation_mask
    2. Y-flip              — MediaPipe Y-down → biomechanics Y-up
    3. Global scale        — median torso length → body-length units
    4. Scale + origin      — two outputs (see _apply_scale_and_origin)
    5. Jolt detection      — jerk on UNSMOOTHED signal, expand mask ±2 frames
    6. Savitzky-Golay      — smooth within continuous non-NaN segments only
    7. Torso alignment     — flag deviations from standing baseline
    8. Confidence array    — single [0, 1] score combining all quality signals

Outputs (saved alongside Stage 1 files):
    landmarks_normalised.npy  (T, 33, 2)  per-frame origin, scaled, smoothed
    landmarks_scaled.npy      (T, 33, 2)  global scale only, smoothed
    interpolation_mask.npy    (T, 33)     bool
    jolt_mask.npy             (T,)        bool
    low_confidence_mask.npy   (T,)        bool
    confidence.npy            (T,)        float32 [0, 1]
    preprocessing_meta.json
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from squat_analysis.config import (
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER, SQUAT_LANDMARKS,
    MIN_FRAME_DETECTION_QUALITY,
    SG_WINDOW, SG_POLY, TORSO_ALIGNMENT_THRESHOLD, JERK_THRESHOLD,
)
from squat_analysis.extraction import load_extraction
from squat_analysis.utils import angle_to_vertical, midpoint

logger = logging.getLogger(__name__)

# Time-based constants — converted to frames at runtime so behaviour is
# consistent across 24, 30, and 60 fps footage.
MAX_INTERP_GAP_SEC = 0.17   # ~5 frames @ 30 fps
JOLT_EXPAND_SEC    = 0.07   # ~2 frames @ 30 fps
BASELINE_SEC       = 1.0    # standing baseline window at video start


def _frames(seconds: float, fps: float) -> int:
    """Convert a duration in seconds to a frame count."""
    return max(1, int(round(seconds * fps)))


# ── Step 1 — NaN interpolation ───────────────────────────────────────────────

def _interpolate_nans(landmarks: np.ndarray, fps: float) -> tuple:
    """Fill short NaN gaps via linear interpolation.

    Only fills gaps ≤ MAX_INTERP_GAP_SEC.  Longer gaps stay NaN — they
    represent genuine occlusion and must not be fabricated.
    Channel 2 (visibility) is never interpolated.

    Returns (filled, interpolation_mask).
    """
    max_gap = _frames(MAX_INTERP_GAP_SEC, fps)
    T = landmarks.shape[0]
    filled      = landmarks.copy()
    interp_mask = np.zeros((T, 33), dtype=bool)

    for lm in range(33):
        for ax in range(2):  # x, y only
            col      = filled[:, lm, ax]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue

            changes = np.diff(nan_mask.astype(int), prepend=0, append=0)
            starts  = np.where(changes ==  1)[0]
            ends    = np.where(changes == -1)[0]

            for s, e in zip(starts, ends):
                if (e - s) > max_gap:
                    continue
                has_left  = s > 0 and not np.isnan(col[s - 1])
                has_right = e < T and not np.isnan(col[e])
                if not (has_left and has_right):
                    continue

                x0, y0 = float(s - 1), col[s - 1]
                x1, y1 = float(e),     col[e]
                for i in range(s, e):
                    col[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
                    interp_mask[i, lm] = True

            filled[:, lm, ax] = col

    logger.info("  interpolation: %d landmark-frames filled", int(interp_mask.sum()))
    return filled, interp_mask


# ── Step 2 — Y-flip ──────────────────────────────────────────────────────────

def _flip_y(landmarks: np.ndarray) -> np.ndarray:
    """Negate Y channel: MediaPipe Y-down → biomechanics Y-up."""
    out = landmarks.copy()
    out[:, :, 1] *= -1.0
    return out


# ── Step 3 — Global scale ────────────────────────────────────────────────────

def _compute_global_scale(
    landmarks: np.ndarray,
    detection_quality: np.ndarray,
) -> float:
    """Median torso length (mid-hip → mid-shoulder) across good frames.

    Only frames above MIN_FRAME_DETECTION_QUALITY are used so that
    partially-detected frames don't corrupt the estimate.
    """
    good = np.where(detection_quality >= MIN_FRAME_DETECTION_QUALITY)[0]
    if len(good) < 5:
        warnings.warn("Fewer than 5 good frames for scale — using all frames.")
        good = np.arange(landmarks.shape[0])

    torso_lengths = []
    for t in good:
        pts = [landmarks[t, idx, :2] for idx in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)]
        if any(np.isnan(p).any() for p in pts):
            continue
        l_hip, r_hip, l_sho, r_sho = pts
        torso_lengths.append(
            float(np.linalg.norm(midpoint(l_sho, r_sho) - midpoint(l_hip, r_hip)))
        )

    if len(torso_lengths) < 3:
        raise RuntimeError(
            "Cannot compute torso length — too many NaN frames at hip/shoulder."
        )

    scale = float(np.median(torso_lengths))
    logger.info("  global scale: %.4f (from %d frames)", scale, len(torso_lengths))
    return scale


# ── Steps 4 + 5 — Scale and per-frame origin shift ───────────────────────────

def _apply_scale_and_origin(
    landmarks: np.ndarray,
    scale: float,
    original_landmarks: np.ndarray,
) -> tuple:
    """Apply global scale and per-frame mid-hip origin shift.

    Returns two arrays with an important distinction:

    **landmarks_scaled** — globally scaled, Y-up, NO origin shift.
        Hip still moves across frames.  Used for rep segmentation
        (valley detection on hip-Y).

    **landmarks_normalised** — scaled + per-frame hip origin at (0, 0).
        Used for joint-angle feature computation.

    Frames where both hips were originally NaN *before interpolation* are
    set to NaN in landmarks_normalised — interpolation may have filled them
    with plausible values, but they were never observed and must not be
    treated as genuine data.
    """
    T  = landmarks.shape[0]
    xy = landmarks[:, :, :2].copy().astype(np.float32)
    if scale > 1e-9:
        xy /= scale

    landmarks_scaled = xy.copy()

    # Identify frames where both hips were missing in the raw Stage 1 data
    both_hips_orig_nan = (
        np.isnan(original_landmarks[:, L_HIP, 0])
        & np.isnan(original_landmarks[:, R_HIP, 0])
    )

    for t in range(T):
        if both_hips_orig_nan[t]:
            xy[t] = np.nan
            continue

        l_hip = xy[t, L_HIP]
        r_hip = xy[t, R_HIP]
        l_nan = np.isnan(l_hip).any()
        r_nan = np.isnan(r_hip).any()

        if l_nan and r_nan:
            xy[t] = np.nan
            continue

        origin = (
            r_hip.copy()           if l_nan else
            l_hip.copy()           if r_nan else
            midpoint(l_hip, r_hip)
        )
        xy[t] -= origin

    return landmarks_scaled, xy  # (T,33,2), (T,33,2)


# ── Step 6 — Smoothing ───────────────────────────────────────────────────────

def _smooth(landmarks: np.ndarray, interpolation_mask: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing per landmark per axis.

    Applied within continuous non-NaN segments only — never bridges
    across a long NaN gap, preventing contamination of real data.
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
                    col, window_length=window, polyorder=SG_POLY, mode="interp",
                )
                continue

            # Smooth each contiguous non-NaN segment independently
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
                    polyorder=SG_POLY, mode="interp",
                )

    return smoothed


# ── Step 5 (runs before smoothing) — Jolt detection ──────────────────────────

def _detect_jolts(landmarks_scaled: np.ndarray, fps: float) -> np.ndarray:
    """Detect camera jolts via jerk on UNSMOOTHED mid-hip Y.

    Jerk (3rd derivative) distinguishes camera knocks from fast human
    movement: a squat produces high velocity but controlled jerk; a
    camera bump produces a sharp jerk spike.

    Detected frames are expanded by JOLT_EXPAND_SEC on each side.
    """
    expand = _frames(JOLT_EXPAND_SEC, fps)
    T      = landmarks_scaled.shape[0]
    l_hip  = landmarks_scaled[:, L_HIP, 1]
    r_hip  = landmarks_scaled[:, R_HIP, 1]

    # Use whichever hip side has fewer NaNs
    hip_y = l_hip if np.isnan(l_hip).sum() <= np.isnan(r_hip).sum() else r_hip

    if np.isnan(hip_y).all():
        logger.warning("  jolt detection: all hip-Y NaN — skipping.")
        return np.zeros(T, dtype=bool)

    # Temporary NaN fill for derivative computation only
    if np.isnan(hip_y).any():
        t_idx = np.arange(T, dtype=float)
        valid = ~np.isnan(hip_y)
        hip_y = np.interp(t_idx, t_idx[valid], hip_y[valid])

    jerk     = np.gradient(np.gradient(np.gradient(hip_y)))
    jolt_raw = np.abs(jerk) > JERK_THRESHOLD

    jolt_mask = jolt_raw.copy()
    for idx in np.where(jolt_raw)[0]:
        lo = max(0, idx - expand)
        hi = min(T, idx + expand + 1)
        jolt_mask[lo:hi] = True

    logger.info("  jolts: %d frames flagged", int(jolt_mask.sum()))
    return jolt_mask


# ── Step 7 — Torso alignment check ───────────────────────────────────────────

def _check_torso_alignment(landmarks_scaled: np.ndarray, fps: float) -> np.ndarray:
    """Flag frames where torso deviates from the standing baseline.

    Uses the first BASELINE_SEC seconds to establish baseline angle.
    A camera that is already slightly tilted at the start is absorbed
    into the baseline — only *changes* from that baseline are flagged.
    """
    baseline_n = _frames(BASELINE_SEC, fps)
    T          = landmarks_scaled.shape[0]
    low_conf   = np.zeros(T, dtype=bool)
    angles     = np.full(T, np.nan)

    for t in range(T):
        pts = [landmarks_scaled[t, idx, :2]
               for idx in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER)]
        if any(np.isnan(p).any() for p in pts):
            continue
        l_hip, r_hip, l_sho, r_sho = pts
        angles[t] = angle_to_vertical(midpoint(l_hip, r_hip), midpoint(l_sho, r_sho))

    valid_baseline = angles[:baseline_n]
    valid_baseline = valid_baseline[~np.isnan(valid_baseline)]
    if len(valid_baseline) < 3:
        logger.warning("  alignment: insufficient baseline frames — skipping.")
        return low_conf

    baseline = float(np.median(valid_baseline))
    for t in range(T):
        if not np.isnan(angles[t]) and abs(angles[t] - baseline) > TORSO_ALIGNMENT_THRESHOLD:
            low_conf[t] = True

    logger.info("  torso baseline: %.1f° | low-confidence: %d frames",
                baseline, int(low_conf.sum()))
    return low_conf


# ── Step 8 — Confidence array ────────────────────────────────────────────────

def _build_confidence(
    detection_quality: np.ndarray,
    interpolation_mask: np.ndarray,
    jolt_mask: np.ndarray,
    low_confidence_mask: np.ndarray,
) -> np.ndarray:
    """Combine quality signals into a single per-frame score in [0, 1].

    conf = detection_quality
         × (1 − interp_fraction)   penalise interpolated frames
         × 0.0  if jolt            zero out camera-jolt frames
         × 0.5  if low_confidence  halve for torso-alignment failures
    """
    squat_interp = interpolation_mask[:, SQUAT_LANDMARKS]
    interp_frac  = squat_interp.mean(axis=1).astype(np.float32)

    conf  = detection_quality.copy().astype(np.float32)
    conf *= (1.0 - interp_frac)
    conf[jolt_mask]            = 0.0
    conf[low_confidence_mask] *= 0.5

    return np.clip(conf, 0.0, 1.0)


# ── Smoothing validation ─────────────────────────────────────────────────────

def _validate_smoothing(
    raw: np.ndarray,
    smoothed: np.ndarray,
    min_ratio: float = 0.80,
) -> None:
    """Warn if smoothing over-dampened the hip-Y motion range."""
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
            f"Consider reducing SG_WINDOW."
        )


# ── Loader (used by Stage 3) ─────────────────────────────────────────────────

def load_preprocessing(session_dir: str) -> dict:
    """Load Stage 2 outputs into memory."""
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


# ── Main entry point ─────────────────────────────────────────────────────────

def preprocess(
    session_dir: str,
    output_dir: Optional[str] = None,
) -> Path:
    """Run the full Stage 2 preprocessing pipeline.

    Args:
        session_dir: Path containing Stage 1 outputs.
        output_dir:  Override save location (defaults to session_dir).

    Returns:
        Path to the output directory.
    """
    session_dir = Path(session_dir)
    out_dir     = Path(output_dir) if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Stage 2] Preprocessing — %s", session_dir)

    data              = load_extraction(session_dir)
    landmarks_raw     = data["landmarks"]          # (T, 33, 3)
    detection_quality = data["detection_quality"]   # (T,)
    meta_s1           = data["metadata"]
    T                 = landmarks_raw.shape[0]
    fps               = float(meta_s1["fps"])

    logger.info("  frames: %d  fps: %.1f", T, fps)

    # 1. NaN interpolation
    filled, interp_mask = _interpolate_nans(landmarks_raw, fps)

    # 2. Y-flip
    yup = _flip_y(filled)

    # 3. Global scale
    scale = _compute_global_scale(yup, detection_quality)

    # 4+5. Scale + origin (pass raw landmarks to identify originally-NaN hips)
    lm_scaled, lm_norm = _apply_scale_and_origin(yup, scale, _flip_y(landmarks_raw))

    # 5. Jolt detection — must run on UNSMOOTHED signal
    jolt_mask = _detect_jolts(lm_scaled, fps)

    # 6. Savitzky-Golay smoothing
    lm_scaled_s = _smooth(lm_scaled, interp_mask)
    lm_norm_s   = _smooth(lm_norm,   interp_mask)
    _validate_smoothing(lm_scaled, lm_scaled_s)

    # 7. Torso alignment
    low_conf_mask = _check_torso_alignment(lm_scaled_s, fps)

    # 8. Confidence
    confidence = _build_confidence(
        detection_quality, interp_mask, jolt_mask, low_conf_mask,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    np.save(out_dir / "landmarks_normalised.npy", lm_norm_s)
    np.save(out_dir / "landmarks_scaled.npy",     lm_scaled_s)
    np.save(out_dir / "interpolation_mask.npy",   interp_mask)
    np.save(out_dir / "jolt_mask.npy",            jolt_mask)
    np.save(out_dir / "low_confidence_mask.npy",  low_conf_mask)
    np.save(out_dir / "confidence.npy",           confidence)

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
        "max_interp_gap_frames":     _frames(MAX_INTERP_GAP_SEC, fps),
        "jerk_threshold":            JERK_THRESHOLD,
        "jolt_expand_sec":           JOLT_EXPAND_SEC,
        "jolt_expand_frames":        _frames(JOLT_EXPAND_SEC, fps),
        "torso_baseline_sec":        BASELINE_SEC,
        "torso_baseline_frames":     _frames(BASELINE_SEC, fps),
        "torso_alignment_threshold": TORSO_ALIGNMENT_THRESHOLD,
    }
    with open(out_dir / "preprocessing_meta.json", "w") as f:
        json.dump(meta_s2, f, indent=2)

    logger.info(
        "  interpolated: %d | jolts: %d | low-conf: %d | mean conf: %.3f",
        int(interp_mask.sum()), int(jolt_mask.sum()),
        int(low_conf_mask.sum()), float(confidence.mean()),
    )
    return out_dir