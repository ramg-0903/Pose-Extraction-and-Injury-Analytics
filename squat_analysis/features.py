"""
features.py  —  Stage 3
=======================
Inputs  : Stage 2 session directory (landmarks_normalised, landmarks_scaled,
          confidence, jolt_mask, low_confidence_mask, preprocessing_meta)
Outputs : features.csv  — one row per rep, all features + binary flags
          features_meta.json — segmentation diagnostics

Entry points
    extract_features(session_dir) -> pd.DataFrame
    load_features(session_dir)    -> pd.DataFrame
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import theilslopes

from squat_analysis.config import (
    L_HIP, R_HIP, L_SHOULDER, R_SHOULDER,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT_INDEX, R_FOOT_INDEX,
    N_FRAMES,
    REP_PROMINENCE, REP_DISTANCE, MIN_REP_FRAMES, MAX_REP_FRAMES,
    MIN_DEPTH_FRACTION,
    DISCRETIZATION_THRESHOLDS,
)
from squat_analysis.preprocessing import load_preprocessing
from squat_analysis.utils import angle_between, angle_to_vertical, midpoint

logger = logging.getLogger(__name__)


# =============================================================================
# Rep segmentation
# =============================================================================

def _landmark_avg_y(landmarks: np.ndarray, idx_l: int, idx_r: int) -> np.ndarray:
    """Average Y trajectory for a bilateral landmark pair. Shape: (T,)."""
    l = landmarks[:, idx_l, 1]
    r = landmarks[:, idx_r, 1]
    return np.where(~np.isnan(l) & ~np.isnan(r), (l + r) / 2.0,
           np.where(~np.isnan(l), l, r))


def _pick_segmentation_signal(landmarks_scaled: np.ndarray) -> tuple:
    """Select the best signal for rep segmentation.

    For front/back view: hip-Y has the most vertical motion.
    For side view: knee-Y travels further as the knee bends forward.

    Automatically picks whichever bilateral signal has the largest range
    across the video. Returns (signal, signal_name).
    """
    candidates = {
        "hip":   _landmark_avg_y(landmarks_scaled, L_HIP,   R_HIP),
        "knee":  _landmark_avg_y(landmarks_scaled, L_KNEE,  R_KNEE),
        "ankle": _landmark_avg_y(landmarks_scaled, L_ANKLE, R_ANKLE),
    }
    ranges = {
        name: float(np.nanmax(sig) - np.nanmin(sig))
        for name, sig in candidates.items()
    }
    best = max(ranges, key=ranges.get)
    logger.info(
        "  segmentation signal: %s (range=%.4f) — hip=%.4f knee=%.4f ankle=%.4f",
        best, ranges[best], ranges["hip"], ranges["knee"], ranges["ankle"]
    )
    return candidates[best], best


def _segment_reps(
    landmarks_scaled: np.ndarray,
    fps: float,
) -> tuple:
    """Detect squat reps via valley or peak detection on the best motion signal.

    Strategy:
      1. Try valley detection first (standard — works when video starts/ends standing)
      2. If no valleys found, try peak detection (works when video starts/ends mid-squat)
         Peaks = standing positions. The rep is the motion between consecutive peaks,
         or from video start to first peak, or from last peak to video end.

    Returns:
        segments : list of (start, bottom, end) frame index tuples
        diag     : dict with segmentation diagnostic counts
    """
    T      = landmarks_scaled.shape[0]
    signal, signal_name = _pick_segmentation_signal(landmarks_scaled)

    if np.isnan(signal).any():
        valid = ~np.isnan(signal)
        if valid.sum() < 4:
            return [], {"n_candidate_valleys": 0, "n_accepted": 0,
                        "n_rejected_duration": 0, "n_rejected_depth": 0,
                        "signal_used": signal_name}
        t_idx  = np.arange(T, dtype=float)
        signal = np.interp(t_idx, t_idx[valid], signal[valid])

    n_rejected_dur   = 0
    n_rejected_depth = 0
    segments         = []

    # ── Try valley detection first ────────────────────────────────────────────
    bottoms, _ = find_peaks(
        -signal,
        prominence=REP_PROMINENCE,
        distance=REP_DISTANCE,
        width=3,
    )

    if len(bottoms) > 0:
        # Standard path — valleys found
        raw_depths = []
        for b in bottoms:
            lb    = int(bottoms[bottoms < b][-1]) if any(bottoms < b) else 0
            rb    = int(bottoms[bottoms > b][0])  if any(bottoms > b) else T
            top_y = float(np.nanmax(signal[lb:b + 1])) if b > lb else float(signal[lb])
            raw_depths.append(top_y - float(signal[b]))
        session_median_depth = float(np.median(raw_depths)) if raw_depths else 0.0

        for i, bottom in enumerate(bottoms):
            lb    = int(bottoms[i - 1]) if i > 0 else 0
            rb    = int(bottoms[i + 1]) if i < len(bottoms) - 1 else T
            start = lb + int(np.argmax(signal[lb:bottom + 1]))
            end   = bottom + int(np.argmax(signal[bottom:rb]))

            duration = end - start
            if duration < MIN_REP_FRAMES or duration > MAX_REP_FRAMES:
                n_rejected_dur += 1
                continue

            depth = float(signal[start]) - float(signal[bottom])
            if session_median_depth > 1e-6 and depth < MIN_DEPTH_FRACTION * session_median_depth:
                n_rejected_depth += 1
                continue

            segments.append((start, int(bottom), end))

    else:
        # ── Fallback: peak detection ──────────────────────────────────────────
        # Video starts/ends mid-squat. Find standing peaks and build reps
        # around the valleys between them, or treat the whole video as one rep.
        peaks, _ = find_peaks(
            signal,
            prominence=REP_PROMINENCE,
            distance=REP_DISTANCE,
            width=3,
        )

        if len(peaks) == 0:
            # No peaks either — try treating entire video as one rep
            # Find the global min as the bottom
            bottom = int(np.nanargmin(signal))
            duration = T - 1
            if MIN_REP_FRAMES <= duration <= MAX_REP_FRAMES:
                segments.append((0, bottom, T - 1))
        else:
            # Build rep segments around each peak
            # Each rep goes from the previous peak (or start) to the next peak (or end)
            boundaries = [0] + list(peaks) + [T - 1]
            for i in range(len(boundaries) - 1):
                start  = boundaries[i]
                end    = boundaries[i + 1]
                duration = end - start
                if duration < MIN_REP_FRAMES or duration > MAX_REP_FRAMES:
                    n_rejected_dur += 1
                    continue
                # Bottom = minimum signal in this window
                bottom = start + int(np.nanargmin(signal[start:end + 1]))
                segments.append((start, bottom, end))

    n_candidates = len(bottoms) if len(bottoms) > 0 else len(segments)

    diag = {
        "n_candidate_valleys": n_candidates,
        "n_accepted":          len(segments),
        "n_rejected_duration": n_rejected_dur,
        "n_rejected_depth":    n_rejected_depth,
        "signal_used":         signal_name,
    }

    if len(segments) < 3:
        warnings.warn(
            f"Only {len(segments)} reps detected. Session features may be unreliable."
        )

    return segments, diag


# =============================================================================
# Rep normalisation
# =============================================================================

def _normalize_rep(
    landmarks_normalised: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Resample rep [start:end] to exactly N_FRAMES using linear interpolation.

    Linear chosen over cubic to avoid overshoot on short or noisy reps.

    Returns: (N_FRAMES, 33, 2)
    """
    raw = landmarks_normalised[start:end]   # (L, 33, 2)
    L   = raw.shape[0]

    if L == N_FRAMES:
        return raw.astype(np.float32)

    src = np.linspace(0.0, 1.0, L)
    tgt = np.linspace(0.0, 1.0, N_FRAMES)
    out = np.empty((N_FRAMES, 33, 2), dtype=np.float32)

    for lm in range(33):
        for ax in range(2):
            col = raw[:, lm, ax].astype(float)
            if np.isnan(col).all():
                out[:, lm, ax] = np.nan
                continue
            fn = interp1d(src, col, kind="linear",
                          bounds_error=False, fill_value=np.nan)
            out[:, lm, ax] = fn(tgt)

    return out


# =============================================================================
# Angle trajectories
# =============================================================================

def _safe_angle(a, vertex, b) -> float:
    """angle_between returning NaN if any input is NaN."""
    if np.isnan(a).any() or np.isnan(vertex).any() or np.isnan(b).any():
        return np.nan
    return angle_between(a, vertex, b)


def _safe_vertical(a, b) -> float:
    """angle_to_vertical returning NaN if any input is NaN."""
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    return angle_to_vertical(a, b)


def _compute_angle_trajectories(rep: np.ndarray) -> dict:
    """Compute per-frame joint angle trajectories for one normalised rep.

    Args:
        rep: (N_FRAMES, 33, 2)

    Returns:
        dict of (N_FRAMES,) float arrays. NaN where landmarks missing.
        Keys: knee_flexion_L/R, trunk_inclination, hip_flexion_L/R (view-dep),
              ankle_dorsiflexion_L/R (view-dep), hip_y
    """
    n = rep.shape[0]
    out = {k: np.full(n, np.nan) for k in [
        "knee_flexion_L", "knee_flexion_R",
        "trunk_inclination",
        "hip_flexion_L",  "hip_flexion_R",
        "ankle_dorsiflexion_L", "ankle_dorsiflexion_R",
        "hip_y",
    ]}

    for f in range(n):
        l_hip = rep[f, L_HIP];  r_hip = rep[f, R_HIP]
        l_sho = rep[f, L_SHOULDER]; r_sho = rep[f, R_SHOULDER]
        l_kne = rep[f, L_KNEE]; r_kne = rep[f, R_KNEE]
        l_ank = rep[f, L_ANKLE]; r_ank = rep[f, R_ANKLE]
        l_ft  = rep[f, L_FOOT_INDEX]; r_ft = rep[f, R_FOOT_INDEX]

        out["knee_flexion_L"][f]       = _safe_angle(l_hip, l_kne, l_ank)
        out["knee_flexion_R"][f]       = _safe_angle(r_hip, r_kne, r_ank)
        out["trunk_inclination"][f]    = _safe_vertical(midpoint(l_hip, r_hip),
                                                        midpoint(l_sho, r_sho))
        out["hip_flexion_L"][f]        = _safe_angle(l_sho, l_hip, l_kne)
        out["hip_flexion_R"][f]        = _safe_angle(r_sho, r_hip, r_kne)
        out["ankle_dorsiflexion_L"][f] = _safe_angle(l_kne, l_ank, l_ft)
        out["ankle_dorsiflexion_R"][f] = _safe_angle(r_kne, r_ank, r_ft)

        if not (np.isnan(l_hip).any() and np.isnan(r_hip).any()):
            mid = midpoint(l_hip, r_hip) if not np.isnan(l_hip).any() and not np.isnan(r_hip).any() \
                  else (r_hip if np.isnan(l_hip).any() else l_hip)
            out["hip_y"][f] = float(mid[1])

    return out


def _find_bottom_frame(traj: dict) -> int:
    """Bottom frame = frame of minimum knee-Y within the normalised rep.

    Knee-Y works for all camera angles — the knee is lowest (most bent)
    at the squat bottom regardless of whether the camera is front, back,
    or side. Falls back to middle frame if knee-Y is entirely NaN.
    """
    knee_avg = np.where(
        ~np.isnan(traj["knee_flexion_L"]) & ~np.isnan(traj["knee_flexion_R"]),
        (traj["knee_flexion_L"] + traj["knee_flexion_R"]) / 2.0,
        np.where(~np.isnan(traj["knee_flexion_L"]),
                 traj["knee_flexion_L"], traj["knee_flexion_R"])
    )
    # Minimum knee flexion angle = most bent = squat bottom
    valid = ~np.isnan(knee_avg)
    if valid.sum() == 0:
        return N_FRAMES // 2
    return int(np.nanargmin(knee_avg))


def _window_median(arr: np.ndarray, centre: int, half: int = 1) -> float:
    """Median of arr over [centre-half, centre+half] clipped to array bounds."""
    lo  = max(0, centre - half)
    hi  = min(len(arr), centre + half + 1)
    vals = arr[lo:hi]
    valid = vals[~np.isnan(vals)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


# =============================================================================
# Category A — static features
# =============================================================================

def _extract_static(traj: dict, bottom: int) -> dict:
    kL  = traj["knee_flexion_L"]
    kR  = traj["knee_flexion_R"]
    tr  = traj["trunk_inclination"]
    hL  = traj["hip_flexion_L"]
    hR  = traj["hip_flexion_R"]
    aL  = traj["ankle_dorsiflexion_L"]
    aR  = traj["ankle_dorsiflexion_R"]

    knee_avg = np.where(~np.isnan(kL) & ~np.isnan(kR), (kL + kR) / 2.0,
               np.where(~np.isnan(kL), kL, kR))

    start = 0
    end   = N_FRAMES - 1
    ha    = (hL + hR) / 2.0
    aa    = (aL + aR) / 2.0

    return {
        # ── Bottom keypoint ───────────────────────────────────────────────────
        "knee_flexion_at_bottom":          _window_median(knee_avg, bottom),
        "knee_flexion_L_at_bottom":        _window_median(kL, bottom),
        "knee_flexion_R_at_bottom":        _window_median(kR, bottom),
        "knee_flexion_range":              float(np.nanmax(knee_avg) - np.nanmin(knee_avg)),
        "trunk_lean_at_bottom":            _window_median(tr, bottom),
        "trunk_lean_mean":                 float(np.nanmean(tr)),
        "trunk_lean_max":                  float(np.nanmax(tr)),
        "trunk_lean_range":                float(np.nanmax(tr) - np.nanmin(tr)),
        "symmetry_knee_at_bottom":         abs(_window_median(kL, bottom) -
                                               _window_median(kR, bottom)),
        "symmetry_knee_mean":              float(np.nanmean(np.abs(kL - kR))),
        "hip_flexion_at_bottom":           _window_median(ha, bottom),
        "ankle_dorsiflexion_at_bottom":    _window_median(aa, bottom),

        # ── Start keypoint (standing before descent) ──────────────────────────
        "knee_flexion_at_start":           _window_median(knee_avg, start),
        "trunk_lean_at_start":             _window_median(tr, start),
        "symmetry_knee_at_start":          abs(_window_median(kL, start) -
                                               _window_median(kR, start)),
        "hip_flexion_at_start":            _window_median(ha, start),
        "ankle_dorsiflexion_at_start":     _window_median(aa, start),

        # ── End keypoint (standing after ascent) ──────────────────────────────
        "knee_flexion_at_end":             _window_median(knee_avg, end),
        "trunk_lean_at_end":               _window_median(tr, end),
        "symmetry_knee_at_end":            abs(_window_median(kL, end) -
                                               _window_median(kR, end)),
        "hip_flexion_at_end":              _window_median(ha, end),
        "ankle_dorsiflexion_at_end":       _window_median(aa, end),
    }


# =============================================================================
# Category B — temporal features
# =============================================================================

def _extract_temporal(traj: dict, bottom: int, fps: float) -> dict:
    kL = traj["knee_flexion_L"]
    kR = traj["knee_flexion_R"]
    tr = traj["trunk_inclination"]

    knee_avg = np.where(~np.isnan(kL) & ~np.isnan(kR), (kL + kR) / 2.0,
               np.where(~np.isnan(kL), kL, kR))

    dt = 1.0 / fps   # seconds per frame

    descent = max(bottom, 1)
    ascent  = max(N_FRAMES - 1 - bottom, 1)

    # Velocity (deg/s) and jerk (deg/s³) on smoothed knee trajectory
    vel  = np.gradient(knee_avg) / dt
    jerk = np.gradient(np.gradient(vel) / dt) / dt

    # Cap jerk at 99th percentile to suppress interpolation spikes
    jerk_cap = np.nanpercentile(np.abs(jerk), 99)
    jerk      = np.clip(jerk, -jerk_cap, jerk_cap)

    jerk_rms = float(np.sqrt(np.nanmean(jerk ** 2)))
    k_range  = float(np.nanmax(knee_avg) - np.nanmin(knee_avg))

    return {
        "descent_frames":         float(descent),
        "ascent_frames":          float(ascent),
        "descent_time_s":         round(descent * dt, 4),
        "ascent_time_s":          round(ascent  * dt, 4),
        "rep_duration_s":         round(N_FRAMES * dt, 4),
        # ratio < 1 = faster descent than ascent (rushed)
        "descent_ascent_ratio":   round(descent / ascent, 4),

        "knee_vel_max_descent":   float(np.nanmax(np.abs(vel[:bottom]))) if bottom > 0 else 0.0,
        "knee_vel_max_ascent":    float(np.nanmax(np.abs(vel[bottom:]))),

        "knee_jerk_max":          float(np.nanmax(np.abs(jerk))),
        "knee_jerk_rms":          jerk_rms,
        # Smoothness metric — NaN if range too small (no meaningful movement)
        "normalized_jerk_cost":   float(np.nansum(jerk ** 2) /
                                        (N_FRAMES * k_range ** 2))
                                  if k_range > 0.01 else np.nan,
    }


# =============================================================================
# Category C — session features
# =============================================================================

def _extract_session(all_rep_feats: list) -> dict:
    """Compute fatigue and consistency metrics across all reps.

    Uses Theil-Sen (median-based) regression for slopes — robust to
    outlier reps which would distort least-squares.
    """
    n = len(all_rep_feats)
    if n < 3:
        warnings.warn("Fewer than 3 reps — session features unreliable.")

    idx = np.arange(n, dtype=float)

    def _slope(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in all_rep_feats], dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            return np.nan
        res = theilslopes(vals[valid], idx[valid])
        return float(res.slope)

    def _cv(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in all_rep_feats], dtype=float)
        m = float(np.nanmean(vals))
        return float(np.nanstd(vals) / m) if abs(m) > 1e-9 else np.nan

    return {
        "n_reps":                    n,
        "fatigue_trunk_lean_slope":  _slope("trunk_lean_mean"),
        "fatigue_knee_depth_slope":  _slope("knee_flexion_at_bottom"),
        "depth_consistency_cv":      _cv("knee_flexion_at_bottom"),
    }


# =============================================================================
# Rep quality metrics (from Stage 2 masks)
# =============================================================================

def _rep_quality(
    start: int, end: int,
    confidence: np.ndarray,
    jolt_mask: np.ndarray,
    low_conf_mask: np.ndarray,
    interp_mask: np.ndarray,
) -> dict:
    slc = slice(start, end)
    n   = end - start
    return {
        "rep_mean_confidence":   round(float(np.mean(confidence[slc])), 4),
        "rep_interp_fraction":   round(float(interp_mask[slc].mean()), 4),
        "rep_jolt_frames":       int(jolt_mask[slc].sum()),
        "rep_low_conf_frames":   int(low_conf_mask[slc].sum()),
    }


# =============================================================================
# Discretisation → binary flags
# =============================================================================

def _discretize(feat: dict) -> dict:
    flags = {}
    for feat_name, (threshold, direction, flag_name) in DISCRETIZATION_THRESHOLDS.items():
        val = feat.get(feat_name, np.nan)
        if np.isnan(val):
            flags[flag_name] = np.nan
            continue
        flags[flag_name] = int(val < threshold if direction == "lt" else val > threshold)
    return flags


# =============================================================================
# Loader
# =============================================================================

def load_features(session_dir: str) -> pd.DataFrame:
    """Load Stage 3 CSV output."""
    return pd.read_csv(Path(session_dir) / "features.csv")


# =============================================================================
# Main entry point
# =============================================================================

def extract_features(
    session_dir: str,
    output_dir: Optional[str] = None,
    save_trajectories: bool = False,
) -> pd.DataFrame:
    """Run Stage 3 on a preprocessed session directory.

    Args:
        session_dir:       Path containing Stage 2 outputs.
        output_dir:        Override save path. Defaults to session_dir.
        save_trajectories: If True, save per-rep angle trajectories to
                           trajectories.npz for debugging and visualisation.

    Returns:
        pd.DataFrame with one row per rep.
    """
    session_dir = Path(session_dir)
    out_dir     = Path(output_dir) if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stage 3] Feature extraction")
    print(f"  session : {session_dir}")

    data           = load_preprocessing(session_dir)
    lm_norm        = data["landmarks_normalised"]   # (T, 33, 2)
    lm_scaled      = data["landmarks_scaled"]       # (T, 33, 2)
    confidence     = data["confidence"]             # (T,)
    jolt_mask      = data["jolt_mask"]              # (T,)
    low_conf_mask  = data["low_confidence_mask"]    # (T,)
    interp_mask    = data["interpolation_mask"]     # (T, 33)
    meta           = data["preprocessing_meta"]
    fps            = float(meta["fps"])
    session_id     = meta["session_id"]

    if not (10.0 <= fps <= 120.0):
        warnings.warn(
            f"Unexpected FPS value ({fps}). Temporal features may be unreliable."
        )

    # Stage 2 saves (T, 33, 2) — interp_mask is (T, 33) bool
    # For rep quality we need a per-frame scalar → mean across landmarks
    interp_per_frame = interp_mask.mean(axis=1)    # (T,)

    # ── Rep segmentation ──────────────────────────────────────────────────────
    print("  [1/4] Rep segmentation")
    segments, seg_diag = _segment_reps(lm_scaled, fps)

    print(f"         {seg_diag['n_accepted']} reps accepted  "
          f"| {seg_diag['n_candidate_valleys']} valleys found  "
          f"| {seg_diag['n_rejected_duration']} rejected (duration)  "
          f"| {seg_diag['n_rejected_depth']} rejected (depth)")

    if not segments:
        raise RuntimeError(
            "No valid reps detected. Check video quality and REP_PROMINENCE in config."
        )

    # ── Per-rep feature extraction ────────────────────────────────────────────
    print("  [2/4] Per-rep angle + feature extraction")
    all_rep_feats = []
    rows          = []
    all_trajs     = []   # populated only when save_trajectories=True

    for i, (start, bottom_raw, end) in enumerate(segments):
        rep  = _normalize_rep(lm_norm, start, end)
        traj = _compute_angle_trajectories(rep)
        bot  = _find_bottom_frame(traj)

        static   = _extract_static(traj, bot)
        temporal = _extract_temporal(traj, bot, fps)
        quality  = _rep_quality(start, end, confidence,
                                jolt_mask, low_conf_mask, interp_per_frame)

        all_rep_feats.append({**static, **temporal})
        if save_trajectories:
            all_trajs.append(traj)

        rows.append({
            "session_id":        session_id,
            "rep_index":         i,
            "frame_start":       start,
            "frame_bottom":      bottom_raw,
            "frame_end":         end,
            # view-dependent flag — True only when camera_view explicitly "side"
            "view_dep_reliable": meta.get("camera_view", "unknown") == "side",
            **quality,
            **static,
            **temporal,
        })

    # ── Session features ──────────────────────────────────────────────────────
    print("  [3/4] Session features")
    session_feats = _extract_session(all_rep_feats)
    for row in rows:
        row.update(session_feats)

    # ── Discretisation ────────────────────────────────────────────────────────
    print("  [4/4] Discretisation → binary flags")
    for row in rows:
        row.update(_discretize(row))

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Pre-save sanity checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().all(axis=1).any():
        warnings.warn("One or more reps have all-NaN features. Check tracking quality.")

    df.to_csv(out_dir / "features.csv", index=False)

    if save_trajectories and all_trajs:
        npz_data = {}
        for key in all_trajs[0]:
            npz_data[key] = np.stack([t[key] for t in all_trajs], axis=0)
        np.savez(out_dir / "trajectories.npz", **npz_data)
        print(f"  trajectories   : saved to {out_dir / 'trajectories.npz'}")

    features_meta = {
        "session_id":       session_id,
        "fps":              fps,
        "n_frames_per_rep": N_FRAMES,
        "feature_version":  "1.0",
        **seg_diag,
        **session_feats,
    }
    with open(out_dir / "features_meta.json", "w") as f:
        json.dump(features_meta, f, indent=2)

    print(f"  reps extracted : {len(df)}")
    print(f"  features       : {len(df.columns)} columns")
    print(f"  saved to       : {out_dir / 'features.csv'}")

    return df
