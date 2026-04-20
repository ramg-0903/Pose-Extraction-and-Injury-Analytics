"""
Stage 3 — Feature Extraction

Inputs:  Stage 2 session directory (landmarks_normalised, landmarks_scaled,
         confidence, jolt_mask, low_confidence_mask, preprocessing_meta).
Outputs: features.csv       — one row per rep, all features + binary flags.
         features_meta.json — segmentation diagnostics.

Entry points:
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
    MIN_DEPTH_FRACTION, DISCRETIZATION_THRESHOLDS,
)
from squat_analysis.preprocessing import load_preprocessing
from squat_analysis.utils import angle_between, angle_to_vertical, midpoint

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bilateral_avg(arr_l: np.ndarray, arr_r: np.ndarray) -> np.ndarray:
    """Average of two bilateral signals, falling back to whichever is valid."""
    return np.where(
        ~np.isnan(arr_l) & ~np.isnan(arr_r), (arr_l + arr_r) / 2.0,
        np.where(~np.isnan(arr_l), arr_l, arr_r),
    )


def _safe_angle(a, vertex, b) -> float:
    """angle_between that returns NaN if any input contains NaN."""
    if np.isnan(a).any() or np.isnan(vertex).any() or np.isnan(b).any():
        return np.nan
    return angle_between(a, vertex, b)


def _safe_vertical(a, b) -> float:
    """angle_to_vertical that returns NaN if any input contains NaN."""
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    return angle_to_vertical(a, b)


def _window_median(arr: np.ndarray, centre: int, half: int = 1) -> float:
    """Median over [centre-half, centre+half], clipped to array bounds."""
    lo    = max(0, centre - half)
    hi    = min(len(arr), centre + half + 1)
    valid = arr[lo:hi]
    valid = valid[~np.isnan(valid)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


# ── Rep Segmentation ─────────────────────────────────────────────────────────

def _landmark_avg_y(landmarks: np.ndarray, idx_l: int, idx_r: int) -> np.ndarray:
    """Average Y trajectory for a bilateral landmark pair. Shape: (T,)."""
    return _bilateral_avg(landmarks[:, idx_l, 1], landmarks[:, idx_r, 1])


def _pick_segmentation_signal(landmarks_scaled: np.ndarray) -> tuple:
    """Pick the landmark whose Y-trajectory has the largest range.

    Side-view → knee-Y dominates (knee travels forward as it bends).
    Front/back → hip-Y dominates (vertical drop).
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
        "  seg signal: %s (%.4f) — hip=%.4f knee=%.4f ankle=%.4f",
        best, ranges[best], ranges["hip"], ranges["knee"], ranges["ankle"],
    )
    return candidates[best], best


def _segment_reps(landmarks_scaled: np.ndarray, fps: float) -> tuple:
    """Detect squat reps via valley (then peak-fallback) detection.

    Strategy:
      1. Valley detection — standard when video starts/ends standing.
      2. Peak fallback — when video starts/ends mid-squat. Peaks are
         standing positions; reps are the motion between consecutive peaks.

    Returns (segments, diagnostics) where each segment is
    (start_frame, bottom_frame, end_frame).
    """
    T = landmarks_scaled.shape[0]
    signal, signal_name = _pick_segmentation_signal(landmarks_scaled)

    # Fill NaNs with linear interp for peak-finding only
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

    # ── Primary: valley detection ─────────────────────────────────────────
    bottoms, _ = find_peaks(
        -signal, prominence=REP_PROMINENCE, distance=REP_DISTANCE, width=3,
    )

    if len(bottoms) > 0:
        # Compute per-valley raw depths for partial-rep rejection
        raw_depths = []
        for b in bottoms:
            lb    = int(bottoms[bottoms < b][-1]) if any(bottoms < b) else 0
            top_y = float(np.nanmax(signal[lb:b + 1])) if b > lb else float(signal[lb])
            raw_depths.append(top_y - float(signal[b]))
        session_median_depth = float(np.median(raw_depths)) if raw_depths else 0.0

        for i, bottom in enumerate(bottoms):
            lb    = int(bottoms[i - 1]) if i > 0             else 0
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

    # ── Fallback: peak detection ──────────────────────────────────────────
    else:
        peaks, _ = find_peaks(
            signal, prominence=REP_PROMINENCE, distance=REP_DISTANCE, width=3,
        )
        if len(peaks) == 0:
            bottom   = int(np.nanargmin(signal))
            duration = T - 1
            if MIN_REP_FRAMES <= duration <= MAX_REP_FRAMES:
                segments.append((0, bottom, T - 1))
        else:
            boundaries = [0] + list(peaks) + [T - 1]
            for i in range(len(boundaries) - 1):
                start    = boundaries[i]
                end      = boundaries[i + 1]
                duration = end - start
                if duration < MIN_REP_FRAMES or duration > MAX_REP_FRAMES:
                    n_rejected_dur += 1
                    continue
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
            f"Only {len(segments)} reps detected — session features may be unreliable."
        )
    return segments, diag


# ── Rep Normalisation ─────────────────────────────────────────────────────────

def _normalize_rep(
    landmarks_normalised: np.ndarray,
    start: int, end: int,
) -> np.ndarray:
    """Resample rep [start:end] to exactly N_FRAMES via linear interpolation.

    Linear chosen over cubic to avoid overshoot on short or noisy reps.
    Returns: (N_FRAMES, 33, 2).
    """
    raw = landmarks_normalised[start:end]  # (L, 33, 2)
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


# ── Angle Trajectories ───────────────────────────────────────────────────────

def _compute_angle_trajectories(rep: np.ndarray) -> dict:
    """Per-frame joint angle trajectories for one normalised rep.

    Returns dict of (N_FRAMES,) arrays.  NaN where landmarks are missing.
    Keys: knee_flexion_L/R, trunk_inclination, hip_flexion_L/R,
          ankle_dorsiflexion_L/R, hip_y.
    """
    n = rep.shape[0]
    out = {k: np.full(n, np.nan) for k in [
        "knee_flexion_L", "knee_flexion_R", "trunk_inclination",
        "hip_flexion_L", "hip_flexion_R",
        "ankle_dorsiflexion_L", "ankle_dorsiflexion_R", "hip_y",
    ]}

    for f in range(n):
        l_hip = rep[f, L_HIP];      r_hip = rep[f, R_HIP]
        l_sho = rep[f, L_SHOULDER];  r_sho = rep[f, R_SHOULDER]
        l_kne = rep[f, L_KNEE];     r_kne = rep[f, R_KNEE]
        l_ank = rep[f, L_ANKLE];    r_ank = rep[f, R_ANKLE]
        l_ft  = rep[f, L_FOOT_INDEX]; r_ft = rep[f, R_FOOT_INDEX]

        out["knee_flexion_L"][f]       = _safe_angle(l_hip, l_kne, l_ank)
        out["knee_flexion_R"][f]       = _safe_angle(r_hip, r_kne, r_ank)
        out["trunk_inclination"][f]    = _safe_vertical(
            midpoint(l_hip, r_hip), midpoint(l_sho, r_sho),
        )
        out["hip_flexion_L"][f]        = _safe_angle(l_sho, l_hip, l_kne)
        out["hip_flexion_R"][f]        = _safe_angle(r_sho, r_hip, r_kne)
        out["ankle_dorsiflexion_L"][f] = _safe_angle(l_kne, l_ank, l_ft)
        out["ankle_dorsiflexion_R"][f] = _safe_angle(r_kne, r_ank, r_ft)

        # hip_y: use midpoint of both hips, fall back to whichever is valid
        if not (np.isnan(l_hip).any() and np.isnan(r_hip).any()):
            mid = (
                midpoint(l_hip, r_hip) if not np.isnan(l_hip).any() and not np.isnan(r_hip).any()
                else (r_hip if np.isnan(l_hip).any() else l_hip)
            )
            out["hip_y"][f] = float(mid[1])

    return out


def _find_bottom_frame(traj: dict) -> int:
    """Frame of minimum knee flexion angle (= deepest squat position).

    Knee-flexion works for all camera angles — it's lowest at the
    squat bottom regardless of front/back/side view.
    Falls back to the middle frame if entirely NaN.
    """
    knee_avg = _bilateral_avg(traj["knee_flexion_L"], traj["knee_flexion_R"])
    valid = ~np.isnan(knee_avg)
    if valid.sum() == 0:
        return N_FRAMES // 2
    return int(np.nanargmin(knee_avg))


# ── Category A — Static Features ─────────────────────────────────────────────

def _extract_static(traj: dict, bottom: int) -> dict:
    """Features computed at specific keypoints (start, bottom, end)."""
    kL = traj["knee_flexion_L"]
    kR = traj["knee_flexion_R"]
    tr = traj["trunk_inclination"]
    hL = traj["hip_flexion_L"];  hR = traj["hip_flexion_R"]
    aL = traj["ankle_dorsiflexion_L"];  aR = traj["ankle_dorsiflexion_R"]

    knee_avg = _bilateral_avg(kL, kR)
    ha = (hL + hR) / 2.0
    aa = (aL + aR) / 2.0

    start = 0
    end   = N_FRAMES - 1

    return {
        # Bottom keypoint
        "knee_flexion_at_bottom":       _window_median(knee_avg, bottom),
        "knee_flexion_L_at_bottom":     _window_median(kL, bottom),
        "knee_flexion_R_at_bottom":     _window_median(kR, bottom),
        "knee_flexion_range":           float(np.nanmax(knee_avg) - np.nanmin(knee_avg)),
        "trunk_lean_at_bottom":         _window_median(tr, bottom),
        "trunk_lean_mean":              float(np.nanmean(tr)),
        "trunk_lean_max":               float(np.nanmax(tr)),
        "trunk_lean_range":             float(np.nanmax(tr) - np.nanmin(tr)),
        "symmetry_knee_at_bottom":      abs(_window_median(kL, bottom) -
                                            _window_median(kR, bottom)),
        "symmetry_knee_mean":           float(np.nanmean(np.abs(kL - kR))),
        "hip_flexion_at_bottom":        _window_median(ha, bottom),
        "ankle_dorsiflexion_at_bottom": _window_median(aa, bottom),

        # Start keypoint (standing before descent)
        "knee_flexion_at_start":        _window_median(knee_avg, start),
        "trunk_lean_at_start":          _window_median(tr, start),
        "symmetry_knee_at_start":       abs(_window_median(kL, start) -
                                            _window_median(kR, start)),
        "hip_flexion_at_start":         _window_median(ha, start),
        "ankle_dorsiflexion_at_start":  _window_median(aa, start),

        # End keypoint (standing after ascent)
        "knee_flexion_at_end":          _window_median(knee_avg, end),
        "trunk_lean_at_end":            _window_median(tr, end),
        "symmetry_knee_at_end":         abs(_window_median(kL, end) -
                                            _window_median(kR, end)),
        "hip_flexion_at_end":           _window_median(ha, end),
        "ankle_dorsiflexion_at_end":    _window_median(aa, end),
    }


# ── Category B — Temporal Features ───────────────────────────────────────────

def _extract_temporal(traj: dict, bottom: int, fps: float) -> dict:
    """Timing, velocity, and smoothness features."""
    knee_avg = _bilateral_avg(traj["knee_flexion_L"], traj["knee_flexion_R"])

    dt      = 1.0 / fps
    descent = max(bottom, 1)
    ascent  = max(N_FRAMES - 1 - bottom, 1)

    vel  = np.gradient(knee_avg) / dt
    jerk = np.gradient(np.gradient(vel) / dt) / dt

    # Cap jerk at 99th percentile to suppress interpolation edge spikes
    jerk_cap = np.nanpercentile(np.abs(jerk), 99)
    jerk     = np.clip(jerk, -jerk_cap, jerk_cap)

    jerk_rms = float(np.sqrt(np.nanmean(jerk ** 2)))
    k_range  = float(np.nanmax(knee_avg) - np.nanmin(knee_avg))

    return {
        "descent_frames":       float(descent),
        "ascent_frames":        float(ascent),
        "descent_time_s":       round(descent * dt, 4),
        "ascent_time_s":        round(ascent * dt, 4),
        "rep_duration_s":       round(N_FRAMES * dt, 4),
        "descent_ascent_ratio": round(descent / ascent, 4),

        "knee_vel_max_descent": float(np.nanmax(np.abs(vel[:bottom]))) if bottom > 0 else 0.0,
        "knee_vel_max_ascent":  float(np.nanmax(np.abs(vel[bottom:]))),

        "knee_jerk_max":        float(np.nanmax(np.abs(jerk))),
        "knee_jerk_rms":        jerk_rms,
        # Smoothness — NaN if range too small (no meaningful movement)
        "normalized_jerk_cost": (
            float(np.nansum(jerk ** 2) / (N_FRAMES * k_range ** 2))
            if k_range > 0.01 else np.nan
        ),
    }


# ── Category C — Session Features ────────────────────────────────────────────

def _extract_session(all_rep_feats: list) -> dict:
    """Fatigue and consistency metrics across all reps.

    Uses Theil-Sen regression (median-based) for slopes — robust to
    outlier reps that would distort least-squares.
    """
    n   = len(all_rep_feats)
    idx = np.arange(n, dtype=float)

    if n < 3:
        warnings.warn("Fewer than 3 reps — session features unreliable.")

    def _slope(key: str) -> float:
        vals  = np.array([r.get(key, np.nan) for r in all_rep_feats], dtype=float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            return np.nan
        return float(theilslopes(vals[valid], idx[valid]).slope)

    def _cv(key: str) -> float:
        vals = np.array([r.get(key, np.nan) for r in all_rep_feats], dtype=float)
        m = float(np.nanmean(vals))
        return float(np.nanstd(vals) / m) if abs(m) > 1e-9 else np.nan

    return {
        "n_reps":                   n,
        "fatigue_trunk_lean_slope": _slope("trunk_lean_mean"),
        "fatigue_knee_depth_slope": _slope("knee_flexion_at_bottom"),
        "depth_consistency_cv":     _cv("knee_flexion_at_bottom"),
    }


# ── Rep Quality ──────────────────────────────────────────────────────────────

def _rep_quality(
    start: int, end: int,
    confidence: np.ndarray,
    jolt_mask: np.ndarray,
    low_conf_mask: np.ndarray,
    interp_mask: np.ndarray,
) -> dict:
    """Quality metrics for a single rep span [start:end]."""
    slc = slice(start, end)
    return {
        "rep_mean_confidence": round(float(np.mean(confidence[slc])), 4),
        "rep_interp_fraction": round(float(interp_mask[slc].mean()), 4),
        "rep_jolt_frames":     int(jolt_mask[slc].sum()),
        "rep_low_conf_frames": int(low_conf_mask[slc].sum()),
    }


# ── Discretisation → Binary Flags ────────────────────────────────────────────

def _discretize(feat: dict) -> dict:
    """Apply biomechanical thresholds to produce binary form-error flags."""
    flags = {}
    for feat_name, (threshold, direction, flag_name) in DISCRETIZATION_THRESHOLDS.items():
        val = feat.get(feat_name, np.nan)
        if np.isnan(val):
            flags[flag_name] = np.nan
            continue
        flags[flag_name] = int(val < threshold if direction == "lt" else val > threshold)
    return flags


# ── Loader ────────────────────────────────────────────────────────────────────

def load_features(session_dir: str) -> pd.DataFrame:
    """Load Stage 3 CSV output."""
    return pd.read_csv(Path(session_dir) / "features.csv")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def extract_features(
    session_dir: str,
    output_dir: Optional[str] = None,
    save_trajectories: bool = False,
) -> pd.DataFrame:
    """Run Stage 3 on a preprocessed session directory.

    Args:
        session_dir:       Path containing Stage 2 outputs.
        output_dir:        Override save path (defaults to session_dir).
        save_trajectories: Save per-rep angle trajectories to .npz
                           for debugging / visualisation.

    Returns:
        DataFrame with one row per rep.
    """
    session_dir = Path(session_dir)
    out_dir     = Path(output_dir) if output_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[Stage 3] Feature extraction — %s", session_dir)

    data          = load_preprocessing(session_dir)
    lm_norm       = data["landmarks_normalised"]
    lm_scaled     = data["landmarks_scaled"]
    confidence    = data["confidence"]
    jolt_mask     = data["jolt_mask"]
    low_conf_mask = data["low_confidence_mask"]
    interp_mask   = data["interpolation_mask"]
    meta          = data["preprocessing_meta"]
    fps           = float(meta["fps"])
    session_id    = meta["session_id"]

    if not (10.0 <= fps <= 120.0):
        warnings.warn(f"Unexpected FPS ({fps}) — temporal features may be unreliable.")

    # interp_mask is (T, 33) bool → per-frame scalar for rep quality
    interp_per_frame = interp_mask.mean(axis=1)

    # ── Rep segmentation ──────────────────────────────────────────────────
    segments, seg_diag = _segment_reps(lm_scaled, fps)
    logger.info(
        "  %d reps accepted | %d valleys | %d rej(dur) | %d rej(depth)",
        seg_diag["n_accepted"], seg_diag["n_candidate_valleys"],
        seg_diag["n_rejected_duration"], seg_diag["n_rejected_depth"],
    )

    if not segments:
        raise RuntimeError(
            "No valid reps detected. Check video quality and REP_PROMINENCE."
        )

    # ── Per-rep features ──────────────────────────────────────────────────
    all_rep_feats = []
    rows          = []
    all_trajs     = []

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
            "view_dep_reliable": meta.get("camera_view", "unknown") == "side",
            **quality, **static, **temporal,
        })

    # ── Session features ──────────────────────────────────────────────────
    session_feats = _extract_session(all_rep_feats)
    for row in rows:
        row.update(session_feats)

    # ── Discretisation ────────────────────────────────────────────────────
    for row in rows:
        row.update(_discretize(row))

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().all(axis=1).any():
        warnings.warn("One or more reps have all-NaN features — check tracking quality.")

    df.to_csv(out_dir / "features.csv", index=False)

    if save_trajectories and all_trajs:
        npz_data = {
            key: np.stack([t[key] for t in all_trajs], axis=0)
            for key in all_trajs[0]
        }
        np.savez(out_dir / "trajectories.npz", **npz_data)

    features_meta = {
        "session_id":       session_id,
        "fps":              fps,
        "n_frames_per_rep": N_FRAMES,
        "feature_version":  "1.0",
        **seg_diag, **session_feats,
    }
    with open(out_dir / "features_meta.json", "w") as f:
        json.dump(features_meta, f, indent=2)

    logger.info("  %d reps × %d columns → %s",
                len(df), len(df.columns), out_dir / "features.csv")
    return df