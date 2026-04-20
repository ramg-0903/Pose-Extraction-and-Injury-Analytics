"""
Central configuration for the squat-analysis pipeline.

All constants, thresholds, and paths live here — change a value
once and it propagates everywhere.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR   = ROOT_DIR / "outputs"

# ── MediaPipe Model ───────────────────────────────────────────────────────────

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "pose_landmarker_heavy.task"
MODEL_NAME = "pose_landmarker_heavy"

# ── Stage 1 — Extraction ─────────────────────────────────────────────────────

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.6
MIN_PRESENCE_CONFIDENCE  = 0.7

# Landmarks below this visibility → NaN (raw visibility value still stored)
VISIBILITY_NAN_THRESHOLD    = 0.65
# Frames where fewer than this fraction of squat landmarks are visible
MIN_FRAME_DETECTION_QUALITY = 0.5

# ── MediaPipe Landmark Indices ────────────────────────────────────────────────

L_SHOULDER   = 11;  R_SHOULDER   = 12
L_HIP        = 23;  R_HIP        = 24
L_KNEE       = 25;  R_KNEE       = 26
L_ANKLE      = 27;  R_ANKLE      = 28
L_HEEL       = 29;  R_HEEL       = 30
L_FOOT_INDEX = 31;  R_FOOT_INDEX = 32

N_LANDMARKS = 33

# Key landmarks for detection-quality scoring
SQUAT_LANDMARKS = [
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
]

# ── Stage 2 — Preprocessing ──────────────────────────────────────────────────

SG_WINDOW = 7       # Savitzky-Golay window (must be odd)
SG_POLY   = 2       # Polynomial order

# Torso deviation from standing baseline beyond this → low-confidence flag
TORSO_ALIGNMENT_THRESHOLD = 30.0  # degrees

# Jerk threshold for camera-jolt detection (body-units / frame³).
# Lower = more sensitive.  Calibrated against side-view powerlifting footage.
JERK_THRESHOLD = 0.08

# ── Stage 3 — Feature Extraction & Rep Segmentation ─────────────────────────

N_FRAMES           = 20    # All reps resampled to this length
REP_PROMINENCE     = 0.08  # Min valley/peak depth (normalised units)
REP_DISTANCE       = 15    # Min frames between rep bottoms
MIN_REP_FRAMES     = 10    # Shortest valid rep  (~0.3 s @ 30 fps)
MAX_REP_FRAMES     = 250   # Longest valid rep   (~8.3 s @ 30 fps)
MIN_DEPTH_FRACTION = 0.60  # Partial-rep rejection threshold

# ── Stage 4 — Mining ─────────────────────────────────────────────────────────

RANDOM_SEED            = 42
DTW_MAX_K              = 6
DTW_N_INIT             = 10
PCA_VARIANCE_THRESHOLD = 0.95
PCA_ANOMALY_SIGMA      = 2.0   # mean + Nσ → high-risk flag

# ARM thresholds kept low to capture rare but important flags
# (e.g. excessive_lean fires on only ~4 % of reps)
ARM_MIN_SUPPORT    = 0.03
ARM_MIN_CONFIDENCE = 0.60
ARM_MIN_LIFT       = 1.05

# Flags too prevalent for informative rules (shallow_squat ≈ 86 %)
ARM_EXCLUDE_FLAGS = ["shallow_squat", "ankle_restricted", "butt_wink"]

RF_N_ESTIMATORS = 200
RF_CV_FOLDS     = 5
RF_N_SEEDS      = 10   # Runs for feature-importance stability

# ── Flag Definitions ──────────────────────────────────────────────────────────

# 6 pipeline flags (written by Stage 3 into features.csv)
PIPELINE_FLAGS = [
    "shallow_squat", "excessive_lean", "asymmetric_depth",
    "ankle_restricted", "rushed_descent", "butt_wink",
]

# 4 extra flags computed at mining time from continuous features.
# Format: flag_name → (feature_col, direction, threshold_or_method)
#   threshold_or_method: float = fixed cutoff, "p75" = 75th percentile
EXTRA_FLAG_DEFS = {
    "high_jerk":        ("knee_jerk_rms",         "gt", "p75"),
    "incomplete_depth": ("knee_flexion_at_bottom", "gt", 120.0),
    "unstable_return":  ("symmetry_knee_at_end",   "gt", 15.0),
    "asymmetric_start": ("symmetry_knee_at_start", "gt", 10.0),
}

ALL_FLAGS = PIPELINE_FLAGS + list(EXTRA_FLAG_DEFS.keys())

# ── Continuous Feature Sets ───────────────────────────────────────────────────

# Features for PCA and classification.
# Excludes view-dependent (hip/ankle symmetry), quality metrics, session
# features, frame indices, binary flags, and identity columns.
PCA_FEATURES = [
    # Knee depth (averaged bilateral — L/R individual too sparse in side-view)
    "knee_flexion_at_bottom",
    "knee_flexion_range",
    # Trunk lean — most reliable signal across all camera angles
    "trunk_lean_at_bottom",
    "trunk_lean_mean",
    "trunk_lean_max",
    "trunk_lean_range",
    "trunk_lean_at_end",
    # Temporal — timing and smoothness
    "descent_ascent_ratio",
    "descent_time_s",
    "ascent_time_s",
    "knee_vel_max_descent",
    "knee_vel_max_ascent",
    "knee_jerk_max",
    "knee_jerk_rms",
    "normalized_jerk_cost",
]

# Subset used for DTW trajectory clustering
DTW_FEATURES = [
    "knee_flexion_at_bottom",
    "trunk_lean_at_bottom",
    "symmetry_knee_at_bottom",
    "descent_ascent_ratio",
    "knee_jerk_rms",
]

# ── Discretisation Thresholds ─────────────────────────────────────────────────
# Maps (feature_col) → (threshold, direction, flag_name).
# Literature sources are documented in the project report (Section 3.4).

DISCRETIZATION_THRESHOLDS = {
    "knee_flexion_at_bottom":       (90.0, "lt", "shallow_squat"),        # Myer 2014
    "trunk_lean_max":               (45.0, "gt", "excessive_lean"),       # Schoenfeld 2010
    "symmetry_knee_at_bottom":      (10.0, "gt", "asymmetric_depth"),     # Mauntel 2017
    "ankle_dorsiflexion_at_bottom": (15.0, "lt", "ankle_restricted"),     # Rabin 2014
    "descent_ascent_ratio":         (0.5,  "lt", "rushed_descent"),       # General biomechanics
    "butt_wink_delta":              (15.0, "gt", "butt_wink"),            # McKean 2010
}

# ── Feature Units (for documentation / notebook use) ──────────────────────────

FEATURE_UNITS = {
    "knee_flexion_at_bottom":       "deg",
    "knee_flexion_L_at_bottom":     "deg",
    "knee_flexion_R_at_bottom":     "deg",
    "knee_flexion_range":           "deg",
    "trunk_lean_at_bottom":         "deg",
    "trunk_lean_mean":              "deg",
    "trunk_lean_max":               "deg",
    "trunk_lean_range":             "deg",
    "symmetry_knee_at_bottom":      "deg",
    "symmetry_knee_mean":           "deg",
    "hip_flexion_at_bottom":        "deg",
    "ankle_dorsiflexion_at_bottom": "deg",
    "descent_frames":               "frames",
    "ascent_frames":                "frames",
    "descent_time_s":               "s",
    "ascent_time_s":                "s",
    "rep_duration_s":               "s",
    "descent_ascent_ratio":         "ratio",
    "knee_vel_max_descent":         "deg/s",
    "knee_vel_max_ascent":          "deg/s",
    "knee_jerk_max":                "deg/s³",
    "knee_jerk_rms":                "deg/s³",
    "normalized_jerk_cost":         "ratio",
    "fatigue_trunk_lean_slope":     "deg/rep",
    "fatigue_knee_depth_slope":     "deg/rep",
    "depth_consistency_cv":         "ratio",
    "rep_mean_confidence":          "ratio",
    "rep_interp_fraction":          "ratio",
    "knee_flexion_at_start":        "deg",
    "trunk_lean_at_start":          "deg",
    "symmetry_knee_at_start":       "deg",
    "hip_flexion_at_start":         "deg",
    "ankle_dorsiflexion_at_start":  "deg",
    "knee_flexion_at_end":          "deg",
    "trunk_lean_at_end":            "deg",
    "symmetry_knee_at_end":         "deg",
    "hip_flexion_at_end":           "deg",
    "ankle_dorsiflexion_at_end":    "deg",
}