"""
test_integration.py
====================
Verifies the entire cleaned codebase for consistency, import integrity,
function signatures, config alignment, and data flow contracts.

Run with:
    python test_integration.py

No video files, trained models, or external data needed.
"""

import importlib
import inspect
import sys
import traceback
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}")
        if detail:
            print(f"    → {detail}")


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# =============================================================================
# 1. Import chain — every module loads without error
# =============================================================================

section("1. Import chain")

MODULES = [
    "squat_analysis",
    "squat_analysis.config",
    "squat_analysis.utils",
    "squat_analysis.extraction",
    "squat_analysis.preprocessing",
    "squat_analysis.features",
    "squat_analysis.pipeline",
    "squat_analysis.inference",
    "squat_analysis.mining",
    "squat_analysis.mining.preparation",
    "squat_analysis.mining.classification",
    "squat_analysis.mining.anomaly",
    "squat_analysis.mining.association",
    "squat_analysis.mining.clustering",
    "squat_analysis.mining.sequential",
]

imported = {}
for mod_name in MODULES:
    try:
        mod = importlib.import_module(mod_name)
        imported[mod_name] = mod
        check(f"import {mod_name}", True)
    except Exception as e:
        check(f"import {mod_name}", False, str(e))


# =============================================================================
# 2. Config — all constants exist and have correct types
# =============================================================================

section("2. Config constants")

if "squat_analysis.config" in imported:
    cfg = imported["squat_analysis.config"]

    # Paths
    for attr in ["ROOT_DIR", "DATA_DIR", "RAW_VIDEO_DIR", "PROCESSED_DIR", "OUTPUTS_DIR"]:
        check(f"config.{attr} exists", hasattr(cfg, attr))
        if hasattr(cfg, attr):
            check(f"config.{attr} is Path", isinstance(getattr(cfg, attr), Path))

    # MediaPipe
    check("config.MODEL_URL is str", isinstance(cfg.MODEL_URL, str))
    check("config.MODEL_PATH is Path", isinstance(cfg.MODEL_PATH, Path))
    check("config.N_LANDMARKS == 33", cfg.N_LANDMARKS == 33)

    # Landmark indices
    for attr in ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE",
                  "L_SHOULDER", "R_SHOULDER", "L_FOOT_INDEX", "R_FOOT_INDEX"]:
        check(f"config.{attr} is int", isinstance(getattr(cfg, attr, None), int))

    check("SQUAT_LANDMARKS has 12 entries", len(cfg.SQUAT_LANDMARKS) == 12)

    # Stage params
    check("N_FRAMES > 0", cfg.N_FRAMES > 0)
    check("SG_WINDOW is odd", cfg.SG_WINDOW % 2 == 1)
    check("SG_POLY < SG_WINDOW", cfg.SG_POLY < cfg.SG_WINDOW)

    # Mining
    check("PCA_FEATURES is list", isinstance(cfg.PCA_FEATURES, list))
    check("PCA_FEATURES has 15 items", len(cfg.PCA_FEATURES) == 15)
    check("PIPELINE_FLAGS has 6 items", len(cfg.PIPELINE_FLAGS) == 6)
    check("EXTRA_FLAG_DEFS has 4 items", len(cfg.EXTRA_FLAG_DEFS) == 4)
    check("ALL_FLAGS == PIPELINE + EXTRA", cfg.ALL_FLAGS == cfg.PIPELINE_FLAGS + list(cfg.EXTRA_FLAG_DEFS.keys()))
    check("DISCRETIZATION_THRESHOLDS has 6 items", len(cfg.DISCRETIZATION_THRESHOLDS) == 6)

    # Feature units covers all PCA features
    for feat in cfg.PCA_FEATURES:
        check(f"FEATURE_UNITS has '{feat}'", feat in cfg.FEATURE_UNITS,
              f"Missing from FEATURE_UNITS")


# =============================================================================
# 3. Utils — function signatures and basic computation
# =============================================================================

section("3. Utils")

if "squat_analysis.utils" in imported:
    import numpy as np
    utils = imported["squat_analysis.utils"]

    # Function existence
    for fn in ["angle_between", "angle_to_vertical", "midpoint", "unit_vec"]:
        check(f"utils.{fn} exists", hasattr(utils, fn))

    # Basic computation checks
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])

    angle = utils.angle_between(a, b, c)
    check("angle_between([1,0],[0,0],[0,1]) ≈ 90°", abs(angle - 90.0) < 0.01,
          f"Got {angle}")

    vert = utils.angle_to_vertical(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    check("angle_to_vertical vertical = 0°", abs(vert) < 0.01, f"Got {vert}")

    mid = utils.midpoint(np.array([0.0, 0.0]), np.array([2.0, 4.0]))
    check("midpoint([0,0],[2,4]) = [1,2]", np.allclose(mid, [1.0, 2.0]),
          f"Got {mid}")

    uv = utils.unit_vec(np.array([0.0, 0.0]), np.array([3.0, 0.0]))
    check("unit_vec([0,0],[3,0]) = [1,0]", np.allclose(uv, [1.0, 0.0]),
          f"Got {uv}")

    # Degenerate inputs
    check("angle_between degenerate = 0", utils.angle_between(a, a, a) == 0.0)
    check("unit_vec degenerate = [0,0]", np.allclose(utils.unit_vec(a, a), [0, 0]))


# =============================================================================
# 4. Function signature consistency — callers match callees
# =============================================================================

section("4. Function signatures")

def get_params(module_name, func_name):
    """Get parameter names for a function."""
    mod = imported.get(module_name)
    if not mod:
        return None
    fn = getattr(mod, func_name, None)
    if not fn:
        return None
    sig = inspect.signature(fn)
    return list(sig.parameters.keys())

# extraction.extract
params = get_params("squat_analysis.extraction", "extract")
if params:
    check("extract() has video_path param", "video_path" in params)
    check("extract() has session_id param", "session_id" in params)
    check("extract() has output_dir param", "output_dir" in params)

# extraction.load_extraction
params = get_params("squat_analysis.extraction", "load_extraction")
if params:
    check("load_extraction() has session_dir param", "session_dir" in params)

# preprocessing.preprocess
params = get_params("squat_analysis.preprocessing", "preprocess")
if params:
    check("preprocess() has session_dir param", "session_dir" in params)

# preprocessing.load_preprocessing
params = get_params("squat_analysis.preprocessing", "load_preprocessing")
if params:
    check("load_preprocessing() has session_dir param", "session_dir" in params)

# features.extract_features
params = get_params("squat_analysis.features", "extract_features")
if params:
    check("extract_features() has session_dir param", "session_dir" in params)
    check("extract_features() has save_trajectories param", "save_trajectories" in params)

# pipeline.run_pipeline
params = get_params("squat_analysis.pipeline", "run_pipeline")
if params:
    check("run_pipeline() has video_path", "video_path" in params)
    check("run_pipeline() has session_id", "session_id" in params)
    check("run_pipeline() has max_frames", "max_frames" in params)
    check("run_pipeline() has save_trajectories", "save_trajectories" in params)

# inference.SquatScorer
if "squat_analysis.inference" in imported:
    inf = imported["squat_analysis.inference"]
    check("SquatScorer class exists", hasattr(inf, "SquatScorer"))

    scorer_cls = inf.SquatScorer
    check("SquatScorer.__init__ takes model_dir",
          "model_dir" in inspect.signature(scorer_cls.__init__).parameters)
    check("SquatScorer.score_video exists", hasattr(scorer_cls, "score_video"))
    check("SquatScorer.score_features exists", hasattr(scorer_cls, "score_features"))

    sv_params = list(inspect.signature(scorer_cls.score_video).parameters.keys())
    check("score_video() has video_path", "video_path" in sv_params)

# mining.preparation
params = get_params("squat_analysis.mining.preparation", "load_and_validate")
if params:
    check("load_and_validate() has dataset_path", "dataset_path" in params)

params = get_params("squat_analysis.mining.preparation", "add_extra_flags")
if params:
    check("add_extra_flags() has df", "df" in params)

params = get_params("squat_analysis.mining.preparation", "prepare_features")
if params:
    check("prepare_features() has df + output_dir", "df" in params and "output_dir" in params)

# mining.classification
params = get_params("squat_analysis.mining.classification", "assign_pseudo_labels")
if params:
    check("assign_pseudo_labels() has df", "df" in params)

params = get_params("squat_analysis.mining.classification", "run_classification")
if params:
    check("run_classification() has df + feat_cols + output_dir",
          all(p in params for p in ["df", "feat_cols", "output_dir"]))

# mining.anomaly
params = get_params("squat_analysis.mining.anomaly", "run_anomaly")
if params:
    check("run_anomaly() has df + X_scaled + feat_cols + output_dir",
          all(p in params for p in ["df", "X_scaled", "feat_cols", "output_dir"]))

# mining.association
params = get_params("squat_analysis.mining.association", "run_association")
if params:
    check("run_association() has df + flag_cols + output_dir",
          all(p in params for p in ["df", "flag_cols", "output_dir"]))

# mining.clustering
params = get_params("squat_analysis.mining.clustering", "run_clustering")
if params:
    check("run_clustering() has df + output_dir", "df" in params and "output_dir" in params)

# mining.sequential
params = get_params("squat_analysis.mining.sequential", "run_sequential")
if params:
    check("run_sequential() has df + flag_cols + output_dir",
          all(p in params for p in ["df", "flag_cols", "output_dir"]))


# =============================================================================
# 5. Cross-module data contracts
# =============================================================================

section("5. Data contracts")

if "squat_analysis.config" in imported:
    cfg = imported["squat_analysis.config"]

    # PCA_FEATURES should not contain any flag columns
    all_flag_names = set(cfg.PIPELINE_FLAGS) | set(cfg.EXTRA_FLAG_DEFS.keys())
    overlap = set(cfg.PCA_FEATURES) & all_flag_names
    check("PCA_FEATURES has no flag columns", len(overlap) == 0,
          f"Overlap: {overlap}")

    # DISCRETIZATION_THRESHOLDS flag names should match PIPELINE_FLAGS
    disc_flags = {v[2] for v in cfg.DISCRETIZATION_THRESHOLDS.values()}
    pipeline_set = set(cfg.PIPELINE_FLAGS)
    check("DISCRETIZATION flags ⊆ PIPELINE_FLAGS", disc_flags <= pipeline_set,
          f"Extra: {disc_flags - pipeline_set}")

    # ARM_EXCLUDE_FLAGS should be subset of ALL_FLAGS
    check("ARM_EXCLUDE_FLAGS ⊆ ALL_FLAGS",
          set(cfg.ARM_EXCLUDE_FLAGS) <= set(cfg.ALL_FLAGS))

    # EXTRA_FLAG_DEFS directions are valid
    for name, (col, direction, thr) in cfg.EXTRA_FLAG_DEFS.items():
        check(f"EXTRA_FLAG '{name}' direction is gt/lt", direction in ("gt", "lt"))

    # DISCRETIZATION_THRESHOLDS directions are valid
    for feat, (thr, direction, flag) in cfg.DISCRETIZATION_THRESHOLDS.items():
        check(f"DISC '{flag}' direction is gt/lt", direction in ("gt", "lt"))


# =============================================================================
# 6. Inference module — label consistency with classification
# =============================================================================

section("6. Inference ↔ Classification consistency")

if ("squat_analysis.inference" in imported and
        "squat_analysis.mining.classification" in imported):
    import numpy as np

    inf = imported["squat_analysis.inference"]
    clf = imported["squat_analysis.mining.classification"]

    # Both should use the same label thresholds
    # Test with known values
    class FakeRow:
        def __init__(self, trunk, knee):
            self._d = {"trunk_lean_max": trunk, "knee_flexion_at_bottom": knee}
        def get(self, key, default=None):
            return self._d.get(key, default)

    # high_risk: trunk > 45
    label_inf = inf.SquatScorer._assign_label(FakeRow(50.0, 60.0))
    check("Inference: trunk=50 → high_risk", label_inf == "high_risk", f"Got {label_inf}")

    # low_risk: knee < 80 AND trunk < 20
    label_inf = inf.SquatScorer._assign_label(FakeRow(15.0, 70.0))
    check("Inference: trunk=15,knee=70 → low_risk", label_inf == "low_risk", f"Got {label_inf}")

    # medium_risk: everything else
    label_inf = inf.SquatScorer._assign_label(FakeRow(30.0, 100.0))
    check("Inference: trunk=30,knee=100 → medium_risk", label_inf == "medium_risk", f"Got {label_inf}")

    # LABEL_FEATURES in classification.py should match what inference excludes
    check("LABEL_FEATURES defined in classification",
          hasattr(clf, "LABEL_FEATURES") and len(clf.LABEL_FEATURES) == 4)


# =============================================================================
# 7. No circular imports
# =============================================================================

section("7. Circular import check")

# If we got here with all imports passing, there are no circular imports
all_imported = all(m in imported for m in MODULES)
check("No circular imports (all modules loaded)", all_imported)


# =============================================================================
# 8. Logging consistency — no print() in library modules
# =============================================================================

section("8. Logging consistency")

import ast

LIBRARY_FILES = [
    "squat_analysis/config.py",
    "squat_analysis/utils.py",
    "squat_analysis/extraction.py",
    "squat_analysis/preprocessing.py",
    "squat_analysis/features.py",
    "squat_analysis/pipeline.py",
    "squat_analysis/inference.py",
    "squat_analysis/mining/preparation.py",
    "squat_analysis/mining/classification.py",
    "squat_analysis/mining/anomaly.py",
    "squat_analysis/mining/association.py",
    "squat_analysis/mining/clustering.py",
    "squat_analysis/mining/sequential.py",
]

for filepath in LIBRARY_FILES:
    p = Path(filepath)
    if not p.exists():
        check(f"{filepath} — no print()", False, "file not found")
        continue
    source = p.read_text()
    try:
        tree = ast.parse(source)
        prints = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        check(f"{filepath} — no print()", len(prints) == 0,
              f"Found {len(prints)} print() calls")
    except SyntaxError as e:
        check(f"{filepath} — valid syntax", False, str(e))


# =============================================================================
# 9. CLI scripts have __main__ guard
# =============================================================================

section("9. CLI entry points")

CLI_SCRIPTS = ["run.py", "batch_process.py", "run_mining.py"]
for script in CLI_SCRIPTS:
    p = Path(script)
    if p.exists():
        source = p.read_text()
        check(f"{script} has if __name__ == '__main__'",
              'if __name__' in source)
        check(f"{script} has main()",
              "def main()" in source)
    else:
        check(f"{script} exists", False, "file not found")


# =============================================================================
# 10. Features module — bilateral_avg helper
# =============================================================================

section("10. Features module internals")

if "squat_analysis.features" in imported:
    feat = imported["squat_analysis.features"]
    import numpy as np

    # _bilateral_avg should exist and work
    check("_bilateral_avg exists", hasattr(feat, "_bilateral_avg"))
    if hasattr(feat, "_bilateral_avg"):
        l = np.array([1.0, np.nan, 3.0])
        r = np.array([2.0, 4.0, np.nan])
        avg = feat._bilateral_avg(l, r)
        check("_bilateral_avg([1,NaN,3],[2,4,NaN]) = [1.5,4,3]",
              np.allclose(avg, [1.5, 4.0, 3.0], equal_nan=True),
              f"Got {avg}")

    # _safe_angle and _safe_vertical should handle NaN
    check("_safe_angle exists", hasattr(feat, "_safe_angle"))
    check("_safe_vertical exists", hasattr(feat, "_safe_vertical"))
    if hasattr(feat, "_safe_angle"):
        result = feat._safe_angle(
            np.array([np.nan, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 1.0]),
        )
        check("_safe_angle with NaN input → NaN", np.isnan(result))


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'='*60}")
print(f"  RESULTS:  {PASS} passed  /  {FAIL} failed  /  {PASS+FAIL} total")
print(f"{'='*60}")

if FAIL > 0:
    print(f"\n  ⚠ {FAIL} test(s) failed — review above.")
    sys.exit(1)
else:
    print(f"\n  All checks passed. Codebase is consistent.")
    sys.exit(0)
