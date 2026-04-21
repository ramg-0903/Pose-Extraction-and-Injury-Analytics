# Automated Exercise Form Assessment & Injury Risk Prediction

An end-to-end pipeline that takes raw squat videos as input and outputs injury risk predictions using data mining techniques on pose-derived features.

**Primary result:** 88.7% binary accuracy (risky vs safe) using a Random Forest classifier on 11 kinematic features — exceeding the 85% target. Trunk lean features are excluded from the primary model because they define the pseudo-labels.

Built for IE 7275 Data Mining in Engineering, Spring 2026, Northeastern University.

## Pipeline overview

```
MP4 squat video
      │
      ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  Stage 1     │    │  Stage 2          │    │  Stage 3            │    │  Stage 4              │
│  Extraction  │───▶│  Preprocessing    │───▶│  Feature Engineering│───▶│  Mining &             │
│              │    │                   │    │                     │    │  Classification       │
│  MediaPipe   │    │  NaN fill, Y-flip │    │  Rep segmentation   │    │  PCA anomaly,         │
│  33 landmarks│    │  scale, smooth    │    │  15 biomechanical   │    │  FP-Growth ARM,       │
│  (x, y, vis) │    │  jolt detection   │    │  features + 6 flags │    │  Random Forest        │
└─────────────┘    └──────────────────┘    └────────────────────┘    └──────────────────────┘
                                                                              │
                                                                              ▼
                                                                    Risk prediction + form errors
```

## Repository structure

```
├── run.py                          CLI — single video through Stages 1→3
├── batch_process.py                CLI — all videos → master_dataset.csv
├── run_mining.py                   CLI — Stage 4 mining on master_dataset.csv
├── generate_report_figures.py      Generate all report figures from real data
│
├── squat_analysis/                 Core package
│   ├── config.py                   All constants, thresholds, paths
│   ├── utils.py                    Geometry helpers (angles, midpoints)
│   ├── extraction.py               Stage 1 — MediaPipe landmark extraction
│   ├── preprocessing.py            Stage 2 — NaN fill, scale, smooth, confidence
│   ├── features.py                 Stage 3 — rep segmentation + 15 features
│   ├── pipeline.py                 Chains Stages 1→3 for a single video
│   ├── inference.py                SquatScorer — real-time scoring with trained models
│   └── mining/
│       ├── preparation.py          Data loading, validation, scaling
│       ├── classification.py       Pseudo-labelling + RF training + CV
│       ├── anomaly.py              PCA reconstruction error scoring
│       ├── association.py          FP-Growth association rule mining
│       ├── clustering.py           DTW k-means (stub — future work)
│       └── sequential.py           Sequential pattern mining (stub — future work)
│
├── data/
│   ├── raw_videos/                 1,525 squat MP4s (not in repo)
│   └── processed/                  Per-session .npy files + master_dataset.csv
│
├── outputs/                        Mining results and trained models
│   ├── scaler.pkl                  Fitted RobustScaler
│   ├── pca_model.pkl               Fitted PCA (7 components, 95.9% variance)
│   ├── enriched_dataset.csv        2,428 reps with risk labels + anomaly scores
│   ├── feature_importance.csv      RF importance across 10 seeds
│   ├── classification_results.json CV metrics for all model variants
│   ├── association_rules.csv       7 FP-Growth rules (6 stable)
│   ├── anomaly_scores.csv          Per-rep reconstruction error
│   ├── anomaly_meta.json           PCA parameters and error distribution
│   └── figures/                    Generated report figures (13 PNGs)
│
├── tests/
│   └── test_integration.py         141 checks — imports, signatures, contracts
│
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Quick start

### Setup

```bash
git clone <repo-url>
cd Pose-Extraction-and-Injury-Analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Process a single video

```bash
python run.py --video data/raw_videos/my_squat.mp4
```

This runs Stages 1→3 and saves features to `data/processed/<session_id>/features.csv`.

### Batch process all videos

```bash
python batch_process.py
```

Processes every MP4 in `data/raw_videos/`, applies quality filters, and produces `data/processed/master_dataset.csv`.

### Run mining (Stage 4)

```bash
python run_mining.py
```

Runs PCA anomaly detection, association rule mining, pseudo-labelling, and Random Forest classification on `master_dataset.csv`. Saves all outputs to `outputs/`.

### Generate report figures

```bash
python generate_report_figures.py
```

Loads real pipeline outputs and generates 13 publication-ready figures to `outputs/figures/`.

### Run tests

```bash
python tests/test_integration.py
```

141 checks covering import chains, function signatures, config contracts, data flow alignment, label consistency, and logging standards.

## Real-time inference

The `SquatScorer` class loads pre-trained artifacts once and scores videos on demand:

```python
from squat_analysis.inference import SquatScorer

scorer = SquatScorer("outputs/")
result = scorer.score_video("path/to/squat.mp4")

print(result["n_reps"])                    # number of reps detected
print(result["reps"][0]["risk_label"])      # "low_risk" / "medium_risk" / "high_risk"
print(result["reps"][0]["form_errors"])     # ["rushed_descent", "asymmetric_depth", ...]
print(result["summary"]["risk_distribution"])
```

## Key results

| Metric | Value |
|--------|-------|
| Binary accuracy (risky vs safe) | **88.7%** |
| 3-class accuracy (honest, kinematic only) | **85.1%** |
| Estimated AUC | **~0.93** |
| Top predictor | `trunk_lean_max` (importance 0.346) |
| Strongest association rule | asymmetric_start + unstable_return → asymmetric_depth (lift 7.64, 100% confidence) |
| PCA components | 7 (95.9% variance explained) |
| Anomalous reps flagged | 243 / 2,428 (10.0%) |

## Dataset

- **Source:** [Pose Estimation for Powerlifting](https://github.com/03y/pose-estimation-for-powerlifting)
- **Size:** 1,525 squat videos → 2,428 valid reps after quality filtering
- **Landmark model:** MediaPipe PoseLandmarker Heavy (float16), VIDEO mode
- **Feature space:** 15 continuous features (8 static angles + 7 temporal) + 6 binary form-error flags

## Design decisions

**Z-coordinate dropped.** Monocular depth from a single camera is unreliable and actively misleading for side-view squats. The entire pipeline operates in 2D.

**RobustScaler over StandardScaler.** Jerk and velocity features have extreme outliers (~72,000 deg/s³) that inflate standard deviation and compress the rest of the feature space. RobustScaler uses the interquartile range, which is resistant to heavy tails.

**Trunk lean excluded from primary model.** Pseudo-labels are defined by `trunk_lean_max > 45°`. Including trunk lean as a predictor lets the model recover the labelling rule, inflating accuracy to 99.6%. The primary model uses only the 11 kinematic features not involved in labelling.

**Valley detection with peak fallback.** Rep segmentation uses valley detection on the knee-Y signal (largest range in side-view). When videos start or end mid-squat, a peak-detection fallback identifies standing positions and builds reps around them.

**Bootstrap stability for ARM.** Association rules are validated across 5 random 80% subsamples. Of 7 rules discovered, 6 are stable (appear in ≥3/5 bootstrap runs).

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- scikit-learn, pandas, numpy, scipy
- mlxtend (for FP-Growth association rules)
- matplotlib, seaborn (for figure generation)

## Authors

- **Alekhya Bulusu** — bulusu.a@northeastern.edu
- **Ramgopal Reddy Alumalla** — alumalla.r@northeastern.edu

Group 20, IE 7275 Data Mining in Engineering, Spring 2026, Northeastern University.
