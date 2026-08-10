# Drowsiness-Detection

Driver drowsiness classification from webcam/video using MediaPipe face-mesh features (EAR, MAR,
head pose) and classical ML models. Supports UTA-RLDD and NTHU-DDD. See `METHODS.md` for the
methodology.

## Setup

Requires Python 3.10-3.12 (3.13 breaks the pinned `mediapipe`/`numpy` wheels).

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## 1. Build the dataset

**UTA-RLDD**: `UTA-RLDD/<subject_id>/{0,5,10}.<ext>` (0 = Alert, 5 = Low Vigilant, 10 = Drowsy).

**NTHU-DDD**: `NTHU/<subject_id>/<category>/<behavior>.<ext>` with a `*_drowsiness.txt` per-frame
label file alongside each video.

See `structure.txt` in each dataset folder for the exact layout.

```bash
python dataset_builder.py --dataset uta_rldd
python dataset_builder.py --dataset nthu
```

Writes `outputs/<dataset>_{raw_landmarks,frame_level,window}.csv` and `_baselines.json`, then
generates feature-progression plots (`outputs/plots/`) and a few annotated debug clips
(`outputs/annotated/`). Resumable — re-run the same command to pick up where it left off.

To re-slice windows at a different `WINDOW_SEC`/`STRIDE_SEC` without re-running MediaPipe:

```bash
python dataset_builder.py --dataset nthu --rewindow
```

## 2. Train models

```bash
python model.py train --window-csv outputs/nthu_window.csv
```

Subject-level cross-validated grid search over all 8 models. Saves `outputs/model_results.json`,
`outputs/best_model.pkl` (single best model), `outputs/all_models.pkl` (every model refit on the
full dataset, for `--ensemble`), `outputs/feature_scaler.pkl`, and a comparison plot in
`outputs/plots/`. Resumable with `--resume` — but if `model.py`'s scoring logic has changed since
the last run, do one fresh (non-`--resume`) run first so the checkpoint isn't stale.

Feature-group ablation:

```bash
python model.py ablation --window-csv outputs/nthu_window.csv --model random_forest
```

Per-feature permutation importance (individual features, not whole groups):

```bash
python model.py feature-importance --window-csv outputs/nthu_window.csv
```

## 3. Run live inference

```bash
python model.py live --camera-id 0
```

Press `c` to calibrate, `q` to quit. Only reports "Drowsy" above `LIVE_DROWSY_CONFIDENCE_THRESHOLD`
(default 70%); otherwise shows the next most likely state. Add `--ensemble` to soft-vote across the
top `ENSEMBLE_TOP_N` models (by CV f1) from `outputs/all_models.pkl` instead of using a single model.

Calibration also picks the driver (rightmost + forward-most face, for a cabin-facing camera) and
fixes a crop around their head position for the rest of the session — see METHODS.md for details
and its caveats (not yet tested against a real vehicle camera).

## 4. Deployment (`drowsiness_live.py`)

Same live inference, but a single self-contained file with no dependency on `dataset_builder.py`
or `model.py` — for copying onto a deployment device (e.g. a Raspberry Pi) without the training
code.

```bash
python drowsiness_live.py --camera-id 0
```

Same flags and behavior as `python model.py live` (`--model-path`, `--ensemble`, `--ensemble-path`,
`--ensemble-top-n`).

**Minimum files needed on the deployment device:**
- `drowsiness_live.py`
- `outputs/best_model.pkl` (single model), or `outputs/all_models.pkl` too if using `--ensemble`
- `models/face_landmarker.task` (or let it auto-download on first run, if the device has internet)

Python packages: `opencv-python`, `numpy`, `pandas`, `joblib`, `mediapipe`, `scipy`, `scikit-learn`
(needed to unpickle the saved model even though it's never imported directly) — plus `xgboost`
and/or `lightgbm` if the deployed model (or any `--ensemble` member) uses one.
