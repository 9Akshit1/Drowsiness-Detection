"""
Trains classifiers on the WINDOW dataset produced by dataset_builder.py, evaluates them,
and runs live webcam inference with the best saved model.
"""

import os
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

import dataset_builder as db

OUTPUT_DIR = "outputs"
FEATURE_COLUMNS = db.WINDOW_FEATURE_COLUMNS
LABEL_NAMES = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}
CALIBRATION_SEC = 8.0

# Outer subject-level fold count for cross-validated evaluation (UTA-RLDD: 60 subjects / 5 folds = 12/fold).
# NOTE: this is a deterministic GroupKFold split, not the original UTA-RLDD paper's official fold
# assignment (that per-subject list isn't available in this repo). If you obtain it, swap this out for
# a PredefinedSplit built from that mapping instead. Automatically clamps to fewer folds for datasets
# with fewer subjects (e.g. a partial NTHU build), so this stays dataset-agnostic.
OUTER_FOLDS = 5


# ---------------------------------------------------------------------------
# Step 1-2: subject-level cross-validation split + normalize
# ---------------------------------------------------------------------------
def outer_subject_folds(df, n_splits=OUTER_FOLDS):
    n_splits = min(n_splits, df["subject_id"].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df["subject_id"]))


def fit_scaler(train_df):
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


# ---------------------------------------------------------------------------
# Step 3-4: models + hyperparameter search
# ---------------------------------------------------------------------------
def build_model_grid():
    grid = {
        "logistic_regression": (LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1, 10]}),
        "decision_tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 8, None]}),
        "random_forest": (RandomForestClassifier(random_state=42),
                           {"n_estimators": [100, 300], "max_depth": [5, 10, None]}),
        "linear_svm": (SVC(kernel="linear", probability=True, random_state=42), {"C": [0.1, 1, 10]}),
    }
    if XGBClassifier is not None:
        grid["xgboost"] = (XGBClassifier(eval_metric="mlogloss", random_state=42),
                            {"n_estimators": [100, 300], "max_depth": [3, 5]})
    if LGBMClassifier is not None:
        grid["lightgbm"] = (LGBMClassifier(random_state=42),
                             {"n_estimators": [100, 300], "max_depth": [3, 5, -1]})
    return grid


# ---------------------------------------------------------------------------
# Step 5: evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    accuracy = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = None
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def _average_metrics(metrics_list):
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    averaged = {}
    for key in keys:
        values = [m[key] for m in metrics_list if m[key] is not None]
        averaged[key] = float(np.mean(values)) if values else None
    averaged["confusion_matrix"] = np.sum([m["confusion_matrix"] for m in metrics_list], axis=0).tolist()
    return averaged


def _feature_importance(model, feature_columns):
    if not hasattr(model, "feature_importances_"):
        return None
    return {name: float(score) for name, score in zip(feature_columns, model.feature_importances_)}


def train_and_evaluate(window_csv_path, output_dir=OUTPUT_DIR, n_splits=OUTER_FOLDS):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(window_csv_path)
    groups = df["subject_id"].values
    model_grid = build_model_grid()

    folds = outer_subject_folds(df, n_splits=n_splits)
    fold_results = {name: [] for name in model_grid}
    fold_importances = {name: [] for name in model_grid}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        scaler = fit_scaler(train_df)
        X_train = scaler.transform(train_df[FEATURE_COLUMNS])
        X_test = scaler.transform(test_df[FEATURE_COLUMNS])
        y_train, y_test = train_df["label"].values, test_df["label"].values
        inner_groups = train_df["subject_id"].values
        inner_cv = GroupKFold(n_splits=min(4, train_df["subject_id"].nunique()))

        for name, (estimator, param_grid) in model_grid.items():
            search = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train, groups=inner_groups)
            best = search.best_estimator_

            fold_results[name].append({
                "fold": fold_idx,
                "best_params": search.best_params_,
                "train": evaluate(best, X_train, y_train),
                "test": evaluate(best, X_test, y_test),
            })
            importance = _feature_importance(best, FEATURE_COLUMNS)
            if importance is not None:
                fold_importances[name].append(importance)

            print(f"[fold {fold_idx}] {name}: test f1={fold_results[name][-1]['test']['f1']:.3f}")

    results = {}
    for name, folds_for_model in fold_results.items():
        results[name] = {
            "folds": folds_for_model,
            "train_avg": _average_metrics([r["train"] for r in folds_for_model]),
            "test_avg": _average_metrics([r["test"] for r in folds_for_model]),
        }
        if fold_importances[name]:
            results[name]["feature_importance_avg"] = pd.DataFrame(fold_importances[name]).mean().to_dict()
        print(f"{name}: avg test f1={results[name]['test_avg']['f1']:.3f} "
              f"(over {len(folds_for_model)} folds)")

    with open(os.path.join(output_dir, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_name = max(results, key=lambda n: results[n]["test_avg"]["f1"])
    print(f"Best model: {best_name} (avg test f1={results[best_name]['test_avg']['f1']:.3f})")

    # Refit the winning model type on the full dataset for deployment (own scaler + hyperparameter search).
    final_scaler = fit_scaler(df)
    X_all = final_scaler.transform(df[FEATURE_COLUMNS])
    y_all = df["label"].values
    final_cv = GroupKFold(n_splits=min(n_splits, df["subject_id"].nunique()))
    estimator, param_grid = model_grid[best_name]
    final_search = GridSearchCV(estimator, param_grid, cv=final_cv, scoring="f1_macro", n_jobs=-1)
    final_search.fit(X_all, y_all, groups=groups)
    final_model = final_search.best_estimator_

    joblib.dump(final_scaler, os.path.join(output_dir, "feature_scaler.pkl"))
    joblib.dump({
        "model": final_model,
        "scaler": final_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_name,
        "best_params": final_search.best_params_,
    }, os.path.join(output_dir, "best_model.pkl"))

    return results, best_name


# ---------------------------------------------------------------------------
# Step 6: live run
# ---------------------------------------------------------------------------
def _draw_stats_panel(width, stats, classification, confidence):
    panel = np.zeros((160, width, 3), dtype=np.uint8)
    lines = [
        f"EAR: {stats['mean_ear']:.2f}   MAR: {stats['mean_mar']:.2f}   PERCLOS: {stats['perclos'] * 100:.0f}%",
        f"Blink Rate: {stats['blink_freq'] * 60:.0f}/min   Blink Count: {stats['blink_count']}   "
        f"Yawn Count: {stats['yawn_count']}",
        f"Pitch Velocity: {stats['avg_pitch_velocity']:.0f}deg/s   Nod Count: {stats['nod_count']}",
        f"Prediction: {classification}   Confidence: {confidence * 100:.1f}%",
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def _center_on_nose(pts):
    """Recenters landmarks on the nose tip without correcting for head rotation -- shows the head's
    actual current pose, as opposed to db.normalize_landmarks() which undoes rotation to face-forward."""
    nose = np.array(pts["nose_tip"])
    return {name: tuple(np.array(p) - nose) for name, p in pts.items()}


def _draw_points_panel(size, pts, label):
    panel = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if not pts:
        return panel
    cx, cy, scale = size // 2, size // 2, size / 400.0
    for x, y, _ in pts.values():
        px, py = int(cx + x * scale), int(cy - y * scale)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(panel, (px, py), 2, (0, 255, 0), -1)
    return panel


def run_live(model_path=os.path.join(OUTPUT_DIR, "best_model.pkl"), camera_id=0):
    bundle = joblib.load(model_path)
    model, scaler, feature_columns = bundle["model"], bundle["scaler"], bundle["feature_columns"]

    face_mesh = db._open_face_mesh()
    cap = cv2.VideoCapture(camera_id)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window_frames = int(db.WINDOW_SEC * fps)

    print("Press 'c' to calibrate (look forward, stay neutral), 'q' to quit.")

    baseline = None
    calibrating_until = None
    calib_rows = []
    rolling_buffer = deque(maxlen=window_frames)

    prev_pitch = prev_yaw = prev_roll = None
    classification, confidence, stats = "Calibrating...", 0.0, None
    last_window_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        norm_pts, raw_pts_centered = {}, {}
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            pts = {name: db._landmark_px(landmarks[idx], w, h) for name, idx in db.LANDMARK_IDS.items()}
            for x, y, _ in pts.values():
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            pose = db.estimate_head_pose(pts, w, h)
            if pose is not None:
                pitch, yaw, roll, rmat, _ = pose
                norm_pts = db.normalize_landmarks(pts, rmat)
                raw_pts_centered = _center_on_nose(pts)
                ear = (db.eye_aspect_ratio(pts, db.LEFT_EYE) + db.eye_aspect_ratio(pts, db.RIGHT_EYE)) / 2.0
                mar = db.mouth_aspect_ratio(pts)
                dt = 1.0 / fps
                pitch_vel = 0.0 if prev_pitch is None else (pitch - prev_pitch) / dt
                yaw_vel = 0.0 if prev_yaw is None else (yaw - prev_yaw) / dt
                roll_vel = 0.0 if prev_roll is None else (roll - prev_roll) / dt
                prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll

                row = {"subject_id": "live", "video": "live", "label": -1,
                       "frame": 0, "time_sec": 0.0,
                       "ear_raw": ear, "mar_raw": mar, "pitch": pitch, "yaw": yaw, "roll": roll,
                       "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel}
                rolling_buffer.append(row)
                if calibrating_until is not None:
                    calib_rows.append(row)

        if calibrating_until is not None and time.time() >= calibrating_until:
            if calib_rows:
                baseline = db.compute_baseline(pd.DataFrame(calib_rows))
                print("Calibration complete:", baseline)
            calibrating_until, calib_rows = None, []

        if baseline is not None and len(rolling_buffer) == window_frames and \
                time.time() - last_window_time >= STRIDE_SEC_LIVE:
            chunk_df = db.apply_baseline(pd.DataFrame(rolling_buffer), baseline)
            stats = db._summarize_window(chunk_df, fps)
            X = scaler.transform([[stats[c] for c in feature_columns]])
            pred = model.predict(X)[0]
            confidence = float(np.max(model.predict_proba(X)[0]))
            classification = LABEL_NAMES.get(pred, str(pred))
            last_window_time = time.time()

        raw_panel = _draw_points_panel(h, raw_pts_centered, "RAW 3D")
        frontal_panel = _draw_points_panel(h, norm_pts, "FRONTAL (normalized)")
        combined = np.hstack([frame, raw_panel, frontal_panel])
        if stats is not None:
            stats_panel = _draw_stats_panel(combined.shape[1], stats, classification, confidence)
            combined = np.vstack([combined, stats_panel])
        cv2.imshow("Driver Drowsiness Monitor", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrating_until = time.time() + CALIBRATION_SEC
            calib_rows = []
            print(f"Calibrating for {CALIBRATION_SEC:.0f}s, look forward and stay neutral...")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


STRIDE_SEC_LIVE = db.STRIDE_SEC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--window-csv", default=os.path.join(OUTPUT_DIR, "uta_rldd_window.csv"))
    train_p.add_argument("--folds", type=int, default=OUTER_FOLDS,
                          help="Number of subject-level outer CV folds (default: 5, per UTA-RLDD's 60 subjects)")

    live_p = sub.add_parser("live")
    live_p.add_argument("--camera-id", type=int, default=0)
    live_p.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))

    args = parser.parse_args()
    if args.command == "train":
        train_and_evaluate(args.window_csv, n_splits=args.folds)
    elif args.command == "live":
        run_live(args.model_path, args.camera_id)
