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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

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
LABEL_NAMES = db.LABEL_NAMES
ALL_LABELS = sorted(LABEL_NAMES)  # fixed label set so confusion matrices are always the same shape,
                                  # even if a given CV fold's split happens to be missing a class
                                  # (e.g. NTHU never produces label 1 -- see dataset_builder.py)

# --- run_live() config ---------------------------------------------------------------------
CALIBRATION_SEC = max(8.0, db.WINDOW_SEC)  # must be >= WINDOW_SEC or the buffer isn't full yet
STRIDE_SEC_LIVE = db.STRIDE_SEC
LIVE_CAPTURE_WIDTH, LIVE_CAPTURE_HEIGHT = 640, 480  # plenty for EAR/MAR/pose, cuts per-frame cost
# Cap MediaPipe calls/sec instead of running on every captured frame -- drowsiness doesn't change
# frame-to-frame, and this is the main lever on real-time cost (MediaPipe inference dominates).
TARGET_PROCESS_HZ = 10.0
LIVE_DROWSY_CONFIDENCE_THRESHOLD = 0.7  # only report Drowsy above this P(Drowsy); else best non-Drowsy guess
ENSEMBLE_TOP_N = 3  # soft-voting ensemble uses the top N models by CV test f1, not all 8 --
                     # weaker models (e.g. decision_tree) would just dilute the vote

FEATURE_GROUPS = {
    "eye_dynamics": ["mean_ear", "std_ear", "min_ear", "max_ear", "blink_count", "blink_freq",
                     "avg_blink_duration", "max_blink_duration", "perclos"],
    "mouth_yawn_dynamics": ["mean_mar", "max_mar", "std_mar", "yawn_count", "avg_yawn_duration"],
    "head_pose": ["mean_pitch", "std_pitch", "max_pitch_velocity", "avg_pitch_velocity",
                  "pitch_oscillation_freq", "nod_count"],
}

# Outer subject-level fold count for cross-validated evaluation (UTA-RLDD: 60 subjects / 5 folds = 12/fold).
# NOTE: this is a class-balance-aware StratifiedGroupKFold split (seeded, reproducible via
# RANDOM_SEED -- not the original UTA-RLDD paper's official fold assignment (that per-subject list
# isn't available in this repo). If you obtain it, swap this out for a PredefinedSplit built from
# that mapping instead. Automatically clamps to fewer folds for datasets with fewer subjects (e.g.
# a partial NTHU build), so this stays dataset-agnostic.
OUTER_FOLDS = 5
# Inner (hyperparameter-search) fold count, used inside each outer fold's training split. Kept
# smaller than OUTER_FOLDS purely as a compute-cost tradeoff (each inner fold multiplies grid
# search cost) -- not a literature-backed value.
INNER_FOLDS = 4
# GridSearchCV's model-selection metric: macro-F1 (unweighted mean of per-class F1), chosen so
# tuning doesn't implicitly favor the majority class -- important here since NTHU windows are
# label-imbalanced (see build results) and UTA-RLDD's "Low Vigilant" class is naturally rarer.
INNER_CV_SCORING = "f1_macro"
# Fixed seed for every model constructor below, for reproducible fold results -- not a modeling
# assumption, just a determinism choice.
RANDOM_SEED = 42
# Not -1 (all cores): crashes with TerminatedWorkerError on this machine once the grid got wide
# (12 cores but limited free RAM; n_jobs=-1 oversubscribes memory, not CPU). 4 is stable.
GRID_SEARCH_N_JOBS = 4


# ---------------------------------------------------------------------------
# Step 1-2: subject-level cross-validation split + normalize
# ---------------------------------------------------------------------------
def outer_subject_folds(df, n_splits=OUTER_FOLDS):
    """Subject-level folds, stratified by label so class balance stays close to the overall
    dataset across folds (plain GroupKFold left this to chance -- measured a 41.5%-60.3% Drowsy
    swing fold-to-fold before switching). shuffle=True + fixed seed: reproducible, not tied to
    row order."""
    n_subjects = df["subject_id"].nunique()
    if n_subjects < 2:
        raise ValueError(
            f"Need at least 2 distinct subjects for subject-level cross-validation, found {n_subjects}. "
            "Add more videos/subjects to this window CSV before training.")
    sgkf = StratifiedGroupKFold(n_splits=max(2, min(n_splits, n_subjects)), shuffle=True, random_state=RANDOM_SEED)
    return list(sgkf.split(df, y=df["label"], groups=df["subject_id"]))


def _fit_best(estimator, param_grid, X, y, groups, n_subjects):
    """Runs GridSearchCV when there are enough distinct subjects to cross-validate hyperparameters
    on; otherwise (a very small/partial test dataset) just fits the estimator with its defaults
    so small runs don't crash outright -- tuning is skipped, not the whole pipeline."""
    if n_subjects < 2:
        model = clone(estimator)
        model.fit(X, y)
        return model, {}
    inner_cv = StratifiedGroupKFold(n_splits=max(2, min(INNER_FOLDS, n_subjects)), shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring=INNER_CV_SCORING, n_jobs=GRID_SEARCH_N_JOBS)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def fit_scaler(train_df):
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


class DenseLabelClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a classifier that requires contiguous 0..k-1 class labels (XGBoost's sklearn API
    raises "Invalid classes inferred" otherwise) so it transparently works with a sparse label
    space -- confirmed to happen in practice: NTHU-only data only ever has labels {0, 2} (no
    "Low Vigilant" equivalent), which crashed XGBoost with exactly that error. LightGBM and the
    other sklearn models here handle sparse labels natively and don't need this."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.classes_ = np.unique(y)
        to_dense = {label: i for i, label in enumerate(self.classes_)}
        y_dense = np.array([to_dense[v] for v in y])
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_dense, **kwargs)
        return self

    def predict(self, X):
        return self.classes_[self.estimator_.predict(X)]

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


# ---------------------------------------------------------------------------
# Step 3-4: models + hyperparameter search
# ---------------------------------------------------------------------------
def build_model_grid():
    # class_weight="balanced" is offered as an option (not forced) for the ~43/57 imbalance;
    # GridSearchCV only uses it where it actually wins.
    grid = {
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
                                 {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}),
        "decision_tree": (DecisionTreeClassifier(random_state=RANDOM_SEED),
                           {"max_depth": [3, 5, 8, None], "class_weight": [None, "balanced"]}),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_SEED),
                           {"n_estimators": [100, 300], "max_depth": [5, 10, 20, None],
                            "class_weight": [None, "balanced"]}),
        # LinearSVC not kernel SVC -- scales to this row count, CalibratedClassifierCV adds
        # predict_proba back.
        "linear_svm": (
            CalibratedClassifierCV(LinearSVC(dual=False, max_iter=5000, random_state=RANDOM_SEED), cv=3),
            {"estimator__C": [0.1, 1, 10], "estimator__class_weight": [None, "balanced"]},
        ),
        "knn": (KNeighborsClassifier(), {"n_neighbors": [5, 15, 31, 51], "weights": ["uniform", "distance"]}),
        # Shallow = one hidden layer. early_stopping avoids wasting iterations past convergence.
        "shallow_nn": (
            MLPClassifier(max_iter=1000, early_stopping=True, random_state=RANDOM_SEED),
            {"hidden_layer_sizes": [(16,), (32,), (64,)], "alpha": [0.0001, 0.001, 0.01]},
        ),
    }
    if XGBClassifier is not None:
        grid["xgboost"] = (
            DenseLabelClassifier(XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_SEED, verbosity=0)),
            {"estimator__n_estimators": [100, 300], "estimator__max_depth": [3, 5],
             "estimator__learning_rate": [0.05, 0.1, 0.3]},
        )
    if LGBMClassifier is not None:
        # num_leaves paired with depth so it can't exceed what the tree depth supports.
        grid["lightgbm"] = (
            LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
            [
                {"n_estimators": [100, 300], "max_depth": [3], "num_leaves": [7], "learning_rate": [0.05, 0.1, 0.3]},
                {"n_estimators": [100, 300], "max_depth": [5], "num_leaves": [31], "learning_rate": [0.05, 0.1, 0.3]},
                {"n_estimators": [100, 300], "max_depth": [-1], "num_leaves": [31], "learning_rate": [0.05, 0.1, 0.3]},
            ],
        )
    return grid


# ---------------------------------------------------------------------------
# Step 5: evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X, y, metric_labels=ALL_LABELS):
    """metric_labels: the classes actually present in this dataset (NTHU only has {0,2}, no "Low
    Vigilant") -- scoring against the full ALL_LABELS regardless drags macro F1 down by averaging
    in a phantom always-zero class, and breaks ROC-AUC entirely since predict_proba only has
    columns for classes the model actually saw. confusion_matrix still uses ALL_LABELS so per-fold
    matrices stay a consistent shape to sum."""
    preds = model.predict(X)
    probs = model.predict_proba(X)
    accuracy = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, labels=metric_labels, average="macro", zero_division=0)
    try:
        if len(metric_labels) == 2:
            # binary roc_auc_score wants one column of positive-class scores; it infers the
            # larger label as positive, matching metric_labels[1] since callers pass it sorted.
            pos_col = list(model.classes_).index(metric_labels[1])
            roc_auc = roc_auc_score(y, probs[:, pos_col])
        else:
            roc_auc = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=metric_labels)
    except ValueError:
        roc_auc = None
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y, preds, labels=ALL_LABELS).tolist(),
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


def _checkpoint_path(window_csv_path, output_dir):
    name = os.path.splitext(os.path.basename(window_csv_path))[0]
    return os.path.join(output_dir, f"{name}_train_progress.json")


def train_and_evaluate(window_csv_path, output_dir=OUTPUT_DIR, n_splits=OUTER_FOLDS, resume=False):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(window_csv_path)
    groups = df["subject_id"].values
    # Labels actually present in this dataset (e.g. NTHU-only data never has label 1, "Low
    # Vigilant") -- see evaluate()'s docstring for why this must NOT just be ALL_LABELS.
    metric_labels = sorted(df["label"].unique().tolist())
    model_grid = build_model_grid()
    model_names = list(model_grid.keys())

    folds = outer_subject_folds(df, n_splits=n_splits)
    checkpoint_path = _checkpoint_path(window_csv_path, output_dir)

    fold_results = {name: [] for name in model_names}
    fold_importances = {name: [] for name in model_names}

    if resume:
        checkpoint = db._load_json(checkpoint_path, default=None)
        if checkpoint is None:
            print(f"[resume] no checkpoint found at {checkpoint_path}, starting fresh")
        elif checkpoint.get("model_names") != model_names or checkpoint.get("n_splits") != len(folds):
            print(f"[resume] checkpoint at {checkpoint_path} doesn't match this run's models/fold "
                  f"count -- starting fresh")
        else:
            fold_results = checkpoint["fold_results"]
            fold_importances = checkpoint["fold_importances"]
            done = sum(len(v) for v in fold_results.values())
            print(f"[resume] loaded checkpoint from {checkpoint_path} ({done} fold/model results already done)")

    def _save_checkpoint():
        db._save_json(checkpoint_path, {
            "model_names": model_names,
            "n_splits": len(folds),
            "fold_results": fold_results,
            "fold_importances": fold_importances,
        })

    interrupted = False
    try:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            scaler = fit_scaler(train_df)
            X_train = scaler.transform(train_df[FEATURE_COLUMNS])
            X_test = scaler.transform(test_df[FEATURE_COLUMNS])
            y_train, y_test = train_df["label"].values, test_df["label"].values
            inner_groups = train_df["subject_id"].values
            n_train_subjects = train_df["subject_id"].nunique()

            for name, (estimator, param_grid) in model_grid.items():
                if len(fold_results[name]) > fold_idx:
                    print(f"[resume] skipping fold {fold_idx} {name} (already done)")
                    continue

                best, best_params = _fit_best(estimator, param_grid, X_train, y_train,
                                               inner_groups, n_train_subjects)

                fold_results[name].append({
                    "fold": fold_idx,
                    "best_params": best_params,
                    "train": evaluate(best, X_train, y_train, metric_labels),
                    "test": evaluate(best, X_test, y_test, metric_labels),
                })
                importance = _feature_importance(best, FEATURE_COLUMNS)
                if importance is not None:
                    fold_importances[name].append(importance)

                print(f"[fold {fold_idx}] {name}: test f1={fold_results[name][-1]['test']['f1']:.3f}")
                _save_checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nStopped early. Progress saved to {checkpoint_path} -- rerun with --resume to continue.")

    if interrupted:
        return None, None

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

    # Refit every model type on the full dataset for deployment (shared scaler, each with its own
    # hyperparameter search) -- not just the winner, so run_live(ensemble=True) has a real set of
    # independently-fit models to vote across, rather than N copies of the same one.
    final_scaler = fit_scaler(df)
    X_all = final_scaler.transform(df[FEATURE_COLUMNS])
    y_all = df["label"].values
    n_subjects_all = df["subject_id"].nunique()

    all_models = {}
    for name, (estimator, param_grid) in model_grid.items():
        fitted, fitted_params = _fit_best(estimator, param_grid, X_all, y_all, groups, n_subjects_all)
        all_models[name] = {
            "model": fitted,
            "best_params": fitted_params,
            "cv_f1": results[name]["test_avg"]["f1"],
        }

    joblib.dump(final_scaler, os.path.join(output_dir, "feature_scaler.pkl"))
    joblib.dump({
        "model": all_models[best_name]["model"],
        "scaler": final_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_name,
        "best_params": all_models[best_name]["best_params"],
    }, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump({
        "models": all_models,
        "scaler": final_scaler,
        "feature_columns": FEATURE_COLUMNS,
    }, os.path.join(output_dir, "all_models.pkl"))

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    generate_model_comparison(results, output_dir)

    return results, best_name


def generate_model_comparison(results, output_dir=OUTPUT_DIR):
    """Bar chart + markdown table comparing every trained model's test-set macro metrics, so they
    can be looked at side by side rather than only as scrolled-past per-fold console output.
    `results` is train_and_evaluate()'s return value (or a freshly re-loaded model_results.json)."""
    names = list(results.keys())
    metrics = ["f1", "accuracy", "precision", "recall", "roc_auc"]
    values = {m: [results[n]["test_avg"].get(m) or 0.0 for n in names] for m in metrics}

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2, 5))
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, values[metric], width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (test, averaged over outer folds)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "plots", "model_comparison.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    print(f"Model comparison plot: {plot_path}")
    return plot_path


def generate_feature_importance_plot(model_path=None, window_csv=None, output_dir=OUTPUT_DIR,
                                      n_repeats=10, sample_size=8000, random_state=RANDOM_SEED):
    """Permutation importance of each individual feature on the already-trained/saved model --
    unlike ablation_study() (which removes a whole feature GROUP and retrains from scratch), this
    shuffles one feature at a time on the existing model and measures the F1 drop, so it ranks
    features individually rather than by group."""
    model_path = model_path or os.path.join(output_dir, "best_model.pkl")
    payload = joblib.load(model_path)
    model, scaler, cols = payload["model"], payload["scaler"], payload["feature_columns"]

    window_csv = window_csv or os.path.join(output_dir, "uta_rldd_window.csv")
    df = pd.read_csv(window_csv)
    X = scaler.transform(pd.DataFrame(df[cols], columns=cols))
    y = df["label"].values

    rng = np.random.RandomState(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X, y = X[idx], y[idx]

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state,
                                     scoring="f1_macro", n_jobs=1)
    order = np.argsort(result.importances_mean)[::-1]
    names = [cols[i] for i in order]
    means = result.importances_mean[order]
    stds = result.importances_std[order]

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.5))
    ypos = np.arange(len(names))
    ax.barh(ypos, means, xerr=stds, color="#4C72B0")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(f"F1-macro drop when shuffled ({payload.get('model_name', 'model')})")
    ax.set_title("Per-feature permutation importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "plots", "feature_importance.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    print(f"Feature importance plot: {plot_path}")
    for name, mean, std in zip(names, means, stds):
        print(f"  {name:28s} {mean:+.4f} +/- {std:.4f}")
    return plot_path


# ---------------------------------------------------------------------------
# Ablation study: automated feature importance (e.g. LightGBM split/gain) only reflects what a
# single already-fit model happened to lean on -- under collinearity (e.g. mean_ear vs. min_ear)
# a feature can look replaceable in isolation even though removing its whole group would hurt.
# This retrains from scratch with each logical feature group actually removed, to measure the
# real impact on macro F1.
# ---------------------------------------------------------------------------
def ablation_study(window_csv_path, output_dir=OUTPUT_DIR, n_splits=OUTER_FOLDS, model_name="random_forest"):
    df = pd.read_csv(window_csv_path)
    metric_labels = sorted(df["label"].unique().tolist())  # see evaluate()'s docstring
    model_grid = build_model_grid()
    if model_name not in model_grid:
        raise ValueError(f"Unknown model '{model_name}', choices: {list(model_grid)}")
    estimator, param_grid = model_grid[model_name]

    variants = {"all_features": list(FEATURE_COLUMNS)}
    for group_name, group_cols in FEATURE_GROUPS.items():
        variants[f"without_{group_name}"] = [c for c in FEATURE_COLUMNS if c not in group_cols]

    folds = outer_subject_folds(df, n_splits=n_splits)
    results = {}
    for variant_name, cols in variants.items():
        fold_f1s = []
        for train_idx, test_idx in folds:
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            scaler = StandardScaler().fit(train_df[cols])
            X_train, X_test = scaler.transform(train_df[cols]), scaler.transform(test_df[cols])
            y_train, y_test = train_df["label"].values, test_df["label"].values
            best, _ = _fit_best(estimator, param_grid, X_train, y_train,
                                 train_df["subject_id"].values, train_df["subject_id"].nunique())
            fold_f1s.append(evaluate(best, X_test, y_test, metric_labels)["f1"])

        results[variant_name] = {
            "n_features": len(cols),
            "mean_f1": float(np.mean(fold_f1s)),
            "std_f1": float(np.std(fold_f1s)),
            "fold_f1s": fold_f1s,
        }
        print(f"{variant_name:24s} ({len(cols):2d} features): mean F1 = {results[variant_name]['mean_f1']:.3f} "
              f"+/- {results[variant_name]['std_f1']:.3f}")

    baseline_f1 = results["all_features"]["mean_f1"]
    print(f"\nImpact of removing each feature group (negative = that group was helping):")
    for group_name in FEATURE_GROUPS:
        f1_without = results[f"without_{group_name}"]["mean_f1"]
        print(f"  removing {group_name:20s}: F1 change = {f1_without - baseline_f1:+.3f}")

    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    return results


# ---------------------------------------------------------------------------
# Step 6: live run
# ---------------------------------------------------------------------------
# _center_on_nose / _draw_points_panel / _draw_metrics_panel now live in dataset_builder.py (db.*)
# -- annotated debug clips need the exact same panels, so they're shared there.


def _decide_with_drowsy_threshold(probs_row, classes, threshold=LIVE_DROWSY_CONFIDENCE_THRESHOLD):
    """Only ever returns Drowsy (label 2) if the model's own P(Drowsy) reaches `threshold`;
    otherwise falls back to whichever non-Drowsy class is most likely. See
    LIVE_DROWSY_CONFIDENCE_THRESHOLD's comment -- this is a deployment decision policy applied on
    top of the model's raw probabilities, not how the model itself is trained or evaluated."""
    class_probs = dict(zip(classes, probs_row))
    drowsy_p = class_probs.get(2, 0.0)
    if drowsy_p >= threshold:
        return 2, drowsy_p
    non_drowsy = {label: p for label, p in class_probs.items() if label != 2}
    if not non_drowsy:  # model only ever saw/predicts Drowsy -- shouldn't happen in practice
        return 2, drowsy_p
    label = max(non_drowsy, key=non_drowsy.get)
    return label, non_drowsy[label]


def _load_ensemble(all_models_path, top_n=ENSEMBLE_TOP_N):
    """Loads the top_n models (by CV test f1) from all_models.pkl for soft-voting. Returns
    (scaler, feature_columns, [(name, model), ...] sorted best-first)."""
    payload = joblib.load(all_models_path)
    ranked = sorted(payload["models"].items(), key=lambda kv: kv[1]["cv_f1"], reverse=True)[:top_n]
    members = [(name, info["model"]) for name, info in ranked]
    print("Ensemble members:", ", ".join(f"{name} (cv f1={info['cv_f1']:.3f})" for name, info in ranked))
    return payload["scaler"], payload["feature_columns"], members


def _ensemble_predict_proba(members, X):
    """Soft-votes by averaging predict_proba across members, aligned by class label (not raw
    array position -- different models can order model.classes_ differently). Returns
    (sorted_classes, avg_probs_row)."""
    classes = sorted(set(c for _, model in members for c in model.classes_))
    acc = {c: 0.0 for c in classes}
    for _, model in members:
        probs = model.predict_proba(X)[0]
        for c, p in zip(model.classes_, probs):
            acc[c] += p / len(members)
    return classes, np.array([acc[c] for c in classes])


def _status_panel(width, text):
    """One-line status panel for run_live() -- shown below the video whenever there's no real
    classification yet (calibrating / buffer still filling), so the window is never just silent."""
    panel = np.zeros((50, width, 3), dtype=np.uint8)
    cv2.putText(panel, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    return panel


MIN_DT_SEC = 1.0 / 60.0  # floor for velocity dt -- a backlog-delivered frame burst can otherwise
                          # give a near-zero dt and blow up angle_change/dt into a fake spike


def run_live(model_path=os.path.join(OUTPUT_DIR, "best_model.pkl"), camera_id=0,
             ensemble=False, ensemble_path=os.path.join(OUTPUT_DIR, "all_models.pkl"),
             ensemble_top_n=ENSEMBLE_TOP_N):
    ensemble_members = None
    if ensemble:
        scaler, feature_columns, ensemble_members = _load_ensemble(ensemble_path, ensemble_top_n)
    else:
        bundle = joblib.load(model_path)
        model, scaler, feature_columns = bundle["model"], bundle["scaler"], bundle["feature_columns"]

    face_landmarker = db._open_face_landmarker()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the freshest frame, not a queued backlog

    print("Press 'c' to calibrate (look forward, stay neutral), 'q' to quit.")

    baseline = None
    calibrating_until = None
    calib_rows = []
    rolling_buffer = deque()  # trimmed to the last WINDOW_SEC of wall-clock time, not a frame count
    buffer_start_time = None

    prev_pitch = prev_yaw = prev_roll = prev_frame_time = None
    classification, confidence, stats, stats_panel = "Calibrating...", 0.0, None, None
    last_window_time = 0.0
    last_process_time = 0.0
    min_process_interval = 1.0 / TARGET_PROCESS_HZ

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrating_until = now + CALIBRATION_SEC
            calib_rows = []
            print(f"Calibrating for {CALIBRATION_SEC:.0f}s, look forward and stay neutral...")

        if now - last_process_time < min_process_interval:
            continue
        last_process_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = db._detect(face_landmarker, rgb, now * 1000.0)

        norm_pts, raw_pts_centered = {}, {}
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            pts = {name: db._landmark_px(landmarks[idx], w, h) for name, idx in db.LANDMARK_IDS.items()}
            for x, y, _ in pts.values():
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            pose = db._rotation_from_result(result)
            if pose is not None:
                pitch, yaw, roll, _ = pose
                norm_pts = db.normalize_landmarks(pts)
                raw_pts_centered = db._center_on_nose(pts)
                ear = (db.eye_aspect_ratio(pts, db.LEFT_EYE) + db.eye_aspect_ratio(pts, db.RIGHT_EYE)) / 2.0
                mar = db.mouth_aspect_ratio(pts)
                dt = max(now - prev_frame_time, MIN_DT_SEC) if prev_frame_time is not None else 1.0 / TARGET_PROCESS_HZ
                pitch_vel = 0.0 if prev_pitch is None else db._circular_diff(pitch, prev_pitch) / dt
                yaw_vel = 0.0 if prev_yaw is None else db._circular_diff(yaw, prev_yaw) / dt
                roll_vel = 0.0 if prev_roll is None else db._circular_diff(roll, prev_roll) / dt
                prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll

                row = {"subject_id": "live", "video": "live", "label": -1,
                       "frame": 0, "time_sec": now,
                       "ear_raw": ear, "mar_raw": mar, "pitch": pitch, "yaw": yaw, "roll": roll,
                       "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel}
                if buffer_start_time is None:
                    buffer_start_time = now
                rolling_buffer.append(row)
                if calibrating_until is not None:
                    calib_rows.append(row)
            prev_frame_time = now

        while rolling_buffer and now - rolling_buffer[0]["time_sec"] > db.WINDOW_SEC:
            rolling_buffer.popleft()

        if calibrating_until is not None and now >= calibrating_until:
            if calib_rows:
                baseline = db.compute_baseline(pd.DataFrame(calib_rows))
                print("Calibration complete:", baseline)
            calibrating_until, calib_rows = None, []

        raw_panel = db._draw_points_panel(db.PANEL_SIZE, raw_pts_centered, "RAW 3D")
        frontal_panel = db._draw_points_panel(db.PANEL_SIZE, norm_pts, "FRONTAL (normalized)")
        frame_resized = cv2.resize(frame, (int(w * db.PANEL_SIZE / h), db.PANEL_SIZE))
        combined = np.hstack([frame_resized, raw_panel, frontal_panel])

        window_span = (rolling_buffer[-1]["time_sec"] - rolling_buffer[0]["time_sec"]) if len(rolling_buffer) > 1 else 0.0
        buffer_age = (now - buffer_start_time) if buffer_start_time is not None else 0.0
        if baseline is not None and buffer_age >= db.WINDOW_SEC and \
                now - last_window_time >= STRIDE_SEC_LIVE:
            chunk_df = db.apply_baseline(pd.DataFrame(rolling_buffer), baseline)
            measured_fps = len(rolling_buffer) / window_span
            stats = db._summarize_window(chunk_df, measured_fps)
            X = scaler.transform(pd.DataFrame([stats], columns=feature_columns))
            if ensemble_members is not None:
                classes, probs = _ensemble_predict_proba(ensemble_members, X)
            else:
                classes, probs = model.classes_, model.predict_proba(X)[0]
            pred, confidence = _decide_with_drowsy_threshold(probs, classes)
            classification = LABEL_NAMES.get(pred, str(pred))
            last_window_time = now
            extra_lines = [f"Prediction: {classification}   Confidence: {confidence * 100:.1f}%"]
            stats_panel = db._draw_metrics_panel(combined.shape[1], stats, window_span, extra_lines)

        if stats_panel is not None:
            combined = np.vstack([combined, stats_panel])
        else:
            if calibrating_until is not None:
                status = f"Calibrating... {max(0.0, calibrating_until - now):.1f}s left"
            elif baseline is None:
                status = "Press 'c' to calibrate"
            else:
                status = f"Warming up: {min(buffer_age, db.WINDOW_SEC):.1f}s / {db.WINDOW_SEC:.0f}s buffered"
            combined = np.vstack([combined, _status_panel(combined.shape[1], status)])
        cv2.imshow("Driver Drowsiness Monitor", combined)

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--window-csv", default=os.path.join(OUTPUT_DIR, "uta_rldd_window.csv"))
    train_p.add_argument("--folds", type=int, default=OUTER_FOLDS,
                          help="Number of subject-level outer CV folds (default: 5, per UTA-RLDD's 60 subjects)")
    train_p.add_argument("--resume", action="store_true",
                          help="Resume from a previous interrupted run's checkpoint for this --window-csv, "
                               "if one exists (default: start fresh)")

    live_p = sub.add_parser("live")
    live_p.add_argument("--camera-id", type=int, default=0)
    live_p.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))
    live_p.add_argument("--ensemble", action="store_true",
                         help="Soft-vote across the top-N models (by CV f1) from outputs/all_models.pkl "
                              "instead of using a single model")
    live_p.add_argument("--ensemble-path", default=os.path.join(OUTPUT_DIR, "all_models.pkl"))
    live_p.add_argument("--ensemble-top-n", type=int, default=ENSEMBLE_TOP_N)

    ablation_p = sub.add_parser("ablation")
    ablation_p.add_argument("--window-csv", default=os.path.join(OUTPUT_DIR, "uta_rldd_window.csv"))
    ablation_p.add_argument("--folds", type=int, default=OUTER_FOLDS)
    ablation_p.add_argument("--model", default="random_forest",
                             help="Which model from build_model_grid() to run the ablation with "
                                  "(default: random_forest)")

    fi_p = sub.add_parser("feature-importance")
    fi_p.add_argument("--window-csv", default=os.path.join(OUTPUT_DIR, "uta_rldd_window.csv"))
    fi_p.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))

    args = parser.parse_args()
    if args.command == "train":
        train_and_evaluate(args.window_csv, n_splits=args.folds, resume=args.resume)
    elif args.command == "live":
        run_live(args.model_path, args.camera_id, ensemble=args.ensemble,
                  ensemble_path=args.ensemble_path, ensemble_top_n=args.ensemble_top_n)
    elif args.command == "ablation":
        ablation_study(args.window_csv, n_splits=args.folds, model_name=args.model)
    elif args.command == "feature-importance":
        generate_feature_importance_plot(model_path=args.model_path, window_csv=args.window_csv)
