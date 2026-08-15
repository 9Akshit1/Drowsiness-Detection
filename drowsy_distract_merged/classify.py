import os
import sys
import time
import argparse
import contextlib
import urllib.request
from collections import deque

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from scipy.signal import find_peaks
from joblib import load as joblib_load

CAMERA_ID = 1
CROP_SIZE = 300
LIVE_NUM_FACES = 8
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
TARGET_PROCESS_HZ = 10.0
DEBUG_PANEL_SIZE = 220

WINDOW_SEC = 10.0
STRIDE_SEC = 1.0

EAR_CLOSED_RATIO = 0.75
MAR_YAWN_RATIO = 1.8
MIN_BLINK_DURATION_SEC = 3 / 30.0
MIN_YAWN_DURATION_SEC = 8 / 30.0
BLINK_MERGE_GAP_SEC = MIN_BLINK_DURATION_SEC
YAWN_MERGE_GAP_SEC = MIN_YAWN_DURATION_SEC

PERCLOS_EAR_RATIO = 0.4
NOD_VELOCITY_DEG_S = 25.0
NOD_MIN_PROMINENCE_DEG = 10.0
NOD_MIN_GAP_SEC = 0.5
NOD_VELOCITY_CHECK_WINDOW_SEC = 0.3
NOD_MAX_CYCLE_SEC = 1.2
MOTION_GATE_DEG_S = 50.0

EAR_CLIP_MAX = 2.0
MAR_CLIP_MAX = 50.0

LIVE_DROWSY_CONFIDENCE_THRESHOLD = 0.7
ENSEMBLE_TOP_N = 3
MIN_DT_SEC = 1.0 / 60.0
MIN_RATE_DISPLAY_SEC = WINDOW_SEC * 0.95
LABEL_NAMES = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "face_landmarker.task")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration", "calibration.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

NOSE_LANDMARK_IDX = 1
LANDMARK_IDS = {
    "nose_tip": 1, "chin": 152,
    "left_eye_outer": 33, "left_eye_top1": 160, "left_eye_top2": 158,
    "left_eye_inner": 133, "left_eye_bottom1": 153, "left_eye_bottom2": 144,
    "right_eye_inner": 362, "right_eye_top1": 385, "right_eye_top2": 387,
    "right_eye_outer": 263, "right_eye_bottom1": 373, "right_eye_bottom2": 380,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top_outer": 0, "mouth_bottom_outer": 17,
    "mouth_top_left": 39, "mouth_bottom_left": 181,
    "mouth_top_right": 269, "mouth_bottom_right": 405,
}
LEFT_EYE = ("left_eye_outer", "left_eye_top1", "left_eye_top2",
            "left_eye_inner", "left_eye_bottom1", "left_eye_bottom2")
RIGHT_EYE = ("right_eye_inner", "right_eye_top1", "right_eye_top2",
             "right_eye_outer", "right_eye_bottom1", "right_eye_bottom2")
MOUTH_VERTICAL_PAIRS = (("mouth_top_left", "mouth_bottom_left"),
                        ("mouth_top_outer", "mouth_bottom_outer"),
                        ("mouth_top_right", "mouth_bottom_right"))
MOUTH_HORIZONTAL = ("mouth_left", "mouth_right")

REFERENCE_FACE_SHAPE = {
    "nose_tip": (0.0, 0.0, 0.0),
    "chin": (0.0, -123.4, 59.6),
    "left_eye_outer": (-63.6, 58.8, 69.2),
    "left_eye_top1": (-46.3, 65.3, 60.5),
    "left_eye_top2": (-47.3, 64.8, 55.2),
    "left_eye_inner": (-25.2, 55.8, 61.2),
    "left_eye_bottom1": (-45.6, 54.0, 56.6),
    "left_eye_bottom2": (-44.4, 53.7, 61.7),
    "right_eye_inner": (25.2, 55.8, 61.2),
    "right_eye_top1": (46.3, 65.3, 60.5),
    "right_eye_top2": (47.3, 64.8, 55.2),
    "right_eye_outer": (63.6, 58.8, 69.2),
    "right_eye_bottom1": (45.6, 54.0, 56.6),
    "right_eye_bottom2": (44.4, 53.7, 61.7),
    "mouth_left": (-51.3, -36.3, 65.0),
    "mouth_right": (51.3, -36.3, 65.0),
    "mouth_top_outer": (0.0, -36.2, 32.3),
    "mouth_bottom_outer": (0.0, -60.0, 42.8),
    "mouth_top_left": (-25.0, -34.5, 39.6),
    "mouth_bottom_left": (-24.5, -56.0, 48.8),
    "mouth_top_right": (25.0, -34.5, 39.6),
    "mouth_bottom_right": (24.5, -56.0, 48.8),
}
REFERENCE_LANDMARK_ORDER = list(LANDMARK_IDS)

LABEL_MAP = {0: "Front", 1: "Left", 2: "Right", 3: "Phone"}
MODEL_FILES = [
    os.path.join(BASE_DIR, "Models", "Cubic_SVM.pkl"),
    os.path.join(BASE_DIR, "Models", "Neural_Network.pkl"),
]
PROB_EMA_ALPHA = 0.55
HOLD_FRAMES = 2
_IRIS_LM = frozenset(range(468, 478))
_EYELID_LM = frozenset({145, 159, 160, 161, 374, 380, 385, 386, 387, 388})
_ALPHA_BASE = 0.60
_ALPHA_EYELID = 0.40
_ALPHA_IRIS = 0.20
GAZE_EAR_OPEN_THRESHOLD = 0.23
GAZE_EAR_CLOSED_THRESHOLD = 0.15
_GAZE_EAR_LEFT = (33, 160, 158, 133, 153, 145)
_GAZE_EAR_RIGHT = (263, 387, 385, 362, 380, 374)
SELECTED_LANDMARKS = [
    0, 1, 2, 2, 4, 5, 10, 17, 33, 37, 39, 40, 54,
    61, 67, 84, 91, 94, 97, 98, 132, 133, 145,
    148, 150, 152, 153, 153, 158, 159, 160, 161,
    162, 163, 168, 172, 185, 195, 234, 251, 263,
    267, 269, 270, 288, 291, 314, 321, 323, 327,
    332, 356, 362, 365, 374, 377, 378, 380, 380,
    385, 386, 387, 388, 390, 409, 468, 469, 470,
    471, 472, 473, 474, 475, 476, 477,
]
_PER_SLOT_ALPHA = np.array(
    [(_ALPHA_IRIS if lm in _IRIS_LM else _ALPHA_EYELID if lm in _EYELID_LM else _ALPHA_BASE)
     for lm in SELECTED_LANDMARKS for _ in (0, 1)], dtype=np.float32)
_IS_IRIS_SLOT = np.array([lm in _IRIS_LM for lm in SELECTED_LANDMARKS for _ in (0, 1)])


def _ensure_face_landmarker_model():
    if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_LANDMARKER_MODEL_PATH), exist_ok=True)
        print(f"downloading face_landmarker.task to {FACE_LANDMARKER_MODEL_PATH} ...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
    return FACE_LANDMARKER_MODEL_PATH


@contextlib.contextmanager
def _suppress_native_stderr():
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def open_face_landmarker(num_faces=LIVE_NUM_FACES):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_face_landmarker_model()),
        running_mode=RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=FACE_MESH_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
        output_facial_transformation_matrixes=True,
    )
    with _suppress_native_stderr():
        return FaceLandmarker.create_from_options(options)


def detect(landmarker, rgb_frame, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, int(timestamp_ms))


def landmarks_to_array(landmarks):
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)


def landmark_px(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z * w)


def face_points(landmarks, w, h):
    return {name: landmark_px(landmarks[idx], w, h) for name, idx in LANDMARK_IDS.items()}


def rotation_from_result(result, face_idx=0):
    if not result.facial_transformation_matrixes or face_idx >= len(result.facial_transformation_matrixes):
        return None
    rmat = np.array(result.facial_transformation_matrixes[face_idx])[:3, :3]
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll


def _dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def eye_aspect_ratio(pts, eye_names):
    p1, p2, p3, p4, p5, p6 = (pts[n] for n in eye_names)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def mouth_aspect_ratio(pts):
    vertical = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL_PAIRS)
    horizontal = _dist(pts[MOUTH_HORIZONTAL[0]], pts[MOUTH_HORIZONTAL[1]])
    return vertical / (3.0 * horizontal)


def circular_diff(a2, a1):
    return (a2 - a1 + 180.0) % 360.0 - 180.0


def apply_baseline(frame_df, baseline):
    df = frame_df.copy()
    df["ear"] = (df["ear_raw"] / baseline["ear_baseline"]).clip(upper=EAR_CLIP_MAX)
    df["mar"] = (df["mar_raw"] / baseline["mar_baseline"]).clip(upper=MAR_CLIP_MAX)
    df["pitch_norm"] = circular_diff(df["pitch"].values, baseline["pitch_baseline"])
    df["yaw_norm"] = circular_diff(df["yaw"].values, baseline["yaw_baseline"])
    return df


def _events(is_active, fps, min_frames, merge_gap_frames=0):
    events = []
    start = last_active = None
    for i, active in enumerate(is_active):
        if active:
            if start is None:
                start = i
            last_active = i
        elif start is not None and (i - last_active) > merge_gap_frames:
            length = last_active - start + 1
            if length >= min_frames:
                events.append(length / fps)
            start = last_active = None
    if start is not None:
        length = last_active - start + 1
        if length >= min_frames:
            events.append(length / fps)
    return events


def _count_nod_events(pitch, pitch_vel, fps):
    distance = max(1, int(NOD_MIN_GAP_SEC * fps))
    max_cycle = max(distance, int(NOD_MAX_CYCLE_SEC * fps))
    peaks_pos, _ = find_peaks(pitch, prominence=NOD_MIN_PROMINENCE_DEG, distance=distance)
    peaks_neg, _ = find_peaks(-pitch, prominence=NOD_MIN_PROMINENCE_DEG, distance=distance)
    tagged = sorted([(int(p), "pos") for p in peaks_pos] + [(int(p), "neg") for p in peaks_neg])

    events = []
    for idx, polarity in tagged:
        if events and polarity != events[-1][-1][1] and idx - events[-1][-1][0] <= max_cycle:
            events[-1].append((idx, polarity))
        else:
            events.append([(idx, polarity)])

    half_window = max(1, int(NOD_VELOCITY_CHECK_WINDOW_SEC * fps))
    count = 0
    for event in events:
        lo = max(0, event[0][0] - half_window)
        hi = min(len(pitch_vel), event[-1][0] + half_window + 1)
        if len(pitch_vel[lo:hi]) and np.max(np.abs(pitch_vel[lo:hi])) >= NOD_VELOCITY_DEG_S:
            count += 1
    return count


def summarize_window(chunk, fps):
    duration = len(chunk) / fps
    ear, mar = chunk["ear"].values, chunk["mar"].values
    pitch, pitch_vel = chunk["pitch_norm"].values, chunk["pitch_vel"].values

    min_blink_frames = max(2, round(MIN_BLINK_DURATION_SEC * fps))
    blink_merge_gap_frames = max(1, round(BLINK_MERGE_GAP_SEC * fps))
    min_yawn_frames = max(2, round(MIN_YAWN_DURATION_SEC * fps))
    yawn_merge_gap_frames = max(1, round(YAWN_MERGE_GAP_SEC * fps))

    motion_ok = (chunk["pitch_vel"].abs().values + chunk["yaw_vel"].abs().values) <= MOTION_GATE_DEG_S

    closed = (ear < EAR_CLOSED_RATIO) & motion_ok
    blinks = _events(closed, fps, min_blink_frames, blink_merge_gap_frames)
    perclos_closed = ear < PERCLOS_EAR_RATIO

    yawning = (mar > MAR_YAWN_RATIO) & motion_ok
    yawns = _events(yawning, fps, min_yawn_frames, yawn_merge_gap_frames)

    nod_count = _count_nod_events(pitch, pitch_vel, fps)
    zero_crossings = np.sum(np.diff(np.sign(pitch - pitch.mean())) != 0)

    return {
        "mean_ear": float(ear.mean()), "std_ear": float(ear.std()),
        "min_ear": float(ear.min()), "max_ear": float(ear.max()),
        "blink_count": len(blinks), "blink_freq": len(blinks) / duration,
        "avg_blink_duration": float(np.mean(blinks)) if blinks else 0.0,
        "max_blink_duration": float(np.max(blinks)) if blinks else 0.0,
        "perclos": float(perclos_closed.mean()),
        "mean_mar": float(mar.mean()), "max_mar": float(mar.max()), "std_mar": float(mar.std()),
        "yawn_count": len(yawns),
        "avg_yawn_duration": float(np.mean(yawns)) if yawns else 0.0,
        "mean_pitch": float(pitch.mean()), "std_pitch": float(pitch.std()),
        "max_pitch_velocity": float(np.max(np.abs(pitch_vel))),
        "avg_pitch_velocity": float(np.mean(np.abs(pitch_vel))),
        "pitch_oscillation_freq": float(zero_crossings / (2 * duration)),
        "nod_count": int(nod_count),
    }


def decide_with_drowsy_threshold(probs_row, classes, threshold=LIVE_DROWSY_CONFIDENCE_THRESHOLD):
    class_probs = dict(zip(classes, probs_row))
    drowsy_p = class_probs.get(2, 0.0)
    if drowsy_p >= threshold:
        return 2, drowsy_p
    non_drowsy = {label: p for label, p in class_probs.items() if label != 2}
    if not non_drowsy:
        return 2, drowsy_p
    label = max(non_drowsy, key=non_drowsy.get)
    return label, non_drowsy[label]


def load_drowsy_model(model_path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["scaler"], bundle["feature_columns"]


def load_drowsy_ensemble(all_models_path, top_n=ENSEMBLE_TOP_N):
    payload = joblib.load(all_models_path)
    ranked = sorted(payload["models"].items(), key=lambda kv: kv[1]["cv_f1"], reverse=True)[:top_n]
    members = [(name, info["model"]) for name, info in ranked]
    print("ensemble members:", ", ".join(f"{name} (cv f1={info['cv_f1']:.3f})" for name, info in ranked))
    return payload["scaler"], payload["feature_columns"], members


def ensemble_predict_proba(members, X):
    classes = sorted(set(c for _, model in members for c in model.classes_))
    acc = {c: 0.0 for c in classes}
    for _, model in members:
        probs = model.predict_proba(X)[0]
        for c, p in zip(model.classes_, probs):
            acc[c] += p / len(members)
    return classes, np.array([acc[c] for c in classes])


def classify_drowsy_window(rolling_buffer, window_span, baseline, scaler, feature_columns,
                            model=None, ensemble_members=None):
    chunk_df = apply_baseline(pd.DataFrame(rolling_buffer), baseline)
    measured_fps = len(rolling_buffer) / window_span
    stats = summarize_window(chunk_df, measured_fps)
    X = scaler.transform(pd.DataFrame([stats], columns=feature_columns))
    if ensemble_members is not None:
        classes, probs = ensemble_predict_proba(ensemble_members, X)
    else:
        classes, probs = model.classes_, model.predict_proba(X)[0]
    pred, confidence = decide_with_drowsy_threshold(probs, classes)
    return stats, LABEL_NAMES.get(pred, str(pred)), confidence


def gaze_compute_ear(lm_2d):
    def ear_one(p1, p2, p3, p4, p5, p6):
        A = np.linalg.norm(lm_2d[p2] - lm_2d[p6])
        B = np.linalg.norm(lm_2d[p3] - lm_2d[p5])
        C = np.linalg.norm(lm_2d[p1] - lm_2d[p4])
        return (A + B) / (2.0 * C + 1e-9)
    return float((ear_one(*_GAZE_EAR_LEFT) + ear_one(*_GAZE_EAR_RIGHT)) * 0.5)


class LandmarkSmoother:
    def __init__(self):
        self._smooth = None

    def update(self, flat_146, ear):
        if self._smooth is None:
            self._smooth = flat_146.copy()
            return self._smooth
        alphas = _PER_SLOT_ALPHA.copy()
        if ear < GAZE_EAR_CLOSED_THRESHOLD:
            alphas = np.where(_IS_IRIS_SLOT, 0.0, alphas).astype(np.float32)
        elif ear < GAZE_EAR_OPEN_THRESHOLD:
            t = (ear - GAZE_EAR_CLOSED_THRESHOLD) / (GAZE_EAR_OPEN_THRESHOLD - GAZE_EAR_CLOSED_THRESHOLD)
            alphas = np.where(_IS_IRIS_SLOT, _ALPHA_IRIS * t, alphas).astype(np.float32)
        self._smooth = alphas * flat_146 + (1.0 - alphas) * self._smooth
        return self._smooth


class PredictionSmoother:
    def __init__(self, n_classes=4, alpha=PROB_EMA_ALPHA, hold=HOLD_FRAMES):
        self.alpha = alpha
        self.hold = hold
        self.smooth = np.ones(n_classes, dtype=np.float32) / n_classes
        self.label = 0
        self.pending = 0
        self.count = 0

    def update(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        probs /= probs.sum() + 1e-9
        self.smooth = self.alpha * probs + (1.0 - self.alpha) * self.smooth
        new_label = int(np.argmax(self.smooth))
        if new_label == self.label:
            self.pending = new_label
            self.count = 0
        elif new_label == self.pending:
            self.count += 1
            if self.count >= self.hold:
                self.label = new_label
                self.count = 0
        else:
            self.pending = new_label
            self.count = 1
        return self.label

    @property
    def confidence(self):
        return float(np.max(self.smooth))


def apply_gaze_bias(raw_478x3, R, t):
    if R is not None and t is not None:
        corrected_3d = (R @ raw_478x3.T + t).T
    else:
        corrected_3d = raw_478x3
    return corrected_3d[:, :2].astype(np.float32)


def load_gaze_models(model_files=MODEL_FILES):
    models = []
    for path in model_files:
        try:
            models.append(joblib_load(path))
            print(f"loaded {path}")
        except Exception as e:
            print(f"could not load {path}: {e}")
    return models


def run_gaze_models(smoothed_flat, models):
    flat = smoothed_flat.reshape(1, -1)
    probs = []
    for m in models:
        try:
            probs.append(m.predict_proba(flat)[0])
        except Exception as e:
            print(f"model error: {e}")
    return np.mean(probs, axis=0) if probs else np.ones(4) / 4.0


def load_calibration(path):
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    if "baseline_keys" not in d.files:
        return None
    out = {"crop": tuple(int(v) for v in d["crop"])}
    out["baseline"] = {k: float(v) for k, v in zip(d["baseline_keys"], d["baseline_vals"])}
    out["ref_3d"] = d["ref_3d"].astype(np.float32)
    if "R" in d.files and "t" in d.files:
        out["R"] = d["R"].astype(np.float32)
        out["t"] = d["t"].astype(np.float32)
    return out


def apply_crop(frame, crop):
    x0, y0, x1, y1 = crop
    return frame[y0:y1, x0:x1]


# -- Kabsch full-frontal reprojection, debug view only, never fed to any model --
def _to_rotation_space(pts):
    return {name: (x, -y, z) for name, (x, y, z) in pts.items()}


def _kabsch_rotation(reference, current):
    ref = reference / np.linalg.norm(reference, axis=1).mean()
    cur = current / np.linalg.norm(current, axis=1).mean()
    h = ref.T @ cur
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def normalize_landmarks(pts):
    rot_pts = _to_rotation_space(pts)
    nose = np.array(rot_pts["nose_tip"])
    centered = {name: np.array(p) - nose for name, p in rot_pts.items()}
    current_arr = np.array([centered[name] for name in REFERENCE_LANDMARK_ORDER])
    reference_arr = np.array([REFERENCE_FACE_SHAPE[name] for name in REFERENCE_LANDMARK_ORDER])
    rmat = _kabsch_rotation(reference_arr, current_arr)
    return {name: tuple(rmat.T @ p) for name, p in centered.items()}


def _project_dense(pts_2d, size, margin=14):
    mn, mx = pts_2d.min(axis=0), pts_2d.max(axis=0)
    rng = np.maximum(mx - mn, 1e-9)
    scaled = (pts_2d - mn) / rng * (size - 2 * margin) + margin
    return scaled.astype(int)


def draw_dense_panel(pts_2d, size, label):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    px = _project_dense(pts_2d, size)
    for x, y in px:
        if 0 <= x < size and 0 <= y < size:
            cv2.circle(canvas, (x, y), 1, (60, 60, 60), -1)
    for idx in set(SELECTED_LANDMARKS):
        x, y = px[idx]
        if 0 <= x < size and 0 <= y < size:
            cv2.circle(canvas, (x, y), 2, (0, 200, 220), -1)
    nx, ny = px[NOSE_LANDMARK_IDX]
    if 0 <= nx < size and 0 <= ny < size:
        cv2.circle(canvas, (nx, ny), 3, (0, 255, 0), -1)
    cv2.putText(canvas, label, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return canvas


def draw_frontal_panel(pts_dict, size, label):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, scale = size // 2, size // 2, size / 400.0
    for x, y, _ in pts_dict.values():
        px, py = int(cx + x * scale), int(cy - y * scale)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)
    cv2.putText(canvas, label, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    return canvas


TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.42
TEXT_THICKNESS = 1
TEXT_LINE_H = 22


def _wrap_line(text, max_width):
    words = text.split(" ")
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(trial, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        if tw > max_width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def _rate_str(count, window_span):
    if window_span < MIN_RATE_DISPLAY_SEC:
        return "..."
    return f"{count / window_span:.2f}/s"


def draw_panel(width, drowsy_stats, window_span, drowsy_label, drowsy_conf, distract_label, distract_conf):
    lines = []
    if drowsy_stats is not None:
        lines.append(f"EAR {drowsy_stats['mean_ear']:.2f}")
        lines.append(f"MAR {drowsy_stats['mean_mar']:.2f}")
        lines.append(f"PERCLOS {drowsy_stats['perclos'] * 100:.0f}%")
        lines.append(f"Blink {drowsy_stats['blink_count']} ({_rate_str(drowsy_stats['blink_count'], window_span)})")
        lines.append(f"Yawn {drowsy_stats['yawn_count']} ({_rate_str(drowsy_stats['yawn_count'], window_span)})")
        lines.append(f"Nod {drowsy_stats['nod_count']} ({_rate_str(drowsy_stats['nod_count'], window_span)})")
        lines.append(f"Drowsy: {drowsy_label} ({drowsy_conf * 100:.0f}%)")
    else:
        lines.append("Drowsy: warming up")
    lines.append(f"Gaze: {distract_label} ({distract_conf * 100:.0f}%)")

    wrapped = [w for line in lines for w in _wrap_line(line, width - 16)]
    panel = np.zeros((TEXT_LINE_H * len(wrapped) + 12, width, 3), dtype=np.uint8)
    for i, line in enumerate(wrapped):
        cv2.putText(panel, line, (8, 18 + i * TEXT_LINE_H), TEXT_FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICKNESS)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int, default=CAMERA_ID)
    parser.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble-path", default=os.path.join(OUTPUT_DIR, "all_models.pkl"))
    parser.add_argument("--ensemble-top-n", type=int, default=ENSEMBLE_TOP_N)
    parser.add_argument("--show-landmarks", action="store_true")
    parser.add_argument("--debug-view", action="store_true")
    args = parser.parse_args()

    calib = load_calibration(CALIBRATION_FILE)
    if calib is None:
        print(f"no calibration found at {CALIBRATION_FILE} -- run calibrate.py first")
        return
    crop, baseline = calib["crop"], calib["baseline"]

    drowsy_model, ensemble_members = None, None
    if args.ensemble:
        scaler, feature_columns, ensemble_members = load_drowsy_ensemble(args.ensemble_path, args.ensemble_top_n)
    else:
        drowsy_model, scaler, feature_columns = load_drowsy_model(args.model_path)

    R, t = calib.get("R"), calib.get("t")
    gaze_models = load_gaze_models()
    if not gaze_models:
        print("no distraction models loaded -- exiting")
        return
    lm_smoother = LandmarkSmoother()
    pred_smoother = PredictionSmoother()

    landmarker = open_face_landmarker()
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print("could not open camera")
        return

    rolling_buffer = deque()
    buffer_start_time = None
    prev_pitch = prev_yaw = prev_frame_time = None
    last_window_time = 0.0
    last_process_time = 0.0
    min_process_interval = 1.0 / TARGET_PROCESS_HZ
    last_stats, last_window_span, drowsy_label, drowsy_conf = None, 0.0, "warming up", 0.0
    distract_label, distract_conf = "Front", 0.0

    print("press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = apply_crop(frame, crop)
        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if now - last_process_time < min_process_interval:
            continue
        last_process_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detect(landmarker, rgb, now * 1000.0)

        if result.face_landmarks:
            face_idx = 0
            landmarks = result.face_landmarks[face_idx]

            raw_3d = landmarks_to_array(landmarks)
            gaze_ear = gaze_compute_ear(raw_3d[:, :2])
            corrected_2d = apply_gaze_bias(raw_3d, R, t)
            flat = corrected_2d[SELECTED_LANDMARKS].flatten().astype(np.float32)
            smooth_flat = lm_smoother.update(flat, gaze_ear)
            raw_probs = run_gaze_models(smooth_flat, gaze_models)
            raw_probs[1], raw_probs[2] = raw_probs[2].copy(), raw_probs[1].copy()
            distract_idx = pred_smoother.update(raw_probs)
            distract_label = LABEL_MAP[distract_idx]
            distract_conf = pred_smoother.confidence

            pts = face_points(landmarks, w, h)
            pose = rotation_from_result(result, face_idx)
            if pose is not None:
                pitch, yaw, roll = pose
                ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                mar = mouth_aspect_ratio(pts)
                dt = max(now - prev_frame_time, MIN_DT_SEC) if prev_frame_time is not None else 1.0 / TARGET_PROCESS_HZ
                pitch_vel = 0.0 if prev_pitch is None else circular_diff(pitch, prev_pitch) / dt
                yaw_vel = 0.0 if prev_yaw is None else circular_diff(yaw, prev_yaw) / dt
                prev_pitch, prev_yaw = pitch, yaw

                row = {"time_sec": now, "ear_raw": ear, "mar_raw": mar,
                       "pitch": pitch, "yaw": yaw, "roll": roll,
                       "pitch_vel": pitch_vel, "yaw_vel": yaw_vel}
                if buffer_start_time is None:
                    buffer_start_time = now
                rolling_buffer.append(row)
            prev_frame_time = now

            if args.show_landmarks:
                for x, y, _ in pts.values():
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            if args.debug_view:
                raw_panel = draw_dense_panel(raw_3d[:, :2], DEBUG_PANEL_SIZE, "RAW")
                corrected_panel = draw_dense_panel(corrected_2d, DEBUG_PANEL_SIZE, "CAMERA-BIAS CORRECTED")
                frontal_panel = draw_frontal_panel(normalize_landmarks(pts), DEBUG_PANEL_SIZE, "FRONTAL (debug only)")
                cv2.imshow("Debug View", np.hstack([raw_panel, corrected_panel, frontal_panel]))

        while rolling_buffer and now - rolling_buffer[0]["time_sec"] > WINDOW_SEC:
            rolling_buffer.popleft()
        if not rolling_buffer:
            buffer_start_time = None

        window_span = (rolling_buffer[-1]["time_sec"] - rolling_buffer[0]["time_sec"]) if len(rolling_buffer) > 1 else 0.0
        buffer_age = (now - buffer_start_time) if buffer_start_time is not None else 0.0
        if buffer_age >= WINDOW_SEC and now - last_window_time >= STRIDE_SEC:
            last_stats, drowsy_label, drowsy_conf = classify_drowsy_window(
                rolling_buffer, window_span, baseline, scaler, feature_columns,
                model=drowsy_model, ensemble_members=ensemble_members)
            last_window_span = window_span
            last_window_time = now

        panel = draw_panel(w, last_stats, last_window_span, drowsy_label, drowsy_conf, distract_label, distract_conf)
        combined = np.vstack([frame, panel])
        cv2.imshow("Driver Monitor", combined)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
