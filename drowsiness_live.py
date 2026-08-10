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

OUTPUT_DIR = "outputs"

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

FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

PANEL_SIZE = 480
LABEL_NAMES = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

CALIBRATION_SEC = max(8.0, WINDOW_SEC)
LIVE_CAPTURE_WIDTH, LIVE_CAPTURE_HEIGHT = 640, 480
TARGET_PROCESS_HZ = 10.0
LIVE_DROWSY_CONFIDENCE_THRESHOLD = 0.7
ENSEMBLE_TOP_N = 3

# num_faces is a fixed cap MediaPipe needs at creation time (no "unlimited" option) -- 8 covers any
# realistic vehicle occupancy. Driver selection still runs over however many are actually found.
LIVE_NUM_FACES = 8
CROP_SIZE = 300

MIN_DT_SEC = 1.0 / 60.0
MIN_RATE_DISPLAY_SEC = WINDOW_SEC * 0.95

LANDMARK_IDS = {
    "nose_tip": 1, "chin": 152,
    "left_eye_outer": 33, "left_eye_top1": 160, "left_eye_top2": 158,
    "left_eye_inner": 133, "left_eye_bottom1": 153, "left_eye_bottom2": 144,
    "right_eye_inner": 362, "right_eye_top1": 385, "right_eye_top2": 387,
    "right_eye_outer": 263, "right_eye_bottom1": 373, "right_eye_bottom2": 380,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top_outer": 13, "mouth_bottom_outer": 14,
    "mouth_top_left": 81, "mouth_bottom_left": 178,
    "mouth_top_right": 311, "mouth_bottom_right": 402,
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


def _ensure_face_landmarker_model():
    if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_LANDMARKER_MODEL_PATH), exist_ok=True)
        print(f"[setup] downloading face_landmarker.task model to {FACE_LANDMARKER_MODEL_PATH} ...")
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


def _open_face_landmarker(num_faces=LIVE_NUM_FACES):
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


def _detect(landmarker, rgb_frame, timestamp_ms):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, int(timestamp_ms))


def _landmark_px(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z * w)


def _rotation_from_result(result, face_idx=0):
    if not result.facial_transformation_matrixes or face_idx >= len(result.facial_transformation_matrixes):
        return None
    rmat = np.array(result.facial_transformation_matrixes[face_idx])[:3, :3]
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll, rmat


def _dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def eye_aspect_ratio(pts, eye_names):
    p1, p2, p3, p4, p5, p6 = (pts[n] for n in eye_names)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def mouth_aspect_ratio(pts):
    vertical = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL_PAIRS)
    horizontal = _dist(pts[MOUTH_HORIZONTAL[0]], pts[MOUTH_HORIZONTAL[1]])
    return vertical / (3.0 * horizontal)


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


def _center_on_nose(pts):
    rot_pts = _to_rotation_space(pts)
    nose = np.array(rot_pts["nose_tip"])
    return {name: tuple(np.array(p) - nose) for name, p in rot_pts.items()}


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


def _draw_metrics_panel(width, stats, duration_sec, extra_lines=None):
    def _rate_str(count):
        if duration_sec < MIN_RATE_DISPLAY_SEC:
            return "..collecting.."
        return f"{count / duration_sec:.2f}/s"

    lines = [
        f"EAR: {stats['mean_ear']:.2f}   MAR: {stats['mean_mar']:.2f}   PERCLOS: {stats['perclos'] * 100:.1f}%",
        f"Blink Rate: {_rate_str(stats['blink_count'])}   Blink Count: {stats['blink_count']}",
        f"Yawn Rate: {_rate_str(stats['yawn_count'])}   Yawn Count: {stats['yawn_count']}",
        f"Nod Rate: {_rate_str(stats['nod_count'])}   Nod Count: {stats['nod_count']}",
        f"Pitch Velocity: avg {stats['avg_pitch_velocity']:.0f} / peak {stats['max_pitch_velocity']:.0f} deg/s",
    ]
    lines.extend(extra_lines or [])
    panel = np.zeros((35 * len(lines) + 15, width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def _circular_diff(a2, a1):
    return (a2 - a1 + 180.0) % 360.0 - 180.0


def compute_baseline(frame_df):
    return {
        "ear_baseline": float(frame_df["ear_raw"].median()),
        "mar_baseline": float(frame_df["mar_raw"].median()),
        "pitch_baseline": float(frame_df["pitch"].median()),
        "yaw_baseline": float(frame_df["yaw"].median()),
        "roll_baseline": float(frame_df["roll"].median()),
    }


def apply_baseline(frame_df, baseline):
    df = frame_df.copy()
    df["ear"] = (df["ear_raw"] / baseline["ear_baseline"]).clip(upper=EAR_CLIP_MAX)
    df["mar"] = (df["mar_raw"] / baseline["mar_baseline"]).clip(upper=MAR_CLIP_MAX)
    df["pitch_norm"] = _circular_diff(df["pitch"].values, baseline["pitch_baseline"])
    df["yaw_norm"] = _circular_diff(df["yaw"].values, baseline["yaw_baseline"])
    df["roll_norm"] = _circular_diff(df["roll"].values, baseline["roll_baseline"])
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


def _summarize_window(chunk, fps):
    duration = len(chunk) / fps
    ear, mar = chunk["ear"].values, chunk["mar"].values
    pitch, pitch_vel = chunk["pitch_norm"].values, chunk["pitch_vel"].values

    min_blink_frames = max(1, round(MIN_BLINK_DURATION_SEC * fps))
    blink_merge_gap_frames = max(1, round(BLINK_MERGE_GAP_SEC * fps))
    min_yawn_frames = max(1, round(MIN_YAWN_DURATION_SEC * fps))
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


def _decide_with_drowsy_threshold(probs_row, classes, threshold=LIVE_DROWSY_CONFIDENCE_THRESHOLD):
    class_probs = dict(zip(classes, probs_row))
    drowsy_p = class_probs.get(2, 0.0)
    if drowsy_p >= threshold:
        return 2, drowsy_p
    non_drowsy = {label: p for label, p in class_probs.items() if label != 2}
    if not non_drowsy:
        return 2, drowsy_p
    label = max(non_drowsy, key=non_drowsy.get)
    return label, non_drowsy[label]


def _load_ensemble(all_models_path, top_n=ENSEMBLE_TOP_N):
    payload = joblib.load(all_models_path)
    ranked = sorted(payload["models"].items(), key=lambda kv: kv[1]["cv_f1"], reverse=True)[:top_n]
    members = [(name, info["model"]) for name, info in ranked]
    print("Ensemble members:", ", ".join(f"{name} (cv f1={info['cv_f1']:.3f})" for name, info in ranked))
    return payload["scaler"], payload["feature_columns"], members


def _ensemble_predict_proba(members, X):
    classes = sorted(set(c for _, model in members for c in model.classes_))
    acc = {c: 0.0 for c in classes}
    for _, model in members:
        probs = model.predict_proba(X)[0]
        for c, p in zip(model.classes_, probs):
            acc[c] += p / len(members)
    return classes, np.array([acc[c] for c in classes])


def _select_driver_face(face_landmarks_list):
    # rightmost + closest to camera == the driver, for a cabin-facing camera, left-hand-drive
    if len(face_landmarks_list) <= 1:
        return 0
    nose_idx = LANDMARK_IDS["nose_tip"]
    scores = [lm[nose_idx].x - lm[nose_idx].z for lm in face_landmarks_list]
    return int(np.argmax(scores))


def _status_panel(width, text):
    panel = np.zeros((50, width, 3), dtype=np.uint8)
    cv2.putText(panel, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
    return panel


def run_live(model_path, camera_id=0, ensemble=False,
             ensemble_path=os.path.join(OUTPUT_DIR, "all_models.pkl"), ensemble_top_n=ENSEMBLE_TOP_N):
    ensemble_members = None
    if ensemble:
        scaler, feature_columns, ensemble_members = _load_ensemble(ensemble_path, ensemble_top_n)
    else:
        bundle = joblib.load(model_path)
        model, scaler, feature_columns = bundle["model"], bundle["scaler"], bundle["feature_columns"]

    face_landmarker = _open_face_landmarker()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Press 'c' to calibrate (look forward, stay neutral), 'q' to quit.")

    baseline = None
    calibrating_until = None
    calib_rows = []
    calib_positions = []
    fixed_crop = None
    rolling_buffer = deque()
    buffer_start_time = None

    prev_pitch = prev_yaw = prev_roll = prev_frame_time = None
    last_stats, last_window_span, last_extra_lines = None, 0.0, None
    last_window_time = 0.0
    last_process_time = 0.0
    min_process_interval = 1.0 / TARGET_PROCESS_HZ

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()

        if fixed_crop is not None:
            x0, y0, x1, y1 = fixed_crop
            frame = frame[y0:y1, x0:x1]

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            calibrating_until = now + CALIBRATION_SEC
            calib_rows = []
            calib_positions = []
            fixed_crop = None
            print(f"Calibrating for {CALIBRATION_SEC:.0f}s, look forward and stay neutral...")

        if now - last_process_time < min_process_interval:
            continue
        last_process_time = now

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _detect(face_landmarker, rgb, now * 1000.0)

        norm_pts, raw_pts_centered = {}, {}
        if result.face_landmarks:
            face_idx = _select_driver_face(result.face_landmarks)
            landmarks = result.face_landmarks[face_idx]
            pts = {name: _landmark_px(landmarks[idx], w, h) for name, idx in LANDMARK_IDS.items()}
            for x, y, _ in pts.values():
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            pose = _rotation_from_result(result, face_idx)
            if pose is not None:
                pitch, yaw, roll, _ = pose
                norm_pts = normalize_landmarks(pts)
                raw_pts_centered = _center_on_nose(pts)
                ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                mar = mouth_aspect_ratio(pts)
                dt = max(now - prev_frame_time, MIN_DT_SEC) if prev_frame_time is not None else 1.0 / TARGET_PROCESS_HZ
                pitch_vel = 0.0 if prev_pitch is None else _circular_diff(pitch, prev_pitch) / dt
                yaw_vel = 0.0 if prev_yaw is None else _circular_diff(yaw, prev_yaw) / dt
                roll_vel = 0.0 if prev_roll is None else _circular_diff(roll, prev_roll) / dt
                prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll

                row = {"time_sec": now, "ear_raw": ear, "mar_raw": mar,
                       "pitch": pitch, "yaw": yaw, "roll": roll,
                       "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel}
                if buffer_start_time is None:
                    buffer_start_time = now
                rolling_buffer.append(row)
                if calibrating_until is not None:
                    calib_rows.append(row)
                    calib_positions.append(pts["nose_tip"][:2])
            prev_frame_time = now

        while rolling_buffer and now - rolling_buffer[0]["time_sec"] > WINDOW_SEC:
            rolling_buffer.popleft()

        if calibrating_until is not None and now >= calibrating_until:
            if calib_rows:
                baseline = compute_baseline(pd.DataFrame(calib_rows))
                print("Calibration complete:", baseline)
            if calib_positions:
                cx = float(np.mean([p[0] for p in calib_positions]))
                cy = float(np.mean([p[1] for p in calib_positions]))
                half = CROP_SIZE // 2
                x0 = int(np.clip(cx - half, 0, max(0, w - CROP_SIZE)))
                y0 = int(np.clip(cy - half, 0, max(0, h - CROP_SIZE)))
                fixed_crop = (x0, y0, min(x0 + CROP_SIZE, w), min(y0 + CROP_SIZE, h))
                print(f"Fixed crop set at {fixed_crop} -- no further continuous face tracking.")
            calibrating_until, calib_rows, calib_positions = None, [], []

        raw_panel = _draw_points_panel(PANEL_SIZE, raw_pts_centered, "RAW 3D")
        frontal_panel = _draw_points_panel(PANEL_SIZE, norm_pts, "FRONTAL (normalized)")
        frame_resized = cv2.resize(frame, (int(w * PANEL_SIZE / h), PANEL_SIZE))
        combined = np.hstack([frame_resized, raw_panel, frontal_panel])

        window_span = (rolling_buffer[-1]["time_sec"] - rolling_buffer[0]["time_sec"]) if len(rolling_buffer) > 1 else 0.0
        buffer_age = (now - buffer_start_time) if buffer_start_time is not None else 0.0
        if baseline is not None and buffer_age >= WINDOW_SEC and now - last_window_time >= STRIDE_SEC:
            chunk_df = apply_baseline(pd.DataFrame(rolling_buffer), baseline)
            measured_fps = len(rolling_buffer) / window_span
            stats = _summarize_window(chunk_df, measured_fps)
            X = scaler.transform(pd.DataFrame([stats], columns=feature_columns))
            if ensemble_members is not None:
                classes, probs = _ensemble_predict_proba(ensemble_members, X)
            else:
                classes, probs = model.classes_, model.predict_proba(X)[0]
            pred, confidence = _decide_with_drowsy_threshold(probs, classes)
            classification = LABEL_NAMES.get(pred, str(pred))
            last_window_time = now
            last_extra_lines = [f"Prediction: {classification}   Confidence: {confidence * 100:.1f}%"]
            last_stats, last_window_span = stats, window_span

        if last_stats is not None:
            combined = np.vstack([combined, _draw_metrics_panel(combined.shape[1], last_stats,
                                                                  last_window_span, last_extra_lines)])
        else:
            if calibrating_until is not None:
                status = f"Calibrating... {max(0.0, calibrating_until - now):.1f}s left"
            elif baseline is None:
                status = "Press 'c' to calibrate"
            else:
                status = f"Warming up: {min(buffer_age, WINDOW_SEC):.1f}s / {WINDOW_SEC:.0f}s buffered"
            combined = np.vstack([combined, _status_panel(combined.shape[1], status)])
        cv2.imshow("Driver Drowsiness Monitor", combined)

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--model-path", default=os.path.join(OUTPUT_DIR, "best_model.pkl"))
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble-path", default=os.path.join(OUTPUT_DIR, "all_models.pkl"))
    parser.add_argument("--ensemble-top-n", type=int, default=ENSEMBLE_TOP_N)
    args = parser.parse_args()

    run_live(args.model_path, args.camera_id, ensemble=args.ensemble,
              ensemble_path=args.ensemble_path, ensemble_top_n=args.ensemble_top_n)
