import os
import sys
import glob
import json
import argparse
import contextlib
import urllib.request
from collections import deque

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from scipy.signal import find_peaks

OUTPUT_DIR = "outputs"

WINDOW_SEC = 10.0
STRIDE_SEC = 1.0

EAR_CLOSED_RATIO = 0.75   # Ersoy et al. 2026 (arXiv:2604.22479)
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
MOTION_GATE_DEG_S = 50.0  # p97 of real |pitch_vel|+|yaw_vel|; excluded from blink/yawn detection

EAR_CLIP_MAX = 2.0
MAR_CLIP_MAX = 50.0

MIN_WINDOW_FILL_RATIO = 0.5
MAX_WINDOW_SPAN_RATIO = 1.5

FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

PANEL_SIZE = 480

LABEL_NAMES = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

UTA_RLDD_ROOT = "UTA-RLDD"
UTA_RLDD_LABELS = {"0": 0, "5": 1, "10": 2}

NTHU_ROOT = "NTHU"
NTHU_CATEGORY_ALIASES = {
    "glasses": "glasses", "noglasses": "noglasses", "sunglasses": "sunglasses",
    "nightglasses": "night_glasses", "night_glasses": "night_glasses",
    "nightnoglasses": "night_noglasses", "night_noglasses": "night_noglasses",
}
NTHU_BASELINE_CATEGORY_PREFERENCE = ["noglasses", "glasses", "sunglasses", "night_glasses", "night_noglasses"]
NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH = {"yawning": 2, "slowBlinkWithNodding": 2}
NTHU_FRAME_LABEL_MAP = {0: 0, 1: 2}

WINDOW_FEATURE_COLUMNS = [
    "mean_ear", "std_ear", "min_ear", "max_ear",
    "blink_count", "blink_freq", "avg_blink_duration", "max_blink_duration", "perclos",
    "mean_mar", "max_mar", "std_mar", "yawn_count", "avg_yawn_duration",
    "mean_pitch", "std_pitch", "max_pitch_velocity", "avg_pitch_velocity",
    "pitch_oscillation_freq", "nod_count",
]

PROGRESSION_PLOT_FEATURES = ["mean_ear", "perclos", "mean_mar", "yawn_count", "mean_pitch", "blink_count", "nod_count"]
PROGRESSION_PLOT_EXAMPLES_PER_LABEL = 2
CLIPS_PER_LABEL_DEFAULT = 2
CLIP_WINDOWS_TARGET = 3

PREVIEW_WINDOW_NAME = "Building dataset - preview (press q to stop)"

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


def _uta_rldd_videos(root=UTA_RLDD_ROOT):
    videos = []
    for subject_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(subject_dir):
            continue
        subject_id = os.path.basename(subject_dir)
        for path in sorted(glob.glob(os.path.join(subject_dir, "*"))):
            prefix = os.path.basename(path).split(".")[0].split("_")[0]
            if prefix not in UTA_RLDD_LABELS:
                continue
            videos.append({
                "path": path,
                "subject_id": subject_id,
                "label": UTA_RLDD_LABELS[prefix],
                "is_baseline": prefix == "0",
                "frame_labels": None,
            })
    return videos


def _nthu_read_frame_labels(path):
    with open(path) as f:
        digits = f.read().strip()
    return np.array([NTHU_FRAME_LABEL_MAP.get(int(c), 0) for c in digits if c.isdigit()], dtype=np.int64)


def _nthu_videos(root=NTHU_ROOT):
    entries = {}
    for subject_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(subject_dir) or os.path.basename(subject_dir).startswith("_"):
            continue
        subject_id = os.path.basename(subject_dir)
        if not subject_id.isdigit():
            continue
        for category_dir in sorted(glob.glob(os.path.join(subject_dir, "*"))):
            if not os.path.isdir(category_dir):
                continue
            category = NTHU_CATEGORY_ALIASES.get(os.path.basename(category_dir), os.path.basename(category_dir))
            for fname in os.listdir(category_dir):
                fpath = os.path.join(category_dir, fname)
                name, ext = os.path.splitext(fname)
                if ext.lower() in (".avi", ".mp4"):
                    key = (subject_id, category, name)
                    entries.setdefault(key, {})["video"] = fpath
                elif ext.lower() == ".txt" and name.startswith(f"{subject_id}_") and name.endswith("_drowsiness"):
                    behavior = name[len(subject_id) + 1: -len("_drowsiness")]
                    key = (subject_id, category, behavior)
                    entries.setdefault(key, {})["drowsiness_txt"] = fpath

    videos = []
    for (subject_id, category, behavior), info in sorted(entries.items()):
        video_path = info.get("video")
        if video_path is None:
            continue

        drowsiness_txt = info.get("drowsiness_txt")
        if drowsiness_txt is not None:
            frame_labels = _nthu_read_frame_labels(drowsiness_txt)
            label = int(np.bincount(frame_labels).argmax()) if len(frame_labels) else 0
        elif behavior in NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH:
            frame_labels = None
            label = NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH[behavior]
        else:
            continue

        videos.append({
            "path": video_path, "subject_id": subject_id, "category": category, "behavior": behavior,
            "label": label, "frame_labels": frame_labels, "is_baseline": False,
        })

    by_subject = {}
    for v in videos:
        by_subject.setdefault(v["subject_id"], []).append(v)
    for subject_videos in by_subject.values():
        candidates = [v for v in subject_videos if v["behavior"] == "nonsleepyCombination"]
        if not candidates:
            candidates = [
                v for v in subject_videos
                if v["behavior"] == "mix" and v["frame_labels"] is not None and (v["frame_labels"] == 0).any()
            ]
        if not candidates:
            continue
        def _pref_rank(v):
            try:
                return NTHU_BASELINE_CATEGORY_PREFERENCE.index(v["category"])
            except ValueError:
                return len(NTHU_BASELINE_CATEGORY_PREFERENCE)
        min(candidates, key=_pref_rank)["is_baseline"] = True

    return videos


def get_video_list(dataset_name):
    if dataset_name == "uta_rldd":
        return _uta_rldd_videos()
    if dataset_name == "nthu":
        return _nthu_videos()
    raise ValueError(f"Unknown dataset: {dataset_name}")


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


def _open_face_landmarker(num_faces=1):
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


MIN_RATE_DISPLAY_SEC = WINDOW_SEC * 0.95


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


_EMPTY_CLIP_STATS = {"mean_ear": 0.0, "mean_mar": 0.0, "perclos": 0.0, "blink_count": 0,
                     "yawn_count": 0, "nod_count": 0, "avg_pitch_velocity": 0.0, "max_pitch_velocity": 0.0}


def _circular_diff(a2, a1):
    return (a2 - a1 + 180.0) % 360.0 - 180.0


def _draw_debug_overlay(frame, pts, ear, mar, pitch):
    for x, y, _ in pts.values():
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}  Pitch: {pitch:.1f}deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def extract_raw_and_frame_features(video_path, subject_id, label, face_landmarker, show=False, max_seconds=None,
                                    frame_labels=None):
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(max_seconds * fps) if max_seconds is not None else None

        video_id = f"{os.path.basename(os.path.dirname(video_path))}/{os.path.basename(video_path)}"

        raw_rows, frame_rows = [], []
        prev_pitch = prev_yaw = prev_roll = None
        prev_detected_frame_idx = None
        frame_idx = 0
        aborted = False

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = _detect(face_landmarker, rgb, frame_idx / fps * 1000.0)

            display_frame = frame
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                pts = {name: _landmark_px(landmarks[idx], w, h) for name, idx in LANDMARK_IDS.items()}
                pose = _rotation_from_result(result)

                if pose is not None:
                    pitch, yaw, roll, _ = pose
                    norm_pts = normalize_landmarks(pts)
                    ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                    mar = mouth_aspect_ratio(pts)
                    dt = (frame_idx - prev_detected_frame_idx) / fps if prev_detected_frame_idx is not None else 1.0 / fps
                    pitch_vel = 0.0 if prev_pitch is None else _circular_diff(pitch, prev_pitch) / dt
                    yaw_vel = 0.0 if prev_yaw is None else _circular_diff(yaw, prev_yaw) / dt
                    roll_vel = 0.0 if prev_roll is None else _circular_diff(roll, prev_roll) / dt
                    prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll
                    prev_detected_frame_idx = frame_idx

                    frame_label = label
                    if frame_labels is not None and len(frame_labels) > 0:
                        frame_label = int(frame_labels[min(frame_idx, len(frame_labels) - 1)])

                    raw_row = {"subject_id": subject_id, "video": video_id, "label": frame_label,
                               "frame": frame_idx, "time_sec": frame_idx / fps}
                    for name in LANDMARK_IDS:
                        x, y, z = pts[name]
                        nx, ny, nz = norm_pts[name]
                        raw_row[f"{name}_x"], raw_row[f"{name}_y"], raw_row[f"{name}_z"] = x, y, z
                        raw_row[f"{name}_norm_x"], raw_row[f"{name}_norm_y"], raw_row[f"{name}_norm_z"] = nx, ny, nz
                    raw_rows.append(raw_row)

                    frame_rows.append({
                        "subject_id": subject_id, "video": video_id, "label": frame_label,
                        "frame": frame_idx, "time_sec": frame_idx / fps,
                        "ear_raw": ear, "mar_raw": mar,
                        "pitch": pitch, "yaw": yaw, "roll": roll,
                        "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel,
                    })

                    display_frame = _draw_debug_overlay(frame.copy(), pts, ear, mar, pitch)

            if show:
                cv2.imshow(PREVIEW_WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True

            frame_idx += 1
            if aborted:
                break

        cap.release()
    return pd.DataFrame(raw_rows), pd.DataFrame(frame_rows), fps, aborted


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

    window_label = int(pd.Series(chunk["label"].values).mode().iloc[0])

    return {
        "subject_id": chunk["subject_id"].iloc[0],
        "video": chunk["video"].iloc[0],
        "label": window_label,
        "start_time": float(chunk["time_sec"].iloc[0]),
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


def compute_window_features(frame_df, fps, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    rows = frame_df.reset_index(drop=True)
    window_frames = int(window_sec * fps)
    stride_frames = max(1, int(stride_sec * fps))

    windows = []
    for start in range(0, max(len(rows) - window_frames, 0) + 1, stride_frames):
        chunk = rows.iloc[start:start + window_frames]
        if len(chunk) < window_frames * MIN_WINDOW_FILL_RATIO:
            continue
        actual_span = chunk["time_sec"].iloc[-1] - chunk["time_sec"].iloc[0]
        if actual_span > window_sec * MAX_WINDOW_SPAN_RATIO:
            continue
        windows.append(_summarize_window(chunk, fps))
    return pd.DataFrame(windows)


def rebuild_windows(dataset_name, output_dir=OUTPUT_DIR):
    frame_path = os.path.join(output_dir, f"{dataset_name}_frame_level.csv")
    window_path = os.path.join(output_dir, f"{dataset_name}_window.csv")
    baselines_path = os.path.join(output_dir, f"{dataset_name}_baselines.json")
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"{frame_path} not found -- run a full `build_dataset` first.")

    frame_df = pd.read_csv(frame_path, dtype={"subject_id": str})
    baselines = _load_json(baselines_path, default={})

    all_windows = []
    for (subject_id, video), g in frame_df.groupby(["subject_id", "video"], sort=False):
        if subject_id not in baselines:
            continue
        g = g.sort_values("time_sec").reset_index(drop=True)
        dt = np.median(np.diff(g["time_sec"].values)) if len(g) > 1 else 1.0 / 30.0
        fps = 1.0 / dt if dt > 0 else 30.0
        bdf = apply_baseline(g, baselines[subject_id])
        wdf = compute_window_features(bdf, fps)
        if len(wdf):
            wdf["subject_id"] = subject_id
            all_windows.append(wdf)

    if not all_windows:
        print(f"[rewindow] no windows produced for {dataset_name} -- nothing written")
        return
    window_df = pd.concat(all_windows, ignore_index=True)
    window_df.to_csv(window_path, index=False)
    print(f"[rewindow] WINDOW_SEC={WINDOW_SEC} STRIDE_SEC={STRIDE_SEC} -> "
          f"{len(window_df)} windows written to {window_path}")


def _load_json(path, default):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _append_csv(df, path):
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


class _StopBuilding(Exception):
    pass


def build_dataset(dataset_name, output_dir=OUTPUT_DIR, show=False, make_clips=True,
                   clips_per_label=CLIPS_PER_LABEL_DEFAULT, clip_windows=CLIP_WINDOWS_TARGET):
    os.makedirs(output_dir, exist_ok=True)
    videos = get_video_list(dataset_name)

    subjects = {}
    for v in videos:
        subjects.setdefault(v["subject_id"], []).append(v)

    raw_path = os.path.join(output_dir, f"{dataset_name}_raw_landmarks.csv")
    frame_path = os.path.join(output_dir, f"{dataset_name}_frame_level.csv")
    window_path = os.path.join(output_dir, f"{dataset_name}_window.csv")
    baselines_path = os.path.join(output_dir, f"{dataset_name}_baselines.json")
    progress_path = os.path.join(output_dir, f"{dataset_name}_progress.json")

    baselines = _load_json(baselines_path, default={})
    completed = set(_load_json(progress_path, default={"completed_videos": []})["completed_videos"])

    if completed:
        print(f"[resume] {len(completed)} video(s) already processed, skipping those")

    try:
        for subject_id, subject_videos in subjects.items():
            if all(v["path"] in completed for v in subject_videos):
                continue

            baseline_video = next((v for v in subject_videos if v["is_baseline"]), None)
            if baseline_video is None:
                print(f"[skip] subject {subject_id}: no baseline (awake) video found")
                continue

            ordered_videos = [baseline_video] + [v for v in subject_videos if v is not baseline_video]
            subject_has_baseline = subject_id in baselines

            for v in ordered_videos:
                if v["path"] in completed:
                    continue

                face_landmarker = _open_face_landmarker()
                try:
                    raw_df, frame_df, fps, aborted = extract_raw_and_frame_features(
                        v["path"], subject_id, v["label"], face_landmarker, show=show,
                        frame_labels=v.get("frame_labels"))
                finally:
                    face_landmarker.close()
                if aborted:
                    raise _StopBuilding()

                if frame_df.empty:
                    print(f"[skip] {v['path']}: no face detected")
                    completed.add(v["path"])
                    _save_json(progress_path, {"completed_videos": sorted(completed)})
                    continue

                if not subject_has_baseline and v is baseline_video:
                    alert_frames = frame_df[frame_df["label"] == 0]
                    baselines[subject_id] = compute_baseline(alert_frames if not alert_frames.empty else frame_df)
                    _save_json(baselines_path, baselines)
                    subject_has_baseline = True

                if not subject_has_baseline:
                    print(f"[skip] subject {subject_id}: baseline video had no usable frames")
                    break

                frame_df = apply_baseline(frame_df, baselines[subject_id])
                window_df = compute_window_features(frame_df, fps)

                _append_csv(raw_df, raw_path)
                _append_csv(frame_df, frame_path)
                _append_csv(window_df, window_path)

                completed.add(v["path"])
                _save_json(progress_path, {"completed_videos": sorted(completed)})
                print(f"[ok] {v['path']} -> {len(frame_df)} frames, {len(window_df)} windows")
    except (_StopBuilding, KeyboardInterrupt):
        print("\nStopped early. Progress has been saved -- re-run the same command to resume.")
    finally:
        if show:
            cv2.destroyAllWindows()

    generate_progression_plots(dataset_name, output_dir)
    if make_clips:
        make_annotated_clips(dataset_name, output_dir, clips_per_label=clips_per_label, clip_windows=clip_windows)


def generate_progression_plots(dataset_name, output_dir=OUTPUT_DIR, max_examples_per_label=PROGRESSION_PLOT_EXAMPLES_PER_LABEL):
    window_path = os.path.join(output_dir, f"{dataset_name}_window.csv")
    if not os.path.isfile(window_path):
        return
    df = pd.read_csv(window_path, dtype={"subject_id": str})
    if df.empty:
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    features = [f for f in PROGRESSION_PLOT_FEATURES if f in df.columns]

    for label in sorted(df["label"].unique()):
        label_df = df[df["label"] == label]
        subject_ids = sorted(label_df["subject_id"].unique())
        if not subject_ids:
            continue
        n_pick = min(max_examples_per_label, len(subject_ids))
        pick_positions = sorted(set(np.linspace(0, len(subject_ids) - 1, n_pick).round().astype(int)))
        chosen_subjects = [subject_ids[i] for i in pick_positions]

        for subject_id in chosen_subjects:
            subject_label_df = label_df[label_df["subject_id"] == subject_id]
            video = subject_label_df["video"].iloc[0]
            clip = subject_label_df[subject_label_df["video"] == video].sort_values("start_time")
            if len(clip) < 2:
                continue
            fig, axes = plt.subplots(len(features), 1, figsize=(9, 2.0 * len(features)), sharex=True)
            axes = np.atleast_1d(axes)
            label_name = LABEL_NAMES.get(label, str(label))
            fig.suptitle(f"subject {subject_id} - {video} - {label_name}")
            for ax, feat in zip(axes, features):
                ax.plot(clip["start_time"], clip[feat])
                ax.set_ylabel(feat, fontsize=8)
            axes[-1].set_xlabel("time (s)")
            fig.tight_layout()
            safe_video = str(video).replace(".", "_").replace("/", "_")
            out_path = os.path.join(plots_dir, f"{dataset_name}_{subject_id}_{safe_video}_progression.png")
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            print(f"[plots] saved {out_path}")


def _open_video_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if writer.isOpened():
        return writer
    print("[clip] H.264 encoder unavailable, falling back to mp4v (won't preview in browser-based viewers)")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def _longest_run(values, target):
    best_start = best_len = cur_start = cur_len = 0
    for i, v in enumerate(values):
        if v == target:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    return best_start, best_len


def _record_annotated_clip(video_path, start_frame, num_frames, baseline, label_name, output_path):
    face_landmarker = _open_face_landmarker()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    window_frames = int(WINDOW_SEC * fps)

    clip_buffer = deque(maxlen=window_frames)
    prev_pitch = prev_yaw = prev_roll = None

    writer = None
    try:
        with _suppress_native_stderr():
            for clip_frame_idx in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = _detect(face_landmarker, rgb, clip_frame_idx / fps * 1000.0)

                raw_pts, frontal_pts = {}, {}
                stats = None
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    pts = {name: _landmark_px(landmarks[idx], w, h) for name, idx in LANDMARK_IDS.items()}
                    for x, y, _ in pts.values():
                        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                    pose = _rotation_from_result(result)
                    if pose is not None:
                        pitch, yaw, roll, _ = pose
                        frontal_pts = normalize_landmarks(pts)
                        raw_pts = _center_on_nose(pts)

                        ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                        mar = mouth_aspect_ratio(pts)
                        dt = 1.0 / fps
                        pitch_vel = 0.0 if prev_pitch is None else _circular_diff(pitch, prev_pitch) / dt
                        yaw_vel = 0.0 if prev_yaw is None else _circular_diff(yaw, prev_yaw) / dt
                        roll_vel = 0.0 if prev_roll is None else _circular_diff(roll, prev_roll) / dt
                        prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll

                        clip_buffer.append({
                            "subject_id": "clip", "video": "clip", "label": 0,
                            "frame": clip_frame_idx, "time_sec": clip_frame_idx / fps,
                            "ear_raw": ear, "mar_raw": mar, "pitch": pitch, "yaw": yaw, "roll": roll,
                            "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel,
                        })
                        if len(clip_buffer) >= 2:
                            clip_df = apply_baseline(pd.DataFrame(clip_buffer), baseline)
                            stats = _summarize_window(clip_df, fps)

                cv2.putText(frame, f"Label: {label_name}", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                raw_panel = _draw_points_panel(PANEL_SIZE, raw_pts, "RAW 3D")
                frontal_panel = _draw_points_panel(PANEL_SIZE, frontal_pts, "FRONTAL (normalized)")
                frame_resized = cv2.resize(frame, (int(w * PANEL_SIZE / h), PANEL_SIZE))
                combined = np.hstack([frame_resized, raw_panel, frontal_panel])
                metrics_panel = _draw_metrics_panel(combined.shape[1], stats or _EMPTY_CLIP_STATS,
                                                     len(clip_buffer) / fps)
                combined = np.vstack([combined, metrics_panel])

                if writer is None:
                    writer = _open_video_writer(output_path, fps, (combined.shape[1], combined.shape[0]))
                writer.write(combined)
    finally:
        cap.release()
        face_landmarker.close()
        if writer is not None:
            writer.release()


def make_annotated_clips(dataset_name, output_dir=OUTPUT_DIR, clips_per_label=CLIPS_PER_LABEL_DEFAULT,
                          clip_windows=CLIP_WINDOWS_TARGET):
    baselines_path = os.path.join(output_dir, f"{dataset_name}_baselines.json")
    baselines = _load_json(baselines_path, default={})
    if not baselines:
        print(f"[clip] no baselines found in {baselines_path} yet -- skipping annotated clips "
              f"(run a build first: python dataset_builder.py --dataset {dataset_name})")
        return []

    videos = [v for v in get_video_list(dataset_name) if v["subject_id"] in baselines]
    by_label = {}
    for v in videos:
        by_label.setdefault(v["label"], []).append(v)

    annotated_dir = os.path.join(output_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    clip_sec = clip_windows * WINDOW_SEC

    saved = []
    for label, label_videos in sorted(by_label.items()):
        label_name = LABEL_NAMES.get(label, str(label))

        first_video_per_subject = {}
        for v in label_videos:
            first_video_per_subject.setdefault(v["subject_id"], v)
        subject_ids = sorted(first_video_per_subject)
        if not subject_ids:
            continue
        n_pick = min(clips_per_label, len(subject_ids))
        pick_positions = sorted(set(np.linspace(0, len(subject_ids) - 1, n_pick).round().astype(int)))
        chosen_subject_ids = [subject_ids[i] for i in pick_positions]

        for count, subject_id in enumerate(chosen_subject_ids):
            v = first_video_per_subject[subject_id]

            with _suppress_native_stderr():
                cap = cv2.VideoCapture(v["path"])
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            if total_frames <= 0:
                continue
            num_frames = min(total_frames, int(clip_sec * fps))

            frame_labels = v.get("frame_labels")
            if frame_labels is not None and len(frame_labels) > 0:
                run_start, run_len = _longest_run(frame_labels, label)
                if run_len < 1:
                    continue
                start_frame = min(run_start + max(0, run_len - num_frames) // 2, max(0, total_frames - num_frames))
            else:
                start_frame = max(0, (total_frames - num_frames) // 2)

            safe_video = os.path.splitext(os.path.basename(v["path"]))[0]
            out_name = f"{dataset_name}_{v['subject_id']}_{safe_video}_{label_name.replace(' ', '')}_{count}.mp4"
            out_path = os.path.join(annotated_dir, out_name)

            print(f"[clip] subject {v['subject_id']} {os.path.basename(v['path'])} ({label_name}) "
                  f"frames {start_frame}-{start_frame + num_frames} -> {out_path}")
            _record_annotated_clip(v["path"], start_frame, num_frames, baselines[v["subject_id"]],
                                    label_name, out_path)
            saved.append(out_path)

    print(f"[clip] saved {len(saved)} annotated clip(s) to {annotated_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["uta_rldd", "nthu"], default="uta_rldd")
    parser.add_argument("--show", action="store_true",
                         help="Show a live preview window while processing (press q to stop early)")
    parser.add_argument("--make-clips", action="store_true",
                         help="Only (re)generate annotated debug clips from an already-built dataset; "
                              "a normal build already does this at the end.")
    parser.add_argument("--no-clips", action="store_true",
                         help="Skip automatic annotated-clip generation at the end of a build")
    parser.add_argument("--clips-per-label", type=int, default=CLIPS_PER_LABEL_DEFAULT)
    parser.add_argument("--clip-windows", type=int, default=CLIP_WINDOWS_TARGET,
                         help="How many full WINDOW_SEC-length windows each clip should span end-to-end "
                              f"(default: {CLIP_WINDOWS_TARGET}, i.e. {CLIP_WINDOWS_TARGET * WINDOW_SEC:.0f}s)")
    parser.add_argument("--rewindow", action="store_true",
                         help="Only recompute <dataset>_window.csv from the already-extracted "
                              "<dataset>_frame_level.csv using the current WINDOW_SEC/STRIDE_SEC. "
                              "Fast (no MediaPipe re-run).")
    args = parser.parse_args()

    if args.rewindow:
        rebuild_windows(args.dataset)
    elif args.make_clips:
        make_annotated_clips(args.dataset, clips_per_label=args.clips_per_label, clip_windows=args.clip_windows)
    else:
        build_dataset(args.dataset, show=args.show, make_clips=not args.no_clips,
                       clips_per_label=args.clips_per_label, clip_windows=args.clip_windows)
