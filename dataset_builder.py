"""
Builds RAW / FRAME-LEVEL / WINDOW datasets from driver videos for drowsiness classification.
Supports UTA-RLLD now; NTHU loading is left for later since its layout isn't finalized.
"""

import os
import glob
import json
import argparse

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"

WINDOW_SEC = 5.0
STRIDE_SEC = 1.0
BASELINE_CALIBRATION_SEC = 8.0  # first N seconds of the Alert video used for per-subject baseline

# ear/mar are normalized by each subject's own awake-baseline, so these ratios
# are meant to be roughly subject-independent (1.0 == baseline behavior).
EAR_CLOSED_RATIO = 0.7
MAR_YAWN_RATIO = 1.8
MIN_BLINK_FRAMES = 2
MIN_YAWN_FRAMES = 8
NOD_VELOCITY_DEG_S = 25.0
NOD_MIN_GAP_SEC = 0.5

UTA_RLLD_ROOT = "UTA-RLLD"
UTA_RLLD_LABELS = {"0": 0, "5": 1, "10": 2}  # Alert, Low Vigilant, Drowsy
NTHU_ROOT = "NTHU"

WINDOW_FEATURE_COLUMNS = [
    "mean_ear", "std_ear", "min_ear", "max_ear",
    "blink_count", "blink_freq", "avg_blink_duration", "max_blink_duration", "perclos",
    "mean_mar", "max_mar", "std_mar", "yawn_count", "avg_yawn_duration",
    "mean_pitch", "std_pitch", "max_pitch_velocity", "avg_pitch_velocity",
    "pitch_oscillation_freq", "nod_count",
]

mp_face_mesh = mp.solutions.face_mesh

# Subset of the 468 face-mesh landmarks actually needed for EAR/MAR/head-pose.
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

# Standard 6-point generic 3D face model (mm) used for solvePnP head-pose estimation.
POSE_LANDMARK_NAMES = ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left", "mouth_right")
POSE_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset video listing (the only dataset-specific logic in this file)
# ---------------------------------------------------------------------------
def _uta_rldd_videos(root=UTA_RLLD_ROOT):
    videos = []
    for subject_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(subject_dir):
            continue
        subject_id = os.path.basename(subject_dir)
        for path in sorted(glob.glob(os.path.join(subject_dir, "*"))):
            prefix = os.path.basename(path).split(".")[0].split("_")[0]
            if prefix not in UTA_RLLD_LABELS:
                continue
            videos.append({
                "path": path,
                "subject_id": subject_id,
                "label": UTA_RLLD_LABELS[prefix],
                "is_baseline": prefix == "0",
            })
    return videos


def _nthu_videos(root=NTHU_ROOT):
    raise NotImplementedError("NTHU dataset file layout is not finalized yet.")


def get_video_list(dataset_name):
    if dataset_name == "uta_rldd":
        return _uta_rldd_videos()
    if dataset_name == "nthu":
        return _nthu_videos()
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Landmarks / features (shared by dataset building and live inference)
# ---------------------------------------------------------------------------
def _open_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=False, refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )


def _landmark_px(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z * w)


def _dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def eye_aspect_ratio(pts, eye_names):
    p1, p2, p3, p4, p5, p6 = (pts[n] for n in eye_names)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4))


def mouth_aspect_ratio(pts):
    vertical = sum(_dist(pts[a], pts[b]) for a, b in MOUTH_VERTICAL_PAIRS)
    horizontal = _dist(pts[MOUTH_HORIZONTAL[0]], pts[MOUTH_HORIZONTAL[1]])
    return vertical / (3.0 * horizontal)


def estimate_head_pose(pts, w, h):
    image_points = np.array([pts[n][:2] for n in POSE_LANDMARK_NAMES], dtype=np.float64)
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(POSE_MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll, rmat, tvec


def normalize_landmarks(pts, rmat):
    """Undo head rotation around the nose tip to get a frontal-pose view of the landmarks."""
    nose = np.array(pts["nose_tip"])
    return {name: tuple(rmat.T @ (np.array(p) - nose)) for name, p in pts.items()}


PREVIEW_WINDOW_NAME = "Building dataset - preview (press q to stop)"


def _draw_debug_overlay(frame, pts, ear, mar, pitch):
    for x, y, _ in pts.values():
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}  Pitch: {pitch:.1f}deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def extract_raw_and_frame_features(video_path, subject_id, label, face_mesh, show=False, demo_output_path=None,
                                    max_seconds=None):
    """Returns (raw_df, frame_df, fps, aborted). aborted is True if the user pressed 'q' in the preview window.

    max_seconds, if given, stops reading after that many seconds of video (used for baseline calibration,
    which only needs the first 5-10s of the Alert video rather than the whole ~10 minute recording).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps) if max_seconds is not None else None

    raw_rows, frame_rows = [], []
    prev_pitch = prev_yaw = prev_roll = None
    frame_idx = 0
    demo_writer = None
    aborted = False

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        display_frame = frame
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            pts = {name: _landmark_px(landmarks[idx], w, h) for name, idx in LANDMARK_IDS.items()}
            pose = estimate_head_pose(pts, w, h)

            if pose is not None:
                pitch, yaw, roll, rmat, _ = pose
                norm_pts = normalize_landmarks(pts, rmat)
                ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                mar = mouth_aspect_ratio(pts)
                dt = 1.0 / fps
                pitch_vel = 0.0 if prev_pitch is None else (pitch - prev_pitch) / dt
                yaw_vel = 0.0 if prev_yaw is None else (yaw - prev_yaw) / dt
                roll_vel = 0.0 if prev_roll is None else (roll - prev_roll) / dt
                prev_pitch, prev_yaw, prev_roll = pitch, yaw, roll

                raw_row = {"subject_id": subject_id, "video": os.path.basename(video_path), "label": label,
                           "frame": frame_idx, "time_sec": frame_idx / fps}
                for name in LANDMARK_IDS:
                    x, y, z = pts[name]
                    nx, ny, nz = norm_pts[name]
                    raw_row[f"{name}_x"], raw_row[f"{name}_y"], raw_row[f"{name}_z"] = x, y, z
                    raw_row[f"{name}_norm_x"], raw_row[f"{name}_norm_y"], raw_row[f"{name}_norm_z"] = nx, ny, nz
                raw_rows.append(raw_row)

                frame_rows.append({
                    "subject_id": subject_id, "video": os.path.basename(video_path), "label": label,
                    "frame": frame_idx, "time_sec": frame_idx / fps,
                    "ear_raw": ear, "mar_raw": mar,
                    "pitch": pitch, "yaw": yaw, "roll": roll,
                    "pitch_vel": pitch_vel, "yaw_vel": yaw_vel, "roll_vel": roll_vel,
                })

                display_frame = _draw_debug_overlay(frame.copy(), pts, ear, mar, pitch)

        if demo_output_path and demo_writer is None:
            demo_writer = cv2.VideoWriter(demo_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if demo_writer is not None:
            demo_writer.write(display_frame)

        if show:
            cv2.imshow(PREVIEW_WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True

        frame_idx += 1
        if aborted:
            break

    cap.release()
    if demo_writer is not None:
        demo_writer.release()
    return pd.DataFrame(raw_rows), pd.DataFrame(frame_rows), fps, aborted


# ---------------------------------------------------------------------------
# Baseline calibration (per subject, from their Alert/awake video)
# ---------------------------------------------------------------------------
def compute_baseline(frame_df):
    return {
        "ear_baseline": float(frame_df["ear_raw"].mean()),
        "mar_baseline": float(frame_df["mar_raw"].mean()),
        "pitch_baseline": float(frame_df["pitch"].mean()),
        "yaw_baseline": float(frame_df["yaw"].mean()),
        "roll_baseline": float(frame_df["roll"].mean()),
    }


def apply_baseline(frame_df, baseline):
    df = frame_df.copy()
    df["ear"] = df["ear_raw"] / baseline["ear_baseline"]
    df["mar"] = df["mar_raw"] / baseline["mar_baseline"]
    df["pitch_norm"] = df["pitch"] - baseline["pitch_baseline"]
    df["yaw_norm"] = df["yaw"] - baseline["yaw_baseline"]
    df["roll_norm"] = df["roll"] - baseline["roll_baseline"]
    return df


# ---------------------------------------------------------------------------
# Sliding windows -> window-level features
# ---------------------------------------------------------------------------
def _events(is_active, fps, min_frames):
    """Run-length encodes consecutive True frames into event durations (seconds)."""
    events, count = [], 0
    for active in is_active:
        if active:
            count += 1
        else:
            if count >= min_frames:
                events.append(count / fps)
            count = 0
    if count >= min_frames:
        events.append(count / fps)
    return events


def _summarize_window(chunk, fps):
    duration = len(chunk) / fps
    ear, mar = chunk["ear"].values, chunk["mar"].values
    pitch, pitch_vel = chunk["pitch_norm"].values, chunk["pitch_vel"].values

    closed = ear < EAR_CLOSED_RATIO
    blinks = _events(closed, fps, MIN_BLINK_FRAMES)

    yawning = mar > MAR_YAWN_RATIO
    yawns = _events(yawning, fps, MIN_YAWN_FRAMES)

    peaks, _ = find_peaks(np.abs(pitch_vel), height=NOD_VELOCITY_DEG_S, distance=max(1, int(NOD_MIN_GAP_SEC * fps)))
    zero_crossings = np.sum(np.diff(np.sign(pitch - pitch.mean())) != 0)

    return {
        "subject_id": chunk["subject_id"].iloc[0],
        "video": chunk["video"].iloc[0],
        "label": chunk["label"].iloc[0],
        "start_time": float(chunk["time_sec"].iloc[0]),
        "mean_ear": float(ear.mean()), "std_ear": float(ear.std()),
        "min_ear": float(ear.min()), "max_ear": float(ear.max()),
        "blink_count": len(blinks), "blink_freq": len(blinks) / duration,
        "avg_blink_duration": float(np.mean(blinks)) if blinks else 0.0,
        "max_blink_duration": float(np.max(blinks)) if blinks else 0.0,
        "perclos": float(closed.mean()),
        "mean_mar": float(mar.mean()), "max_mar": float(mar.max()), "std_mar": float(mar.std()),
        "yawn_count": len(yawns),
        "avg_yawn_duration": float(np.mean(yawns)) if yawns else 0.0,
        "mean_pitch": float(pitch.mean()), "std_pitch": float(pitch.std()),
        "max_pitch_velocity": float(np.max(np.abs(pitch_vel))),
        "avg_pitch_velocity": float(np.mean(np.abs(pitch_vel))),
        "pitch_oscillation_freq": float(zero_crossings / (2 * duration)),
        "nod_count": int(len(peaks)),
    }


def compute_window_features(frame_df, fps, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    rows = frame_df.reset_index(drop=True)
    window_frames = int(window_sec * fps)
    stride_frames = max(1, int(stride_sec * fps))

    windows = []
    for start in range(0, max(len(rows) - window_frames, 0) + 1, stride_frames):
        chunk = rows.iloc[start:start + window_frames]
        if len(chunk) < window_frames * 0.5:
            continue
        windows.append(_summarize_window(chunk, fps))
    return pd.DataFrame(windows)


# ---------------------------------------------------------------------------
# Incremental persistence (resumable runs)
# ---------------------------------------------------------------------------
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
    """Raised to stop early (user pressed 'q' in the preview window)."""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_dataset(dataset_name, output_dir=OUTPUT_DIR, show=False, save_demo=False):
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
    demo_path = os.path.join(output_dir, f"{dataset_name}_demo_annotated.mp4")

    baselines = _load_json(baselines_path, default={})
    completed = set(_load_json(progress_path, default={"completed_videos": []})["completed_videos"])
    demo_saved = save_demo and os.path.isfile(demo_path)

    if completed:
        print(f"[resume] {len(completed)} video(s) already processed, skipping those")

    face_mesh = _open_face_mesh()
    try:
        for subject_id, subject_videos in subjects.items():
            if all(v["path"] in completed for v in subject_videos):
                continue

            baseline_video = next((v for v in subject_videos if v["is_baseline"]), None)
            if baseline_video is None:
                print(f"[skip] subject {subject_id}: no baseline (awake) video found")
                continue

            if subject_id in baselines:
                baseline = baselines[subject_id]
            else:
                _, baseline_frames, _, aborted = extract_raw_and_frame_features(
                    baseline_video["path"], subject_id, baseline_video["label"], face_mesh, show=show,
                    max_seconds=BASELINE_CALIBRATION_SEC)
                if aborted:
                    raise _StopBuilding()
                if baseline_frames.empty:
                    print(f"[skip] subject {subject_id}: no face detected in baseline video")
                    continue
                baseline = compute_baseline(baseline_frames)
                baselines[subject_id] = baseline
                _save_json(baselines_path, baselines)

            for v in subject_videos:
                if v["path"] in completed:
                    continue

                want_demo = save_demo and not demo_saved
                raw_df, frame_df, fps, aborted = extract_raw_and_frame_features(
                    v["path"], subject_id, v["label"], face_mesh, show=show,
                    demo_output_path=demo_path if want_demo else None)
                if aborted:
                    raise _StopBuilding()

                if want_demo:
                    demo_saved = True
                    print(f"[demo] saved annotated proof video to {demo_path}")

                if frame_df.empty:
                    print(f"[skip] {v['path']}: no face detected")
                    completed.add(v["path"])
                    _save_json(progress_path, {"completed_videos": sorted(completed)})
                    continue

                frame_df = apply_baseline(frame_df, baseline)
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
        face_mesh.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["uta_rldd", "nthu"], default="uta_rldd")
    parser.add_argument("--show", action="store_true",
                         help="Show a live preview window while processing (press q to stop early)")
    parser.add_argument("--save-demo", action="store_true",
                         help="Save one annotated proof video to outputs/<dataset>_demo_annotated.mp4")
    args = parser.parse_args()
    build_dataset(args.dataset, show=args.show, save_demo=args.save_demo)
