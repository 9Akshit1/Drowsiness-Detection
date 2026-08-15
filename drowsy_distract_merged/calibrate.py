import os
import sys
import time
import argparse
import contextlib
import urllib.request

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

CAMERA_ID = 1
CROP_SIZE = 300
LIVE_NUM_FACES = 8
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480

NOSE_CROP_SEC = 3.0
CALIBRATION_SEC = 10.0

FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "face_landmarker.task")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration", "calibration.npz")

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


def derive_crop(nose_positions, frame_w, frame_h, crop_size=CROP_SIZE):
    cx = float(np.mean([p[0] for p in nose_positions]))
    cy = float(np.mean([p[1] for p in nose_positions]))
    half = crop_size // 2
    x0 = int(np.clip(cx - half, 0, max(0, frame_w - crop_size)))
    y0 = int(np.clip(cy - half, 0, max(0, frame_h - crop_size)))
    return (x0, y0, min(x0 + crop_size, frame_w), min(y0 + crop_size, frame_h))


def getMargin(cam:cv2.VideoCapture, IS_DISPLAY:bool, WIDTH:int, HEIGHT:int, FRAMES:int) -> tuple[int, int, int, int]:
    """
    Estimates the average face center from webcam frames using MediaPipe FaceMesh 
    and returns a bounding box (margin) centered around the face.

    Parameters
    ----------
    cam : cv2.VideoCapture
        OpenCV camera object used to capture frames from a webcam.
    IS_DISPLAY : bool
        Whether to display frames and draw landmarks during processing.
    WIDTH : int
        Width of the desired cropped output around the face center.
    HEIGHT : int
        Height of the desired cropped output around the face center.
    FRAMES : int
        Number of consecutive frames with detected faces to average over.

    Returns
    -------
    tuple[int, int, int, int]
        A tuple representing the cropping margins in the format:
        (top, bottom, left, right), ensuring the region stays within the frame bounds.
    """

    #Def var
    counter = 0
    x = np.empty(FRAMES, dtype=np.float32)
    y = np.empty(FRAMES, dtype=np.float32)

    #get image dimensions
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Initialize Mediapipe FaceMesh detector
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=False#new
        )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # loop until the face if found FRAMES times in a row
    while counter < FRAMES:
        
        #get camera in BGR
        ret, frame = cam.read() 
        frame = cv2.flip(frame, 1)

        #check if frame is recieved
        if not ret:
            print("Error: Could not read frame.")
            continue

        # Convert the frame to RGB
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect facial landmarks
        results = face_mesh.process(frame)

        #if face is found
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Convert landmark list to numpy array of shape (N, 2)
            coords = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
            

            # Multiply once using broadcasting
            coords[:, 0] *= w  # x
            coords[:, 1] *= h  # y

            min_x = float(coords[:, 0].min())
            max_x = float(coords[:, 0].max())
            min_y = float(coords[:, 1].min())
            max_y = float(coords[:, 1].max())
            face_w = max_x - min_x
            face_h = max_y - min_y
            frame_w_ratio = face_w / w
            frame_h_ratio = face_h / h

            # Compute averages
            avg_x, avg_y = coords.mean(axis=0)

            # Append to history
            x[counter] = avg_x
            y[counter] = avg_y

            counter += 1
            print(f"Frame {counter}: x = {avg_x:.1f}, y = {avg_y:.1f}")
            print(f" - Face size: {face_w:.1f} x {face_h:.1f} px")

            # Draw the face landmarks for visualization
            if IS_DISPLAY:
                
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list = landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                            (0, 255, 255), 2)   # yellow box
                cv2.putText(frame,
                            f"Face {face_w:.1f}x{face_h:.1f}",
                            (int(min_x), max(0, int(min_y) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Reset if no face detected
            counter = 0

            
        # Display the frame
        if IS_DISPLAY:
            cv2.imshow('Head Orientation Detection', frame)
           
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #calculate mean of x and y
    x,y = int(np.mean(x)), int(np.mean(y))
    print(f"Average face center (full 640 x 480 frame): x = {x}, y = {y}")

    #calculate margine
    half_w = WIDTH // 2
    half_h = HEIGHT // 2

    #ensure bounding box size
    left = max(0, min(x - half_w, w - WIDTH))
    top = max(0, min(y - half_h, h - HEIGHT))
    right = left + WIDTH
    bottom = top + HEIGHT

    print(f"Average face center (in cropped 300 x 300 frame): x = {x - left}, y = {y - top}")

    # Display the frames
    if IS_DISPLAY:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.imshow('Head Orientation Detection', frame)

    #return margins
    return left, top, right, bottom

def apply_crop(frame, crop):
    x0, y0, x1, y1 = crop
    return frame[y0:y1, x0:x1]


def euler_from_rotation(R):
    sin_pitch = np.clip(R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([pitch, roll, yaw], dtype=np.float32)


def calibrate_reference(raw_frames_3d):
    stacked = np.stack(raw_frames_3d, axis=0)
    return stacked.mean(axis=0).astype(np.float32), float(stacked.std(axis=0).mean())


def calibrate_position(raw_frames_3d, ref_3d):
    stacked = np.stack(raw_frames_3d, axis=0)
    cur_3d = stacked.mean(axis=0).astype(np.float32)

    cur_center = cur_3d.mean(axis=0, keepdims=True)
    ref_center = ref_3d.mean(axis=0, keepdims=True)
    cur_centered = cur_3d - cur_center
    ref_centered = ref_3d - ref_center

    H = cur_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = (ref_center.T - R @ cur_center.T).reshape(3, 1)

    angles_deg = np.degrees(euler_from_rotation(R))
    info = {
        "stability_std": float(stacked.std(axis=0).mean()),
        "rotation_pitch_deg": float(angles_deg[0]),
        "rotation_roll_deg": float(angles_deg[1]),
        "rotation_yaw_deg": float(angles_deg[2]),
        "translation_magnitude": float(np.linalg.norm(t)),
    }
    return R.astype(np.float32), t.astype(np.float32), info


def compute_baseline(rows):
    df = pd.DataFrame(rows)
    return {
        "ear_baseline": float(df["ear_raw"].median()),
        "mar_baseline": float(df["mar_raw"].median()),
        "pitch_baseline": float(df["pitch"].median()),
        "yaw_baseline": float(df["yaw"].median()),
        "roll_baseline": float(df["roll"].median()),
    }


def save_reference(path, ref_3d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ref_3d=ref_3d)


def save_full_calibration(path, crop, baseline, ref_3d, R, t):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(baseline)
    np.savez(path, crop=np.array(crop, dtype=np.int32), ref_3d=ref_3d, R=R, t=t,
              baseline_keys=np.array(keys), baseline_vals=np.array([baseline[k] for k in keys], dtype=np.float64))


def load_calibration(path):
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    out = {"ref_3d": d["ref_3d"].astype(np.float32)}
    if "R" in d.files and "t" in d.files:
        out["R"] = d["R"].astype(np.float32)
        out["t"] = d["t"].astype(np.float32)
    return out


TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
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


def draw_text_panel(width, lines, progress=None):
    wrapped = [w for line in lines for w in _wrap_line(line, width - 16)]
    height = TEXT_LINE_H * len(wrapped) + 12
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    for i, line in enumerate(wrapped):
        cv2.putText(panel, line, (8, 18 + i * TEXT_LINE_H), TEXT_FONT, TEXT_SCALE, (0, 255, 255), TEXT_THICKNESS)
    if progress is not None:
        cv2.rectangle(panel, (0, height - 6), (int(width * progress), height), (0, 255, 255), -1)
    return panel


def _detect_and_draw(landmarker, frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detect(landmarker, rgb, time.time() * 1000.0)
    landmarks, face_idx, pts = None, None, None
    if result.face_landmarks:
        face_idx = 0
        landmarks = result.face_landmarks[face_idx]
        pts = face_points(landmarks, w, h)
        for x, y, _ in pts.values():
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    return result, landmarks, face_idx, pts, w, h


def wait_for_key(cap, landmarker, crop, instructions):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        if crop is not None:
            frame = apply_crop(frame, crop)
        _, landmarks, _, _, _, _ = _detect_and_draw(landmarker, frame)

        lines = list(instructions) + ["", "press C when ready, Q to abort"]
        if landmarks is None:
            lines.insert(0, "no face detected")
        panel = draw_text_panel(frame.shape[1], lines)
        cv2.imshow("Calibration", np.vstack([frame, panel]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("c"), ord("C")):
            return True
        if key in (ord("q"), ord("Q")):
            return False


def record_stage(cap, landmarker, duration_sec, crop, instructions, collect_3d):
    rows, raw_3d_frames = [], []
    t_start = time.time()
    w = h = None
    while time.time() - t_start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        if crop is not None:
            frame = apply_crop(frame, crop)
        result, landmarks, face_idx, pts, w, h = _detect_and_draw(landmarker, frame)

        if landmarks is not None:
            if collect_3d:
                raw_3d_frames.append(landmarks_to_array(landmarks))
                pose = rotation_from_result(result, face_idx)
                if pose is not None:
                    pitch, yaw, roll = pose
                    ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                    mar = mouth_aspect_ratio(pts)
                    rows.append({"ear_raw": ear, "mar_raw": mar, "pitch": pitch, "yaw": yaw, "roll": roll})
            else:
                rows.append({"nose_xy": pts["nose_tip"][:2]})

        progress = min(1.0, (time.time() - t_start) / duration_sec)
        panel_lines = list(instructions)
        if landmarks is None:
            panel_lines.append("no face detected")
        panel = draw_text_panel(w, panel_lines, progress)
        cv2.imshow("Calibration", np.vstack([frame, panel]))
        cv2.waitKey(1)

    return rows, raw_3d_frames, w, h


def run_reference_only(cap, landmarker):
    stage1_instructions = [
        "STAGE 1/2: FIND HEAD POSITION",
        "Sit normally, look straight ahead at the road",
        "Keep your head where it naturally rests while driving",
    ]
    if not wait_for_key(cap, landmarker, None, stage1_instructions):
        print("aborted")
        return

    print("stage 1: finding crop from the face center...")
    crop = getMargin(cap, True, CROP_SIZE, CROP_SIZE, 5)
    print(f"crop set to {crop}")

    stage2_instructions = [
        "STAGE 2/2: REFERENCE CALIBRATION",
        "Look straight ahead at the road, neutral expression, stay still",
        "Keep the face in the crop while the reference is being recorded",
    ]
    if not wait_for_key(cap, landmarker, crop, stage2_instructions):
        print("aborted")
        return

    print(f"recording {CALIBRATION_SEC:.0f}s for crop-based reference...")
    _, raw_3d_frames, _, _ = record_stage(cap, landmarker, CALIBRATION_SEC, crop, stage2_instructions, collect_3d=True)
    if not raw_3d_frames:
        print("no usable frames captured, aborting")
        return

    ref_3d, std = calibrate_reference(raw_3d_frames)
    print(f"reference captured (stability std={std:.4f})")
    save_reference(CALIBRATION_FILE, ref_3d)
    print(f"saved reference -> {CALIBRATION_FILE}")
    print("run calibrate.py again to do the full per-vehicle calibration (crop + bias + baseline).")


def run_full_calibration(cap, landmarker, ref_3d):
    stage1_instructions = [
        "STAGE 1/2: FIND HEAD POSITION",
        "Sit normally, look straight ahead at the road",
        "Keep your head where it naturally rests while driving",
    ]
    if not wait_for_key(cap, landmarker, None, stage1_instructions):
        print("aborted")
        return
    print(f"stage 1: recording ({NOSE_CROP_SEC:.0f}s)...")
    stage1_rows, _, w, h = record_stage(cap, landmarker, NOSE_CROP_SEC, None, stage1_instructions, collect_3d=False)
    nose_positions = [r["nose_xy"] for r in stage1_rows if r.get("nose_xy") is not None]
    if not nose_positions:
        print("no face detected, aborting")
        return
    #crop = derive_crop(nose_positions, w, h)        # assumes nose tip is the center of the face
    crop = getMargin(cap, True, CROP_SIZE, CROP_SIZE, 5)     # calcualtes the face center from the landmarks instead of nose tip
    print(f"crop set to {crop}")

    stage2_instructions = [
        "STAGE 2/2: RECORDING BIAS + BASELINE",
        "Keep looking straight ahead, face relaxed",
        "Eyes open normally, mouth closed, don't talk or move",
    ]
    if not wait_for_key(cap, landmarker, crop, stage2_instructions):
        print("aborted")
        return
    print(f"stage 2: recording {CALIBRATION_SEC:.0f}s...")
    calib_rows, raw_3d_frames, _, _ = record_stage(cap, landmarker, CALIBRATION_SEC, crop, stage2_instructions, collect_3d=True)
    if not calib_rows or not raw_3d_frames:
        print("no usable frames captured, aborting")
        return

    baseline = compute_baseline(calib_rows)
    print("drowsy baseline:", baseline)
    R, t, info = calibrate_position(raw_3d_frames, ref_3d)
    print(f"position bias: pitch={info['rotation_pitch_deg']:.2f} roll={info['rotation_roll_deg']:.2f} "
          f"yaw={info['rotation_yaw_deg']:.2f} translation={info['translation_magnitude']:.4f}")
    save_full_calibration(CALIBRATION_FILE, crop, baseline, ref_3d, R, t)
    print(f"saved crop + baseline + bias -> {CALIBRATION_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", type=int, default=CAMERA_ID)
    parser.add_argument("--seed-reference", default=None,
                         help="path to the original reference .npz (the ref_3d the distraction "
                              "models were actually trained against), use once instead of letting "
                              "this run define its own reference")
    args = parser.parse_args()

    existing = load_calibration(CALIBRATION_FILE)
    ref_3d = existing["ref_3d"] if existing else None

    if args.seed_reference:
        seed = np.load(args.seed_reference)
        if "ref_3d" not in seed.files:
            print(f"{args.seed_reference} has no 'ref_3d' key (found: {seed.files}), aborting")
            return
        ref_3d = seed["ref_3d"].astype(np.float32)
        print(f"using reference from {args.seed_reference}")

    landmarker = open_face_landmarker()
    cap = cv2.VideoCapture(args.camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print("could not open camera")
        return

    if ref_3d is None:
        run_reference_only(cap, landmarker)
    else:
        run_full_calibration(cap, landmarker, ref_3d)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
