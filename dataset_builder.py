"""
dataset_builder.py
Builds RAW / FRAME-LEVEL / WINDOW datasets from driver videos for drowsiness classification.
Supports UTA-RLDD and NTHU-DDD. 
"""

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"

# Window/stride for feature aggregation. 10s tested best among {5,10,20}s on real CV (F1
# 0.763/0.798/0.815); picked over 20s for faster live response. `--rewindow` re-slices an
# already-extracted dataset if this changes.
WINDOW_SEC = 10.0
STRIDE_SEC = 1.0

# ear/mar are ratios to each subject's own baseline (1.0 == baseline behavior).
EAR_CLOSED_RATIO = 0.75   # Ersoy et al. 2026 (arXiv:2604.22479): 75% of personal baseline EAR
MAR_YAWN_RATIO = 1.8      # loosely bracketed by Ersoy et al.'s 140%, not matched exactly

# Duration-based (not frame-count) so they scale correctly at any fps, live included.
MIN_BLINK_DURATION_SEC = 3 / 30.0   # ~100ms, Soukupova & Cech 2016 / Singh & Kaur 2012
MIN_YAWN_DURATION_SEC = 8 / 30.0    # ~267ms, no literature precedent, engineering choice
BLINK_MERGE_GAP_SEC = MIN_BLINK_DURATION_SEC  # debounce: a brief reopening doesn't end the event
YAWN_MERGE_GAP_SEC = MIN_YAWN_DURATION_SEC

PERCLOS_EAR_RATIO = 0.4  # stricter than EAR_CLOSED_RATIO; data-calibrated (see notes), not literature
NOD_VELOCITY_DEG_S = 25.0
NOD_MIN_PROMINENCE_DEG = 10.0
NOD_MIN_GAP_SEC = 0.5
NOD_VELOCITY_CHECK_WINDOW_SEC = 0.3

EAR_CLIP_MAX = 2.0
MAR_CLIP_MAX = 50.0  # baselines vary ~30x across subjects, so ratios legitimately get large

MIN_WINDOW_FILL_RATIO = 0.5   # drop a video's short trailing partial window
# Rows in a window aren't guaranteed temporally contiguous -- MediaPipe can lose the face for a
# real stretch (min detection rate seen: 11%). Drop a window if its actual timespan exceeds this
# multiple of window_sec (loose on purpose, only catches genuine gaps).
MAX_WINDOW_SPAN_RATIO = 1.5

FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5  # MediaPipe defaults
FACE_MESH_MIN_PRESENCE_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# Head pose comes from MediaPipe's own facial transformation matrix (Procrustes fit over all 478
# landmarks), not a manual solvePnP estimate -- more stable at non-frontal angles.
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task")

# Fixed size (px) for the RAW/FRONTAL 3D debug panels, independent of the source video's own
# resolution. Previously these panels reused the source frame's pixel HEIGHT as their size, which
# for portrait/high-res phone videos (e.g. UTA-RLDD's 1080x1920 clips) produced enormous
# 1920x1920 panels -- landmark positions (scaled for a ~400px panel) then rendered as a handful of
# tiny, scattered dots lost in a mostly-black canvas, and the inflated combined-frame width could
# exceed OpenH264's encoder resolution cap (width*height <= 9,437,184), silently falling back to
# the non-browser-playable mp4v codec.
PANEL_SIZE = 480

LABEL_NAMES = {0: "Alert", 1: "Low Vigilant", 2: "Drowsy"}

UTA_RLDD_ROOT = "UTA-RLDD"
UTA_RLDD_LABELS = {"0": 0, "5": 1, "10": 2}  # Alert, Low Vigilant, Drowsy

NTHU_ROOT = "NTHU"
# Category folder names are spelled inconsistently in the dataset itself
# (nightglasses / night_glasses, nightnoglasses / night_noglasses) -- normalize them.
NTHU_CATEGORY_ALIASES = {
    "glasses": "glasses", "noglasses": "noglasses", "sunglasses": "sunglasses",
    "nightglasses": "night_glasses", "night_glasses": "night_glasses",
    "nightnoglasses": "night_noglasses", "night_noglasses": "night_noglasses",
}
# Preferred category to source each subject's Alert/baseline clip from (cleanest conditions first).
NTHU_BASELINE_CATEGORY_PREFERENCE = ["noglasses", "glasses", "sunglasses", "night_glasses", "night_noglasses"]
# Each subject/category folder has up to 4 behavior clips. "*Combination" clips are mixed
# alert+drowsy recordings -- only usable with their per-frame _drowsiness.txt ground truth
# (never blanket-labeled, since that would reintroduce the exact video-level label-noise problem
# found in UTA-RLDD). "yawning"/"slowBlinkWithNodding" are short, deliberately single-behavior
# clips -- safe to blanket-label Drowsy if their _drowsiness.txt happens to be missing locally.
NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH = {"yawning": 2, "slowBlinkWithNodding": 2}  # 2 = Drowsy
# NTHU's own per-frame ground truth is binary (0=not drowsy, 1=drowsy). Mapped into our 3-class
# space as 0->Alert(0), 1->Drowsy(2) -- NTHU has no "Low Vigilant" equivalent, so label 1 never
# appears for NTHU windows.
NTHU_FRAME_LABEL_MAP = {0: 0, 1: 2}

WINDOW_FEATURE_COLUMNS = [
    "mean_ear", "std_ear", "min_ear", "max_ear",
    "blink_count", "blink_freq", "avg_blink_duration", "max_blink_duration", "perclos",
    "mean_mar", "max_mar", "std_mar", "yawn_count", "avg_yawn_duration",
    "mean_pitch", "std_pitch", "max_pitch_velocity", "avg_pitch_velocity",
    "pitch_oscillation_freq", "nod_count",
]

# Progression plots (generate_progression_plots) and annotated debug clips (make_annotated_clips)
# both run automatically at the end of build_dataset(). Clip settings are exposed on the CLI
# (--clips-per-label / --clip-windows); PROGRESSION_PLOT_EXAMPLES_PER_LABEL/FEATURES aren't
# currently CLI flags -- edit them here directly, or call generate_progression_plots() yourself.
PROGRESSION_PLOT_FEATURES = ["mean_ear", "perclos", "mean_mar", "yawn_count", "mean_pitch", "blink_count", "nod_count"]
PROGRESSION_PLOT_EXAMPLES_PER_LABEL = 2
CLIPS_PER_LABEL_DEFAULT = 2
# Number of full WINDOW_SEC-length windows each clip should span end-to-end (e.g. 3 -> 30s clips
# at the current WINDOW_SEC=10.0) -- not a count of 1s-stride slides. Set to 3 so a clip's first
# WINDOW_SEC is spent filling the rolling buffer (see _record_annotated_clip/MIN_RATE_DISPLAY_SEC)
# while still leaving a couple of full windows' worth of real, rate-accurate footage to look at.
CLIP_WINDOWS_TARGET = 3

PREVIEW_WINDOW_NAME = "Building dataset - preview (press q to stop)"

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

# Generic frontal reference shape (all 22 LANDMARK_IDS points, nose-tip-centered, in the same
# +y-up/pixel-derived convention as _to_rotation_space) used to compute the frontal-normalization
# rotation -- see normalize_landmarks(). Derived from a real, low-yaw/pitch/roll frame (NTHU
# subject 001), then made exactly left-right symmetric by averaging mirrored point pairs (a real
# frontal face IS left-right symmetric; a single real frame never quite is, due to residual pose
# and individual asymmetry, so this symmetrization step matters -- an un-symmetrized reference
# would bias every subject's "frontalized" output toward that one frame's own leftover tilt).
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
REFERENCE_LANDMARK_ORDER = list(LANDMARK_IDS)  # fixed order shared with the Kabsch fit


# ---------------------------------------------------------------------------
# Dataset video listing -- the ONLY dataset-specific logic in this file lives in
# _uta_rldd_videos / _nthu_videos, both reached only through get_video_list().
# ---------------------------------------------------------------------------
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


# NTHU-DDD local copy lives flat under NTHU/<subject_id>/<category>/ (videos + per-frame
# *_drowsiness.txt annotations together). Originally the local download was split across 3
# non-overlapping shard folders with no path collisions between them; those have been merged
# on disk into this flat layout (see git history for the one-off migration). NTHU_CATEGORY_ALIASES
# / NTHU_BASELINE_CATEGORY_PREFERENCE / NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH /
# NTHU_FRAME_LABEL_MAP are defined in the Config section above.


def _nthu_read_frame_labels(path):
    """NTHU per-frame annotation files are a single line, one digit per frame, no delimiters."""
    with open(path) as f:
        digits = f.read().strip()
    return np.array([NTHU_FRAME_LABEL_MAP.get(int(c), 0) for c in digits if c.isdigit()], dtype=np.int64)


def _nthu_videos(root=NTHU_ROOT):
    # (subject_id, category, behavior) -> {"video": path, "drowsiness_txt": path}
    entries = {}
    for subject_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(subject_dir) or os.path.basename(subject_dir).startswith("_"):
            continue  # skip NTHU/_unused and any other non-subject folders
        subject_id = os.path.basename(subject_dir)
        if not subject_id.isdigit():
            continue  # e.g. "CVLab Drowsiness Dataset" (raw zip archives, not extracted data)
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
            continue  # label-only entry (no video present locally) -- nothing to extract

        drowsiness_txt = info.get("drowsiness_txt")
        if drowsiness_txt is not None:
            frame_labels = _nthu_read_frame_labels(drowsiness_txt)
            label = int(np.bincount(frame_labels).argmax()) if len(frame_labels) else 0
        elif behavior in NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH:
            frame_labels = None
            label = NTHU_BLANKET_LABEL_IF_NO_GROUND_TRUTH[behavior]
        else:
            continue  # mixed/combination clip with no ground truth available -- skip, don't guess

        videos.append({
            "path": video_path, "subject_id": subject_id, "category": category, "behavior": behavior,
            "label": label, "frame_labels": frame_labels, "is_baseline": False,
        })

    # Pick exactly one Alert/baseline video per subject: their nonsleepyCombination clip, from
    # the cleanest available category. Subjects with no such clip (the Testing/Evaluation-split
    # subjects, which only have a single mixed "mix" recording per category) fall back to their
    # "mix" clip instead -- build_dataset() restricts the baseline computation to that clip's
    # genuinely Alert-labeled frames (via its real per-frame ground truth), so the reference is
    # still a clean alert sample rather than being contaminated by the video's drowsy segments.
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


# ---------------------------------------------------------------------------
# Landmarks / features (shared by dataset building and live inference)
# ---------------------------------------------------------------------------
def _ensure_face_landmarker_model():
    if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_LANDMARKER_MODEL_PATH), exist_ok=True)
        print(f"[setup] downloading face_landmarker.task model to {FACE_LANDMARKER_MODEL_PATH} ...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
    return FACE_LANDMARKER_MODEL_PATH


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silences native (C++-level) log/warning spam written straight to the OS stderr file
    descriptor, bypassing Python's logging/absl bindings entirely -- so it isn't reachable from
    Python-level logging config, only by redirecting the raw fd. Two known sources both wrapped
    with this: (1) MediaPipe's "Sets FaceBlendshapesGraph acceleration to xnnpack by default"
    startup line, emitted once per FaceLandmarker instance -- i.e. once per video, since each video
    gets its own instance (see _open_face_landmarker); confirmed benign, see
    https://github.com/google/mediapipe/issues/4944. (2) OpenCV's FFmpeg backend logging
    corrupted-macroblock/decode-error warnings when a source video file has a damaged frame
    (observed on some NTHU-DDD .avi files) -- also confirmed benign, frame reading continues
    normally afterward, this just hides the noise."""
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


def _open_face_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_face_landmarker_model()),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
        min_face_presence_confidence=FACE_MESH_MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
        output_facial_transformation_matrixes=True,
    )
    with _suppress_native_stderr():
        return FaceLandmarker.create_from_options(options)


def _detect(landmarker, rgb_frame, timestamp_ms):
    """Runs FaceLandmarker in VIDEO mode on one frame. timestamp_ms must strictly increase across
    calls to the SAME landmarker instance -- each video (or live session) needs its own instance."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return landmarker.detect_for_video(mp_image, int(timestamp_ms))


def _landmark_px(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z * w)


def _rotation_from_result(result):
    """Extracts (pitch, yaw, roll, rotation_matrix) from FaceLandmarker's facial transformation
    matrix, or None if unavailable. That matrix is MediaPipe's own rotation+translation fit of ALL
    478 landmarks against its canonical 3D face model (see FACE_LANDMARKER_MODEL_URL comment) --
    far more stable than a 6-point solvePnP estimate, especially at non-frontal head angles."""
    if not result.facial_transformation_matrixes:
        return None
    rmat = np.array(result.facial_transformation_matrixes[0])[:3, :3]
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
    """Pixel space is +y down; flip to +y up so drawing/rotation math stays consistent."""
    return {name: (x, -y, z) for name, (x, y, z) in pts.items()}


def _kabsch_rotation(reference, current):
    """Best-fit rotation (Kabsch/SVD) mapping reference onto current, both (N,3) nose-centered
    and scale-normalized first so a size mismatch doesn't bias the fit."""
    ref = reference / np.linalg.norm(reference, axis=1).mean()
    cur = current / np.linalg.norm(current, axis=1).mean()
    h = ref.T @ cur
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def normalize_landmarks(pts):
    """Undo head rotation around the nose tip for a frontal view of the landmarks.

    Fits against REFERENCE_FACE_SHAPE in our own pixel-derived coordinate space (Kabsch, see
    _kabsch_rotation) rather than reusing MediaPipe's facial_transformation_matrixes, which live
    in a different (metric, model-fit) space and don't frontalize pixel-space points correctly.
    """
    rot_pts = _to_rotation_space(pts)
    nose = np.array(rot_pts["nose_tip"])
    centered = {name: np.array(p) - nose for name, p in rot_pts.items()}
    current_arr = np.array([centered[name] for name in REFERENCE_LANDMARK_ORDER])
    reference_arr = np.array([REFERENCE_FACE_SHAPE[name] for name in REFERENCE_LANDMARK_ORDER])
    rmat = _kabsch_rotation(reference_arr, current_arr)
    return {name: tuple(rmat.T @ p) for name, p in centered.items()}


def _center_on_nose(pts):
    """Recenters landmarks on the nose tip WITHOUT correcting rotation -- shows the head's actual
    current pose, as opposed to normalize_landmarks() which undoes rotation to face-forward."""
    rot_pts = _to_rotation_space(pts)
    nose = np.array(rot_pts["nose_tip"])
    return {name: tuple(np.array(p) - nose) for name, p in rot_pts.items()}


def _draw_points_panel(size, pts, label):
    """Generic small-dots visualization of a landmark set, used for both the raw (pose-preserving)
    and frontal (normalized) 3D panels in live inference and annotated debug clips. `size` should
    be PANEL_SIZE (a fixed constant), not the source video's own frame height -- see PANEL_SIZE."""
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


# Below this, show raw counts only, not an extrapolated rate. Callers pass a duration that's
# capped at WINDOW_SEC (deque/trim logic) so it basically never reaches WINDOW_SEC exactly -- use
# 95% as "close enough" instead of an exact comparison.
MIN_RATE_DISPLAY_SEC = WINDOW_SEC * 0.95


def _draw_metrics_panel(width, stats, duration_sec, extra_lines=None):
    """EAR/MAR/PERCLOS/blink/yawn/nod/pitch-velocity text panel below the video. Shared by
    annotated debug clips and live inference. Rates are count/duration_sec, shown as "/s"."""

    def _rate_str(count):
        if duration_sec < MIN_RATE_DISPLAY_SEC:
            return "..collecting.."
        return f"{count / duration_sec:.2f}/s"

    lines = [
        f"EAR: {stats['mean_ear']:.2f}   MAR: {stats['mean_mar']:.2f}   PERCLOS: {stats['perclos'] * 100:.1f}%",
        f"Blink Rate: {_rate_str(stats['blink_count'])}   Blink Count: {stats['blink_count']}",
        f"Yawn Rate: {_rate_str(stats['yawn_count'])}   Yawn Count: {stats['yawn_count']}",
        f"Nod Rate: {_rate_str(stats['nod_count'])}   Nod Count: {stats['nod_count']}",
        # avg is diluted for a brief fast movement (averaged over the whole window); peak surfaces it.
        f"Pitch Velocity: avg {stats['avg_pitch_velocity']:.0f} / peak {stats['max_pitch_velocity']:.0f} deg/s",
    ]
    lines.extend(extra_lines or [])
    panel = np.zeros((35 * len(lines) + 15, width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 30 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


# Placeholder for _draw_metrics_panel before enough frames have accumulated to compute real stats
# (e.g. a clip's very first frame) -- keeps every written frame the same size (see
# _record_annotated_clip) instead of only drawing the panel once real numbers exist.
_EMPTY_CLIP_STATS = {"mean_ear": 0.0, "mean_mar": 0.0, "perclos": 0.0, "blink_count": 0,
                     "yawn_count": 0, "nod_count": 0, "avg_pitch_velocity": 0.0, "max_pitch_velocity": 0.0}


def _circular_diff(a2, a1):
    """Real angular difference in degrees, wrapped to [-180, 180]. pitch/yaw/roll are stored in
    [-180, 180], so naive subtraction explodes across that boundary (e.g. -179 to +179 is really
    a ~2 degree movement but naive subtraction says 358) -- confirmed on real data to spike to
    >40,000 deg/s and to be the direct cause of spurious nod-count triggers."""
    return (a2 - a1 + 180.0) % 360.0 - 180.0


def _draw_debug_overlay(frame, pts, ear, mar, pitch):
    for x, y, _ in pts.values():
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}  Pitch: {pitch:.1f}deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def extract_raw_and_frame_features(video_path, subject_id, label, face_landmarker, show=False, max_seconds=None,
                                    frame_labels=None):
    """Returns (raw_df, frame_df, fps, aborted). aborted is True if the user pressed 'q' in the preview window.

    max_seconds, if given, stops reading after that many seconds of video.
    frame_labels, if given, overrides `label` per-frame (clamped to the array's length) -- used
    for datasets like NTHU that provide genuine per-frame ground truth rather than one label for
    the whole video.
    face_landmarker must be a fresh instance for THIS video (VIDEO-mode timestamps must strictly
    increase within one instance's lifetime, and each video's timestamps restart at 0) -- see
    _open_face_landmarker().
    """
    # Wrapped in _suppress_native_stderr for the whole video, not just cap.read(): FFmpeg's
    # decoder writes corrupted-macroblock/frame-corruption warnings straight to the OS stderr file
    # descriptor (same as MediaPipe's native log line) whenever a source file has a damaged frame
    # -- confirmed harmless/recoverable (frame reading continues normally afterward) but noisy
    # across a full dataset build, and some NTHU source .avi files are affected. Re-entering the fd
    # redirection every single frame would add per-frame syscall overhead across a multi-million-
    # frame build, so this suppresses once for the video's whole read loop instead.
    with _suppress_native_stderr():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(max_seconds * fps) if max_seconds is not None else None

        # Parent-dir + basename, not just basename -- NTHU reuses identical behavior filenames
        # (e.g. "sleepyCombination.avi") across different category folders for the same subject, so
        # basename alone isn't unique and would silently merge unrelated recordings when later code
        # groups by (subject_id, video).
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
                    # Gap-aware: prev_pitch may be several frames stale if MediaPipe missed a
                    # detection in between, so dt must reflect the actual frame gap, not 1/fps --
                    # confirmed on real data that treating every gap as 1 frame inflated velocity
                    # ~40x on the 0.4% of frames right after a gap.
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


# ---------------------------------------------------------------------------
# Baseline calibration (per subject, from their Alert/awake reference video)
# ---------------------------------------------------------------------------
def compute_baseline(frame_df):
    """Median (not mean) over the WHOLE reference video -- robust to the handful of subjects
    whose first few seconds happen to catch them talking/head-tilted; confirmed on real data to
    fix implausible baselines (e.g. one subject's mar_baseline was 4.5x the population median
    from a short mean-based window, and came back in-line using the full-video median)."""
    return {
        "ear_baseline": float(frame_df["ear_raw"].median()),
        "mar_baseline": float(frame_df["mar_raw"].median()),
        "pitch_baseline": float(frame_df["pitch"].median()),
        "yaw_baseline": float(frame_df["yaw"].median()),
        "roll_baseline": float(frame_df["roll"].median()),
    }


def apply_baseline(frame_df, baseline):
    """pitch/yaw/roll-relative-to-baseline use _circular_diff, not naive subtraction: a subject
    whose neutral head position happens to read near the +-180 wrap boundary (confirmed on real
    data -- one subject's pitch_baseline was 173.9 degrees) would otherwise get wildly wrong
    "normalized" angles for perfectly normal head movement, the exact same wraparound bug as the
    frame-to-frame velocity computation, just applied to baseline-relative position instead."""
    df = frame_df.copy()
    df["ear"] = (df["ear_raw"] / baseline["ear_baseline"]).clip(upper=EAR_CLIP_MAX)
    df["mar"] = (df["mar_raw"] / baseline["mar_baseline"]).clip(upper=MAR_CLIP_MAX)
    df["pitch_norm"] = _circular_diff(df["pitch"].values, baseline["pitch_baseline"])
    df["yaw_norm"] = _circular_diff(df["yaw"].values, baseline["yaw_baseline"])
    df["roll_norm"] = _circular_diff(df["roll"].values, baseline["roll_baseline"])
    return df


# ---------------------------------------------------------------------------
# Sliding windows -> window-level features
# ---------------------------------------------------------------------------
def _events(is_active, fps, min_frames, merge_gap_frames=0):
    """Run-length encodes is_active into distinct event durations (seconds), treating a reopening
    of merge_gap_frames or fewer consecutive False frames as noise/jitter rather than a genuine end
    of the event -- see BLINK_MERGE_GAP_SEC/YAWN_MERGE_GAP_SEC for why this matters: without
    it, a single long held blink/yawn whose EAR/MAR hovers right at the threshold gets fragmented
    into many spuriously short "events", inflating both the count and (count / elapsed duration)
    rate features. An event's duration spans from its first to its last active frame -- the
    frame(s) inside a tolerated gap don't themselves count as active, but don't end the event
    either.
    """
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
    """Peaks in the pitch angle (either direction) with enough prominence, corroborated by a
    nearby velocity spike so a slow head-tilt doesn't count as a nod."""
    distance = max(1, int(NOD_MIN_GAP_SEC * fps))
    peaks_pos, _ = find_peaks(pitch, prominence=NOD_MIN_PROMINENCE_DEG, distance=distance)
    peaks_neg, _ = find_peaks(-pitch, prominence=NOD_MIN_PROMINENCE_DEG, distance=distance)
    merged = sorted(np.concatenate([peaks_pos, peaks_neg])) if len(peaks_pos) or len(peaks_neg) else []
    # find_peaks only enforces `distance` within each polarity separately -- a dip immediately
    # followed by a rebound overshoot (one physical nod) shows up as one peak in each direction,
    # close together, and wasn't deduplicated across the two lists. Merge here too.
    candidates = []
    for p in merged:
        if not candidates or p - candidates[-1] >= distance:
            candidates.append(p)

    half_window = max(1, int(NOD_VELOCITY_CHECK_WINDOW_SEC * fps))
    count = 0
    for p in candidates:
        lo, hi = max(0, p - half_window), min(len(pitch_vel), p + half_window + 1)
        if len(pitch_vel[lo:hi]) and np.max(np.abs(pitch_vel[lo:hi])) >= NOD_VELOCITY_DEG_S:
            count += 1
    return count


def _summarize_window(chunk, fps):
    duration = len(chunk) / fps
    ear, mar = chunk["ear"].values, chunk["mar"].values
    pitch, pitch_vel = chunk["pitch_norm"].values, chunk["pitch_vel"].values

    # Duration constants -> frame counts using the fps ACTUALLY in effect for this chunk (not a
    # hardcoded assumption) -- see MIN_BLINK_DURATION_SEC's comment for why this matters.
    min_blink_frames = max(1, round(MIN_BLINK_DURATION_SEC * fps))
    blink_merge_gap_frames = max(1, round(BLINK_MERGE_GAP_SEC * fps))
    min_yawn_frames = max(1, round(MIN_YAWN_DURATION_SEC * fps))
    yawn_merge_gap_frames = max(1, round(YAWN_MERGE_GAP_SEC * fps))

    closed = ear < EAR_CLOSED_RATIO
    blinks = _events(closed, fps, min_blink_frames, blink_merge_gap_frames)
    perclos_closed = ear < PERCLOS_EAR_RATIO  # PERCLOS uses its own, stricter threshold -- see config

    yawning = mar > MAR_YAWN_RATIO
    yawns = _events(yawning, fps, min_yawn_frames, yawn_merge_gap_frames)

    nod_count = _count_nod_events(pitch, pitch_vel, fps)
    zero_crossings = np.sum(np.diff(np.sign(pitch - pitch.mean())) != 0)

    # Majority-vote label: identical to chunk["label"].iloc[0] whenever the label is constant
    # across the video (every UTA-RLDD case, and NTHU's blanket-labeled clips), but correctly
    # summarizes NTHU's per-frame-varying ground truth for windows straddling a state transition.
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
        # See MAX_WINDOW_SPAN_RATIO -- window_frames CONSECUTIVE SURVIVING ROWS aren't guaranteed
        # to be temporally contiguous; skip a window whose actual real-time span (accounting for a
        # detection gap landing inside it) is badly out of proportion to the intended window_sec.
        actual_span = chunk["time_sec"].iloc[-1] - chunk["time_sec"].iloc[0]
        if actual_span > window_sec * MAX_WINDOW_SPAN_RATIO:
            continue
        windows.append(_summarize_window(chunk, fps))
    return pd.DataFrame(windows)


def rebuild_windows(dataset_name, output_dir=OUTPUT_DIR):
    """Recomputes <dataset>_window.csv from the already-extracted <dataset>_frame_level.csv, using
    whatever WINDOW_SEC/STRIDE_SEC are CURRENTLY configured above -- for when only the
    windowing/aggregation step needs to change (e.g. testing/adopting a different WINDOW_SEC), not
    the underlying per-frame landmark extraction, which is the genuinely expensive part (MediaPipe
    inference; re-windowing here is pure pandas/numpy over data already on disk). Does not touch
    the raw-landmark or frame-level CSVs, and never calls MediaPipe. Per-video fps is inferred from
    that video's own frame timestamps (median inter-frame gap) rather than assumed, so this stays
    correct even if a future dataset mixes videos recorded at different frame rates."""
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

            # Process the baseline video first so its (now full-video-median) baseline is ready
            # before any of the subject's other videos need it -- also avoids extracting it twice.
            ordered_videos = [baseline_video] + [v for v in subject_videos if v is not baseline_video]
            subject_has_baseline = subject_id in baselines

            for v in ordered_videos:
                if v["path"] in completed:
                    continue

                # A fresh landmarker per video: VIDEO-mode timestamps must strictly increase
                # within one instance's lifetime, and each video's timestamps restart at 0.
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
                    # Restrict to genuinely Alert-labeled frames -- a no-op for pure-alert
                    # reference videos (nonsleepyCombination), but essential for the "mix"
                    # fallback (mixed alert/drowsy recording) so drowsy segments don't
                    # contaminate the subject's baseline.
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


# ---------------------------------------------------------------------------
# Automatic stat-progression plots -- run right after a build completes, so you can visually
# sanity-check that features actually track behavior over the course of a video, not just look
# at aggregate distributions. PROGRESSION_PLOT_FEATURES / PROGRESSION_PLOT_EXAMPLES_PER_LABEL are
# defined in the Config section above.
# ---------------------------------------------------------------------------
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
        # Spread example picks across the whole subject range (not just alphabetically-first) --
        # groupby on a filtered df otherwise always returns the same handful of subjects first
        # (e.g. NTHU subject "001" sorts before every other subject), so every plot batch used to
        # come from a single subject no matter how many subjects/labels existed.
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


# ---------------------------------------------------------------------------
# Short annotated debug clips: a few clips per label, showing the label, live landmarks, and
# both the raw-pose and frontal-normalized 3D panels, so tracking correctness can be spot-checked
# without watching an entire long video. CLIP_WINDOWS_TARGET / CLIPS_PER_LABEL_DEFAULT are defined
# in the Config section above.
# ---------------------------------------------------------------------------
def _open_video_writer(path, fps, size):
    """Tries real H.264 first (playable in browsers/VS Code's built-in preview); falls back to
    mp4v (playable in File Explorer/most desktop players, but not in browser-based viewers) if
    the OpenH264 codec DLL isn't available on this machine."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if writer.isOpened():
        return writer
    print("[clip] H.264 encoder unavailable, falling back to mp4v (won't preview in browser-based viewers)")
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def _longest_run(values, target):
    """Returns (start_idx, length) of the longest contiguous run where values == target."""
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

    # Grows from the clip's start (stats visible from frame 2 on), then rolls once full at
    # window_frames -- same buffer semantics as run_live()'s rolling_buffer.
    clip_buffer = deque(maxlen=window_frames)
    prev_pitch = prev_yaw = prev_roll = None

    writer = None
    try:
        # See extract_raw_and_frame_features for why this suppresses FFmpeg's native stderr
        # decode-warning spam for the whole read loop rather than per-frame.
        with _suppress_native_stderr():
            for clip_frame_idx in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Timestamps count from this clip's own start (a fresh landmarker per clip), not the
                # source video's absolute frame position.
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
                # Always draw the metrics panel (even before enough frames have accumulated to compute
                # real stats) so every written frame has the SAME combined size -- a video writer's
                # frame size is fixed by its first frame, and a later frame of a different size would
                # either fail to write or corrupt the output.
                # duration passed here is the buffer's REAL current elapsed length (grows up to
                # WINDOW_SEC, then stays pinned there once the deque is full/rolling) -- not always
                # WINDOW_SEC -- so MIN_RATE_DISPLAY_SEC inside _draw_metrics_panel can tell whether
                # enough time has actually elapsed to trust a derived rate yet.
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
        # Called automatically at the end of build_dataset() as well as standalone via
        # --make-clips, so this can't be a hard SystemExit -- an interrupted build with zero
        # subjects completed yet should degrade gracefully, not crash on its way out.
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

        # Spread picks across the whole subject range (not just alphabetically-first subjects) --
        # same fix as generate_progression_plots, and for the same reason: video lists here are
        # sorted by subject_id, so always taking the first N would always pick the same subjects.
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
                         help="Only (re)generate short annotated debug clips in outputs/annotated/ from an "
                              "already-built dataset; does not rebuild anything. A normal build (no flag) "
                              "already generates clips automatically at the end, so this is only needed to "
                              "regenerate clips on their own, e.g. after changing --clips-per-label.")
    parser.add_argument("--no-clips", action="store_true",
                         help="Skip automatic annotated-clip generation at the end of a build (plots are "
                              "still generated either way)")
    parser.add_argument("--clips-per-label", type=int, default=CLIPS_PER_LABEL_DEFAULT)
    parser.add_argument("--clip-windows", type=int, default=CLIP_WINDOWS_TARGET,
                         help="How many full WINDOW_SEC-length windows each clip should span end-to-end "
                              f"(default: {CLIP_WINDOWS_TARGET}, i.e. {CLIP_WINDOWS_TARGET * WINDOW_SEC:.0f}s)")
    parser.add_argument("--rewindow", action="store_true",
                         help="Only recompute <dataset>_window.csv from the already-extracted "
                              "<dataset>_frame_level.csv using the CURRENT WINDOW_SEC/STRIDE_SEC config -- "
                              "for after changing window length. Fast (no MediaPipe re-run); does not touch "
                              "raw/frame-level CSVs or regenerate plots/clips.")
    args = parser.parse_args()

    if args.rewindow:
        rebuild_windows(args.dataset)
    elif args.make_clips:
        make_annotated_clips(args.dataset, clips_per_label=args.clips_per_label, clip_windows=args.clip_windows)
    else:
        build_dataset(args.dataset, show=args.show, make_clips=not args.no_clips,
                       clips_per_label=args.clips_per_label, clip_windows=args.clip_windows)
