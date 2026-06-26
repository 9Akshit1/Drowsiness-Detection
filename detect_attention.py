#from: https://github.com/yaninsanity/openeyes/tree/main
# only: EAR, MAR (yawn), head pose (direction looking), head tilt/drooping (nod)

'''
What you should see
When it works, a window titled Attention Monitor opens and shows your live webcam feed with overlays like EAR, MAR, yaw, pitch, status dots, and alerts. If you press:

h = help text.

m = face mesh overlay.

f = FPS.

p = threshold panel.

b = blink-rate panel.

t = mute voice.

c = recalibrate.

q or Esc = quit.'''


#!/usr/bin/env python3
"""
main.py

- Auto‑calibrates EAR/MAR/Yaw/Pitch baselines and standard deviations
- Real‑time detection of Drowsiness, Yawning, Head Turn, Head Nod
- Blink Rate (BRR) monitoring
- OpenCV UI + pyttsx3 voice alerts + CSV logging
- Interactive sliders, keyboard shortcuts, help/legend overlay
"""
import time
import argparse
import csv
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from collections import deque

# ======== Default parameters ========
DEFAULTS = {
    'window_size':      60,    # sliding window length (frames)
    'calib_frames':     120,   # frames to collect during calibration
    'calib_delay':      2.0,   # seconds before calibration begins

    'drowsy_sec':       2.0,   # continuous eye‑closure seconds → drowsiness
    'brr_window':      60.0,   # blink‑rate calculation window (seconds)

    'ear_thresh':       0.15,  # EAR threshold offset
    'mar_thresh':       0.20,  # MAR threshold offset
    'yaw_thresh':       15.0,  # static yaw threshold (degrees)
    'pitch_thresh':     15.0,  # static pitch threshold (degrees)
    'yaw_duration':     1.0,   # sustained head turn seconds → alert
    'nod_duration':     1.0,   # down‑up nod transition seconds → nod event
    'yaw_mult':         2.0,   # dynamic yaw threshold multiplier
    'pitch_mult':       2.0,   # dynamic pitch threshold multiplier

    'perclos_thresh':   0.30,  # proportion threshold for drowsiness
    'mar_thresh_prop':  0.20,  # proportion threshold for yawning
    'yaw_prop':         0.20,  # proportion threshold for head turn
    'pitch_prop':       0.20,  # proportion threshold for nodding

    'smooth':           0.6,   # smoothing factor for composite score
    'alert_cooldown':   5.0,   # minimum seconds between voice alerts

    'log_csv':          'attention_log.csv'
}

HELP_TEXT = [
    "Keys: c=Recalibrate | h=Help | m=Mesh | f=FPS | p=Thresholds |",
    "      t=Mute | b=BRR panel | q/Esc=Quit",
    "Sliders: EAR_thresh, MAR_thresh, PERC_thresh, Yaw_mult, Pitch_mult,",
    "         Drowsy_sec, Yaw_duration, Nod_duration, BRR_window"
]

# ======== Text‑to‑Speech helper ========
class TTS:
    def __init__(self, cooldown):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.cooldown = cooldown
        self.last_time = 0
        self.muted = False

    def speak(self, text):
        if self.muted:
            return
        now = time.time()
        if now - self.last_time < self.cooldown:
            return
        self.last_time = now
        self.engine.say(text)
        self.engine.runAndWait()

# ======== Metric computations ========
def eye_aspect_ratio(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3]) + 1e-6
    return (A + B) / (2 * C)

def compute_ear(landmarks, w, h):
    left = [33,159,153,145,144,133]
    right = [362,386,385,263,373,380]
    def coords(idxs):
        return np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in idxs], dtype=np.float64)
    return (eye_aspect_ratio(coords(left)) + eye_aspect_ratio(coords(right))) / 2

def compute_mar(landmarks, w, h):
    top   = np.array([landmarks[13].x*w, landmarks[13].y*h])
    bot   = np.array([landmarks[14].x*w, landmarks[14].y*h])
    left  = np.array([landmarks[78].x*w,  landmarks[78].y*h])
    right = np.array([landmarks[308].x*w, landmarks[308].y*h])
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6)

def get_head_pose(landmarks, w, h):
    model3d = np.array([
        (0,0,0), (0,-63.6,-12.5),
        (-43.3,32.7,-26), (43.3,32.7,-26),
        (-28.9,-28.9,-24.1), (28.9,-28.9,-24.1)
    ], dtype=np.float64)
    imgpts = np.array([
        (landmarks[1].x*w, landmarks[1].y*h),
        (landmarks[199].x*w, landmarks[199].y*h),
        (landmarks[33].x*w, landmarks[33].y*h),
        (landmarks[263].x*w, landmarks[263].y*h),
        (landmarks[61].x*w, landmarks[61].y*h),
        (landmarks[291].x*w, landmarks[291].y*h)
    ], dtype=np.float64)
    K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(model3d, imgpts, K, np.zeros((4,1)),
                              flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.hypot(R[0,0], R[1,0])
    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
    pitch = np.degrees(np.arctan2(-R[2,0], sy))
    return yaw, pitch

# ======== Calibration routine ========
def calibrate(cap, face_mesh, params):
    print(f"Waiting {params['calib_delay']}s before calibration...")
    time.sleep(params['calib_delay'])
    containers = {'ear':[], 'mar':[], 'yaw':[], 'pit':[]}
    count = 0
    while count < params['calib_frames']:
        ret, frame = cap.read()
        if not ret:
            continue
        flipped = cv2.flip(frame, 1)
        h, w = flipped.shape[:2]
        result = face_mesh.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            containers['ear'].append(compute_ear(lm, w, h))
            containers['mar'].append(compute_mar(lm, w, h))
            y, p = get_head_pose(lm, w, h)
            containers['yaw'].append(y)
            containers['pit'].append(p)
            count += 1
        cv2.putText(flipped, f"Calibrating {count}/{params['calib_frames']}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.imshow("Attention Monitor", flipped)
        cv2.waitKey(1)
    baseline = {k: np.median(v) for k, v in containers.items()}
    stddev   = {k: np.std(v)    for k, v in containers.items()}
    print("Calibration complete. Baselines:", baseline, "StdDev:", stddev)
    return baseline, stddev

# ======== Main loop ========
def main():
    # Parse command‑line arguments
    params = DEFAULTS.copy()
    parser = argparse.ArgumentParser(description="Real‑time attention & drowsiness monitor")
    for key, val in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    params.update(vars(args))

    # Initialize TTS, camera, MediaPipe FaceMesh
    tts = TTS(params['alert_cooldown'])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    cv2.namedWindow("Attention Monitor", cv2.WINDOW_NORMAL)

    # Create sliders (trackbars)
    def add_slider(name, val, mx):
        cv2.createTrackbar(name, "Attention Monitor", val, mx, lambda v: None)

    add_slider('EAR_thresh',    int(params['ear_thresh']*1000),       500)
    add_slider('MAR_thresh',    int(params['mar_thresh']*1000),       500)
    add_slider('PERC_thresh',   int(params['perclos_thresh']*100),    100)
    add_slider('Yaw_mult',      int(params['yaw_mult']*10),           50)
    add_slider('Pitch_mult',    int(params['pitch_mult']*10),         50)
    add_slider('Drowsy_sec',    int(params['drowsy_sec']*10),         50)
    add_slider('Yaw_duration',  int(params['yaw_duration']*10),       50)
    add_slider('Nod_duration',  int(params['nod_duration']*10),       50)
    add_slider('BRR_window',    int(params['brr_window']),            300)

    # Perform initial calibration
    baseline, stddev = calibrate(cap, face_mesh, params)

    # Sliding windows for events
    window_len = params['window_size']
    win_ear = deque(maxlen=window_len)
    win_mar = deque(maxlen=window_len)
    win_yaw = deque(maxlen=window_len)
    win_nod = deque(maxlen=window_len)

    # State variables
    eye_close_start = None
    turn_start      = None
    nod_state       = 0
    nod_start       = None
    nod_count       = 0
    blink_times     = deque()

    # Open CSV log
    csv_file = open(params['log_csv'], 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp","ear","mar","yaw","pitch","p_ear","p_mar","p_yaw","p_nod","brr","score"])

    # Display toggles
    show_help = show_mesh = show_fps = show_panel = show_brr = False
    composite_score = 0.0
    prev_time = time.time()

    def proportion(win): return sum(win)/len(win) if win else 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 1)
        h, w = flipped.shape[:2]
        now = time.time()

        # Update params from sliders
        params['ear_thresh']    = cv2.getTrackbarPos('EAR_thresh','Attention Monitor')/1000.0
        params['mar_thresh']    = cv2.getTrackbarPos('MAR_thresh','Attention Monitor')/1000.0
        params['perclos_thresh']= cv2.getTrackbarPos('PERC_thresh','Attention Monitor')/100.0
        params['yaw_mult']      = cv2.getTrackbarPos('Yaw_mult','Attention Monitor')/10.0
        params['pitch_mult']    = cv2.getTrackbarPos('Pitch_mult','Attention Monitor')/10.0
        params['drowsy_sec']    = cv2.getTrackbarPos('Drowsy_sec','Attention Monitor')/10.0
        params['yaw_duration']  = cv2.getTrackbarPos('Yaw_duration','Attention Monitor')/10.0
        params['nod_duration']  = cv2.getTrackbarPos('Nod_duration','Attention Monitor')/10.0
        params['brr_window']    = cv2.getTrackbarPos('BRR_window','Attention Monitor')

        result = face_mesh.process(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
        alert_text = None

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            ear = compute_ear(lm, w, h)
            mar = compute_mar(lm, w, h)
            yaw, pitch = get_head_pose(lm, w, h)

            # dynamic thresholds
            yaw_thr   = max(params['yaw_thresh'],   params['yaw_mult']   * stddev['yaw'])
            pitch_thr = max(params['pitch_thresh'], params['pitch_mult'] * stddev['pit'])

            # 1) Drowsiness detection
            closed = (ear < baseline['ear'] - params['ear_thresh'])
            win_ear.append(closed)
            if closed:
                if eye_close_start is None:
                    eye_close_start = now
                elif now - eye_close_start >= params['drowsy_sec']:
                    alert_text = "You appear drowsy. Please take a break."
                    tts.speak(alert_text)
            else:
                if eye_close_start and now - eye_close_start > 0.2:
                    blink_times.append(now)
                eye_close_start = None

            # 2) Yawning
            yawning = (mar > baseline['mar'] + params['mar_thresh'])
            win_mar.append(yawning)

            # 3) Head turn detection
            turned = abs(yaw - baseline['yaw']) > yaw_thr
            win_yaw.append(turned)
            if turned:
                if turn_start is None:
                    turn_start = now
                elif now - turn_start >= params['yaw_duration'] and not alert_text:
                    alert_text = "Please face forward to stay focused."
                    tts.speak(alert_text)
            else:
                turn_start = None

            # 4) Nod detection state machine
            if nod_state == 0 and pitch < baseline['pit'] - pitch_thr:
                nod_state = 1
                nod_start = now
            elif nod_state == 1 and pitch > baseline['pit'] + pitch_thr:
                nod_state = 2
            elif nod_state == 2 and abs(pitch - baseline['pit']) < pitch_thr * 0.5:
                if now - nod_start >= params['nod_duration'] and not alert_text:
                    alert_text = "You nodded your head. Stay attentive!"
                    tts.speak(alert_text)
                    nod_count += 1
                nod_state = 0
            win_nod.append(nod_state == 2)

            # 5) Blink-rate (BPM)
            while blink_times and now - blink_times[0] > params['brr_window']:
                blink_times.popleft()
            brr = len(blink_times) / params['brr_window'] * 60.0

            # 6) Composite attention score
            p_ear = proportion(win_ear)
            p_mar = proportion(win_mar)
            p_yaw = proportion(win_yaw)
            p_nod = proportion(win_nod)
            flags = [
                p_ear > params['perclos_thresh'],
                p_mar > params['mar_thresh_prop'],
                p_yaw > params['yaw_prop'],
                p_nod > params['pitch_prop']
            ]
            raw_score = sum(flags) / 4.0
            composite_score = (params['smooth'] * raw_score +
                               (1 - params['smooth']) * composite_score)

            # 7) fallback yawning alert if none sent
            if not alert_text and p_mar > params['mar_thresh_prop']:
                alert_text = "Looks like you are yawning. Keep focused!"
                tts.speak(alert_text)

            # --- Render UI ---
            # status circles
            status_labels = ["DROWSY","YAWN","TURN","NOD"]
            status_vals   = [win_ear[-1], win_mar[-1], win_yaw[-1], win_nod[-1]]
            for i,(st,lb) in enumerate(zip(status_vals, status_labels)):
                color = (0,0,255) if st else (100,100,100)
                cv2.circle(flipped, (30+60*i,30), 10, color, -1)
                cv2.putText(flipped, lb, (15+60*i,55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # mesh overlay
            if show_mesh:
                mp.solutions.drawing_utils.draw_landmarks(
                    flipped, result.multi_face_landmarks[0],
                    mp.solutions.face_mesh.FACEMESH_TESSELATION)

            # alert text
            if alert_text:
                cv2.putText(flipped, alert_text, (w//6, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

            # FPS counter
            if show_fps:
                fps = 1.0 / (now - prev_time) if prev_time else 0.0
                prev_time = now
                cv2.putText(flipped, f"FPS: {fps:.1f}", (w-120,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            # progress bars
            def draw_bar(x, y, v):
                cv2.rectangle(flipped, (x,y), (x+int(v*200),y+8),
                              (int(255*(1-v)), int(255*v), 0), -1)
                cv2.rectangle(flipped, (x,y), (x+200,y+8), (255,255,255), 1)
            draw_bar(10, 70,  p_ear)
            draw_bar(10, 85,  p_mar)
            draw_bar(10,100,  p_yaw)
            draw_bar(10,115,  p_nod)

            # BRR panel
            if show_brr:
                cv2.putText(flipped, f"Blink Rate: {brr:.1f} BPM",
                            (w-240,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,0),2)

            # thresholds panel
            if show_panel:
                overlay = flipped.copy()
                cv2.rectangle(overlay, (50,50), (w-50,h-50), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, flipped, 0.4, 0, flipped)
                lines = [
                    f"EAR_thresh={params['ear_thresh']:.3f}",
                    f"MAR_thresh={params['mar_thresh']:.3f}",
                    f"PERC_thresh={params['perclos_thresh']:.2f}",
                    f"Yaw_mult={params['yaw_mult']:.1f}",
                    f"Pitch_mult={params['pitch_mult']:.1f}",
                    f"Drowsy_sec={params['drowsy_sec']:.1f}",
                    f"Yaw_duration={params['yaw_duration']:.1f}",
                    f"Nod_duration={params['nod_duration']:.1f}",
                    f"BRR_window={params['brr_window']:.0f}s"
                ]
                for i, line in enumerate(lines):
                    cv2.putText(flipped, line, (60, 80+20*i),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

            # numeric readouts
            cv2.putText(flipped, f"EAR: {ear:.3f}", (10,140),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(flipped, f"MAR: {mar:.3f}", (10,160),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(flipped, f"Yaw: {yaw:.1f}",  (10,180),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(flipped, f"Pitch: {pitch:.1f}",(10,200),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(flipped, f"Nods: {nod_count}", (10,220),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.putText(flipped, f"Score: {composite_score*100:.1f}%", (10,240),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            # write CSV log
            writer.writerow([
                time.strftime("%H:%M:%S"), ear, mar, yaw, pitch,
                p_ear, p_mar, p_yaw, p_nod, brr, composite_score
            ])

        else:
            cv2.putText(flipped, "No face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        # help overlay
        if show_help:
            for i, line in enumerate(HELP_TEXT):
                cv2.putText(flipped, line, (10, h-80+20*i),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        else:
            cv2.putText(flipped, "Press 'h' for help", (10,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

        cv2.imshow("Attention Monitor", flipped)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('m'):
            show_mesh = not show_mesh
        elif key == ord('f'):
            show_fps = not show_fps
        elif key == ord('p'):
            show_panel = not show_panel
        elif key == ord('t'):
            tts.muted = not tts.muted
        elif key == ord('b'):
            show_brr = not show_brr
        elif key == ord('c'):
            baseline, stddev = calibrate(cap, face_mesh, params)

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()