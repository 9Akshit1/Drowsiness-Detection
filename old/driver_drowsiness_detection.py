
#from: https://github.com/SeajalUniyal/driver_fatigue_detection/tree/main
#only EAR, MAR (yawn), and head tilt/drooping (nod)

import cv2
import time
import csv
import math
import mediapipe as mp
import winsound 

# --------------------------
# CONFIGURATION
# --------------------------
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 15

MOUTH_THRESHOLD = 0.6
NOD_THRESHOLD = 15     # Degrees
LOG_FILE = "fatigue_log.csv"

ALARM_SOUND = "alarm.wav"   

# --------------------------
# MEDIAPIPE SETUP
# --------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# --------------------------
# DISTANCE FUNCTION
# --------------------------
def euclidean(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

# --------------------------
# EYE ASPECT RATIO (EYE_AR)
# --------------------------
def eye_aspect_ratio(landmarks, p1, p2, p3, p4, p5, p6):
    A = euclidean(landmarks[p2], landmarks[p6])
    B = euclidean(landmarks[p3], landmarks[p5])
    C = euclidean(landmarks[p1], landmarks[p4])
    EYE_AR = (A + B) / (2.0 * C)
    return EYE_AR

# --------------------------
# MOUTH OPEN (YAWN)
# --------------------------
def mouth_open_ratio(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    v_dist = euclidean(top, bottom)
    h_dist = euclidean(left, right)

    return v_dist / h_dist

# --------------------------
# HEAD TILT
# --------------------------
def head_tilt(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dy = right_eye.y - left_eye.y
    dx = right_eye.x - left_eye.x
    angle = math.degrees(math.atan2(dy, dx))
    return angle

# --------------------------
# LOGGING
# --------------------------
def log_event(event):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.ctime(), event])

# --------------------------
# SAFE ALARM FUNCTION
# --------------------------
def play_alarm():
    try:
        winsound.PlaySound(ALARM_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except:
        print("⚠ Could not play alarm sound.")

# --------------------------
# VIDEO STREAM
# --------------------------
cap = cv2.VideoCapture(0)
blink_counter = 0
alarm_on = False

print("📷 Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            # ---- EYE_AR FOR BOTH EYES ----
            left_EYE_AR = eye_aspect_ratio(landmarks, 33, 160, 158, 133, 153, 144)
            right_EYE_AR = eye_aspect_ratio(landmarks, 362, 385, 387, 263, 373, 380)
            EYE_AR = (left_EYE_AR + right_EYE_AR) / 2

            # ---- MOUTH OPEN ----
            mouth_ratio = mouth_open_ratio(landmarks)

            # ---- HEAD TILT ----
            angle = head_tilt(landmarks)

            # --------------------------
            # DROWSINESS DETECTION
            # --------------------------
            if EYE_AR < EYE_AR_THRESHOLD:
                blink_counter += 1
            else:
                blink_counter = 0

            # Trigger alarm
            if blink_counter >= EYE_AR_CONSEC_FRAMES and not alarm_on:
                alarm_on = True
                log_event("Drowsiness detected (eyes closed)")
                play_alarm()

            if EYE_AR > EYE_AR_THRESHOLD:
                alarm_on = False

            # --------------------------
            # YAWN
            # --------------------------
            if mouth_ratio > MOUTH_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                log_event("Yawn detected")

            # --------------------------
            # HEAD NOD
            # --------------------------
            if abs(angle) > NOD_THRESHOLD:
                cv2.putText(frame, "HEAD NOD DETECTED!", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                log_event("Head nod detected")

            # --------------------------
            # DISPLAY
            # --------------------------
            cv2.putText(frame, f"EYE_AR: {EYE_AR:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mouth: {mouth_ratio:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Tilt: {angle:.1f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
