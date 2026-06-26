# Features: EAR, MAR, PERCLOS, blink frequency, head pose, head tilt/nod

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import csv
from collections import deque

eye_ar_threshold = 0.25
eye_ar_consec_frames = 15
mouth_threshold = 0.6
nod_threshold = 15  
perclos_threshold = 0.8
blink_rate_low = 3
blink_rate_high = 30
drowsy_seconds = 2.0
yawn_counter_thresh = 40
distraction_counter_max = 100
head_turn_threshold = 20
log_file = "driver_detection_log.csv"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def euclidean_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_ear(landmarks, p1, p2, p3, p4, p5, p6):
    A = euclidean_distance(landmarks[p2], landmarks[p6])
    B = euclidean_distance(landmarks[p3], landmarks[p5])
    C = euclidean_distance(landmarks[p1], landmarks[p4])
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def compute_mar(landmarks):
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    v_dist = euclidean_2d(top, bottom)
    h_dist = euclidean_2d(left, right)
    return v_dist / (h_dist + 1e-6)


def compute_head_pose(landmarks, w, h):
    model_3d = np.array([
        (0, 0, 0),
        (0, -63.6, -12.5),
        (-43.3, 32.7, -26),
        (43.3, 32.7, -26),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype=np.float64)
    
    img_pts = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[199].x * w, landmarks[199].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w, landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype=np.float64)
    
    K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    try:
        success, rvec, _ = cv2.solvePnP(model_3d, img_pts, K, dist_matrix,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            sy = np.hypot(R[0, 0], R[1, 0])
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            return yaw, pitch
    except:
        pass
    
    return 0, 0


def compute_head_tilt(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    dy = right_eye.y - left_eye.y
    dx = right_eye.x - left_eye.x
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def log_event(event, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime("%H:%M:%S")
    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event])
    except:
        pass


class DriverMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            exit()
        
        self.blink_counter = 0
        self.alarm_on = False
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.nod_counter = 0
        self.sleep_alert_count = 0
        self.yawn_alert_count = 0
        self.distraction_alert_count = 0
        
        self.baseline_ear = 0.4
        self.baseline_mar = 0.2
        self.baseline_yaw = 0
        self.baseline_pitch = 0
        self.stddev_ear = 0.1
        self.stddev_yaw = 5
        self.stddev_pitch = 5
        self.calibrated = False
        
        self.eye_close_start = None
        self.last_alarm_time = 0
        self.frame_count = 0
        self.last_blink_time = 0
        
        self.ear_history = deque(maxlen=60)
        self.perclos_frames = deque(maxlen=int(60 * 25))
        self.blink_times = deque()
        
        self._init_csv()
    
    def _init_csv(self):
        try:
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "EAR", "MAR", "Yaw", "Pitch", "Tilt",
                    "PERCLOS", "Blink_Rate", "Status"
                ])
        except:
            pass
    
    def calibrate(self, duration=5):
        print("Calibrating for %d seconds... Keep head still and look forward" % duration)
        
        ear_vals = []
        mar_vals = []
        yaw_vals = []
        pitch_vals = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                
                ear_l = compute_ear(landmarks, 33, 160, 158, 133, 153, 144)
                ear_r = compute_ear(landmarks, 362, 385, 387, 263, 373, 380)
                ear = (ear_l + ear_r) / 2
                
                mar = compute_mar(landmarks)
                yaw, pitch = compute_head_pose(landmarks, w, h)
                
                ear_vals.append(ear)
                mar_vals.append(mar)
                yaw_vals.append(yaw)
                pitch_vals.append(pitch)
            
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, "Calibrating: %ds" % remaining,
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Driver Detection - Calibration", frame)
            cv2.waitKey(30)
        
        if ear_vals:
            self.baseline_ear = np.median(ear_vals)
            self.stddev_ear = np.std(ear_vals)
            self.baseline_mar = np.median(mar_vals)
            self.baseline_yaw = np.median(yaw_vals)
            self.baseline_pitch = np.median(pitch_vals)
            self.stddev_yaw = np.std(yaw_vals)
            self.stddev_pitch = np.std(pitch_vals)
            self.calibrated = True
            print("Calibration complete!")
            print("EAR baseline: %.3f (%.3f)" % (self.baseline_ear, self.stddev_ear))
            print("Yaw baseline: %.1f (%.1f)" % (self.baseline_yaw, self.stddev_yaw))
            print("Pitch baseline: %.1f (%.1f)" % (self.baseline_pitch, self.stddev_pitch))
    
    def run(self):
        if not self.calibrated:
            self.calibrate()
        
        print("Driver monitoring started. Press q to quit, c to recalibrate")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            self.frame_count += 1
            current_time = time.time()
            alert_status = ""
            
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                
                ear_l = compute_ear(landmarks, 33, 160, 158, 133, 153, 144)
                ear_r = compute_ear(landmarks, 362, 385, 387, 263, 373, 380)
                ear = (ear_l + ear_r) / 2
                
                mar = compute_mar(landmarks)
                yaw, pitch = compute_head_pose(landmarks, w, h)
                tilt = compute_head_tilt(landmarks)
                
                self.ear_history.append(ear)
                
                if ear < eye_ar_threshold:
                    self.blink_counter += 1
                    if self.eye_close_start is None:
                        self.eye_close_start = current_time
                    self.perclos_frames.append(1)
                else:
                    if self.eye_close_start and current_time - self.eye_close_start > 0.15:
                        self.blink_times.append(current_time)
                        self.last_blink_time = current_time
                    self.eye_close_start = None
                    self.blink_counter = 0
                    self.perclos_frames.append(0)
                
                if self.blink_counter >= eye_ar_consec_frames and not self.alarm_on:
                    if current_time - self.eye_close_start >= drowsy_seconds:
                        self.alarm_on = True
                        alert_status = "DROWSY - EYES CLOSED"
                        log_event(alert_status)
                        self.sleep_alert_count += 1
                
                if ear > eye_ar_threshold:
                    self.alarm_on = False
                
                if len(self.perclos_frames) > 0:
                    perclos = sum(self.perclos_frames) / len(self.perclos_frames)
                    if perclos > perclos_threshold:
                        if alert_status == "":
                            alert_status = "PERCLOS CRITICAL %.2f" % perclos
                            log_event(alert_status)
                else:
                    perclos = 0
                
                while self.blink_times and current_time - self.blink_times[0] > 60:
                    self.blink_times.popleft()
                blink_rate = len(self.blink_times)
                
                if blink_rate < blink_rate_low and len(self.blink_times) > 5:
                    if alert_status == "":
                        alert_status = "ABNORMAL BLINK RATE: %d BPM" % blink_rate
                elif blink_rate > blink_rate_high:
                    if alert_status == "":
                        alert_status = "HIGH BLINK RATE: %d BPM" % blink_rate
                
                if mar > mouth_threshold:
                    self.yawn_counter += 1
                else:
                    self.yawn_counter = 0
                
                if self.yawn_counter > yawn_counter_thresh:
                    if alert_status == "":
                        alert_status = "YAWNING DETECTED"
                        log_event(alert_status)
                        self.yawn_alert_count += 1
                    self.yawn_counter = 0
                
                yaw_deviation = abs(yaw - self.baseline_yaw)
                pitch_deviation = abs(pitch - self.baseline_pitch)
                head_pose_text = "FORWARD"
                
                if yaw_deviation > head_turn_threshold:
                    if yaw > self.baseline_yaw:
                        head_pose_text = "LOOKING RIGHT"
                    else:
                        head_pose_text = "LOOKING LEFT"
                    self.distraction_counter += 1
                elif pitch_deviation > head_turn_threshold:
                    if pitch > self.baseline_pitch:
                        head_pose_text = "LOOKING DOWN"
                    else:
                        head_pose_text = "LOOKING UP"
                    self.distraction_counter += 1
                else:
                    self.distraction_counter = max(0, self.distraction_counter - 1)
                
                if self.distraction_counter > distraction_counter_max:
                    if alert_status == "":
                        alert_status = "DISTRACTION: " + head_pose_text
                        log_event(alert_status)
                        self.distraction_alert_count += 1
                    self.distraction_counter = 0
                
                if abs(tilt) > nod_threshold:
                    alert_status = "HEAD NOD/DROOP DETECTED"
                    log_event(alert_status)
                    self.nod_counter += 1
                
                try:
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%H:%M:%S"),
                            "%.3f" % ear,
                            "%.3f" % mar,
                            "%.1f" % yaw,
                            "%.1f" % pitch,
                            "%.1f" % tilt,
                            "%.2f" % perclos,
                            blink_rate,
                            alert_status
                        ])
                except:
                    pass
                
                cv2.putText(frame, "EAR: %.3f" % ear, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "MAR: %.3f" % mar, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Yaw: %.1f | Pitch: %.1f" % (yaw, pitch), (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Tilt: %.1f" % tilt, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "PERCLOS: %.2f" % perclos, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Blink Rate: %d BPM" % blink_rate, (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, head_pose_text, (10, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(frame, "Sleep: %d" % self.sleep_alert_count, (w-180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Yawn: %d" % self.yawn_alert_count, (w-180, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Distraction: %d" % self.distraction_alert_count, (w-180, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "Nods: %d" % self.nod_counter, (w-180, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if alert_status:
                    cv2.rectangle(frame, (10, h-80), (w-10, h-10), (0, 0, 255), -1)
                    cv2.putText(frame, alert_status, (30, h-30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                mp_drawing.draw_landmarks(
                    frame, 
                    result.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(circle_radius=1, thickness=1),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1)
                )
            
            else:
                cv2.putText(frame, "NO FACE DETECTED", (w//2-150, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            cv2.imshow("Driver Drowsiness and Distraction Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate()
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = DriverMonitor()
    monitor.run()