#from: https://github.com/erlendskinnemoen/Driver-Monitoring-Systems-using-Google-s-MediaPipe-FaceMesh/tree/main
#only EAR and PERCLOS

import cv2 
import numpy as np
import mediapipe as mp
import time
#import playsound

import pygame
pygame.mixer.init()
sound = pygame.mixer.Sound('beep-01a.mp3')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

frames = []
time_elapsed = 0

EAR_calibration_data_eyes_open = []
EAR_calibration_data_eyes_closed = []
threshold_EAR = 0

face_pitch_calibration_data = []
face_yaw_calibration_data = []
face_angle_baseline = []

capture = cv2.VideoCapture(0)

start_capture = time.time()

while capture.isOpened():
    success, frame = capture.read()
    start = time.time()

    if not success:
        print("Capture failed")
        break

    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True

    frame_height, frame_width, frame_channels = frame.shape

    # The camera matrix
    focal_length = 1 * frame_width
    cam_matrix = np.array([ [focal_length, 0, frame_height / 2],
    [0, focal_length, frame_width / 2],
    [0, 0, 1]])
    # The distorsion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)


    point_right_eye_right = []
    point_right_eye_bottom = [] 
    point_right_eye_left = [] 
    point_right_eye_top = [] 

    point_left_eye_right = [] 
    point_left_eye_bottom = []
    point_left_eye_left= []
    point_left_eye_top = []

    point_right_eye_iris_center = []
    point_left_eye_iris_center = []

    face_2d = []
    face_3d = []
    prev_face_pitch = 0
    prev_face_yaw = 0

    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index, landmark in enumerate(face_landmarks.landmark):

                if index == 33:
                    point_right_eye_right = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 145:
                    point_right_eye_bottom = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 133:
                    point_right_eye_left = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 159:
                    point_right_eye_top = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 362:
                    point_left_eye_right = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 374:
                    point_left_eye_bottom = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 263:
                    point_left_eye_left = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 386:
                    point_left_eye_top = (landmark.x * frame_width, landmark.y * frame_height)
                    cv2.circle(frame, (int(landmark.x * frame_width), int(landmark.y * frame_height)), radius=5, color=(0, 0, 255), thickness=-1)

                if index == 468:
                    point_right_eye_iris_center = (landmark.x * frame_width, landmark.y * frame_height)                    

                if index == 473:
                    point_left_eye_iris_center = (landmark.x * frame_width, landmark.y * frame_height)


                if index == 33 or index == 263 or index == 1 or index == 61 or index == 291 or index == 199:

                    if index == 1:
                        nose_2d = (landmark.x * frame_width, landmark.y * frame_height)
                        nose_3d = (landmark.x * frame_width, landmark.y * frame_height, landmark.z * 3000)

                    x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    face_2d.append((x, y))
                    face_3d.append((x, y, landmark.z))
                    

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
            euler_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

            face_pitch = np.rad2deg(euler_angles[0])
            face_yaw = -np.rad2deg(euler_angles[1])


            left_eye_width = point_left_eye_left[0] - point_left_eye_right[0]
            left_eye_height = point_left_eye_bottom[1] - point_left_eye_top[1]
            left_eye_center = [(point_left_eye_left[0] + point_left_eye_right[0])/2 ,(point_left_eye_bottom[1] + point_left_eye_top[1])/2]

            right_eye_width = point_right_eye_left[0] - point_right_eye_right[0]
            right_eye_height = point_right_eye_bottom[1] - point_right_eye_top[1]
            right_eye_center = [(point_right_eye_left[0] + point_right_eye_right[0])/2 ,(point_right_eye_bottom[1] + point_right_eye_top[1])/2]

            right_eye_gaze_vector = np.array([point_right_eye_iris_center[0] - right_eye_center[0], point_right_eye_iris_center[1] - right_eye_center[1]])
            left_eye_gaze_vector = np.array([point_left_eye_iris_center[0] - left_eye_center[0], point_left_eye_iris_center[1] - left_eye_center[1]])

            if right_eye_height > 0:
                pitch_right_eye = (right_eye_gaze_vector[1] / (right_eye_height/2)) * 30
            else:
                pitch_right_eye = (right_eye_gaze_vector[1] / (right_eye_height/2)) * 45
            yaw_right_eye = (right_eye_gaze_vector[0] / (right_eye_width / 2)) * 45

            if left_eye_gaze_vector[1] > 0:
                pitch_left_eye = (left_eye_gaze_vector[1] / (left_eye_height/2)) * 30
            else:
                pitch_left_eye = (left_eye_gaze_vector[1] / (left_eye_height/2)) * 45
            yaw_left_eye = (left_eye_gaze_vector[0] / (left_eye_width / 2)) * 45                

            # 4.4. - Draw the positions on the frame
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + face_yaw * 10), int(nose_2d[1] - face_pitch * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            cv2.circle(frame, (int(point_left_eye_iris_center[0]), int(point_left_eye_iris_center[1])), radius=3, color=(0, 255, 0), thickness=-1) # Center of iris
            cv2.circle(frame, (int(left_eye_center[0]), int(left_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1)
            
            cv2.circle(frame, (int(point_right_eye_iris_center[0]), int(point_right_eye_iris_center[1])), radius=3, color=(0, 0, 255), thickness=-1) # Center of iris
            cv2.circle(frame, (int(right_eye_center[0]), int(right_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) # Center of eye 

            # 4.5 - Calculate the EAR
            right_eye_EAR = right_eye_height / right_eye_width 
            left_eye_EAR = left_eye_height / left_eye_width
            EAR = (right_eye_EAR + left_eye_EAR) / 2
            
            if time_elapsed < 3:
                cv2.putText(frame, "Calibrating ", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Eyes open and head forward", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                EAR_calibration_data_eyes_open.append(EAR)
                face_pitch_calibration_data.append(face_pitch)
                face_yaw_calibration_data.append(face_yaw)

                face_pitch_baseline = np.mean(face_pitch_calibration_data)
                face_yaw_baseline = np.mean(face_yaw_calibration_data)
                baseline_eyes_open = np.mean(EAR_calibration_data_eyes_open)

            elif time_elapsed > 3 and time_elapsed < 6:
                
                cv2.putText(frame, "Calibrating EAR baseline", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Eyes closed", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                EAR_calibration_data_eyes_closed.append(EAR)
                baseline_eyes_closed = np.mean(EAR_calibration_data_eyes_closed)

                threshold_EAR = baseline_eyes_open - (baseline_eyes_open - baseline_eyes_closed) * 0.8
            elif time_elapsed > 6 and time_elapsed < 8:
                #playsound.playsound('beep-01a.mp3')
                sound.play()
                cv2.putText(frame, "Calibration complete", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if EAR < threshold_EAR: # eyes are 80% closed
                    frames.append(1)
                else:
                    frames.append(0)

                if abs(np.mean([pitch_right_eye, pitch_left_eye]) + (face_pitch - face_pitch_baseline)) > 30 or abs(np.mean([yaw_right_eye, yaw_left_eye]) + (face_yaw - face_yaw_baseline)) > 30:
                    cv2.putText(frame, "DISTRACTED", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # speed reduction (comment out for full speed
            # PERCLOS works best with lower framerate
            time.sleep(1/25) # [s]

        end = time.time()
        total_time = end-start
        time_elapsed = time.time() - start_capture

        if total_time>0:
            fps = 1 / total_time
        else:
            fps=0
            
        if len(frames) > 10 * fps:
            PERCLOS = frames.count(1) / len(frames)

            if PERCLOS > 0.8: # 80% of the time eyes are closed
                cv2.putText(frame, "Drowsy", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                sound.play()
            frames.pop(0)

        cv2.putText(frame, f'FPS : {int(fps)}', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(frame, f'TIME : {round(time_elapsed, 2)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI', frame)             

    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly source and eventual log file
capture.release()