# Script to test media pipe's pose detection

import cv2 as cv
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture(0)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    exit()
while True:

    ret, frame = cap.read()
    if ret:
        imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(imgRgb)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for loc in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cX, cY = int(w*loc.x), int(loc.y*h)
                if loc:
                    cv.circle(frame, (cX, cY), 10, (255, 0, 0), cv.FILLED)
        cv.imshow("Webcam", frame)
        cv.waitKey(5)


cap.release()
cv.destroyAllWindows()