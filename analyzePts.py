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
        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            pts = {}
            ptsKey = {}
            with open("landmarksKey.txt", "r") as myKey:
                rawKey = myKey.read()
            for i in rawKey.split("\n"):
                parsedKey = i.split(" ")
                name = ""
                for j in range(1, len(parsedKey)):
                    name += str(j) + "_"
                name.removesuffix("_")
                ptsKey[parsedKey[0]] = name

            for i, pt in enumerate(results.pose_landmarks.landmark):
                pts[ptsKey[i]] = pt
                h, w, c = frame.shape
                cX, cY = int(w*pt.x), int(pt.y*h)
                if pt:
                    cv.circle(frame, (cX, cY), 10, (255, 0, 0), cv.FILLED)
            print(pts)
        cv.imshow("Webcam", frame)
        cv.waitKey(5)



cap.release()
cv.destroyAllWindows()


'''
            for pt in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cX, cY = int(w*pt.x), int(pt.y*h)
                if pt:
                    cv.circle(frame, (cX, cY), 10, (255, 0, 0), cv.FILLED)
'''