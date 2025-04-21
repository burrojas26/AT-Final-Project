# Script to test media pipe's pose detection

import cv2 as cv
import mediapipe as mp


class PoseDetector:
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    def __init__(self, videoPath):
        self.videoPath = videoPath
        self.cap = cv.VideoCapture(self.videoPath)
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    def showPose(self):
        ret, frame = self.cap.read()
        if ret:
            imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRgb)
            print(self.results.pose_landmarks)
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for loc in self.results.pose_landmarks.landmark:
                    h, w, c = frame.shape
                    cX, cY = int(w * loc.x), int(loc.y * h)
                    if loc:
                        cv.circle(frame, (cX, cY), 10, (255, 0, 0), cv.FILLED)
            cv.imshow("Webcam", frame)
            cv.waitKey(5)


if __name__ == "__main__":
    poseDetector = PoseDetector(0)
    if not poseDetector.cap.isOpened():
        exit()
    while True:
        poseDetector.showPose()



    poseDetector.cap.release()
    cv.destroyAllWindows()

