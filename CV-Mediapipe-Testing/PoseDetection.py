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

    def getFrame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame

    def getPts(self):
        frame = self.getFrame()
        if frame.any():
            imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.pose.process(imgRgb)
            return results

    def showPose(self):
        frame = self.getFrame()
        results = self.getPts()
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            h, w, c = frame.shape
            for id, loc in enumerate(results.pose_landmarks.landmark):
                cX, cY = int(w * loc.x), int(loc.y * h)
                if loc and id in [11, 13, 15, 23, 25, 27, 29]:
                    cv.circle(frame, (cX, cY), 20, (255, 0, 0), cv.FILLED)
            pts = results.pose_landmarks.landmark
            cv.line(frame, (int(pts[11].x*w), int(pts[11].y*h)), (int(pts[13].x*w), int(pts[13].y*h)), (0, 0, 255), 5)
        cv.imshow("Webcam", frame)
        cv.waitKey(5)

    #def drawLine(self, pt1, pt2, ):


    def drawFigure(self):
        canvas = cv.imread("/Users/jasper/Desktop/ATFinal/AT-Final-Project/canvasBlk.avif")
        results = self.getPts()
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(canvas, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, loc in enumerate(results.pose_landmarks.landmark):
                h, w, c = canvas.shape
                cX, cY = int(w * loc.x), int(loc.y * h)
                #if loc and id == 7:
                    #cv.circle(canvas, (cX, cY), 30, (255, 0, 0))
                if loc and id in [11, 13, 15, 23, 25, 27, 29]:
                    cv.circle(canvas, (cX, cY), 5, (255, 0, 0), cv.FILLED)
        cv.imshow("Stick Figure", canvas)
        cv.waitKey(5)

if __name__ == "__main__":
    poseDetector = PoseDetector(0)
    if not poseDetector.cap.isOpened():
        exit()
    while True:
        poseDetector.showPose()



    poseDetector.cap.release()
    cv.destroyAllWindows()

