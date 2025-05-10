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

    def showPose(self, color, pts, frame):
        # Allows for the changing of the color
        drawing_spec = self.mpDraw.DrawingSpec(thickness=5, circle_radius=5, color=color)
        if pts:
            self.mpDraw.draw_landmarks(frame, pts, self.mpPose.POSE_CONNECTIONS, drawing_spec, drawing_spec)
        cv.imshow("Webcam", frame)
        cv.waitKey(2)
        cv.destroyAllWindows()

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

