from PoseDetection import PoseDetector
from buildData import AngleAnalyzer
from UseNN import User
import time

def getPose():
    # Instantiate the poseDetector class to use the webcam for live video feed
    start = time.time()
    poseDetector = PoseDetector(0)
    angleAnalyzer = AngleAnalyzer()

    if not poseDetector.cap.isOpened():
        exit()

    angleCombos = angleAnalyzer.getCombos()
    frameCount = 0
    pose_color = (255, 255, 255)
    pts = None
    while True:
        frame = poseDetector.getFrame()
        stop = time.time()
        print("Frame received: " + str((stop-start)*10**3))
        start = stop
        if frameCount % 1 == 0:
            # Get the key angles and the angles that effect form
            anglesAndLandmarks = angleAnalyzer.getAngles(frame, angleCombos, start)
            stop = time.time()
            print("Angles received: " + str((stop - start) * 10 ** 3))
            start = stop

            angles = anglesAndLandmarks[0]
            pts = anglesAndLandmarks[1]
            angles_to_analyze = angles[4:]
            if angles_to_analyze[0] and angles_to_analyze[1] >= 178:
                pose_color = (255, 255, 255)
            else:
                end = False
                # Ensure all the angles are measured
                for angle in angles_to_analyze:
                    if angle is None:
                        end = True
                if not end:
                    # Run the angles through the neural network
                    outcome = User().use(angles_to_analyze)
                    stop = time.time()
                    print("Outcome received: " + str((stop - start) * 10 ** 3))
                    start = stop

                    # Change the color of the pose based on whether the form is good or bad
                    if outcome == 1:
                        pose_color = (0, 255, 0)
                    else:
                        pose_color = (0, 0, 255)
                    # Display the pose
        if pts is not None:
            poseDetector.showPose(pose_color, pts, frame)
            stop = time.time()
            print("Pose displayed: " + str((stop - start) * 10 ** 3))
            start = stop

        start = time.time()
        frameCount += 1


if __name__ == "__main__":
    getPose()
