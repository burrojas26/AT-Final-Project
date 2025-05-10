from PoseDetection import PoseDetector
import buildData
from BuildNN import User

def getPose():
    # Instantiate the poseDetector class to use the webcam for live video feed
    poseDetector = PoseDetector(0)
    if not poseDetector.cap.isOpened():
        exit()
    while True:
        frame = poseDetector.getFrame()
        # Get the key angles and the angles that effect form
        angles = buildData.getAngles(frame)
        angles_to_analyze = angles[4:]
        end = False
        # Ensure all the angles are measured
        for angle in angles_to_analyze:
            if angle is None:
                end = True
        if not end:
            # Run the angles through the neural network
            outcome = User().use(angles_to_analyze)
            # Change the color of the pose based on whether the form is good or bad
            if outcome == 1:
                pose_color = (0, 255, 0)
            else:
                pose_color = (0, 0, 255)
            # Display the pose
            poseDetector.showPose(pose_color)


if __name__ == "__main__":
    getPose()
