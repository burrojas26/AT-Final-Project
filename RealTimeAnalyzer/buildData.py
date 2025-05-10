import cv2 as cv
import os
import mediapipe as mp
import numpy as np


def getPts(frame):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    if frame.any():
        imgRgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(imgRgb)
    if results.pose_landmarks:
        pts = results.pose_landmarks.landmark
    return results.pose_landmarks


def getAngles(frame):
    landmarks = getPts(frame)
    angles = []
    for i in range(10):
        angles.append(None)
    '''
    Elbow Angle - shoulder - elbow - wrist
        11 - 13 - 15 - left
        12 - 14 - 16 - right
    Shoulder angle - elbow - shoulder - hip
        13 - 11 - 23 - left
        14 - 12 - 24 - right
    Hip angle - shoulder - hip - knee
        11 - 23 - 25 - left
        12 - 24 - 26 - right
    Knee angle - hip - knee - ankle
        23 - 25 - 27 - left
        24 - 26 - 28 - right
    Ankle angle - knee - ankle - index
        25 - 27 - 31 - left
        26 - 28 - 32 - right
  '''
    if landmarks:
        with open("angles.txt", "r") as angleReference:
            anglesLines = angleReference.readlines()
        i = 0
        for line in anglesLines:
            if line[0].isdigit():
                combo = line.split(" - ")
                # Convert points to np arrays for easier math
                point = landmarks.landmark
                a = np.array([point[int(combo[0])].x, point[int(combo[0])].y, point[int(combo[0])].z])
                b = np.array([point[int(combo[1])].x, point[int(combo[1])].y, point[int(combo[1])].z])
                c = np.array([point[int(combo[2])].x, point[int(combo[2])].y, point[int(combo[2])].z])

                # Creating vectors
                ba = a - b
                bc = c - b

                # Calculate angle based on vector formula
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # clip avoids numerical errors
                angles[i] = (np.degrees(angle))
                i += 1

    return angles


def makeData():
    # Receive the paths of each video
    currPath = os.getcwd()
    vidPath = os.path.join(currPath, "Edited_Videos")
    fileNames = os.listdir(vidPath)
    print(fileNames)
    paths = []
    for filename in fileNames:
        paths.append(os.path.join(vidPath, filename))
    # Remove the file .ds store
    paths.remove(os.path.join(vidPath, ".DS_Store"))
    # Capture Each video
    for i, vid in enumerate(paths):
        capture = cv.VideoCapture(vid)
        check = True
        currFrame = 0
        fps = capture.get(cv.CAP_PROP_FPS)
        while check:
            check, frame = capture.read()
            if not check:
                break
            cv.imshow("frame", frame)
            cv.waitKey(2)
            binary = int(input("Is this form good or bad? 1 for good, 0 for bad, 2 for do not include: "))
            if binary != 2:
                # Get the body angles
                angles = getAngles(frame)
                # Write the data to the csv
                with open("squatData.csv", "a") as data:
                    data.write(str(i) + "," + str(currFrame) + ",")
                    for angle in angles:
                        data.write(str(angle) + ",")
                    data.write(str(binary) + "\n")
                print("Successfully added frame to dataset!")
            currFrame += 1

        capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    makeData()
