import cv2 as cv
import os

def getAngles(frame):
  return [1, 2, 3]


# Receive the paths of each video
path = "/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Edited_Videos"
fileNames = os.listdir(path)
paths = []
for filename in fileNames:
  paths.append(os.path.join(path, filename))
paths.remove("/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Edited_Videos/.DS_Store")
# Capture Each video
for vid in paths:
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
    if binary == 2:
      break
    angles = getAngles(frame)
    with open("squatData.csv", "a") as data:
      data.write(str(currFrame) + ", " + str(angles) + ", " + str(binary) + "\n")
    currFrame += 1

# TODO: Make the function to calculate the angles

capture.release()
cv.destroyAllWindows()