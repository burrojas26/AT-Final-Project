import cv2 as cv
import os

# Receive the paths of each video
path = "/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Videos"
fileNames = os.listdir(path)
paths = []
for filename in fileNames:
  paths.append(os.path.join(path, filename))

# Capture Each video
for vid in paths:
  capture = cv.VideoCapture(vid)
  check = True
  while check:
    check, frame = capture.read()
    print(check)
    if not check:
      break
    fps = capture.get(cv.CAP_PROP_FPS)
    print(fps)
    #cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('q'):
      break

# TODO: decrease the frame rate of the videos and make them go frame by frame to allow for human input

capture.release()
cv.destroyAllWindows()