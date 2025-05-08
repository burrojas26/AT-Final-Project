import yt_dlp as ydl
import cv2 as cv
import os

def downloadOrig():
    # Open the links file
    with open("links2.txt", "r") as theLinks:
        links = theLinks.read().split("\n")

    # Configuring ydl options
    ydl_opts = {
        'format': 'bestvideo[height<=720]',
        'outtmpl': '/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Videos/%(id)s.%(ext)s'
    }

    # download each video
    with ydl.YoutubeDL(ydl_opts) as yd:
        yd.download(links)

def getDict():
    with open("links2.txt", "r") as links:
        keys = links.read().split("\n")
    with open("times.txt", "r") as times:
        values = times.read().split("\n")
    return dict(zip(keys, values))


def alterVids():
    # Alters the videos to decrease frame rate and cut to specified time
    path = "/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Videos"
    pathOut = "/Users/jasper/Desktop/ATFinal/AT-Final-Project/Dataset/Edited_Videos"
    fileNames = os.listdir(path)
    paths = []
    # for filename in fileNames:
    #     paths.append(os.path.join(path, filename))

    timesDict = getDict()
    print(timesDict)



    for i, filename in enumerate(fileNames):
        # Create capture object and split the start and end times for the current video
        capture = cv.VideoCapture(os.path.join(path, filename))
        times = timesDict[filename[:11]].split(",")
        fps = capture.get(cv.CAP_PROP_FPS)
        # Calculate the start and end frames based on the times and fps
        start = (int(times[0][0]) * 60 + int(times[0][2:])) * fps
        end = (int(times[1][0]) * 60 + int(times[1][2:])) * fps

        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object and create path for output
        fourcc = cv.VideoWriter_fourcc(*"avc1")
        outPath = pathOut + "/" + str(i) + ".mov"
        out = cv.VideoWriter(outPath, fourcc, 30.0, (frame_width, frame_height))



        check = True
        # current frame number
        currFrame = 0
        # Whether to add the frame or skip
        currAdd = 0
        # Number of frames in the edited version
        totalFrames = 0
        print(path)
        while check:
            check, frame = capture.read()
            if not check:
                break
            # If the frame is between the start and end frames and is not being skipped
            if start <= currFrame <= end and currAdd <= 0:
                # Add the frame to the edited version and display the frame
                totalFrames += 1
                out.write(frame)
                currAdd = 4
                cv.imshow("frame", frame)
                print(currFrame)
            elif currFrame > end:
                cv.destroyAllWindows()
                break
            currAdd -= 1
            currFrame += 1
        capture.release()
        out.release()
        print(totalFrames)


if __name__ == "__main__":
    alterVids()


