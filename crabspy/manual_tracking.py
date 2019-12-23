#!/usr/bin/env python3

import cv2
import csv
from datetime import datetime
import argparse
import os

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

startTime = datetime.now().strftime("%Y%m%d-%H%M%S")

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="video/GP010016.mov", help="Provide path to video file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide time in seconds of target video section showing the key points")
# ap.add_argument("-c", "--crab_id", default="crab_", help="Provide a name for the crab to be tracked")
args = vars(ap.parse_args())





video_name = os.path.basename(args["video"])
video_name = os.path.splitext(video_name)[0]
print(video_name)

#Open bin for data
resultFile = open("results/" + video_name + "_MT_" + startTime +  ".csv", "w", newline='\n')
wr = csv.writer(resultFile, delimiter=",")
wr.writerow(['Coord x','Coord y', 'Frame clicked'])

drawing = False
track_points = []
Counter_points = 0
position = (0,0)
posmouse = (0,0)


def click(event, x, y, flags, param):
    global drawing, track_points, position, posmouse, resultFile

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        position = (x, y)
    elif event== cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            position = (x, y)
            track_points.append(position)
            info = x, y, vid.get(1)
            wr.writerow(info)
            # wr.writerow([position, vid.get(1)])
            resultFile.flush()
            print(position)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    if event == cv2.EVENT_MOUSEMOVE:
        posmouse = (x, y)

vid = cv2.VideoCapture(args['video'])
fgbg = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=225, detectShadows=False)

Nframes = vid.get(7)
print("Total number of frames = " + str(Nframes))

while (vid.isOpened()):
    ret, frame = vid.read()
    frame2 = cv2.resize(frame, (1440,810))
    hsl = cv2.cvtColor(frame2, cv2.COLOR_BGR2HLS_FULL)
    one, two, three = cv2.split(hsl)
    blobs = fgbg.apply(two)
    blobs = cv2.erode(blobs, (3,3), iterations=1)

    res = cv2.bitwise_or(blobs, two)

    res = cv2.merge((res,res,blobs))


    if ret == True:
        cv2.namedWindow('Movement detections')
        # cv2.namedWindow('Video')
        cv2.setMouseCallback('Movement detections', click)
        # cv2.setMouseCallback('Video', click)

        for i, val in enumerate(track_points):
            cv2.circle(res, val, 1, (0, 255, 255), 1)
            Counter_points = i + 1

        cv2.putText(res, "Number of frames tracked {}".format(Counter_points), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        cv2.putText(res, "Last position coordinate {}".format(position), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.putText(res, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    1)

        cv2.imshow('Movement detections', res)
        # cv2.imshow('Video', frame2)
        # cv2.imshow("blobs", blobs)

        key = cv2.waitKey(1) & 0xFF
        if key ==27:
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
resultFile.close()