'''
This code does a visualization of how SSD MobileNet looks like.
You have to check that out.write dimensions match dimensions of resulted frame.
'''
import argparse
import cv2
import sys
import csv
from datetime import datetime
import os
from collections import deque
import numpy as np

from random import randint

'''
Parser arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="GP010016.MP4", help='Provide path to video file')
ap.add_argument('-s', '--seconds', default=None,
                help='Provide time in seconds of target video section showing the key points')
args = vars(ap.parse_args())

'''

'''
fname = os.path.basename(args['video'])
filename, file_ext = os.path.splitext(fname)
print(fname)
dirname = 'samples_pos'

startTime = datetime.now()

resultFile = open("Tracking.csv", "w", newline='\n')
wr = csv.writer(resultFile, delimiter=",")
wr.writerow(['Coord x', 'Coord y'])
position = (0, 0)
center = (0, 0)

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, MIL, KCF, TLD, MEDIANFLOW or GOTURN

# tracker = cv2.Tracker_create('BOOSTING')
tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMedianFlow_create()
# tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()

# Read video
# vid = cv2.VideoCapture(args['video'])
vid = cv2.VideoCapture('video/' + args['video'])

####SECONDS
fps = vid.get(cv2.CAP_PROP_FPS)
if args['seconds'] is None:
    target_frame = 1
else:
    target_frame = int(int(args['seconds']) * fps)
vid.set(1, target_frame-1)


# Exit if video not opened.
if not vid.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.q
ok, frame = vid.read()
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
# bbox = (650, 355, 25, 25)
bbox = cv2.selectROI('tracking select', frame, fromCenter=False)

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=10000)
counter = 0
(dX, dY) = (0, 0)
direction = ""

out = cv2.VideoWriter('Uca_detection+tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (960,720))

while True:
    # Read a new frame
    ok, frame = vid.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    _, crab_frame = vid.read()
    # frame = cv2.resize(frame,)
    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)
    # print(position)
    position = (bbox[0], bbox[1])

    startTime1 = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]


    ##randint

    r_box0 = randint(5,40)
    r_box1 = randint(5,20)
    r_box2 = randint(5,40)
    r_box3 = randint(5,20)
    r_det = randint(59,95)
    t_det = 'Uca spp: ' + str(r_det) + '%'
    # print(t_det)

    # wr.writerow(position)
    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        # p1 = (int(bbox[0]+r_box0), int(bbox[1]+r_box1))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # p2 = (int(bbox[0] + bbox[2] + r_box2), int(bbox[1] + bbox[3]+r_box3))
        # cv2.rectangle(frame, p1, p2, (0, 0, 255))

        center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
        cv2.rectangle(frame, (center[0]-40-r_box0, center[1]-40-r_box1), (center[0]+40+r_box2, center[1]+40+r_box3) , (0, 0, 255))
        cv2.rectangle(frame, (center[0]-40-r_box0, center[1]-40-r_box1-7), (center[0]-40-r_box0+60, center[1]-40-r_box1), (0, 0, 255),
                      cv2.FILLED)
        cv2.putText(frame, t_det, (center[0]-40-r_box0, center[1]-40-r_box1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (1,1,1),
                    1, 8)

        crab = crab_frame[center[1]-15:center[1]+15, center[0]-15:center[0]+15]
        # crab = frame[int(bbox[0] + bbox[2]/2):100, int(bbox[1] + bbox[3]/2):100]
        # crab = frame[100:(100 + 50), 250:(250 + 50)]
        # filename = os.path.join(dirname, fname, str(center), startTime1)
        cv2.imwrite(dirname + '/' + filename + '_' + startTime1 + str(center) + '_' + '.jpg', crab)

        pts.appendleft(center)
        # print(center)
        wr.writerow(center)
        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 5 and i == 1 and pts[-5] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = pts[-5][0] - pts[i][0]
                dY = pts[-5][1] - pts[i][1]

                # print(dX, dY)
                (dirX, dirY) = ("", "")

                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > 2:
                    dirX = "East" if np.sign(dX) == 1 else "West"

                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 2:
                    dirY = "North" if np.sign(dY) == 1 else "South"

                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)

                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(10 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the movement deltas and the direction of movement on
        # the frame
        # cv2.putText(frame, direction, (560, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (255, 0, 255), 3)
        # cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        #             (560, 60), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (255, 0, 255), 3)

    # Display result
    # print(frame.shape)
    # frame = frame[100:600,100:800]
    # print(frame.shape)
    out.write(frame)
    cv2.imshow("Tracking", frame)
    # cv2.imshow('Crab', crab)
    counter += 1

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
vid.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime)
