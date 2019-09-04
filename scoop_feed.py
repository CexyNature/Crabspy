'''
This code intends to count feeding activity in crabs
'''
import argparse
import cv2
import sys
import csv
from datetime import datetime
import os
from collections import deque
import numpy as np

# from random import randint

import random as rng

'''
Parser arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="VIRB0037-3.MP4", help='Provide path to video file')
ap.add_argument('-s', '--seconds', default=640,
                help='Provide time in seconds of target video section showing the key points')
args = vars(ap.parse_args())


# Read video
# vid = cv2.VideoCapture(args['video'])
vid = cv2.VideoCapture('video/' + args['video'])

fname = os.path.basename(args['video'])
filename, file_ext = os.path.splitext(fname)
print(fname)
dirname = 'samples_pos'

startTime = datetime.now()

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

'''
This section creates a background model for do background substraction'''

### ave
(rAvg, gAvg, bAvg) = (None, None, None)
total_ave = 0

# while True:
#     ok, frame_ori = vid.read()
#     frame = cv2.resize(frame_ori, (0,0), fx=0.5, fy=0.5)
#     if not ok:
#         break
#     ### ave
#     (B, G, R) = cv2.split(frame.astype("float"))
#     # if the frame averages are None, initialize them
#     if rAvg is None:
#         rAvg = R
#         bAvg = B
#         gAvg = G
#
#     # otherwise, compute the weighted average between the history of
#     # frames and the current frames
#     else:
#         rAvg = ((total_ave * rAvg) + (1 * R)) / (total_ave + 1.0)
#         gAvg = ((total_ave * gAvg) + (1 * G)) / (total_ave + 1.0)
#         bAvg = ((total_ave * bAvg) + (1 * B)) / (total_ave + 1.0)
#
#     # increment the total number of frames read thus far
#     total_ave += 1
#     # merge the RGB averages together and write the output image to disk
#     avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")
#     cv2.imshow('Video averaging', frame)
#
#     cv2.imwrite('BG_model.jpg', avg)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#         break
#
# cv2.destroyAllWindows()

# resultFile = open("Tracking.csv", "w", newline='\n')
# wr = csv.writer(resultFile, delimiter=",")
# wr.writerow(['Coord x', 'Coord y'])
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


# Read first frame.q
ok, frame = vid.read()
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
# bbox = (650, 355, 25, 25)
bbox = cv2.selectROI('tracking select', frame, fromCenter=False)
# bbox = (357, 431, 182, 108)

print(bbox)
cv2.destroyAllWindows()

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

fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=20)
# fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold=10)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history= 5000, dist2Threshold=250)

# for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20, 20))
for_di1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3, 3))

# out = cv2.VideoWriter('Uca_detection+tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (960,720))
# BG_MODEL = cv2.imread('BG_model.jpg')
hull_list = []
while True:
    # Read a new frame
    ok, frame_ori = vid.read()
    frame = cv2.resize(frame_ori, (0,0), fx=0.5, fy=0.5)
    _, crab_frame = vid.read()
    crab_frame = cv2.resize(frame_ori, (0,0), fx=0.5, fy=0.5)

    # gray = cv2.cvtColor(crab_frame, cv2.COLOR_BGR2GRAY)
    # hsl = cv2.cvtColor(crab_frame, cv2.COLOR_BGR2HLS_FULL)
    # one, two, three = cv2.split(hsl)
    # fb_res_two3 = fgbg3.apply(gray, learningRate=-1)
    # fb_res_two3 = cv2.erode(fb_res_two3, for_er)
    # fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
    # masked = cv2.bitwise_and(crab_frame, crab_frame, mask=fb_res_two3)
    # edge = cv2.Canny(gray, threshold1=100, threshold2=230)


    # crab_frame = BG_MODEL - crab_frame
    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)
    # print(position)
    position = (bbox[0], bbox[1])

    startTime1 = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

    # wr.writerow(position)
    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        # p1 = (int(bbox[0]+r_box0), int(bbox[1]+r_box1))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # p2 = (int(bbox[0] + bbox[2] + r_box2), int(bbox[1] + bbox[3]+r_box3))
        # cv2.rectangle(frame, p1, p2, (0, 0, 255))

        center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))


        crab = crab_frame[center[1]-100:center[1]+100, center[0]-100:center[0]+100]
        # crab = frame[int(bbox[0] + bbox[2]/2):100, int(bbox[1] + bbox[3]/2):100]
        # crab = frame[100:(100 + 50), 250:(250 + 50)]
        # filename = os.path.join(dirname, fname, str(center), startTime1)
        cv2.imwrite(dirname + '/' + filename + '_' + startTime1 + str(center) + '_' + '.jpg', crab)

        pts.appendleft(center)
        # print(center)
        # wr.writerow(center)
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

    # gray = cv2.cvtColor(crab, cv2.COLOR_BGR2GRAY)
    # hsl = cv2.cvtColor(crab, cv2.COLOR_BGR2HLS_FULL)
    # one, two, three = cv2.split(hsl)
    # fb_res_two3 = fgbg3.apply(gray, learningRate=-1)
    # fb_res_two3 = cv2.erode(fb_res_two3, for_er)
    # fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
    # masked = cv2.bitwise_and(crab, crab, mask=fb_res_two3)
    # edge = cv2.Canny(gray, threshold1=100, threshold2=230)

    crab_smooth = cv2.bilateralFilter(frame, 9, 5, 5)
    crab_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crab_fb = fgbg3.apply(frame, learningRate=-1)

    crab_smooth0 = cv2.bilateralFilter(frame, 9, 5, 5)
    crab_smooth1 = cv2.bilateralFilter(frame, 9, 25, 25)
    crab_smooth2 = cv2.bilateralFilter(frame, 9, 50, 55)
    # crab_smooth3 = cv2.bilateralFilter(crab_gray, 9, 50, 5)

    result = crab_smooth0 - crab_smooth1 - crab_smooth2
    result_blur = cv2.GaussianBlur(result, (21,21), 0)
    edge = cv2.Canny(result, threshold1=10, threshold2=150)

    result_f = result - result_blur

    # _, contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # # Find the convex hull object for each contour
    # for i, cnt in enumerate(contours):
    #
    #     x, y, w, h = cv2.boundingRect(cnt)
    #
    #     if w > 70 and h > 70 and w < 120 and h < 120:
    #         for i in range(len(contours)):
    #             hull = cv2.convexHull(contours[i])
    #             hull_list.append(hull)
    #         # Draw contours + hull results
    #         drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
    #         for i in range(len(contours)):
    #             color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #             cv2.drawContours(drawing, contours, i, color)
    #             cv2.drawContours(drawing, hull_list, i, color)
    #         # Show in a window
    # cv2.imshow('Contours', drawing)

    # blur = cv2.GaussianBlur(crab_fb, (5, 5), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # crab_fb = cv2.erode(crab_fb, for_er)

    # Display result
    # print(frame.shape)
    # frame = frame[100:600,100:800]
    # print(frame.shape)
    # out.write(frame)
    # cv2.imshow("Tracking", frame)
    # cv2.imshow("BG model", crab_frame)
    # cv2.imshow('Crab', crab)
    # cv2.imshow('Crab Smooth', crab_smooth)
    cv2.imshow('Crab Smooth0', crab_smooth0)
    cv2.imshow('Crab Smooth1', crab_smooth1)
    cv2.imshow('Crab Smooth2', crab_smooth2)
    # cv2.imshow('Crab Smooth3', crab_smooth3)
    cv2.imshow('result', result)
    cv2.imshow('result_blur', result_blur)
    cv2.imshow('result_f', result_f)
    # cv2.imshow('Crab Smooth FB', crab_fb)
    # cv2.imshow('Crab blur', blur)
    # cv2.imshow('Crab th', th3)
    # cv2.imshow('gray', gray)
    # cv2.imshow('hsl', hsl)
    # cv2.imshow('background substraction', fb_res_two3)
    # cv2.imshow('masked', masked)
    cv2.imshow('edge', edge)
    # cv2.imshow('ave', avg)
    counter += 1

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
vid.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime)

