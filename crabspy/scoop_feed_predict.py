#!/usr/bin/env python3

"""
Predcit feeding activity based on SVM.
"""

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
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import measure, filters, feature, exposure, segmentation, color
from skimage.future import graph
from sklearn import svm
from joblib import load


__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"


'''
Parser arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="VIRB0037-3.MP4", help='Provide path to video file')
ap.add_argument('-s', '--seconds', default=640,
                help='Provide time in seconds of target video section showing the key points')
args = vars(ap.parse_args())

clf = load("scoop_feed_model_v3.sav")

# Read video
# vid = cv2.VideoCapture(args['video'])
vid = cv2.VideoCapture('video/' + args['video'])

fname = os.path.basename(args['video'])
filename, file_ext = os.path.splitext(fname)
print(fname)
dirname = 'samples_pos'

startTime = datetime.now()

# SECONDS
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

# ave
(rAvg, gAvg, bAvg) = (None, None, None)
total_ave = 0


position = (0, 0)
center = (0, 0)

tracker = cv2.TrackerBoosting_create()

# Read first frame.q
ok, frame = vid.read()
frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
# bbox = (650, 355, 25, 25)
# bbox = cv2.selectROI('tracking select', frame, fromCenter=False)
bbox = (357, 431, 182, 108)

print(bbox)
cv2.destroyAllWindows()

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=10000)
counter = 0
(dX, dY) = (0, 0)
direction = ""

fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20)
# fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold=10)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)

# for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
for_di1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# out = cv2.VideoWriter('Uca_detection+tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (960,720))
# BG_MODEL = cv2.imread('BG_model.jpg')
hull_list = []
feeding_counter = 0
switch = False
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

while True:
    # Read a new frame
    ok, frame_ori = vid.read()

    if not ok:
        break

    if ok:
        frame = cv2.resize(frame_ori, (0, 0), fx=0.5, fy=0.5)
        frame_norec = frame.copy()

        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        one, two, three = cv2.split(hsl)
        blue, green, red = cv2.split(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Update tracker
        ok, bbox = tracker.update(frame)
        # print(position)
        position = (bbox[0], bbox[1])

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

        # crab = gray[center[1]-100:center[1]+100, center[0]-100:center[0]+100]
        crab = three[int(bbox[1]) + 5: int(bbox[1] + bbox[3]) - 5, int(bbox[0]) + 5: int(bbox[0] + bbox[2]) - 5]
        # crab = three[center[1]-100:center[1]+100, center[0]-100:center[0]+100]
        crab_color = frame[int(bbox[1]) + 5: int(bbox[1] + bbox[3]) - 5, int(bbox[0]) + 5: int(bbox[0] + bbox[2]) - 5]
        # crab_color = frame[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]
        crab_red = red[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]

        crab_ori = crab.copy()
        crab = cv2.resize(crab, (200, 200), interpolation=cv2.INTER_CUBIC)
        crab_color_ori = crab_color.copy()
        crab_color = cv2.resize(crab_color, (200, 200), interpolation=cv2.INTER_CUBIC)


        opening = cv2.morphologyEx(crab, cv2.MORPH_OPEN, (11, 11))
        blur = cv2.GaussianBlur(opening, (5, 5), 0)
        blur1 = cv2.GaussianBlur(opening, (9, 9), 0)

        frame_ana = cv2.resize(crab_color, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        hsl_ana = cv2.cvtColor(frame_ana, cv2.COLOR_BGR2HLS_FULL)
        one_ana, two_ana, three_ana = cv2.split(hsl_ana)
        new_sat_ana = exposure.adjust_sigmoid(two_ana, 0.75, inv=True)
        # new_img = cv2.merge([new_sat, two, three])
        canny = cv2.Canny(new_sat_ana, 200, 255)

        new_fd, new_hog = feature.hog(canny, orientations=9, pixels_per_cell=(20, 20), block_norm="L1",
                                      cells_per_block=(3, 3), transform_sqrt=False, visualize=True, multichannel=False,
                                      feature_vector=True)
        new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 20))

        new_fd = np.array(new_fd)
        # print(new_fd.shape)
        new_fd = new_fd.reshape(1, -1)
        # print(new_fd.shape)
        result = clf.predict(new_fd)[0]

        if switch:
            if result == "claw_down":
                pass
            else:
                switch = False
        else:
            if result == "claw_up":
                pass
            else:
                feeding_counter =+ 1
                switch = True

        print(result, "__", switch)

        text = "Feeding scoop counter {}".format(feeding_counter)
        pts.appendleft(center)
        # print(text)
        cv2.imshow("Crab color2", new_hog.astype("uint8")*255)
        # crab_color_ori = cv2.resize(crab_color_ori, (0, 0), fx=2.5, fy=2.5)
        cv2.putText(crab_color_ori, str(feeding_counter), (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 10), 1)
        cv2.imshow("Crab color", crab_color_ori)

        counter += 1

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

vid.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime)
