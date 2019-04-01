#!/usr/bin/env python3

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
from skimage.morphology import skeletonize
from skimage.util import invert

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


position = (0, 0)
center = (0, 0)

tracker = cv2.TrackerBoosting_create()

# Read first frame.q
ok, frame = vid.read()
frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
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

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))


while True:
    # Read a new frame
    ok, frame_ori = vid.read()

    if not ok:
        break

    if ok:
        frame = cv2.resize(frame_ori, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Update tracker
        ok, bbox = tracker.update(frame)
        # print(position)
        position = (bbox[0], bbox[1])

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))


        center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

        crab = gray[center[1]-100:center[1]+100, center[0]-100:center[0]+100]


        opening = cv2.morphologyEx(crab, cv2.MORPH_OPEN, (3,3))
        blur = cv2.GaussianBlur(opening, (5,5), 0)
        blur1 = cv2.GaussianBlur(opening, (9,9), 0)

        result = blur1

        _, th4 = cv2.threshold(result, 240, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        th4[th4 == 255] = 1



        th5 = np.array(th4, dtype=np.uint8)
        th5[th5 == 1] = 255
        # image = invert(th4)
        skeleton = skeletonize(th4)
        skeleton = np.array(skeleton, dtype= np.uint8)
        skeleton[skeleton == 1] = 255

        row0 = np.hstack((crab, result))
        row1 = np.hstack((opening, blur))
        row2 = np.hstack((th5, skeleton))

        res1 = np.vstack((row1, row2))
        res = np.vstack((row0, res1))

        # skel = np.zeros(crab.shape, np.uint8)
        # eroded = cv2.erode(crab, element)
        # dilated =cv2.dilate(eroded, element)
        # result = cv2.subtract(crab, dilated)
        # skel = cv2.bitwise_or(skel, result)

        pts.appendleft(center)

        # cv2.imshow("Tracking", frame)
        # cv2.imshow('Crab', crab)

        # cv2.imshow('eroded', eroded)
        # cv2.imshow('dilated', dilated)
        # cv2.imshow('result', result)
        # cv2.imshow('Skel', skel)

        # cv2.imshow("opening", opening)
        # cv2.imshow("blur", blur)
        # cv2.imshow("th4", th4)
        # cv2.imshow("th5", th5)
        # cv2.imshow("image", image)
        # cv2.imshow("skeleton", skeleton)

        cv2.imshow("res", res)

        counter += 1

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

vid.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime)
