#!/usr/bin/env python3

"""
This code intends to count feeding activity in crabs.
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
from skimage import measure, filters, feature, exposure, segmentation, color, io
from skimage.future import graph

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.gridspec as gridspec
import time
import pywt

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

# SECONDS
fps = vid.get(cv2.CAP_PROP_FPS)
if args['seconds'] is None:
    target_frame = 1
else:
    target_frame = int(int(args['seconds']) * fps)
vid.set(1, target_frame - 1)

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
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

new_img_bgr = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = new_img_bgr[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array([pixel[0] + 40, pixel[1] + 40, pixel[2] + 40])
        lower =  np.array([pixel[0] - 40, pixel[1] - 40, pixel[2] - 40])
        print(lower, upper)

        image_mask = cv2.inRange(new_img_bgr, lower, upper)
        cv2.imshow("Mask", image_mask)


while True:
    # Read a new frame
    ok, frame_ori = vid.read()

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
        crab = three[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]
        crab_color = frame[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]
        crab_red = red[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]

        hue = one[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]
        sat = two[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]
        val = three[center[1] - 100:center[1] + 100, center[0] - 100:center[0] + 100]


        new_val = exposure.adjust_log(val, 1.25, inv=True)
        new_sat = exposure.adjust_sigmoid(sat, 0.75, inv=True)

        new_img = cv2.merge([new_sat, sat, val])
        new_img_bgr = cv2.cvtColor(new_img, cv2.COLOR_HLS2BGR_FULL)

        bl, gr, re = cv2.split(new_img_bgr)

        canny = cv2.Canny(new_sat, 200, 255)
        _, cnts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(new_img_bgr, cnts, -1, (100, 250, 0), 3)

        #Li Threshold
        thresh_li = filters.threshold_li(new_sat)
        img_thresh_li = sat > thresh_li
        img_tli_ero = cv2.erode(img_thresh_li.astype("uint8") * 255, (3, 3))



        new_fd, new_hog = feature.hog(canny, orientations=9, pixels_per_cell=(20, 20), block_norm="L1",
                                      cells_per_block=(3, 3), transform_sqrt=False, visualize=True, multichannel=False,
                                      feature_vector=False)
        # np.savetxt("array_hog.txt", new_fd, fmt="%s", delimiter=";")
        new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 20))

        # print(crab_red.shape)
        # print(canny.shape)
        # print(new_fd.shape)
        fancy = np.full((8, 8), new_fd[:, :, 0, 0, 0])
        print(fancy)
        plt.clf()
        plt.imshow(fancy, cmap="hot", interpolation="nearest")
        plt.show()
        plt.pause(0.001)


        coeffs = pywt.wavedec2(new_val, wavelet="haar", level=1)
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0
        # coeffs_H[0] = tuple([np.zeros_like(v) for v in coeffs_H[0]])
        imggwv = pywt.waverec2(coeffs_H, wavelet="haar")


        cv2.imshow("Sat0", sat)
        cv2.imshow("Val0", val)
        cv2.imshow("ValSig", new_val.astype("uint8") * 255)
        cv2.imshow("SatSig", new_sat.astype("uint8") * 255)
        cv2.imshow("NBGR", new_img_bgr)
        cv2.imshow("NHSL", new_img)
        cv2.imshow("HOG", new_hog.astype("uint8") * 255)
        cv2.imshow("Ch1", re)
        cv2.imshow("CaDt", canny)
        # cv2.imshow("Wavelet", imggwv.astype("uint8") * 255)
        cv2.imshow("ThreshLI", img_thresh_li.astype("uint8") * 255)
        cv2.imshow("TLI Ero", img_tli_ero)
        cv2.setMouseCallback("NBGR", pick_color)


        # opening = cv2.morphologyEx(crab, cv2.MORPH_OPEN, (11, 11))
        # blur = cv2.GaussianBlur(opening, (5, 5), 0)
        # blur1 = cv2.GaussianBlur(opening, (9, 9), 0)
        #
        # result = blur1
        #
        # _, th4 = cv2.threshold(result, 240, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # th4[th4 == 255] = 1
        #
        # th5 = np.array(th4, dtype=np.uint8)
        # th5[th5 == 1] = 255
        #
        # # image = invert(th4)
        # skeleton = skeletonize(th4)
        # skeleton = np.array(skeleton, dtype=np.uint8)
        # skeleton[skeleton == 1] = 255
        #
        # row0 = np.hstack((crab, result))
        # # row0 = np.hstack((crab_ch, result))
        # row1 = np.hstack((opening, blur))
        # row2 = np.hstack((th5, skeleton))
        #
        # res1 = np.vstack((row1, row2))
        # res = np.vstack((row0, res1))
        #
        # # contours = measure.find_contours(red, 0.8)
        # # new = filters.sobel(crab_red)
        # # new = np.array(new, dtype=np.uint8)
        # # new[new == 1] = 255
        #
        # # Sobel filter
        # # new_x = cv2.Sobel(crab_red, cv2.CV_32F, 1, 0)
        # # new_y = cv2.Sobel(crab_red, cv2.CV_32F, 0, 1)
        # # new_xcvt = cv2.convertScaleAbs(new_x)
        # # new_ycvt = cv2.convertScaleAbs(new_y)
        # # new = cv2.addWeighted(new_xcvt, 0.5, new_ycvt, 0.5, 0)
        #
        # # Adjust exposure with Gamma and Logarithmic correction
        # # new_gamma = exposure.adjust_gamma(opening, 3)
        # # new_log = exposure.adjust_log(opening, 2, inv=True)
        # new_sigmoid = exposure.adjust_sigmoid(opening, cutoff=0.1, gain=15, inv=False)
        # new_sigmoid = cv2.GaussianBlur(new_sigmoid, (5, 5), 0)
        #
        # coeffs = pywt.wavedec2(crab_red, wavelet="haar", level=1)
        # # coeffs = pywt.dwt2(crab_red, wavelet="sym9")
        #
        # coeffs_H = list(coeffs)
        # # coeffs_H[0] *= 0
        # # coeffs_H[0] = tuple([np.zeros_like(v) for v in coeffs_H[0]])
        # imggwv = pywt.waverec2(coeffs_H, wavelet="haar")
        # # imggwv = pywt.idwt2(coeffs_H, wavelet="sym9")
        #
        # # HOG
        # new_fd, new_hog = feature.hog(imggwv, orientations=8, pixels_per_cell=(22, 22), block_norm="L1",
        #                               cells_per_block=(5, 5), transform_sqrt=False, visualize=True, multichannel=False)
        # new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 11))
        #
        # cv2.imshow("res", res)
        # cv2.imshow("Crab color", crab_color)
        #
        # # cv2.imshow("Crab color3", new_gamma.astype("uint8")*255)
        # # cv2.imshow("Crab color3", new_log.astype("uint8")*255)
        # cv2.imshow("Sigmoid correction", new_sigmoid.astype("uint8")*255)
        # cv2.imshow("HOG", new_hog.astype("uint8") * 255)
        # cv2.imshow("PYWT", imggwv.astype("uint8") * 255)
        # cv2.imshow("Crab red", crab_red)
        print(counter)
        pts.appendleft(center)
        counter += 1

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime)
