#!/usr/bin/env python3

"""
Module for generating a binary video only representing individuals movements as blobs"
"""

import os
import cv2
import numpy as np
import argparse
from collections import deque
import sys
from datetime import datetime
from statistics import mean
import math

import methods
import constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="GP010016.mov", help="Provide path to video file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide time in seconds of target video section showing the key points")
# ap.add_argument("-c", "--crab_id", default="crab_", help="Provide a name for the crab to be tracked")
args = vars(ap.parse_args())

# Return video information
video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(args["video"])
local_creation, creation = methods.get_file_creation(args["video"])
# Set frame where video should start to be read
vid, target_frame = methods.set_video_star(vid, args["seconds"], fps)

if target_frame > length_vid:
    print("You have provided a time beyond the video duration.\n"
          "Video duration is {} seconds".format(round(vid_duration, 2)))
    sys.exit("Crabspy halted")

while vid.isOpened():
    ret, frame = vid.read()

    methods.enable_point_capture(constant.CAPTURE_VERTICES)
    frame = methods.draw_points_mousepos(frame, methods.quadratpts, methods.posmouse)
    cv2.imshow("Vertices selection", frame)

    if len(methods.quadratpts) == 4:
        print("Vertices were captured. Coordinates in pixels are: top-left {}, top-right {}, "
              "bottom-left {}, and bottom-right {}".format(*methods.quadratpts))
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # print("Q - key pressed. Window quit by user")
        break

# vid.release()
cv2.destroyAllWindows()

M, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)
center = (0, 0)
mini = np.amin(vertices_draw, axis=0)
maxi = np.amax(vertices_draw, axis=0)

ok, frame = vid.read()
frame = cv2.warpPerspective(frame, M, (side, side))

if not ok:
    print("Cannot read video file")
    sys.exit()


counter = 0
startTime = datetime.now()
cv2.destroyAllWindows()

# From warp.py
# fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20, detectShadows=False)
# fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)
for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))


info = [methods.CompileInformation("name_video", video_name),
        methods.CompileInformation("local_creation", local_creation),
        methods.CompileInformation("creation", creation),
        methods.CompileInformation("length_vid", length_vid),
        methods.CompileInformation("fps", fps),
        methods.CompileInformation("vid_duration", vid_duration),
        methods.CompileInformation("target_frame", target_frame),
        methods.CompileInformation("side", side),
        methods.CompileInformation("conversion", conversion)]

info_video = {}
for i in info:
    info_video[i.name] = i.value

start, end, step, _, _ = methods.frame_to_time(info_video)
print("Recording was started at: ", start, "\nRecording was ended at: ", end,
      "\nThis information might not be precise as it depends on your computer file system, and file meta information")

while vid.isOpened():
    _, img = vid.read()

    if img is None:
        break

    else:
        crop_img = img[mini[1]-10:maxi[1]+10, mini[0]-10:maxi[0]+10]
        result = cv2.warpPerspective(img, M, (side, side))

        methods.draw_quadrat(img, vertices_draw)

        startTime1 = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        hsl = cv2.cvtColor(result, cv2.COLOR_BGR2HLS_FULL)
        one, two, three = cv2.split(hsl)
        model = fgbg3.apply(two, learningRate=-1)
        model = cv2.erode(model, for_er)
        model = cv2.dilate(model, for_di)

        masked = cv2.bitwise_and(result, result, mask=model)
        masked = cv2.addWeighted(result, 0.2, masked, 0.8, 0)

        percentage_vid = (target_frame + counter) / length_vid * 100
        text = "Video {0:.1f} %".format(percentage_vid)
        cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        cv2.putText(result, "Frame n. {0:d}".format(target_frame + counter), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        cv2.imshow("original", result)
        cv2.imshow("model", model)
        cv2.imshow("result", masked)

        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

vid.release()
cv2.destroyAllWindows()
