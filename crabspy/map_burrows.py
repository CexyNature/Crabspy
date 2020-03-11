#!/usr/bin/env python3

"""
This script allows users to map burrow positions
"""

import os
import cv2
import numpy as np
import argparse
import sys
from datetime import datetime
import time
import pandas as pd

import methods
import constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="GP010016.mov", help="Provide path to video file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide the targeted time in seconds of video section you want to jump to")
ap.add_argument("-f", "--frame", default=None, type=int,
                help="Provide the targeted frame of video section you want to jump to")
ap.add_argument("-t", "--timesleep", default=0, type=float,
                help="Provide time in seconds to wait before showing next frame")
# ap.add_argument("-c", "--crab_id", default="crab_", help="Provide a name for the crab to be tracked")
args = vars(ap.parse_args())

# Return video information
video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(args["video"])
local_creation, creation = methods.get_file_creation(args["video"])
# Set frame where video should start to be read
vid, target_frame = methods.set_video_star(vid, args["seconds"], args["frame"], fps)

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

existing_burrows = []
burrows = []
burrows_counter = 0
existing_burrows_counter = 0
new_burrows_counter = 0
position = (0, 0)
posmouse = (0, 0)

def click(event, x, y, flags, param):
    global burrows, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        position = (x, y)
        burrow_info = position, vid.get(1)
        burrows.append(burrow_info)

    if event == cv2.EVENT_MOUSEMOVE:
        posmouse = (x, y)


counter = 0
startTime = datetime.now()

fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.ERODE)
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.DILATE)

info = [methods.CompileInformation("name_video", video_name),
        methods.CompileInformation("local_creation", local_creation),
        methods.CompileInformation("creation", creation),
        methods.CompileInformation("length_vid", length_vid),
        methods.CompileInformation("fps", fps),
        methods.CompileInformation("vid_duration", vid_duration),
        methods.CompileInformation("target_frame", target_frame),
        methods.CompileInformation("side", side),
        methods.CompileInformation("conversion", conversion),
        methods.CompileInformation("tracker", "Manual_tracking"),
        methods.CompileInformation("Crab_ID", None)]

info_video = {}
for i in info:
    info_video[i.name] = i.value


if os.path.isfile("results/" + video_name + "_burrows_map.csv"):
    try:
        print("Burrows map file found. Burrows coordinates will be loaded.")
        head_true = False
        burrows_meta = pd.read_csv("results/" + video_name + "_burrows_map.csv", header=0, nrows=1)
        burrows_coord = pd.read_csv("results/" + video_name + "_burrows_map.csv", header=2, skiprows=range(0, 1))
        # print(burrows_coord)
        for i, rows in burrows_coord.iterrows():
            row_values = [(int(rows.Burrow_coord_x), int(rows.Burrow_coord_y)), int(rows.Frame_number)]
            # print(row_values)
            existing_burrows.append(row_values)

    except (TypeError, RuntimeError):
        print("Exiting because of TypeError or RuntimeError")
        pass

else:
    print("There is not burrows map file for this video, creating one.")
    head_true = True
    methods.burrow_writer(args["video"], info_video, None, head_true)


start, end, step, _, _ = methods.frame_to_time(info_video)
print("Recording was started at: ", start, "\nRecording was ended at: ", end,
      "\nThis information might not be precise as it depends on your computer file system, and file meta information.",
      "\nPress the 'p' key to pause and play the video.")

pause = True

while vid.isOpened():
    _, img = vid.read()
    key = cv2.waitKey(1) & 0xFF

    if img is None:
        break
    else:
        if pause:
            while True:

                key2 = cv2.waitKey(1) & 0xff
                cv2.namedWindow('Burrow counter')
                cv2.setMouseCallback('Burrow counter', click)
                result = cv2.warpPerspective(img, M, (side, side))
                cv2.imshow('Burrow counter', result)
                if key2 == ord('p'):
                    pause = False
                    break

        crop_img = img[mini[1]-10:maxi[1]+10, mini[0]-10:maxi[0]+10]

        result = cv2.warpPerspective(img, M, (side, side))
        crab_frame = cv2.warpPerspective(img, M, (side, side))
        methods.draw_quadrat(img, vertices_draw)
        cv2.namedWindow('Burrow counter')
        cv2.setMouseCallback('Burrow counter', click)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        hsl = cv2.cvtColor(result, cv2.COLOR_BGR2HLS_FULL)
        one, two, three = cv2.split(hsl)
        fb_res_two3 = fgbg3.apply(two, learningRate=-1)
        fb_res_two3 = cv2.erode(fb_res_two3, for_er)
        fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
        masked = cv2.bitwise_and(result, result, mask=fb_res_two3)

        masked = cv2.addWeighted(result, 0.3, masked, 0.7, 0)
        edge = cv2.Canny(masked, threshold1=100, threshold2=230)

        startTime1 = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

        _, _, _, time_absolute, time_since_start = methods.frame_to_time(info_video)

        for i, val in enumerate(existing_burrows):
            cv2.circle(result, val[0], 3, (255, 5, 205), 2)
            cv2.circle(masked, val[0], 3, (255, 5, 205), 2)
            existing_burrows_counter = i + 1

        for i, val in enumerate(burrows):
            cv2.circle(result, val[0], 3, (0, 255, 0), 2)
            cv2.circle(masked, val[0], 3, (0, 255, 0), 2)
            new_burrows_counter = i + 1

        burrows_counter = existing_burrows_counter + new_burrows_counter

        result_1 = cv2.warpPerspective(result, IM, (img.shape[1], img.shape[0]))
        result_1 = cv2.addWeighted(img, 0.5, result_1, 0.5, 0)

        cv2.putText(result_1, "Number of burrows {}".format(burrows_counter), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(result_1, "Last burrow coordinate {}".format(position), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.putText(result_1, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)

        percentage_vid = (target_frame + counter) / length_vid * 100
        text = "Video {0:.1f} %".format(percentage_vid)
        cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        cv2.putText(result, "Frame n. {0:d}".format(target_frame + counter),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)

        cv2.imshow("Original perspective", result_1)
        cv2.imshow("Background subtracted", masked)
        cv2.imshow('Burrow counter', result)

        time.sleep(args["timesleep"])
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("p"):
            pause = True

vid.release()
cv2.destroyAllWindows()

head_true = False

if len(burrows) > 0:
    print('Writing new coordinates to file.')
    for i in burrows:
        try:
            methods.burrow_writer(args["video"], info_video, i, head_true)
        except (TypeError, RuntimeError):
            print('A TypeError or RuntimeError was caught while writing burrows coordinates')
            pass
else:
    print('No new coordinates to write.')
    pass