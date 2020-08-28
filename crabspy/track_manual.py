#!/usr/bin/env python3

"""
Module for tracking one individual in a video using the mouse pointer.
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
import time

import methods, constant

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

resz_val = constant.RESIZE
rect_val = constant.RECT_SIZE

while vid.isOpened():
    ret, frame = vid.read()
    frame = cv2.resize(frame, (0, 0), fx=resz_val, fy=resz_val)
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

M, width, height, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)
center = (0, 0)
mini = np.amin(vertices_draw, axis=0)
maxi = np.amax(vertices_draw, axis=0)

ok, frame = vid.read()
frame = cv2.resize(frame, (0, 0), fx=resz_val, fy=resz_val)
frame = cv2.warpPerspective(frame, M, (width, height))

if not ok:
    print("Cannot read video file")
    sys.exit()

####### MANUAL TRACKING
drawing = False
track_points = []
# Counter_points = 0
position = (0,0)
posmouse = (0,0)

def click(event, x, y, flags, param):
    global drawing, position, posmouse, track_points

    if event== cv2.EVENT_MOUSEMOVE:

        if drawing == False:
            posmouse = (x, y)
            position = (x, y)

        if drawing == True:
            position = (x, y)
            track_points.append(position)

    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        track_points.append(position)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False



# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=100000)
posx = deque(maxlen=constant.DECK)
posy = deque(maxlen=constant.DECK)

counter = 0
(dX, dY) = (0, 0)
startTime = datetime.now()

# From warp.py
fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.ERODE)
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.DILATE)

if constant.MANUAL_ANNOTATION is True:
    try:
        name = video_name + "_" + str(input("* Please enter name for this individual: "))
        # species = str(input("* Please enter species name for this individual: "))
        # sex = str(input("* Please enter sex for this individual: "))
        # handedness = str(input(" *Please enter handedness for this individual: "))
    except ValueError:
        print("Error in input. Using pre-defined information")
        name = video_name + "_" + methods.random_name()
        # species = "unknown"
        # sex = "unknown"
        # handedness = "unknown"
else:
    name = video_name + "_" + methods.random_name()
    # species = "unknown"
    # sex = "unknown"
    # handedness = "unknown"

info = [methods.CompileInformation("name_video", video_name),
        methods.CompileInformation("local_creation", local_creation),
        methods.CompileInformation("creation", creation),
        methods.CompileInformation("length_vid", length_vid),
        methods.CompileInformation("fps", fps),
        methods.CompileInformation("vid_duration", vid_duration),
        methods.CompileInformation("target_frame", target_frame),
        methods.CompileInformation("side", (width, height, side)),
        methods.CompileInformation("conversion", conversion),
        methods.CompileInformation("tracker", "Manual_tracking"),
        methods.CompileInformation("Crab_ID", name)]

info_video = {}
for i in info:
    info_video[i.name] = i.value

if os.path.isfile("results/" + video_name):
    try:
        database = methods.CrabNames.open_crab_names(info_video)
        # target_name = video_name + "_" + name
        print("I am looking this crab name in the database: ", name)

        if name in methods.CrabNames.get_crab_names("results/" + video_name):
            print("Yes, file exists and crab name found")


            for i in database:
                if i.crab_name == name:
                    head_true = False
                    sex = i.sex
                    species = i.species
                    handedness = i.handedness

                else:
                    pass

        else:
            print("Crab name not found in database")
            species = str(input("* Please enter species name for this individual: "))
            sex = str(input("* Please enter sex for this individual: "))
            handedness = str(input(" *Please enter handedness for this individual: "))
            crab_id = methods.CrabNames(name, str("NULL (This is a Manual_tracking)"), species, sex, handedness)
            print(crab_id)
            head_true = True
            methods.data_writer(args["video"], info_video, head_true)

    except (TypeError, RuntimeError):
        pass

else:
    print(video_name, "No, file does not exists")
    head_true = True
    species = str(input("* Please enter species name for this individual: "))
    sex = str(input("* Please enter sex for this individual: "))
    handedness = str(input(" *Please enter handedness for this individual: "))
    crab_id = methods.CrabNames(name, str("Manual_tracking"), species, sex, handedness)
    print(crab_id)
    methods.data_writer(args["video"], info_video, head_true)

methods.CrabNames.save_crab_names(methods.CrabNames.instances, info_video)

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
        img = cv2.resize(img, (0, 0), fx=resz_val, fy=resz_val)
        if pause:
            while True:

                # posmouse = (0, 0)
                key2 = cv2.waitKey(1) & 0xff
                cv2.namedWindow('Manual tracking')
                cv2.setMouseCallback('Manual tracking', click)
                result = cv2.warpPerspective(img, M, (width, height))
                cv2.imshow('Manual tracking', result)
                if key2 == ord('p'):
                    pause = False
                    break

        # crop_img = img[mini[1]:maxi[1], mini[0]:maxi[0]]

        result = cv2.warpPerspective(img, M, (width, height))
        crab_frame = cv2.warpPerspective(img, M, (width, height))
        methods.draw_quadrat(img, vertices_draw)
        cv2.namedWindow('Manual tracking')
        cv2.setMouseCallback('Manual tracking', click)

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

        info = [methods.CompileInformation("Width", ""),
                methods.CompileInformation("Height", ""),
                methods.CompileInformation("Area", ""),
                methods.CompileInformation("Frame", target_frame + counter),
                methods.CompileInformation("Time_absolute", str(time_absolute)),
                methods.CompileInformation("Time_since_start", str(time_since_start)),
                methods.CompileInformation("Crab_ID", name),
                methods.CompileInformation("Crab_Position_x", ""),
                methods.CompileInformation("Crab_Position_y", ""),
                methods.CompileInformation("Crab_Position_cx", ""),
                methods.CompileInformation("Crab_Position_cy", ""),
                methods.CompileInformation("Counter", counter),
                methods.CompileInformation("Species", species),
                methods.CompileInformation("Sex", sex),
                methods.CompileInformation("Handedness", handedness)]

        for i in info:
            info_video[i.name] = i.value

        if len(track_points) >= 1:
            if drawing:
                center_bbox = track_points[-1]

                p1 = (int(center_bbox[0]) - rect_val, int(center_bbox[1]) - rect_val)
                p2 = (int(center_bbox[0]) + rect_val, int(center_bbox[1]) + rect_val)
                cv2.rectangle(result, p1, p2, (204, 204, 100), 2)
                cv2.rectangle(masked, p1, p2, (204, 204, 0))
                cv2.putText(masked, str(track_points[-1]), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 100))
                posx.appendleft(center_bbox[0])
                posy.appendleft(center_bbox[1])
                centx = mean(posx)
                centy = mean(posy)
                center = (int(centx), int(centy))

                for i in info:
                    info_video[i.name] = i.value

                if center[0]-rect_val <= 0:
                    crab = crab_frame[center[1] - rect_val:center[1] + rect_val,
                           0:center[0] + rect_val]
                    crab_snapshot = crab.copy()
                elif center[1]-rect_val <= 0:
                    crab = crab_frame[0:center[1] + rect_val,
                           center[0] - rect_val:center[0] + rect_val]
                    crab_snapshot = crab.copy()
                else:
                    crab = crab_frame[center[1] - rect_val:center[1] + rect_val, center[0] - rect_val:center[0] + rect_val]
                    crab_snapshot = crab.copy()
                pts.appendleft(center)

                blob = fb_res_two3[center[1] - rect_val:center[1] + rect_val, center[0] - rect_val:center[0] + rect_val]
                ret, blob = cv2.threshold(blob, 150, 255, cv2.THRESH_BINARY)
                output = cv2.connectedComponentsWithStats(blob, 4, cv2.CV_32S)
                num_labels = output[0]
                stats = output[2]

                # Computing the connected components for image, and show them in window.
                _, label = cv2.connectedComponents(blob)
                if np.max(label) != 0:
                    try:
                        label_hue = np.uint8(179 * label / np.max(label))
                        # print(stats)
                        blank_ch = 255 * np.ones_like(label_hue)
                        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
                        # cvt to BGR for display
                        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
                        # set bg label to black
                        labeled_img[label_hue == 0] = 0
                        cv2.imshow('labeled components', labeled_img)

                        M_blob = cv2.moments(blob)
                        Mx_blob = int(M_blob["m10"] / M_blob["m00"])
                        My_blob = int(M_blob["m01"] / M_blob["m00"])
                        cx = Mx_blob + int(center_bbox[0]-rect_val)
                        cy = My_blob + int(center_bbox[1]-rect_val)
                        # if (cx, cy) is not None:
                        cv2.circle(result, (cx, cy), 3, (240, 240, 255), -1)
                        cv2.circle(result, (cx, cy), rect_val, (180, 210, 10), 1)
                        # print(cx, cy)
                    except TypeError as e:
                        pass

                crab_size = cv2.Canny(blob, threshold1=100, threshold2=200)
                _, cnts_size, _ = cv2.findContours(crab_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(cnts_size) != 0:

                    cnts_size_sorted = sorted(cnts_size, key=lambda x: cv2.contourArea(x))
                    # Grab second largest contour
                    contour_size = cnts_size_sorted[-1]
                    # Grab larger contour from contours list
                    # contour = max(cnts, key=cv2.contourArea)
                    # print('This is maximum contour ', contour.shape)
                    # Finding min and max coordinates in left-right axis
                    min_LR_size = tuple(contour_size[contour_size[:, :, 0].argmin()][0])
                    max_LR_size = tuple(contour_size[contour_size[:, :, 0].argmax()][0])
                    # Finding min and max coordinates in top-bottom axis
                    min_TB_size = tuple(contour_size[contour_size[:, :, 1].argmin()][0])
                    max_TB_size = tuple(contour_size[contour_size[:, :, 1].argmax()][0])

                    # # Create dictionary of moments for the contour
                    # Mo = cv2.moments(contour_size)
                    #
                    # if 0 in (Mo["m10"], Mo["m00"], Mo['m01'], Mo['m00']):
                    #     centroid_x = 0
                    #     centroid_y = 0
                    #     pass
                    # # if Mo["m00"] != 0: # and then else for other cases: cx, cy = 0, 0
                    # else:
                    #     # Calculate centroid coordinates
                    #     # https://docs.opencv.org/4.0.0/dd/d49/tutorial_py_contour_features.html
                    #     centroid_x = int(Mo["m10"] / Mo["m00"])
                    #     centroid_y = int(Mo['m01'] / Mo['m00'])

                    cv2.circle(crab, min_LR_size, 1, (0, 0, 255), -1)
                    cv2.circle(crab, max_LR_size, 1, (0, 255, 0), -1)
                    cv2.circle(crab, min_TB_size, 1, (255, 0, 0), -1)
                    cv2.circle(crab, max_TB_size, 1, (255, 255, 0), -1)

                    cv2.line(crab, min_LR_size, max_LR_size, (0, 100, 255), 1)
                    cv2.line(crab, min_TB_size, max_TB_size, (255, 100, 0), 1)

                    dist_LRx = (max_LR_size[0] - min_LR_size[0]) ** 2
                    dist_LRy = (max_LR_size[1] - min_LR_size[1]) ** 2
                    dist_LRman = math.sqrt(dist_LRx + dist_LRy)

                    dist_TBx = (max_TB_size[0] - min_TB_size[0]) ** 2
                    dist_TBy = (max_TB_size[1] - min_TB_size[1]) ** 2
                    dist_TBman = math.sqrt(dist_TBx + dist_TBy)

                for label in range(1, num_labels):
                    blob_area = stats[label, cv2.CC_STAT_AREA] * (conversion[0]*conversion[1])
                    # print("This is the area ", blob_area, "ID=", label)
                    blob_width = stats[label, cv2.CC_STAT_WIDTH] * conversion[0]
                    # print("This is the width ", blob_width, "ID=", label)
                    blob_height = stats[label, cv2.CC_STAT_HEIGHT] * conversion[1]
                    # print("This is the height ", blob_height, "ID=", label)
                    # print("CV2 width ", blob_width, " height ", blob_height)

                if num_labels == 1:
                    info = [methods.CompileInformation("Width", ""),
                            methods.CompileInformation("Height", ""),
                            methods.CompileInformation("Area", ""),
                            methods.CompileInformation("Frame", target_frame + counter),
                            methods.CompileInformation("Time_absolute", str(time_absolute)),
                            methods.CompileInformation("Time_since_start", str(time_since_start)),
                            methods.CompileInformation("Crab_ID", name),
                            methods.CompileInformation("Crab_Position_x", center[0]),
                            methods.CompileInformation("Crab_Position_y", center[1]),
                            methods.CompileInformation("Crab_Position_cx", ""),
                            methods.CompileInformation("Crab_Position_cy", ""),
                            methods.CompileInformation("Counter", counter),
                            methods.CompileInformation("Species", species),
                            methods.CompileInformation("Sex", sex),
                            methods.CompileInformation("Handedness", handedness)]

                else:
                    info = [methods.CompileInformation("Width", blob_width),
                            methods.CompileInformation("Height", blob_height),
                            methods.CompileInformation("Area", blob_area),
                            methods.CompileInformation("Frame", target_frame + counter),
                            methods.CompileInformation("Time_absolute", str(time_absolute)),
                            methods.CompileInformation("Time_since_start", str(time_since_start)),
                            methods.CompileInformation("Crab_ID", name),
                            methods.CompileInformation("Crab_Position_x", center[0]),
                            methods.CompileInformation("Crab_Position_y", center[1]),
                            methods.CompileInformation("Crab_Position_cx", cx),
                            methods.CompileInformation("Crab_Position_cy", cy),
                            methods.CompileInformation("Counter", counter),
                            methods.CompileInformation("Species", species),
                            methods.CompileInformation("Sex", sex),
                            methods.CompileInformation("Handedness", handedness)]

                for i in info:
                    info_video[i.name] = i.value

                if constant.SNAPSHOT == True:
                    methods.save_snapshot(crab_snapshot, args["video"], info_video)

                try:
                    crab = cv2.resize(crab, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_LANCZOS4)
                    cv2.imshow("Crab", crab)
                    cv2.imshow("Blob", blob)
                except cv2.error as e:
                    pass


                methods.data_writer(args["video"], info_video, False)

        percentage_vid = (target_frame + counter) / length_vid * 100
        text = "Video {0:.1f} %".format(percentage_vid)
        cv2.putText(masked, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 2)
        cv2.putText(masked, "Frame n. {0:d}".format(target_frame + counter),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 2)

        result_1 = cv2.warpPerspective(result, IM, (img.shape[1], img.shape[0]))
        result_1 = cv2.addWeighted(img, 0.5, result_1, 0.5, 0)

        # cv2.imshow("background substraction", fb_res_two3)
        cv2.imshow("masked", masked)
        cv2.imshow("cropped", result_1)
        cv2.imshow('Manual tracking', result)
        # cv2.imshow("result", result)
        time.sleep(args["timesleep"])
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("p"):
            pause = True

vid.release()
cv2.destroyAllWindows()
