#!/usr/bin/env python3

"""
Module for tracking one individual in a video.
"""

import os
import cv2
import numpy as np
import argparse
from collections import deque
import sys
from datetime import datetime
from statistics import mean

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

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, MIL, KCF, TLD, MEDIANFLOW or GOTURN

# tracker = cv2.Tracker_create("BOOSTING")
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMedianFlow_create()
tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# print(tracker)
# Define an initial bounding box
# bbox = (650, 355, 25, 25)
bbox = cv2.selectROI("tracking select", frame, fromCenter=False)
crab_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
print(crab_center)

if constant.MANUAL_ANNOTATION is True:
    try:
        name = video_name + "_" + str(input("* Please enter name for this individual: "))
        species = str(input("* Please enter species name for this individual: "))
        sex = str(input("* Please enter sex for this individual: "))
        handedness = str(input(" *Please enter handedness for this individual: "))
    except ValueError:
        print("Error in input. Using pre-defined information")
        name = video_name + "_" + methods.random_name()
        species = "unknown"
        sex = "unknown"
        handedness = "unknown"
else:
    name = video_name + "_" + methods.random_name()
    species = "unknown"
    sex = "unknown"
    handedness ="unknown"


# crab_id = args["video"] + "_" + args["crab_id"] + str(crab_center)
# print(crab_id)

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=100000)
posx = deque(maxlen=constant.DECK)
posy = deque(maxlen=constant.DECK)

counter = 0
(dX, dY) = (0, 0)
direction = ""

startTime = datetime.now()

cv2.destroyAllWindows()

# From warp.py
fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.ERODE)
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.DILATE)
# for_di1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# out = cv2.VideoWriter("Uca_detection.avi",
#                       cv2.VideoWriter_fourcc("M", "J", "P", "G"), 24, (464, 464))

info = [methods.CompileInformation("name_video", video_name),
        methods.CompileInformation("local_creation", local_creation),
        methods.CompileInformation("creation", creation),
        methods.CompileInformation("length_vid", length_vid),
        methods.CompileInformation("fps", fps),
        methods.CompileInformation("vid_duration", vid_duration),
        methods.CompileInformation("target_frame", target_frame),
        methods.CompileInformation("side", side),
        methods.CompileInformation("conversion", conversion),
        methods.CompileInformation("tracker", str(tracker)),
        methods.CompileInformation("Crab_ID", name)]

info_video = {}
for i in info:
    info_video[i.name] = i.value
# print(info_video)

# print(name, "\n" + species, "\n" + sex, "\n" + handedness)

# crab_id = methods.CrabNames(name, str(crab_center), species, sex, handedness)
# print(crab_id)

# try:
#     methods.CrabNames.open_crab_names(info_video)
# except:
#     pass


if os.path.isfile("results/" + video_name):
    try:
        methods.CrabNames.open_crab_names(info_video)
        # target_name = video_name + "_" + name
        print("I am looking this crab name in the database: ", name)
        if name in methods.CrabNames.get_crab_names("results/" + video_name ):
            print("Yes, file exists and crab name found")
            head_true = False
        else:
            crab_id = methods.CrabNames(name, str(crab_center), species, sex, handedness)
            print(crab_id)
            head_true = True
            print("No, file exists and crab name was not found")
            methods.data_writer(args["video"], info_video, head_true)
    except (TypeError, RuntimeError):
        pass
# if not os.path.isfile("results" + video_name):
else:
    print(video_name, "No, file does not exists")
    head_true = True
    crab_id = methods.CrabNames(name, str(crab_center), species, sex, handedness)
    print(crab_id)
    methods.data_writer(args["video"], info_video, head_true)

methods.CrabNames.save_crab_names(methods.CrabNames.instances, info_video)

# methods.data_writer(args["video"], info_video, head_true)
# result_file.close()

# if name in methods.CrabNames.get_crab_names("results/GP010016"):
#     print("head_true set to False")
#     head_true = False
# else:
#     head_true = True
#     print("head_true set to True")


start, end, step, _, _ = methods.frame_to_time(info_video)
print("The video recording was started at: ", start, "\nThe video recording was ended at: ", end,
      "\nThis information might not be precise as it depends on your computer file system")

while vid.isOpened():
    _, img = vid.read()
    # print(img.shape)
    # img = cv2.resize(img, (640,400))
    crop_img = img[mini[1]-10:maxi[1]+10, mini[0]-10:maxi[0]+10]

    result = cv2.warpPerspective(img, M, (side, side))
    crab_frame = cv2.warpPerspective(img, M, (side, side))
    # result_speed = result
    # print(crop_img.shape)
    # print("Dimensions for result are: ", result.shape)
    # result_1 = cv2.warpPerspective(result, IM, (682,593))

    methods.draw_quadrat(img, vertices_draw)
    # cv2.polylines(img, np.int32([quadratpts]), True, (204, 204, 0), thickness=2)

    # From warp.py
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    hsl = cv2.cvtColor(result, cv2.COLOR_BGR2HLS_FULL)
    one, two, three = cv2.split(hsl)
    fb_res_two3 = fgbg3.apply(two, learningRate=-1)
    fb_res_two3 = cv2.erode(fb_res_two3, for_er)
    fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
    masked = cv2.bitwise_and(result, result, mask=fb_res_two3)

    masked = cv2.addWeighted(result, 0.3, masked, 0.7, 0)
    edge = cv2.Canny(masked, threshold1=100, threshold2=230)

    # cv2.circle(masked, (52,85), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (382,13), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (225,132), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (313,298), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (291,205), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (446,249), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (369,98), 2, (240, 10, 10), 2)
    # cv2.circle(masked, (163, 335), 2, (240, 10, 10), 2)

    # Update tracker
    ok, bbox = tracker.update(masked)
    # ok, bbox = tracker.update(result)
    # ok, bbox = tracker.update(result)
    # print(position)
    position1 = (bbox[0], bbox[1])

    startTime1 = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

    # wr.writerow(position)
    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(result, p1, p2, (204, 204, 100), 2)
        cv2.rectangle(masked, p1, p2, (204, 204, 0))
        # cv2.circle(result, (180,180), 3, (0, 204, 100), 3)

        # center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        posx.appendleft(int(bbox[0] + bbox[2] / 2))
        posy.appendleft(int(bbox[1] + bbox[3] / 2))
        centx = mean(posx)
        centy = mean(posy)
        center = (int(centx), int(centy))

        _, _, _, time_absolute, time_since_start = methods.frame_to_time(info_video)

        # info = [methods.CompileInformation("Frame", target_frame + counter),
        #         methods.CompileInformation("Time_absolute", str(time_absolute)),
        #         methods.CompileInformation("Time_since_start", str(time_since_start)),
        #         methods.CompileInformation("Crab_ID", name),
        #         methods.CompileInformation("Crab_Position_x", center[0]),
        #         methods.CompileInformation("Crab_Position_y", center[1]),
        #         methods.CompileInformation("Counter", counter),
        #         methods.CompileInformation("Species", species),
        #         methods.CompileInformation("Sex", sex),
        #         methods.CompileInformation("Handedness", handedness)]

        for i in info:
            info_video[i.name] = i.value

        crab = crab_frame[center[1] - 15:center[1] + 15, center[0] - 15:center[0] + 15]
        crab_snapshot = crab.copy()
        # crab = masked[center[1] - 15:center[1] + 15, center[0] - 15:center[0] + 15]
        # crab = frame[int(bbox[0] + bbox[2]/2):100, int(bbox[1] + bbox[3]/2):100]
        # crab = frame[100:(100 + 50), 250:(250 + 50)]
        # filename = os.path.join(dirname, fname, str(center), startTime1)
        # cv2.imwrite(dirname + "/" + filename + "_" + startTime1 + str(center) + "_" + ".jpg", crab)

        crab_edge = cv2.Canny(crab, threshold1=100, threshold2=200)
        _, cnts, _ = cv2.findContours(crab_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) != 0:

            cnts_sorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
            # Grab second largest contour
            contour = cnts_sorted[-1]
            # Grab larger contour from contours list
            # contour = max(cnts, key=cv2.contourArea)
            # print('This is maximum contour ', contour.shape)
            # Finding min and max coordinates in left-right axis
            min_LR = tuple(contour[contour[:, :, 0].argmin()][0])
            max_LR = tuple(contour[contour[:, :, 0].argmax()][0])
            # Finding min and max coordinates in top-bottom axis
            min_TB = tuple(contour[contour[:, :, 1].argmin()][0])
            max_TB = tuple(contour[contour[:, :, 1].argmax()][0])

            # Create dictionary of moments for the contour
            Mo = cv2.moments(contour)

            if 0 in (Mo["m10"], Mo["m00"], Mo['m01'], Mo['m00']):
                centroid_x = 0
                centroid_y = 0
                pass
            # if Mo["m00"] != 0: # and then else for other cases: cx, cy = 0, 0
            else:
                # Calculate centroid coordinates
                # https://docs.opencv.org/4.0.0/dd/d49/tutorial_py_contour_features.html
                centroid_x = int(Mo["m10"] / Mo["m00"])
                centroid_y = int(Mo['m01'] / Mo['m00'])

            cv2.circle(crab, min_LR, 1, (0, 0, 255), -1)
            cv2.circle(crab, max_LR, 1, (0, 255, 0), -1)
            cv2.circle(crab, min_TB, 1, (255, 0, 0), -1)
            cv2.circle(crab, max_TB, 1, (255, 255, 0), -1)

            cv2.line(crab, min_LR, max_LR, (0, 100, 255), 1)
            cv2.line(crab, min_TB, max_TB, (255, 100, 0), 1)

        pts.appendleft(center)
        # print(center)
        # print(pts)
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
                # dX = pts[i][0] - pts[-5][0]
                dY = pts[-5][1] - pts[i][1]
                # dY = pts[i][1] - pts[-5][1]
                dX = int(dX*0.11)
                dY = int(dY*0.11)
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
            if thickness == 0:
                thickness = 1

            # cv2.line(result, pts[i - 1], pts[i], (204, 204, 0), thickness)
            cv2.line(result, pts[i - 1], pts[i], (54, 54, 250), thickness)

        # show the movement deltas and the direction of movement on
        # the frame
        direction = "Uca movement " + str(direction)
        # cv2.putText(result, direction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (10, 10, 10), 2)
        # cv2.putText(result, "Displacement (cm) dx: {}, dy: {}".format(dX, dY),
        #             (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (10, 10, 10), 2)

        # Back transform and show tracker and data in original image

    blob = fb_res_two3[center[1] - 15:center[1] + 15, center[0] - 15:center[0] + 15]
    ret, blob = cv2.threshold(blob, 85, 255, cv2.THRESH_BINARY)
    output = cv2.connectedComponentsWithStats(blob, 4, cv2.CV_32S)
    num_labels = output[0]
    stats = output[2]
    # print("Number of labels ", num_labels)
    # Stat matrix contains in order: leftmost coord, topmost coord, width, height, and area
    # print("Stat matrix is ", stats)
    for label in range(1, num_labels):
        blob_area = stats[label, cv2.CC_STAT_AREA] * conversion
        # print("This is the area ", blob_area, "ID=", label)
        blob_width = stats[label, cv2.CC_STAT_WIDTH] * conversion
        # print("This is the width ", blob_width, "ID=", label)
        blob_height = stats[label, cv2.CC_STAT_HEIGHT] * conversion
        # print("This is the height ", blob_height, "ID=", label)

    if num_labels == 1:
        info = [methods.CompileInformation("Width", ""),
                methods.CompileInformation("Height", ""),
                methods.CompileInformation("Area", ""),
                methods.CompileInformation("Frame", target_frame + counter),
                methods.CompileInformation("Time_absolute", str(time_absolute)),
                methods.CompileInformation("Time_since_start", str(time_since_start)),
                methods.CompileInformation("Crab_ID", name),
                methods.CompileInformation("Crab_Position_x", ""),
                methods.CompileInformation("Crab_Position_y", ""),
                methods.CompileInformation("Counter", counter),
                methods.CompileInformation("Species", species),
                methods.CompileInformation("Sex", sex),
                methods.CompileInformation("Handedness", handedness)]
        # do not save position

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
                methods.CompileInformation("Counter", counter),
                methods.CompileInformation("Species", species),
                methods.CompileInformation("Sex", sex),
                methods.CompileInformation("Handedness", handedness)]

    for i in info:
        info_video[i.name] = i.value


    if constant.SNAPSHOT == True:
        methods.save_snapshot(crab_snapshot, args["video"], info_video)


    percentage_vid = (target_frame + counter) / length_vid * 100
    text = "Video {0:.1f} %".format(percentage_vid)
    cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
    cv2.putText(result, "Frame n. {0:d}".format(target_frame + counter), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)

    # counter_f += 1
    # print("Frame count ", counter_f)
    # if counter_f == 60:
    #     counter_f = 0
    #     cv2.imshow("One every ten", result)

    # DISPLAY (Multiple panels video)
    # edge_3_ch = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # fb_res_two3_3_ch = cv2.cvtColor(fb_res_two3, cv2.COLOR_GRAY2BGR)
    # original_reshape = cv2.resize(crop_img, (809, 928))
    # display00 = np.hstack((edge_3_ch, fb_res_two3_3_ch))
    # display01 = np.hstack((masked, result))
    # display03 = np.vstack((display00, display01))
    #
    # display = np.hstack((original_reshape, display03))
    #
    # cv2.line(display, (809, 0), (809,928), (239, 170, 0), 6)
    # cv2.line(display, (1273, 0), (1273,928), (239, 170, 0), 4)
    # cv2.line(display, (1737, 464), (809,464), (239, 170, 0), 4)
    #
    # display = cv2.resize(display, (0,0), fx=.5, fy=.5)
    # print(display.shape)
    # out.write(result)

    # cv2.imshow("result_1", result_1)
    # cv2.imshow("original", img)
    # cv2.imshow("cropped", crop_img)
    # crab = cv2.resize(crab, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    # cv2.imshow("Crab", crab)
    # cv2.imshow("Crab Edge", crab_edge)
    # cv2.imshow("result", result)
    cv2.imshow("Blob", blob)

    # From warp.py
    cv2.imshow("background substraction", fb_res_two3)
    cv2.imshow("masked", masked)
    cv2.imshow("result", result)
    # cv2.imshow("Canny Edges", edge)
    # cv2.imshow("display00", display00)
    # cv2.imshow("display01", display01)
    # cv2.imshow("display", display)
    methods.data_writer(args["video"], info_video, False)

    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
# result_file.close()
