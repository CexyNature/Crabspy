# from scipy import ndimage
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import argparse
import csv
import os
import time
from collections import deque
import sys
from datetime import datetime

import methods

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="GP010016_fast.mp4", help='Provide path to video file')
ap.add_argument('-d', '--dimension', default=[50, 50, 50, 50], nargs='+', type=int, help='Provide dimension each side')
ap.add_argument('-s', '--seconds', default=None,
                help='Provide time in seconds of target video section showing the key points')
ap.add_argument('-q', '--quadrat_position', type=bool, default=False,
                help='Should quadrat vertices be selected')
ap.add_argument('-c', '--crab_id', default='crab_', help='Provide a name for the crab to be tracked')
args = vars(ap.parse_args())

"""
This section creates list and/or tuples
"""
# Create list for holding quadrat position
quadrat_pts = []
# Set initial default position of first point and mouse
position = (0, 0)
posmouse = (0, 0)

"""
This section creates the video object from default or parser argument.
It also creates the result file, and include high level metadata associated to video.
"""
vid = cv2.VideoCapture('video/' + args['video'])
dim = args['dimension']
# Print quadrat size sides in cm
print(dim)

# Total length of video in frames
length_vid = vid.get(cv2.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv2.CAP_PROP_FPS)

if args['seconds'] is None:
    target_frame = 1
else:
    target_frame = int(int(args['seconds']) * fps)

# print('video length is ', length_vid, '\nFPS is', fps,  '\ntarget frame is ', target_frame)

# # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
vid.set(1, target_frame-1)

path = os.path.basename(args['video'])
file_name, file_ext = os.path.splitext(path)
dir_results = 'results'
# print(file_name)
# print(file_ext)

date_now = time.strftime('%d%m%Y')
time_now = time.strftime('%H%M')
name_resultFile = 'results/' + file_name + '_' + str(date_now) + '_' + str(time_now) + '.csv'
resultFile = open(name_resultFile, 'w', newline='\n')
wr = csv.writer(resultFile, delimiter=',')
wr.writerow(['file_name', 'processed_at_date', 'processed_at_time', 'length_video', 'fps_video',
             'target_frame_used', 'vertice_1', 'vertice_2', 'vertice_3', 'vertice_4',
             'projected_q_side', 'q_factor_distance', 'tracker_method'])


"""
This section call the mouse interacting function to select quadrat's vertices in video. 
Then, depending in the parser argument open an interacting window so the user can select vertices.
Otherwise assign quadrat's vertices as predefined arguments.
"""
# Define mouse click function
def click(event, x, y, flags, param):
    global quadrat_pts, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        position = (x, y)
        quadrat_pts.append(position)
        # print(quadrat_pts)

    if event == cv2.EVENT_MOUSEMOVE:
        posmouse = (x, y)

if args['quadrat_position'] is False:
    quadrat_pts = [(628, 105), (946, 302), (264, 393), (559, 698)]

else:
    while vid.isOpened():
        ret, frame = vid.read()
        # quadrat = [(628, 105), (946, 302), (264, 393), (559, 698)]
        # cv2.circle(frame, quadrat[0], 5, (0, 255, 0), -1)
        # cv2.circle(frame, quadrat[1], 5, (255, 0, 0), -1)
        # cv2.circle(frame, quadrat[2], 5, (0, 0, 255), -1)
        # cv2.circle(frame, quadrat[3], 5, (255, 255, 255), -1)

        cv2.namedWindow('Select vertices quadrat')
        cv2.setMouseCallback('Select vertices quadrat', click)
        for i, val in enumerate(quadrat_pts):
            cv2.circle(frame, val, 3, (204, 204, 0), 2)
        cv2.putText(frame, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204, 204, 0), 2)

        cv2.imshow('Select vertices quadrat', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # print("Q - key pressed. Window quit by user")
            break

    vid.release()
    cv2.destroyAllWindows()

print(quadrat_pts)

# Re-arrange (i.e. sort) quadrat's vertices so they can be plotted later as polyline.
vertices = np.array([quadrat_pts[0],quadrat_pts[1], quadrat_pts[3], quadrat_pts[2]], np.int32)
print('The vertices are ', vertices)

"""
Mathematical calculations are done to estimate centimer to pixel ratio.
"""

vid = cv2.VideoCapture('video/' + args['video'])

if args['seconds'] is None:
    target_frame = 1
else:
    target_frame = int(int(args['seconds']) * fps)

# print('video length is ', length_vid, '\nFPS is', fps,  '\ntarget frame is ', target_frame)

# # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
vid.set(1, target_frame-1)

orig_pts = np.float32([quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]])
counter_f = 0
# frame_r = vid.get(cv2.CAP_PROP_FPS)
# print(frame_r)

# dist = math.hypot(x2-x1, y2-y1)
dist_a = math.sqrt((quadrat_pts[0][0] - quadrat_pts[1][0])**2 + (quadrat_pts[0][1] - quadrat_pts[1][1])**2)
dist_b = math.sqrt((quadrat_pts[0][0] - quadrat_pts[2][0])**2 + (quadrat_pts[0][1] - quadrat_pts[2][1])**2)
dist_c = math.sqrt((quadrat_pts[2][0] - quadrat_pts[3][0])**2 + (quadrat_pts[2][1] - quadrat_pts[3][1])**2)
dist_d = math.sqrt((quadrat_pts[3][0] - quadrat_pts[2][0])**2 + (quadrat_pts[3][1] - quadrat_pts[2][1])**2)

width = int(max(dist_a, dist_c))
width_10 = int(max(dist_a, dist_c) + 10)
height = int(max(dist_b, dist_d))
height_10 = int(max(dist_b, dist_d) + 10)

print(dist_a, dist_b, dist_c, dist_d)
print('This is the width ', width, 'This is the height ', height)

# Conversion factors from pixel to cm per each side
side_a_c = dim[0]/dist_a
side_b_c = dim[1]/dist_b
side_c_c = dim[2]/dist_c
side_d_c = dim[3]/dist_d

print('Conversion factor per side', side_a_c, " ", side_b_c, " ", side_c_c, " ", side_d_c)

# Average conversion factors from pixel to cm for quadrat height and wide
q_w = float(side_a_c + side_c_c) / 2
q_h = float(side_b_c + side_d_c) / 2
area = q_w * q_h
side = np.max([width, height], axis=0)
conversion = dim[0]/side

print('Quadrat wide factor is ', q_w, '\nQuadrat height factor is ', q_h,
      '\nQuadrat area factor is ', area, '\nDistance coversion factor is ', conversion)



print('The selected side vertices is ', side)
dest_pts = np.float32([[0, 0], [side, 0], [0, side], [side, side]])
M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
# IM = cv2.getPerspectiveTransform(dest_pts, orig_pts)

mini = np.amin(vertices, axis=0)
maxi = np.amax(vertices, axis=0)
print(mini, "and ", maxi)

position1 = (0, 0)
center = (0, 0)

# Read first frame.q
ok, frame = vid.read()
frame = cv2.warpPerspective(frame, M, (side, side))
if not ok:
    print('Cannot read video file')
    sys.exit()

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, MIL, KCF, TLD, MEDIANFLOW or GOTURN

# tracker = cv2.Tracker_create('BOOSTING')
# tracker = cv2.TrackerBoosting_create()
# tracker = cv2.TrackerMedianFlow_create()
tracker = cv2.TrackerMIL_create()
# tracker = cv2.TrackerKCF_create()
# print(tracker)
# Define an initial bounding box
# bbox = (650, 355, 25, 25)
bbox = cv2.selectROI('tracking select', frame, fromCenter=False)
crab_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
print(crab_center)

crab_id = args['video'] + '_' + args['crab_id'] + str(crab_center)
print(crab_id)

# Uncomment the line below to select a different bounding box
# bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=10000)

wr.writerow([path, date_now, time_now, length_vid, fps, target_frame,
             quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3], side, conversion, tracker])
wr.writerow(['\n'])
wr.writerow(['Crab_ID', 'Crab_Position', 'Crab_frame'])
# print(M)

counter = 0
(dX, dY) = (0, 0)
direction = ""

startTime = datetime.now()

cv2.destroyAllWindows()

### From warp.py
fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history= 5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))
for_di1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3, 3))

out = cv2.VideoWriter('Uca_detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (464,464))

while vid.isOpened():
    ret, img = vid.read()
    # print(img.shape)
    # img = cv2.resize(img, (640,400))
    crop_img = img[mini[1]-10:maxi[1]+10, mini[0]-10:maxi[0]+10]
    result = cv2.warpPerspective(img, M, (side, side))
    crab_frame = cv2.warpPerspective(img, M, (side, side))
    # result_speed = result
    # print(crop_img.shape)
    print('Dimensions for result are: ', result.shape)
    # result_1 = cv2.warpPerspective(result, IM, (682,593))

    methods.draw_quadrat(img, vertices)
    # cv2.polylines(img, np.int32([quadrat_pts]), True, (204, 204, 0), thickness=2)


    ### From warp.py
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
    # ok, bbox = tracker.update(masked)
    ok, bbox = tracker.update(result)
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

        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))

        wr.writerow([crab_id, center, counter])

        crab = crab_frame[center[1] - 15:center[1] + 15, center[0] - 15:center[0] + 15]
        # crab = frame[int(bbox[0] + bbox[2]/2):100, int(bbox[1] + bbox[3]/2):100]
        # crab = frame[100:(100 + 50), 250:(250 + 50)]
        # filename = os.path.join(dirname, fname, str(center), startTime1)
        # cv2.imwrite(dirname + '/' + filename + '_' + startTime1 + str(center) + '_' + '.jpg', crab)

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
            # cv2.line(result, pts[i - 1], pts[i], (204, 204, 0), thickness)
            cv2.line(result, pts[i - 1], pts[i], (54, 54, 250), thickness)

        # show the movement deltas and the direction of movement on
        # the frame
        direction = 'Uca movement ' + str(direction)
        cv2.putText(result, direction, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (10, 10, 10), 2)
        cv2.putText(result, "Displacement (cm) dx: {}, dy: {}".format(dX, dY),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (10, 10, 10), 2)


        ### Back transform and show tracker and data in original image
        ###


    # counter_f += 1
    # print('Frame count ', counter_f)
    # if counter_f == 60:
    #     counter_f = 0
    #     cv2.imshow('One every ten', result)


    ## DISPLAY (Multiple panels video)
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
    out.write(result)


    # cv2.imshow('result_1', result_1)
    # cv2.imshow('original', img)
    # cv2.imshow('cropped', crop_img)
    # cv2.imshow('Crab', crab)
    # cv2.imshow('result', result)

    ### From warp.py
    # cv2.imshow('background substraction', fb_res_two3)
    # cv2.imshow('masked', masked)
    cv2.imshow('result', result)
    # cv2.imshow('Canny Edges', edge)
    # cv2.imshow('display00', display00)
    # cv2.imshow('display01', display01)
    # cv2.imshow('display', display)
    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
resultFile.close()
