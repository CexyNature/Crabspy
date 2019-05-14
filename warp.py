# from scipy import ndimage
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import argparse
import csv
import os
import time

import methods

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="GP010016.MP4", help='Provide path to video file')
ap.add_argument('-s', '--seconds', default=None,
                help='Provide time in seconds of target video section showing the key points')
args = vars(ap.parse_args())


# Create list for holding quadrat position
quadrat_pts = []
# Create list to hold bounding boxes positions
bbox_list = []
# Create list to hold frames of interest
frames = []

# Set initial default position of  first point and mouse
position = (0, 0)
posmouse = (0, 0)


# Define mouse click function
def click(event, x, y, flags, param):
    global quadrat_pts, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        position = (x, y)
        quadrat_pts.append(position)
        # print(quadrat_pts)

    if event == cv2.EVENT_MOUSEMOVE:
        posmouse = (x, y)


vid = cv2.VideoCapture('video/' + args['video'])


# Total length of video in frames
length_vid = vid.get(cv2.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv2.CAP_PROP_FPS)

if args['seconds'] is None:
    target_frame = 1
else:
    target_frame = int(int(args['seconds']) * fps)

print('Video length is ', length_vid,
      '\nFPS is', fps,
      '\nTarget frame is ', target_frame)

# # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
vid.set(1, target_frame-1)

path = os.path.basename(args['video'])
file_name, file_ext = os.path.splitext(path)
dir_results = 'results'
print(file_name)
print(file_ext)

date_now = time.strftime('%d%m%Y')
time_now = time.strftime('%H%M')
name_resultFile = 'results/' + file_name + '_' + str(date_now) + '_' + str(time_now) + '.csv'
resultFile = open(name_resultFile, 'w', newline='\n')
wr = csv.writer(resultFile, delimiter=',')
wr.writerow(['file_name', 'processed_at_date', 'processed_at_time', 'length_video', 'fps_video',
             'target_frame_used', 'point_1', 'point_2', 'point_3', 'point_4'])

while vid.isOpened():
    ret, frame = vid.read()

    cv2.namedWindow('Select vertices quadrat')
    cv2.setMouseCallback('Select vertices quadrat', click)
    for i, val in enumerate(quadrat_pts):
        cv2.circle(frame, val, 3, (200, 255, 185), 2)
    cv2.putText(frame, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Select vertices quadrat', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Q - key pressed. Window quit by user")
        break


vid.release()
cv2.destroyAllWindows()
print(quadrat_pts)

vid = cv2.VideoCapture('video/' + args['video'])
M, side, vertices_draw, IM, conversion = methods.calc_proj(quadrat_pts)




# orig_pts = np.float32([quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]])
#
# # dist = math.hypot(x2-x1, y2-y1)
# dist_a = math.sqrt((quadrat_pts[0][0] - quadrat_pts[1][0])**2 + (quadrat_pts[0][1] - quadrat_pts[1][1])**2)
# dist_b = math.sqrt((quadrat_pts[1][0] - quadrat_pts[2][0])**2 + (quadrat_pts[1][1] - quadrat_pts[2][1])**2)
# dist_c = math.sqrt((quadrat_pts[2][0] - quadrat_pts[3][0])**2 + (quadrat_pts[2][1] - quadrat_pts[3][1])**2)
# dist_d = math.sqrt((quadrat_pts[0][0] - quadrat_pts[3][0])**2 + (quadrat_pts[0][1] - quadrat_pts[3][1])**2)
#
#
# width = int(max(dist_a, dist_c) + 10)
# height = int(max(dist_b, dist_d) + 10)
#
# print(dist_a, dist_b, dist_c, dist_d)
# print(width, height)
#
# dest_pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
# M = cv2.getPerspectiveTransform(orig_pts, dest_pts)

wr.writerow([path, date_now, time_now, length_vid, fps, target_frame,
             quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]])
# print(M)

fgbg1 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history= 5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 21))
for_di1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3, 3))

while vid.isOpened():
    ret, img = vid.read()
    # img = cv2.resize(img, (640,400))
    result = cv2.warpPerspective(img, M, (side, side))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    hsl = cv2.cvtColor(result, cv2.COLOR_BGR2HLS_FULL)
    one, two, three = cv2.split(hsl)

    # frames.append(gray)

    # if len(frames) > 20:
    #     del frames[0]
    #
    #     # print(len(frames))
    #     for _ in range(20):
    #         median = np.median(frames, axis=0).astype(dtype=np.uint8)
    #         # alter = gray - median
    #         # alter = np.subtract(gray, median)
    #
    #         alter = cv2.subtract(gray, median)
    #         alter1 = cv2.dilate(alter, for_di)
    #         _, alter2 = cv2.threshold(alter1, 40, 255, cv2.THRESH_BINARY)
    #         alter2 = cv2.dilate(alter2, for_di1)
    #         masked1 = cv2.bitwise_and(result, result, mask=alter2)
    #
    #
    #         # cv2.imshow('median frame', median)
    #         cv2.imshow('alter', alter)
    #         cv2.imshow('alter1', alter1)
    #         cv2.imshow('alter2', alter2)
    #         cv2.imshow('masked1', masked1)


    # fb_res_two1 = fgbg1.apply(two)
    # fb_res_two2 = fgbg2.apply(two)
    fb_res_two3 = fgbg3.apply(two, learningRate=-1)
    #
    fb_res_two3 = cv2.erode(fb_res_two3, for_er)
    fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
    #
    masked = cv2.bitwise_and(result, result, mask=fb_res_two3)

    edge = cv2.Canny(masked, threshold1=100, threshold2=230)
    #
    # _, contours, _ = cv2.findContours(fb_res_two3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for i, cnt in enumerate(contours):
    #
    #     x, y, w, h = cv2.boundingRect(cnt)
    #
    #     if w > 8 and h > 8 and w < 40 and h < 40:
    #
    #         cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #         center = cv2.circle(result, (int(x + w / 2), int(y + h / 2)), 1, (0, 0, 255), -1)
    #         cx = int(x + w / 2)
    #         cy = int(y + h / 2)
    #         cv2.putText(result, "#{}".format(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
    #                     1.0, (255, 255, 255), 2)
    #         points = (cx, cy)
    #         # print(i)
    #
    #         bbox_list.append((x,y,w+5,h+5))
    #         # print(bbox_list[0])
    #
    #         for item, pt in enumerate(points):
    #             pass

    # if len(bbox_list) > 3:
    #     bbox = bbox_list[2]
    #     tra = tracker.track(masked, bbox)
    #     tracker.tracker_update(tra, masked)


    cv2.imshow('background substraction', fb_res_two3)
    cv2.imshow('masked', masked)
    cv2.imshow('result', result)
    cv2.imshow('Canny Edges', edge)


    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
resultFile.close()
