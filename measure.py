#!/usr/bin/env python3

import cv2
import math
import argparse

import methods
import constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="GP010016.mp4", help="Provide path to video file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide time in seconds of target video section showing the key points")
ap.add_argument("-z", "--zoom", default=1.5, help="Provide zoom factor e.g. zoom out 0.5, zoom in 2")
args = vars(ap.parse_args())

print('Loading video, please wait')
video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(args["video"])
local_creation, creation = methods.get_file_creation(args["video"])
# Set frame where video should start to be read
vid, target_frame = methods.set_video_star(vid, args["seconds"], fps)

print('Total number of frames in video = ' + str(length_vid))

print('\n' '(1) Use the bar to navigate frames by click, hold and move bar pointer using the mouse',
      '\n' '(2) Once you select the targeted frame press ESC to initialize measure')

def onChange(trackbarValue):
    vid.set(1, trackbarValue)
    ret, frame = vid.read()
    methods.enable_point_capture(constant.CAPTURE_VERTICES)
    frame = methods.draw_points_mousepos(frame, methods.quadratpts, methods.posmouse)
    if len(methods.quadratpts) == 4:
        print("Vertices were captured. Coordinates in pixels are: top-left {}, top-right {}, "
              "bottom-left {}, and bottom-right {}".format(*methods.quadratpts))
    M, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)
    frame = cv2.warpPerspective(frame, M, (side, side))
    frame = cv2.resize(frame, dsize=(0, 0), fx=args["zoom"], fy=args["zoom"], interpolation=cv2.INTER_NEAREST)

    # frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Measure object length', frame)
    pass

coord = []
draw = False

def mouse_line(event, x, y, flags, params):
    global coord, draw

    if event == cv2.EVENT_LBUTTONDOWN:
        coord = [(x, y)]
        cv2.circle(frame, coord[0], 1, (0, 0, 255), 3)
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        coord.append((x, y))
        draw = False
        cv2.circle(frame, coord[1], 1, (0, 0, 255), 3)

        cv2.line(frame, coord[0], coord[1], (0, 0, 255), 1)
        cv2.imshow('Measure object length', frame)

cv2.namedWindow('Measure object length')
cv2.createTrackbar('Selector', 'Measure object length', 0, int(length_vid), onChange)

onChange = 0
cv2.waitKey()

frame_selector = cv2.getTrackbarPos('Selector', 'Measure object length')
vid.set(1, frame_selector)

ret, frame = vid.read()
methods.enable_point_capture(constant.CAPTURE_VERTICES)
frame = methods.draw_points_mousepos(frame, methods.quadratpts, methods.posmouse)
if len(methods.quadratpts) == 4:
    print("Vertices were captured. Coordinates in pixels are: top-left {}, top-right {}, "
          "bottom-left {}, and bottom-right {}".format(*methods.quadratpts))
M, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)
frame = cv2.warpPerspective(frame, M, (side, side))
frame = cv2.resize(frame, dsize=(0, 0), fx=args["zoom"], fy=args["zoom"], interpolation=cv2.INTER_NEAREST)

frame_size_zoomed = frame.shape[0]
new_conversion = constant.DIM[0] / frame_size_zoomed
# print("Size of zoomed frame ", frame_size_zoomed)

reset = frame.copy()

cv2.setMouseCallback('Measure object length', mouse_line)
print('\n' '(3) On the image click and hold mouse left button to set starting point of reference',
      '\n' '(4) Move and release mouse left button to set ending point of reference',
      '\n' '(5) Press key ''r'' to reset selection, or Press key ''s'' to save selection and exit window',
      '\n' '(6) Several lines might be created in the frame, but only the last one will be used',
      '\n' '(7) Press key ESC to exit the window')

while True:
    cv2.imshow('Measure object length', frame)
    key = cv2.waitKey(1) & 0XFF

    if key == ord('r'):
        coord = []
        frame = reset.copy()
    if key == ord('s'):
        dist = math.sqrt((coord[0][0] - coord[1][0]) ** 2 + (coord[0][1] - coord[1][1]) ** 2)
        # pixel_to_meters = args['length'] / dist
        # width_m = vid.get(3) * conversion
        # height_m = vid.get(4) * conversion
        # area = width_m * height_m
        size = dist * new_conversion
        print('\n' 'Results', '\n' 'Line started at', coord[0], '\n' 'Line ended at ', coord[1])
        print('Pixel length of reference = ', dist)
        print('Pixel to meter conversion factor = ', new_conversion)
        # print('Width of field of view in meters = ', width_m)
        # print('Height of field of view in meters = ', height_m)
        # print('Area of field of view in square meters = ', area)
        print('Size in cm is = ', size)

        break
    if key == 27:
        print('\n' 'ESC key pressed. Window quit by user')
        break

cv2.destroyAllWindows()
vid.release()