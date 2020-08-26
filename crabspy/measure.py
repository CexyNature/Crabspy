#!/usr/bin/env python3

"""
Navigate through frames in video and measure objects in the field of view
"""

import os
import csv
import time
import cv2
import math
import argparse

import methods, constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

# Define parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="GP010016.mp4", help="Provide path to video file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide time in seconds of target video section showing the key points")
ap.add_argument("-z", "--zoom", default=1.5, type=float, help="Provide zoom factor e.g. zoom out 0.5, zoom in 2")
args = vars(ap.parse_args())

current_time = time.strftime("%d%m%Y") + "_" + time.strftime("%H%M")
resz_val = constant.RESIZE


def save_measures(video_path, info_capture, head_true):
    """
    This function save information about teh object measured to a .csv file.

    Parameters
    ----------
    video_path: str
        A relative path to the video to be analyzed. The video should be placed inside folder 'video/'.

    info_capture: list
        List containing information about the frame number, line starting and ending point,
        line length in pixels and centimeters and conversion factor.
    head_true: bool
        If True a header will be written at top of file. It is only used to write a header in the file after creation
        If False only new data will be append as a new row.

    Returns
    -------

    """
    try:
        os.mkdir("results/measures")
    except FileExistsError:
        # print("Folder already exits")
        pass

    # create file name with name
    name = os.path.basename(video_path)
    vid_name, file_extension = os.path.splitext(name)
    name_result_file = "results/measures/" + vid_name + "_measures_" + current_time + ".csv"

    if head_true:
        with open(name_result_file, "w", newline="\n") as result_file:
            wr = csv.writer(result_file, delimiter=",")
            date_now = time.strftime("%d%m%Y")
            time_now = time.strftime("%H%M")
            wr.writerow(["file_name", "processed_at_date", "processed_at_time",
                         "vertice_1", "vertice_2", "vertice_3", "vertice_4"])

            wr.writerow([name, date_now, time_now,
                         methods.quadratpts[0], methods.quadratpts[1],
                         methods.quadratpts[2], methods.quadratpts[3]])
            wr.writerow(["\n"])
            wr.writerow(["frame_number", "coordinate_0", "coordinate_1",
                         "length_pixels", "conversion_factor", "length_cm"])

    if not head_true:
        # save track_info to file
        with open(name_result_file, "a+", newline="\n") as result_file:
            wr = csv.writer(result_file, delimiter=",")

            wr.writerow([info_capture["frame_number"], info_capture["coordinate_0"],
                         info_capture["coordinate_1"], info_capture["length_pixels"],
                         info_capture["conversion_factor"], info_capture["length_cm"]])


def onChange(trackbarValue):
    vid.set(1, trackbarValue)
    ret, frame = vid.read()
    frame = cv2.resize(frame, (0, 0), fx=resz_val, fy=resz_val)
    methods.enable_point_capture(constant.CAPTURE_VERTICES)
    frame = methods.draw_points_mousepos(frame, methods.quadratpts, methods.posmouse)
    M, width, height, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)
    frame = cv2.warpPerspective(frame, M, (width, height))
    # frame = cv2.resize(frame, dsize=(0, 0), fx=args["zoom"], fy=args["zoom"], interpolation=cv2.INTER_NEAREST)

    # frame = cv2.resize(frame, (960, 540))
    cv2.imshow('Measure object length', frame)
    pass


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


def mouse_do_nothing(*args, **kwargs):
    pass


# Load video frames
print('Loading video, please wait')
_, vid, length_vid, fps, vid_width, vid_height, vid_duration, _ = methods.read_video(args["video"])
# Set frame where video should start to be read
vid, target_frame = methods.set_video_star(vid, args["seconds"], None, fps)
print('Total number of frames in video = ' + str(length_vid))


while vid.isOpened():
    ret, frame = vid.read()
    frame = cv2.resize(frame, (0, 0), fx=resz_val, fy=resz_val)
    time.sleep(1)
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
cv2.destroyAllWindows()


print('\n' '(1) Use the bar to navigate through frames. Click, hold and move bar marker using the mouse pointer.',
      '\n' '(2) Once you have selected a frame press ESC to initialize measure.',
      '\n' '(3) Press key ''c'' to activate capture mode. Click and hold mouse left button to set measure starting point.',
      '\n' '(4) Move and release mouse left button to set ending point.',
      '\n' '(5) Press key ''r'' to reset selection, or Press key ''s'' to save selection.',
      '\n' '(6) Several lines could be created in the selected frame, but only these saved using key ''s'' will be recorded in the CSV file.',
      '\n' '(7) Press key ESC to exit the window.')

cv2.namedWindow('Measure object length')
if args["seconds"] is None:
    cv2.createTrackbar('Selector', 'Measure object length', 0, int(length_vid)-1, onChange)
else:
    cv2.createTrackbar("Selector", 'Measure object length', target_frame, int(length_vid)-1, onChange)

coord = []
draw = False
onChange(0)
cv2.waitKey()

selected_f = cv2.getTrackbarPos('Selector', 'Measure object length')
vid.set(1, selected_f)
save_measures(args["video"], None, True)

methods.enable_point_capture(constant.CAPTURE_VERTICES)
ret, frame = vid.read()
frame = methods.draw_points_mousepos(frame, methods.quadratpts, methods.posmouse)
if len(methods.quadratpts) == 4:
    print("Vertices were captured. Coordinates in pixels are: top-left {}, top-right {}, "
          "bottom-left {}, and bottom-right {}".format(*methods.quadratpts))
M, width, height, side, vertices_draw, IM, conversion = methods.calc_proj(methods.quadratpts)

while True:
    key = cv2.waitKey(1) & 0XFF
    selected_f = cv2.getTrackbarPos('Selector', 'Measure object length')
    vid.set(1, selected_f)
    ret, frame = vid.read()
    frame = cv2.resize(frame, (0, 0), fx=resz_val, fy=resz_val)
    frame = cv2.warpPerspective(frame, M, (width, height))

    # frame = cv2.resize(frame, dsize=(0, 0), fx=args["zoom"], fy=args["zoom"], interpolation=cv2.INTER_NEAREST)

    frame_size_zoomed = frame.shape[0]
    new_conversion = constant.DIM[0] / frame_size_zoomed
    # print("Size of zoomed frame ", frame_size_zoomed)

    reset = frame.copy()

    if key == ord("c"):
        mouse_task = mouse_line
        capture = True

        while capture:
            cv2.setMouseCallback('Measure object length', mouse_task)

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
                print('\n' 'Measure capture:', '\n' 'Starting point', coord[0], '\n' 'Ending point', coord[1])
                print('Lenght in pixels = ', round(dist,2))
                print('Pixel to centimeter conversion factor = ', round(new_conversion,2))
                # print('Width of field of view in meters = ', width_m)
                # print('Height of field of view in meters = ', height_m)
                # print('Area of field of view in square meters = ', area)
                print('Size in cm = ', round(size,2))

                info = [methods.CompileInformation("frame_number", selected_f),
                        methods.CompileInformation("coordinate_0", coord[0]),
                        methods.CompileInformation("coordinate_1", coord[1]),
                        methods.CompileInformation("length_pixels", dist),
                        methods.CompileInformation("conversion_factor", new_conversion),
                        methods.CompileInformation("length_cm", size)]
                info_video = {}

                for i in info:
                    info_video[i.name] = i.value

                save_measures(args["video"], info_video, False)

                # break
            if key == ord("m"):
                print('\n' 'Move again')
                # break
                capture = False
                mouse_task = mouse_do_nothing
                cv2.setMouseCallback('Measure object length', mouse_task)
                # cv2.destroyWindow("Measure object length")


            if key == 27:
                # print('\n' 'ESC key pressed. Window quit by user')
                break

    if key == 27:
        print('\n' 'ESC key pressed. Window quit by user')
        break

cv2.destroyAllWindows()
vid.release()
# Fix trackbar position when capturing and saving measurements
