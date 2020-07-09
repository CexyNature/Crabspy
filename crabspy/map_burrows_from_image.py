#!/usr/bin/env python3

"""
This script allows users to map burrow positions from an image
"""

import os
import cv2
import numpy as np
import argparse
from datetime import datetime
import time
import pandas as pd
import csv
import ast
import math

import methods

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="video/test.png", help="Provide path to image file")
args = vars(ap.parse_args())

existing_burrows = []
burrows = []
burrows_counter = 0
existing_burrows_counter = 0
new_burrows_counter = 0
position = (0, 0)
posmouse = (0, 0)
counter = 0
startTime = datetime.now()
create_new_file = True

draw = False
xi, yi = 0, 0
radii = []

image_name = args["image"]
image_path = os.path.dirname(image_name)
image_name = os.path.basename(image_name)
image_name, _ = os.path.splitext(image_name)
print(image_name)
print(image_path)

# Return image information
try:
    img = cv2.imread(args["image"])

    if img.size == 0:
        print("Empty matrix returned. Please check that image and image path are valid.")
    else:

        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        print(img.shape)
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

        if os.path.isfile(image_path + "/" + image_name + "_burrows_map.csv"):
            try:
                print("Burrows map file found. Burrows coordinates will be loaded.")
                create_new_file = False
                head_true = False
                burrows_meta = pd.read_csv(image_path + "/" + image_name + "_burrows_map.csv", header=0, nrows=1,
                                           converters={"vertice_1": ast.literal_eval, "vertice_2": ast.literal_eval,
                                                       "vertice_3": ast.literal_eval, "vertice_4": ast.literal_eval})
                burrows_coord = pd.read_csv(image_path + "/" + image_name + "_burrows_map.csv", header=2,
                                            skiprows=range(0, 1))

                quadrat_pts_used = [burrows_meta.iloc[0]["vertice_1"], burrows_meta.iloc[0]["vertice_2"],
                                    burrows_meta.iloc[0]["vertice_3"], burrows_meta.iloc[0]["vertice_4"]]

                vertices = quadrat_pts_used
                # print(quadrat_pts_used)
                # print(burrows_coord)

                for i, rows in burrows_coord.iterrows():
                    row_values = [int(rows.ID), (int(rows.Burrow_coord_x), int(rows.Burrow_coord_y)), int(rows.Radius)]
                    # print(row_values)
                    existing_burrows.append(row_values)

            except (TypeError, RuntimeError):
                print("Exiting because of TypeError or RuntimeError")
                pass

        else:
            while True:
                img_preview = img.copy()
                methods.enable_point_capture(True)
                img_preview = methods.draw_points_mousepos(img_preview, methods.quadratpts, methods.posmouse)
                cv2.imshow("Vertices selection", img_preview)

                if len(methods.quadratpts) == 4:
                    print("Vertices were captured. Coordinates in pixels are: top-left {}, top-right {}, "
                          "bottom-left {}, and bottom-right {}".format(*methods.quadratpts))
                    vertices = methods.quadratpts
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    # print("Q - key pressed. Window quit by user")
                    break

            cv2.destroyAllWindows()

        M, side, vertices_draw, IM, conversion = methods.calc_proj(vertices)
        center = (0, 0)
        mini = np.amin(vertices_draw, axis=0)
        maxi = np.amax(vertices_draw, axis=0)

        img = cv2.imread(args["image"])
        img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
        img = cv2.warpPerspective(img, M, (side, side))

        info = [methods.CompileInformation("name_image", image_name),
                methods.CompileInformation("image_dimx", height),
                methods.CompileInformation("image_dimy", width),
                methods.CompileInformation("image_dimz", channels),
                methods.CompileInformation("conversion", conversion),
                methods.CompileInformation("side", side)]

        info_image = {}
        for i in info:
            info_image[i.name] = i.value

        if create_new_file:
            print("There is not burrows map file for this video, creating one.")
            head_true = True
            name_result_file = image_path + "/" + info_image["name_image"] + "_burrows_map.csv"

            if head_true:
                with open(name_result_file, "w", newline="\n") as result_file:
                    wr = csv.writer(result_file, delimiter=",")
                    date_now = time.strftime("%d%m%Y")
                    time_now = time.strftime("%H%M")
                    wr.writerow(["file_name", "processed_at_date", "processed_at_time",
                                 "vertice_1", "vertice_2", "vertice_3", "vertice_4",
                                 "projected_q_side", "q_conversion_factor_distance",
                                 "img_height", "img_width", "img_channels"])

                    wr.writerow([info_image["name_image"], date_now, time_now,
                                 methods.quadratpts[0], methods.quadratpts[1], methods.quadratpts[2],
                                 methods.quadratpts[3],
                                 info_image["side"], info_image["conversion"],
                                 info_image["image_dimx"], info_image["image_dimy"], info_image["image_dimz"]])
                    wr.writerow(["\n"])
                    wr.writerow(["ID", "Burrow_coord_x", "Burrow_coord_y", "Radius"])

except Exception as e:
    print(e)


def draw_circle(event, x, y, flags, param):
    global xi, yi, radii, draw, burrows, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        xi, yi = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            position = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        burrow_info = (int((xi + x) / 2), int((yi + y) / 2))
        burrows.append(burrow_info)

        radius = int((math.sqrt(((xi - x) ** 2) + ((yi - y) ** 2)))/2)
        radii.append(radius)

        print("Radius {}".format(radius))
        cv2.circle(img, (int((xi + x) / 2), int((yi + y) / 2)), radius, (0, 0, 255), 1)

        draw = False


while True:
    img_preview = img.copy()
    img_preview2 = img.copy()

    for i, val in enumerate(existing_burrows):
        # print(val)
        cv2.circle(img_preview, val[1], val[2], (255, 5, 205), 1)
        cv2.putText(img_preview, "{}".format(val[0]), val[1],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.circle(masked, val[0], 3, (255, 5, 205), 2)
        existing_burrows_counter = i + 1

    for i, val in enumerate(burrows):
        # print(val)
        cv2.circle(img_preview, val, 3, (0, 255, 0), 1)
        # cv2.circle(masked, val[0], 3, (0, 255, 0), 2)
        new_burrows_counter = i + 1

    burrows_counter = existing_burrows_counter + new_burrows_counter

    cv2.namedWindow('Burrow counter')
    # cv2.setMouseCallback('Burrow counter', click)
    cv2.setMouseCallback('Burrow counter', draw_circle)

    # result_1 = cv2.warpPerspective(img_preview, IM, (img.shape[1], img.shape[0]))
    # result_1 = cv2.addWeighted(img_preview, 0.5, result_1, 0.5, 0)

    cv2.putText(img_preview2, "Number of burrows {}".format(burrows_counter), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img_preview2, "Last burrow coordinate {}".format(position), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)
    cv2.putText(img_preview2, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)

    # cv2.imshow("Original perspective", result_1)
    # cv2.imshow("Background subtracted", masked)
    cv2.imshow('Burrow counter', img_preview)
    cv2.imshow('Burrow counter preview', img_preview2)

    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()

print(burrows_counter)


if len(burrows) > 0:
    print('Writing new coordinates to file.')
    name_result_file = image_path + "/" + info_image["name_image"] + "_burrows_map.csv"
    for i, j, k in zip(range(existing_burrows_counter, burrows_counter), burrows, radii):
        try:
            print(i+1, " ", j[0], " ", j[1], k)
            # save track_info to file
            with open(name_result_file, "a+", newline="\n") as result_file:
                wr = csv.writer(result_file, delimiter=",")
                wr.writerow([i+1, j[0], j[1], k])

        except (TypeError, RuntimeError):
            print('A TypeError or RuntimeError was caught while writing burrows coordinates')
            pass
else:
    print('No new coordinates to write.')
    pass
