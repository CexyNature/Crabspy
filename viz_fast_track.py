#!/usr/bin/env python3

"""
Viz fast track
"""

import argparse
import cv2
import csv

import methods

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="VIRB0037-3.MP4", help="Provide path to video file")
ap.add_argument("-f", "--file", default="VIRB0037-3.MP4fast_track.csv", help="Provide path to file")
args = vars(ap.parse_args())

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

"""
Wish list:
1. Introduce conditional to define if result video should be cropped to bbox.
2. Introduce conditional to define if result video should be saved as new video.
"""

video_name, vid, length_vid, fps, vid_width, vid_height, vid_duration, _ = methods.read_video(args["video"])
capture_n_frame = 0
nth_frame = 12
target_frame = 19180

print(vid_width, vid_height)

with open("results/" + args["file"], "r") as f:
    rd = csv.reader(f, delimiter = ",")

    while vid.isOpened():

        line = 0
        for row in rd:

            if line == 0:
                y = 0
                x = 0
                h = int(vid_height / 2)
                w = int(vid_width / 2)
                pass

            elif line <= 50:
                x = int(row[1])
                y = int(row[2])
                w = int(row[3])
                h = int(row[4])

            else:
                break

            ret, img = vid.read()
            print(vid.get(3), vid.get(4))
            current_frame = target_frame + capture_n_frame
            vid.set(1, current_frame)
            img50 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.rectangle(img50, (x, y), (x + w, y + h), (204, 204, 100), 2)
            cv2.imshow("50%", img50)
            # print(line)
            line += 1
            capture_n_frame += nth_frame

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        vid.release()
        cv2.destroyAllWindows()