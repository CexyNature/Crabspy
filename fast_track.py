#!/usr/bin/env python3

'''
This code track target individual skipping nth frames.
It returns the opposite vertices of a polygon enclosing the target individual
'''

import argparse
import cv2

import methods

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="VIRB0037-3.MP4", help='Provide path to video file')
ap.add_argument('-s', '--seconds', default=640,
                help='Provide time in seconds of target video section showing the key points')
ap.add_argument('-n', '--nth_frame', default=12, help='Number of frames to skip')
args = vars(ap.parse_args())


video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(args["video"])
vid, target_frame = methods.set_video_star(vid, args["seconds"], fps)

# tracker, _ = methods.single_target_track(vid)
trackers = methods.multi_target_track(vid, number=5)

capture_n_frame = 0
nth_frame = args["nth_frame"]

while vid.isOpened():

    ret, img = vid.read()

    if ret is True:

        current_frame = target_frame + capture_n_frame
        vid.set(1, current_frame)

        img50 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        # ok, bbox = tracker.update(img50)
        # # print(position)
        # position1 = (bbox[0], bbox[1])
        #
        # if ok:
        #     p1 = (int(bbox[0]), int(bbox[1]))
        #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        #     cv2.rectangle(img50, p1, p2, (204, 204, 100), 2)
        #     # cv2.rectangle(masked, p1, p2, (204, 204, 0))
        #     # cv2.circle(result, (180,180), 3, (0, 204, 100), 3)
        #     print(p1, p2, current_frame)
        #     center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))


        success, boxes = trackers.update(img50)
        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(img50, p1, p2, (100, 150, 200), 2, 1)


        cv2.imshow("original", img50)

    else:
        vid.release()
        break

    capture_n_frame += nth_frame

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
