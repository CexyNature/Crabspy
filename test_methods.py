#!/usr/bin/env python3

import cv2
import time

import methods


vid, vid_length, vid_fps, vid_width, vid_length, vid_fourcc = methods.read_video('VIRB0002.MP4')
# methods.save_tracks('GP010016.MP4')
print(vid_length, vid_fps, vid_width, vid_length, vid_fourcc)

capture_vertices = True

while vid.isOpened():
    ret, frame = vid.read()

    methods.enable_point_capture(capture_vertices)
    frame = methods.draw_points_mousepos(frame, methods.quadrat_pts, methods.posmouse, capture_vertices)
    cv2.imshow("Vertices selection", frame)

    if len(methods.quadrat_pts) == 4:
        print("Vertices were captured")
        break


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # print("Q - key pressed. Window quit by user")
        break

cv2.destroyAllWindows()

while vid.isOpened():
    ret, frame = vid.read()
    M, side = methods.calc_proj(methods.quadrat_pts)
    print(side)
    frame = cv2.warpPerspective(frame, M, (side, side))
    cv2.imshow("Video stream warped", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


vid.release()
cv2.destroyAllWindows()
print("Script runs")
