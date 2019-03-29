#!/usr/bin/env python3

import cv2

import methods
import constant
import pickle

# Working
# uca_01 = methods.CrabNames("Uca_01", (101, 205), "Uca coarctata", "male", "right")
# meto_01 = methods.CrabNames("Metopograpsus_01", (301, 203), "Metopograpsus sp", "male", None)
# methods.CrabNames.save_crab_names(methods.CrabNames.instances)
# for instance in methods.CrabNames.instances:
#     print(instance.sex)

# Working
# methods.CrabNames.open_crab_names("results/GP010016")
# for instance in methods.CrabNames.instances:
#     variable = instance.__dict__
#     print(variable)

# Working
# methods.CrabNames.print_crab_names("results/GP010016")

# Working
# methods.CrabNames.open_crab_names("results/GP010016")
# for instance in methods.CrabNames.instances:
#     # variable = instance.__dict__
#     print("This is individual {crab_name}: sex {sex} and {handedness} handedness," \
#                " from species {species}. Its tracking started at posiiton {start_position}.".format(**instance.__dict__))

# Working
methods.CrabNames.print_crab_names("results/GP010016")



# vid, vid_length, vid_fps, vid_width, vid_length, vid_fourcc = methods.read_video('VIRB0002.MP4')
# # methods.save_tracks('GP010016.MP4')
# print(vid_length, vid_fps, vid_width, vid_length, vid_fourcc)
#
#
# while vid.isOpened():
#     ret, frame = vid.read()
#
#     methods.enable_point_capture(constant.CAPTURE_VERTICES)
#     frame = methods.draw_points_mousepos(frame, methods.quadrat_pts, methods.posmouse)
#     cv2.imshow("Vertices selection", frame)
#
#     if len(methods.quadrat_pts) == 4:
#         print("Vertices were captured")
#         break
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         # print("Q - key pressed. Window quit by user")
#         break
#
# cv2.destroyAllWindows()
#
# while vid.isOpened():
#     _, frame = vid.read()
#     M, side, vertices_draw, IM = methods.calc_proj(methods.quadrat_pts)
#     # print(side)
#     frame = cv2.warpPerspective(frame, M, (side, side))
#     cv2.imshow("Video stream warped", frame)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
#
#
# vid.release()
# cv2.destroyAllWindows()
# print("Script runs")
