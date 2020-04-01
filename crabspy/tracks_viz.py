#!/usr/bin/env python3

"""
Display all tracked individuals in the video
"""

import os
import argparse
import pandas as pd
from collections import deque
import cv2

import methods

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--uni_file", default="Unify_tracks_GP010016_05022020_115125.csv", help="Provide path to file")
ap.add_argument("-m", "--metafile", default="Unify_metaInfo_GP010016_05022020_115125.csv", help="Provide path to metafile")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide the targeted time in seconds of video section you want to jump to")
ap.add_argument("-f", "--frame", default=None, type=int,
                help="Provide the targeted frame of video section you want to jump to")
ap.add_argument("-o", "--outcome", default="",
                help="Should the video outcome be saved")
args = vars(ap.parse_args())


tracks_meta = pd.read_csv("results/unified_tracks/" + args["metafile"], header=0, nrows=1)
tracks = pd.read_csv("results/unified_tracks/" + args["uni_file"], header=0, skiprows=0)

file_name = tracks_meta["file_name"].values[0]
quadrat_vertices = [tracks_meta["vertice_1"].values[0],
                    tracks_meta["vertice_2"].values[0],
                    tracks_meta["vertice_3"].values[0],
                    tracks_meta["vertice_4"].values[0]]

df = pd.DataFrame(quadrat_vertices, columns=["vertices_x"])
df["vertices_x"] = df["vertices_x"].map(lambda x: x.lstrip("(").rstrip(")"))
df[["vertices_x", "vertices_y"]] = df["vertices_x"].str.split(",", expand=True)

quadrat_vertices = [(int(df.iloc[0, 0]), int(df.iloc[0, 1])),
                    (int(df.iloc[1, 0]), int(df.iloc[1, 1])),
                    (int(df.iloc[2, 0]), int(df.iloc[2, 1])),
                    (int(df.iloc[3, 0]), int(df.iloc[3, 1]))]
# print(df)
# print(quadrat_vertices)
# print(file_name)
video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(file_name)
vid, target_frame = methods.set_video_star(vid, args["seconds"], args["frame"], fps)
M, side, vertices_draw, IM, conversion = methods.calc_proj(quadrat_vertices)
q_factor = 0.107758620689655
# print(video_name)
print("Quadrat size: ", side)
# print(vertices_draw)
print("Pixel to centimeter conversion factor: ", conversion)

if args["outcome"] is True:
    os.makedirs("results/processed_videos/", exist_ok=True)
    file_outcome = "results/processed_videos/" + video_name + "_tracks_viz.avi"
    out_vid = cv2.VideoWriter(file_outcome, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (side, side))
else:
    pass


pts = deque(maxlen=int(tracks_meta["length_video"].values[0])+250)
(dX, dY) = (0, 0)
# target = 1
# vid.set(1, target)
counter = target_frame

individuals = tracks.Crab_ID.unique()
colours = methods.select_color(len(individuals))
crab_colors = dict(zip(individuals, colours))
print("\n".join("{}\t{}".format(k, v) for k, v in crab_colors.items()))

# print(individuals)
dict_ind = dict(tuple(tracks.groupby("Crab_ID")))
# for i in individuals:
#     print(len(dict_ind[i]))
f_number = dict(tuple(tracks.groupby("Frame_number")))
# for i in range(0, 1):
#     # print(len(f_number[i]))
#     df = f_number[i]
#     for index, row in df.iterrows():
#         print(row["x_coord"], row["y_coord"])

f_max = tracks["Frame_number"].max()
# print(f_max)

while vid.isOpened():
    ret, img = vid.read()

    if ret:
        result = cv2.warpPerspective(img, M, (side, side))
        result2 = result.copy()

        if counter <= f_max:

            df_f = f_number[counter]


            for index, row in df_f.iterrows():

                try:
                    crab = row["Crab_ID"]
                    x = int(row["Crab_position_x"])
                    y = int(row["Crab_position_y"])
                    bgr = crab_colors[crab][1]
                    cv2.circle(result, (x, y), 15, bgr, 2)

                    # print(crab, x, row["Crab_position_x"], y, row["Crab_position_y"], bgr)
                except ValueError:
                    pass
            # try:
            #     # print(counter, track["Frame_number"].values[counter])
            #     # coord_x = int(track["Crab_position_x"].iloc[[counter-15, counter+15]].mean())
            #     # coord_y = int(track["Crab_position_y"].iloc[[counter-15, counter+15]].mean())
            #     coord_x = int(tracks["Crab_position_x"].values[counter])
            #     coord_y = int(tracks["Crab_position_y"].values[counter])
            #     center = (coord_x, coord_y)
            #     pts.appendleft(center)
            #
            #     for i in np.arange(1, len(pts)):
            #         cv2.line(result, pts[i - 1], pts[i], color1, 1)
            #         cv2.circle(result, (coord_x, coord_y), 15, color1, 2)
            # except (ValueError, IndexError):
            #     pass
        else:
            pass

        # percentage_vid = counter/track_meta["length_video"].values[0]*100
        # text = "Video {0:.1f} %".format(percentage_vid)
        # cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        # cv2.putText(result, "Frame n. {0:d}".format(counter), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)

        # result2 = cv2.addWeighted(result, 0.6, result2, 0.4, 0)
        result_1 = cv2.warpPerspective(result, IM, (img.shape[1], img.shape[0]))
        result_1 = cv2.addWeighted(img, 0.5, result_1, 0.5, 0)

        if args["outcome"] is True:
            out_vid.write(result)

        cv2.imshow("Original FoV", result_1)
        cv2.imshow("Perspective FoV", result)
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()
