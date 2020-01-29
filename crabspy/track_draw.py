#!/usr/bin/env python3

"""
Display a tracked individual in the video.
"""

import argparse
import pandas as pd
from collections import deque
import cv2

import methods

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--track_file", default="GP010016_Us_focal_01.csv", help="Provide path to file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide the targeted time in seconds of video section you want to jump to")
ap.add_argument("-f", "--frame", default=None, type=int,
                help="Provide the targeted frame of video section you want to jump to")
args = vars(ap.parse_args())

track_meta = pd.read_csv("results/" + args["track_file"], header=0, nrows=1)
track = pd.read_csv("results/" + args["track_file"], header=2, skiprows=range(0, 1))

file_name = track_meta["file_name"].values[0]
quadrat_vertices = [track_meta["vertice_1"].values[0],
                    track_meta["vertice_2"].values[0],
                    track_meta["vertice_3"].values[0],
                    track_meta["vertice_4"].values[0]]

df = pd.DataFrame(quadrat_vertices, columns=["vertices_x"])
df["vertices_x"] = df["vertices_x"].map(lambda x: x.lstrip("(").rstrip(")"))
df[["vertices_x", "vertices_y"]] = df["vertices_x"].str.split(",", expand=True)

quadrat_vertices = [(int(df.iloc[0, 0]), int(df.iloc[0, 1])),
                    (int(df.iloc[1, 0]), int(df.iloc[1, 1])),
                    (int(df.iloc[2, 0]), int(df.iloc[2, 1])),
                    (int(df.iloc[3, 0]), int(df.iloc[3, 1]))]

video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(file_name)
vid, target_frame = methods.set_video_star(vid, args["seconds"], args["frame"], fps)
M, side, vertices_draw, IM, conversion = methods.calc_proj(quadrat_vertices)

pts = deque(maxlen=int(track_meta["length_video"].values[0])+250)
(dX, dY) = (0, 0)
counter = target_frame

individuals = track.Crab_ID.unique()
colours = methods.select_color(len(individuals))
crab_colors = dict(zip(individuals, colours))
print("\n".join("{}\t{}".format(k, v) for k, v in crab_colors.items()))

dict_ind = dict(tuple(track.groupby("Crab_ID")))
f_number = dict(tuple(track.groupby("Frame_number")))
f_max = track["Frame_number"].max()


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

                except (ValueError, IndexError):
                    pass
        else:
            pass

        percentage_vid = counter/track_meta["length_video"].values[0]*100
        text = "Video {0:.1f} %".format(percentage_vid)
        cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        cv2.putText(result, "Frame n. {0:d}".format(counter), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)

        result2 = cv2.addWeighted(result, 0.6, result2, 0.4, 0)

        cv2.imshow("result", result2)
        counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
