#!/usr/bin/env python3

import argparse
import pandas as pd
from collections import deque
import cv2
import numpy as np

import methods

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default="GP010016_GP010016_focal1.csv", help="Provide path to file")
args = vars(ap.parse_args())

track_meta = pd.read_csv("results/" + args["file"], header=0, nrows=1)
track = pd.read_csv("results/" + args["file"], header=2, skiprows=range(0, 1))
# track = track.replace(np.nan, "", regex=True)
# print(track.head())
# print(track.tail())
# print(track_meta.head())
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
# print(df)
# print(quadrat_vertices)
video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(file_name)
M, side, vertices_draw, IM, conversion = methods.calc_proj(quadrat_vertices)

pts = deque(maxlen=int(track_meta["length_video"].values[0])+250)
(dX, dY) = (0, 0)
target = 9000
vid.set(1, target)
counter = target
# color1 = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
name_color, color1 = methods.select_color()
# print(name_color)


while vid.isOpened():
    _, img = vid.read()
    result = cv2.warpPerspective(img, M, (side, side))
    result2 = result.copy()

    if counter <= len(track.index)-1:

        try:
            # print(counter, track["Frame_number"].values[counter])
            # coord_x = int(track["Crab_position_x"].iloc[[counter-15, counter+15]].mean())
            # coord_y = int(track["Crab_position_y"].iloc[[counter-15, counter+15]].mean())
            coord_x = int(track["Crab_position_x"].values[counter])
            coord_y = int(track["Crab_position_y"].values[counter])
            center = (coord_x, coord_y)
            pts.appendleft(center)

            for i in np.arange(1, len(pts)):
                cv2.line(result, pts[i - 1], pts[i], color1, 1)
                cv2.circle(result, (coord_x, coord_y), 15, color1, 1)
        except (ValueError, IndexError):
            pass
    else:
        pass

    percentage_vid = counter/track_meta["length_video"].values[0]*100
    text = "Video {0:.1f} %".format(percentage_vid)
    cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
    cv2.putText(result, "Frame n. {0:d}".format(counter), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)

    result2 = cv2.addWeighted(result, 0.5, result2, 0.5, 0)


    cv2.imshow("result", result2)
    counter += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
