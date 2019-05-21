#!/usr/bin/env python3

import argparse
import pandas as pd
import re
import cv2

import methods

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default="GP010016_GP010016_focal1.csv", help="Provide path to file")
args = vars(ap.parse_args())

track_meta = pd.read_csv("results/" + args["file"], header=0, nrows=1)
track = pd.read_csv("results/" + args["file"], header=2, skiprows=range(0,1))

print(track.head())
print(track_meta.head())

file_name = track_meta["file_name"].values[0]

quadrat_vertices = [track_meta["vertice_1"].values[0],
                    track_meta["vertice_2"].values[0],
                    track_meta["vertice_3"].values[0],
                    track_meta["vertice_4"].values[0]]

df = pd.DataFrame(quadrat_vertices, columns=["vertices_x"])
df["vertices_x"] = df["vertices_x"].map(lambda x: x.lstrip("(").rstrip(")"))
df[["vertices_x", "vertices_y"]] = df["vertices_x"].str.split(",", expand =True)

# quadrat_vertices = [(int(df.loc[0,0]), int(df.loc[0,1]))]

print(df)

# QUADRAT_POINTS = [i.replace("''", "") for i in QUADRAT_POINTS]
# print(quadrat_vertices)
# a = re.sub("[()]", "", track_meta["vertice_1"].values[0])
# print(a)

# video_name, vid, length_vid, fps, _, _, vid_duration, _ = methods.read_video(file_name)
# M, side, vertices_draw, IM, conversion = methods.calc_proj(quadrat_vertices)

