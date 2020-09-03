#!/usr/bin/env python3

"""
This script grabs all tracks for a specific video and unify these in a single file
"""

import os
import argparse
import pandas as pd
from datetime import datetime

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="GP010016_fast.mp4", help="Provide video name")
args = vars(ap.parse_args())

video_name = os.path.splitext(args["video"])[0]


def list_files(directory, extension, name):
    files = []
    for f in os.listdir(directory):
        if f.endswith("." + extension):
            if name in f:
                files.append(f)
    return files


my_files = list_files("results/", "csv", video_name)
# print(my_files)

tracks = []
tracks_meta = []
for f in my_files:
    print(f)
    # df = pd.read_csv(str("results/" + f), header=0)
    df_m = pd.read_csv("results/" + f, header=0, nrows=1)
    df_t = pd.read_csv("results/" + f, header=2, skiprows=range(0, 1))
    print(df_t.shape)
    duplicates = df_t[df_t.duplicated(["Frame_number"])]
    if len(duplicates) == 0:
        tracks.append(df_t)
        df_m["Individual"] = f
        tracks_meta.append(df_m)
    else:
        print("Duplicate positions were found in file: {}".format(f),
              "Please run the script 'fixing_duplicates'")

# print(len(tracks))
# print(tracks[0].shape)
tracks_unify = pd.concat(tracks, sort=False)
tracks_meta_unify = pd.concat(tracks_meta, sort=True)
# print(len(tracks_unify))
# print(tracks_unify.shape)
os.makedirs("results/unified_tracks/", exist_ok=True)
time_now = datetime.now().strftime("_%d%m%Y_%H%M%S")
tracks_unify.to_csv("results/unified_tracks/Unify_tracks_" + video_name + str(time_now) + ".csv",
                    index=False)
tracks_meta_unify.to_csv("results/unified_tracks/Unify_metaInfo_" + video_name + str(time_now) + ".csv",
                    index=False)
