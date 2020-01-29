#!/usr/bin/env python3

"""
This scripts deals with duplicated rows in a tracking file.
This is normally product of tracking the same individual twice using the same or different methods.
"""

import os
import argparse
import pandas as pd
import csv
import matplotlib.pyplot as plt

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--track_file", default="tracking_file.csv", help="Provide video name")
args = vars(ap.parse_args())

# video_name = os.path.splitext(args["video"])[0].format()
track = pd.read_csv("results/" + args["track_file"], header=2, skiprows=range(0, 1))
track_meta = pd.read_csv("results/" + args["track_file"], header=None, nrows=3)
duplicates = track[track.duplicated(["Frame_number"], keep=False)]

# Find breaking points: index where
if (duplicates["Frame_number"].diff(periods=1) != 1).any():
    ind = duplicates.index[duplicates["Frame_number"].diff(periods=1) != 1].tolist()
    
    # print(ind)
    # print(ind[1]-ind[0])
    # print(track.iloc[ind[1]:ind[1]+65])
else:
    pass
# Find overlapping frames
# Count how many sections overlap
# Find the tracking method used for each section
# Plot overlapping sections side by side
# Request user to select one
# Keep selection and delete overlap
if len(duplicates) > 0:
    if (duplicates["tracker_method"] == "Manual_tracking").any():
        index_2_del = duplicates.index[duplicates.tracker_method != "Manual_tracking"].tolist()
        print("Priority is given to track positions (i.e. rows) that were tracked using the method 'Manual_tracking'")

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        ax1.plot(duplicates.loc[duplicates.tracker_method == "Manual_tracking", "Crab_position_x"],
                 duplicates.loc[duplicates.tracker_method == "Manual_tracking", "Crab_position_y"])
        ax1.title.set_text("Points got by manual tracking")
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Position Y')
        ax1.set_aspect("equal", "box")

        ax2.plot(duplicates.loc[duplicates.tracker_method != "Manual_tracking", "Crab_position_x"],
                 duplicates.loc[duplicates.tracker_method != "Manual_tracking", "Crab_position_y"])
        ax2.title.set_text("Points got by MIL tracking (to be deleted)")
        ax2.set_xlabel('Position X')
        ax2.set_aspect("equal", "box")

        plt.gca().invert_yaxis()
        plt.show()

        # duplicates = duplicates[duplicates.tracker_method == "Manual_tracking"]
    else:
        print("You have duplicates, but none of these came from the method 'Manual_tracking'.\n"
              "Priority will be given to the last duplicates.")
        tracks_obj = duplicates.tracker_method.unique()
        print("These are the duplicated trackers {}".format(tracks_obj))

        if len(tracks_obj) >= 2:
            index_2_del = duplicates.index[duplicates.tracker_method != tracks_obj[1]].tolist()

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

            ax1.plot(duplicates.loc[duplicates.tracker_method == tracks_obj[0], "Crab_position_x"],
                     duplicates.loc[duplicates.tracker_method == tracks_obj[0], "Crab_position_y"])
            ax1.title.set_text("Points from tracker {}".format(tracks_obj[0]))
            ax1.set_xlabel('Position X')
            ax1.set_ylabel('Position Y')
            ax1.set_aspect("equal", "box")

            ax2.plot(duplicates.loc[duplicates.tracker_method == tracks_obj[1], "Crab_position_x"],
                     duplicates.loc[duplicates.tracker_method == tracks_obj[1], "Crab_position_y"])
            ax2.title.set_text("Points from tracker {}".format(tracks_obj[1]))
            ax2.set_xlabel('Position X')
            ax2.set_aspect("equal", "box")

            plt.gca().invert_yaxis()
            plt.show()

    # print(duplicates)
    # print(index_2_del)
    # print(len(index_2_del))
    # print(track.shape)
    track = track.drop(index_2_del)
    print("{} duplicated registers were deleted".format(len(index_2_del)))
    # print(track.shape)
    # print(track_meta.head())

    os.makedirs("results/processed_tracks/", exist_ok=True)

    new_file = "results/processed_tracks/Pro_" + args["track_file"]
    track.to_csv(new_file, index=False)

    # with open("results/" + args["file"], "r") as file:
    #     meta_information = [next(file) for x in range(3)]

    meta_information = track_meta.values.tolist()
    # print(meta_information)
    print("######################## ##################### ################")

    with open(new_file, "r") as result_in:
        reader = list(csv.reader(result_in))
        # print(reader[:10])
        reader.insert(0, meta_information[2])
        reader.insert(0, meta_information[1])
        reader.insert(0, meta_information[0])
        # print(reader[:10])

    with open(new_file, "w", newline="") as result_out:
        writer = csv.writer(result_out)
        for line in reader:
            writer.writerow(line)

else:
    print("Not duplicates found")
