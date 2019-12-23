#!/usr/bin/env python3

"""
Get stats for a particular track
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default="test_track.csv", help="Provide path to file")
args = vars(ap.parse_args())

try:
    track = pd.read_csv("results/" + args["file"], header=2, skiprows=range(0, 1))
    track_meta = pd.read_csv("results/" + args["file"], header=0, nrows=1)
    fps = track_meta["fps_video"][0]
    max_frame_number = track["Frame_number"].max()
    last_frame_number = track["Frame_number"].iloc[-1]
    # print(max_frame_number)
    # print(last_frame_number)

    min_frame = track["Frame_number"].min()
    tc_empties = track["Crab_position_x"].isna().sum()
    bmc_empties = track["Crab_position_cx"].isna().sum()
    num_frames_attemp = len(track.index)

    print("You have tracked up to frame index {} (or {} seconds)".format(max_frame_number,
                                                                          round(max_frame_number / fps)),
          "The last observation was at frame {} (or {} seconds)".format(last_frame_number,
                                                                        round(last_frame_number / fps)),
          "A total of {} and {} NA values are present in tracker's " \
          "center and blob mass's center, respectively.".format(tc_empties, bmc_empties))

    text_plot0 = "Tracking started at frame {}.\n You attempt to track for {} frames.\n".format(min_frame, num_frames_attemp)

    text_plot1 = "You have tracked up to frame\n index {} (or {} seconds).\n" \
                "The last observation was at frame\n {} (or {} seconds).".format(max_frame_number, round(max_frame_number / fps),
                                                                              last_frame_number, round(last_frame_number / fps))
    text_plot2 = "\nA total of {} and {} \nNA values are present in tracker's " \
                 "\ncenter and blob mass's center, \nrespectively.".format(tc_empties, bmc_empties)

    text_plot = text_plot0 + text_plot1 + text_plot2

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(track.Crab_position_x, track.Crab_position_y, color="r", label = "Tracker's center")
    ax1.plot(track.Crab_position_cx, track.Crab_position_cy, color="b", label = "Blob mass's center")
    ax1.text(0.2, 0.5, text_plot, fontsize=8, ha='center', va='top', transform=plt.gcf().transFigure)
    ax1.title.set_text("Tracking path")
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')
    ax1.set_aspect("equal", "box")
    ax1.legend(loc = "lower center", bbox_to_anchor=(0.5, -0.35))

    plt.gca().invert_yaxis()
    # plt.gca().text(0.5, 0.5, text_plot, fontsize=8)
    plt.subplots_adjust(left=0.5)
    plt.show()

except IOError:
    print("File not found, please double check path and file name")
    pass