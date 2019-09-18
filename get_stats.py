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
ap.add_argument("-f", "--file", default="GP010016_Us_focal_01.csv", help="Provide path to file")
args = vars(ap.parse_args())

try:
    track = pd.read_csv("results/" + args["file"], header=2, skiprows=range(0, 1))
    track_meta = pd.read_csv("results/" + args["file"], header=0, nrows=1)
    fps = track_meta["fps_video"][0]
    max_frame_number = track["Frame_number"].max()
    last_frame_number = track["Frame_number"].iloc[-1]
    # print(max_frame_number)
    # print(last_frame_number)
    print("You have tracked up to frame number {} (or {} seconds)".format(max_frame_number,
                                                                          round(max_frame_number / fps)),
          "The last observation was at frame {} (or {} seconds)".format(last_frame_number,
                                                                        round(last_frame_number / fps)))
    text_plot = "You have tracked up to frame\n number {} (or {} seconds).\n" \
                "The last observation was at frame\n {} (or {} seconds).".format(max_frame_number, round(max_frame_number / fps),
                                                                              last_frame_number, round(last_frame_number / fps))
    # text_plot = "My message"
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(track.Crab_position_x, track.Crab_position_y, color="r")
    ax1.plot(track.Crab_position_cx, track.Crab_position_cy, color="b")
    ax1.text(0.2, 0.5, text_plot, fontsize=8, ha='center', va='top', transform=plt.gcf().transFigure)
    ax1.title.set_text("Tracking path")
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')
    ax1.set_aspect("equal", "box")

    plt.gca().invert_yaxis()
    # plt.gca().text(0.5, 0.5, text_plot, fontsize=8)
    plt.subplots_adjust(left=0.5)
    plt.show()

except IOError:
    print("File not found, please double check path and file name")
    pass