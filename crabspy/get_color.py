#!/usr/bin/env python3

"""
Extract crab color histogram from its track
"""

import argparse
import pandas as pd
from collections import deque
import cv2
import numpy as np
import matplotlib.pyplot as plt

import methods
import constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--track_file", default="GP010016_Us_focal_01.csv", help="Provide path to file")
ap.add_argument("-s", "--seconds", default=None,
                help="Provide the targeted time in seconds of video section you want to jump to")
ap.add_argument("-f", "--frame", default=None, type=int,
                help="Provide the targeted frame of video section you want to jump to")
ap.add_argument("-b", "--bins_number", default=10, type=int,
                help="Provide number of wished bins")
args = vars(ap.parse_args())

resz_val = constant.RESIZE

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

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.set_title("Histogram of crab colour")
ax0.set_xlabel("Color intensity")
ax0.set_ylabel("Frequency")
ax1.set_title("Histogram of light reference")
ax1.set_xlabel("Color intensity")
ax1.set_ylabel("Frequency")

bins_num = args["bins_number"]

ch_red, = ax0.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'r', lw = 2, alpha = 0.65)
ch_blue, = ax0.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'b', lw = 2, alpha = 0.65)
ch_green, = ax0.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'g', lw = 2, alpha = 0.65)

lr_red, = ax1.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'r', lw = 2, alpha = 0.65)
lr_blue, = ax1.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'b', lw = 2, alpha = 0.65)
lr_green, = ax1.plot(np.arange(bins_num), np.zeros((bins_num)),  c = 'g', lw = 2, alpha = 0.65)

ax0.set_xlim(0, bins_num-1)
ax0.set_ylim(0, 10)
ax1.set_xlim(0, bins_num-1)
ax1.set_ylim(0, 10)
plt.ion()
plt.show()

# From warp.py
fgbg1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=20)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=100)
fgbg3 = cv2.createBackgroundSubtractorKNN(history=5000, dist2Threshold=250)

for_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, constant.ERODE)
for_di = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

methods.hist_writer(video_name, individuals, bins_num, None, None, None, header = True)

while vid.isOpened():
    ret, img = vid.read()

    if ret:

        result = cv2.warpPerspective(img, M, (side, side))
        result2 = result.copy()

        crab_frame = cv2.warpPerspective(img, M, (side, side))

        methods.draw_quadrat(img, vertices_draw)

        # From warp.py
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        hsl = cv2.cvtColor(result, cv2.COLOR_BGR2HLS_FULL)
        one, two, three = cv2.split(hsl)
        fb_res_two3 = fgbg3.apply(two, learningRate=-1)
        fb_res_two3 = cv2.erode(fb_res_two3, for_er)
        fb_res_two3 = cv2.dilate(fb_res_two3, for_di)
        masked = cv2.bitwise_and(result, result, mask=fb_res_two3)

        masked = cv2.addWeighted(result, 0, masked, 1, 0)
        edge = cv2.Canny(masked, threshold1=100, threshold2=230)

        if counter <= f_max:

            try:

                df_f = f_number[counter]

                for index, row in df_f.iterrows():

                    try:
                        crab = row["Crab_ID"]
                        x = int(row["Crab_position_x"])
                        y = int(row["Crab_position_y"])
                        bgr = crab_colors[crab][1]
                        cv2.circle(result, (x, y), 15, bgr, 2)

                        crab_window = masked[y-25:y+25, x-25:x+25]
                        crab_window_blob = fb_res_two3[y-25:y+25, x-25:x+25]
                        output = cv2.connectedComponentsWithStats(crab_window_blob, 8, cv2.CV_32S)
                        stats = output[2]
                        total_pixels = sum(stats[1:, 4])

                        light_ref = result2[320:350, 395:425]
                        lr_total_pixels = np.prod(light_ref.shape[:2])

                        channels0, mat0 = methods.split_colour(crab_window, "BW")
                        channels1, mat1 = methods.split_colour(light_ref, "BW")

                        hist0 = methods.get_hist(channels0, crab_window_blob, bins_num, total_pixels, normalize=True)
                        hist1 = methods.get_hist(channels1, None, bins_num, lr_total_pixels, normalize=True)

                        try:
                            ch_red.set_ydata(hist0[0])
                            ch_blue.set_ydata(hist0[1])
                            ch_green.set_ydata(hist0[2])

                            lr_red.set_ydata(hist1[0])
                            lr_blue.set_ydata(hist1[1])
                            lr_green.set_ydata(hist1[2])

                            cv2.imshow("Crab window", mat0)
                            cv2.imshow("Light reference", mat1)
                            fig.canvas.draw()

                            methods.hist_writer(video_name, individuals, bins_num, total_pixels, hist0, counter,
                                                header=False)

                        except (ValueError):
                            pass

                    except (ValueError, IndexError):
                        pass

            except KeyError:
                pass

        else:
            pass

        percentage_vid = counter/track_meta["length_video"].values[0]*100
        text = "Video {0:.1f} %".format(percentage_vid)
        cv2.putText(result, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        cv2.putText(result, "Frame n. {0:d}".format(counter), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)

        result2 = cv2.addWeighted(result, 0.6, result2, 0.4, 0)

        cv2.imshow("result", result2)
        # cv2.imshow("background substraction", fb_res_two3)
        cv2.imshow("masked", masked)
        # cv2.imshow("Crab window", crab_window)
        # cv2.imshow("Light reference", light_ref)
        # fig.canvas.draw()

        # methods.hist_writer(video_name, individuals, bins_num, total_pixels, hist_val, counter, header = False)

        counter += 1

        ch_red.set_ydata(np.zeros(bins_num))
        ch_blue.set_ydata(np.zeros(bins_num))
        ch_green.set_ydata(np.zeros(bins_num))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    else:
        break

vid.release()
cv2.destroyAllWindows()
