#!/usr/bin/env python3

"""
This module creates a dictionary for all crab snapshots
"""

import os
import json
import pandas as pd

root_path = "results/snapshots/GP010016"

name = []
# species = []
handedness = []
# sex = []
video = []
pic_number = []
path_img = []
sample = []

for root, sub_directories, files in os.walk(root_path):
    for img in files:
        if ".jpg" in img:
            video.append(img.split("_")[0])
            handedness.append(img.split("_")[1])
            name.append(img.split("_")[2])
            pic_number.append(img.split("_")[3])
            path_img.append(os.path.join(root, img).replace(os.sep, "/"))
            sample.append(img)
            # print("This is {} a {} handed individual from {} video, pic number {}".format(name, handedness, video, pic_number))

d = {"Name": name, "Handedness": handedness, "Video": video,
     "Pic_reference": pic_number, "Path": path_img, "Sample": sample}
df = pd.DataFrame(d, columns=["Video", "Name", "Handedness", "Path", "Sample", "Pic_reference"])
df.to_csv("result101.csv", index=False)
df.to_json("result101.json", orient="columns")
# samples_dict = dict(zip(name, sample))
# with open("sample.json", "w") as fp:
#     json.dump(samples_dict, fp, sort_keys=True, indent=4)
# for i in samples_dict:
#     print(i, samples_dict[i])
