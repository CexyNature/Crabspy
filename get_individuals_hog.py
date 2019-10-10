#!/usr/bin/env python3

"""
Module for assessing handedness in male crabs"
"""

import argparse
import cv2
import os
from skimage import feature, exposure

import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Provide path to video file")
ap.add_argument("-c", "--crab_id", help="Provide a name for the crab to be tracked")
args = vars(ap.parse_args())


def main():
    videoname, _ = os.path.splitext(args["video"])
    img_path = "results/snapshots/" + videoname + "/" + args["crab_id"]
    img_res = "results/snapshots/" + videoname + "_HOG/" + args["crab_id"]

    for img in os.listdir(img_path):
        # print(img)

        frame = cv2.imread(os.path.join(img_path, img))
        frame = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        one, two, three = cv2.split(hsl)
        new_sat = exposure.adjust_sigmoid(two, 0.75, inv=True)
        # new_img = cv2.merge([new_sat, two, three])
        canny = cv2.Canny(new_sat, 200, 255)
        new_fd, new_hog = feature.hog(canny, orientations=5, pixels_per_cell=(2, 2), block_norm="L1",
                                      cells_per_block=(3, 3), transform_sqrt=False, visualise=True,
                                      feature_vector=False)
        # np.savetxt("array_hog.txt", new_fd, fmt="%s", delimiter=";")

        os.makedirs(img_res, exist_ok=True)
        fullpath = os.path.join(img_res, 'hog_' + img)
        # misc.imsave(fullpath, new_hog)

        # new_hog = exposure.rescale_intensity(new_hog, in_range=(0, 20))
        cv2.imshow("Original", frame)
        cv2.imshow("HSL", hsl)
        cv2.imshow("Sat", two)
        cv2.imshow("NewSat", new_sat)
        cv2.imshow("Canny", canny)
        cv2.imshow("HOG", new_hog.astype("uint8") * 255)
        cv2.imwrite(fullpath, new_hog.astype("uint8") * 255)

        cv2.waitKey(6)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

if __name__ == '__main__':
    main()
