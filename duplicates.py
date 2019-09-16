#!/usr/bin/env python3

"""
This scripts deals with duplicate rows in tracking file.
This is normally product of tracking the same individual twice using the same or different methods.
"""

import os
import argparse
import pandas as pd
from datetime import datetime

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", default="tracking_file.csv", help="Provide video name")
args = vars(ap.parse_args())

video_name = os.path.splitext(args["video"])[0]