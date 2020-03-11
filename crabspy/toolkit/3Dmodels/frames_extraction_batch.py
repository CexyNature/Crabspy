#!/usr/bin/python

"""
This script recursively runs frames_extraction.py in all videos inside the specified directory.
"""

import os
import argparse
from datetime import datetime

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d', type=str, help='Directory name')
parser.add_argument('--frames', '-f', default = 6, type=int, help='Provide number of frames to skip')
args = vars(parser.parse_args())

video_files = []

for root, dirs, files in os.walk(args['directory']):
    for file in files:
        if file.endswith('.MP4'):
            path = os.path.join(root, file)
            video_files.append(path)
            print(path)

for video in video_files:
    print('Extracting frames for video {}'.format(video))
    os.system ('python frames_extraction.py -p {} -f {}'.format(video, args['frames']))

print('Total time {}'.format(datetime.now() - start_time))
