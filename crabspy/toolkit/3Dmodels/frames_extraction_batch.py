import os
import argparse
import time
from datetime import datetime

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
    os.system ('python save_frame.py -p {} -f {}'.format(video, args['frames']))

print('Total time {}'.format(datetime.now() - start_time))
