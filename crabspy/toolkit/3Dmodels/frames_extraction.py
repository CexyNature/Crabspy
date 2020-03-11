#!/usr/bin/python

"""
This scripts takes a video and it returns every nth frame
"""

import cv2
import argparse
import os

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, help='A path to the video')
parser.add_argument('--frames', '-f', type=int, help='Provide number of frames to skip')
args = vars(parser.parse_args())

vid = cv2.VideoCapture(args['path'])
count = 0

path = os.path.basename(args['path'])
file_name, file_ext = os.path.splitext(path)

t_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'Total number of frames in the video is {t_frames}')

# number of frames to skip
numFrameToSave = args['frames']
directory = 'Data_frames'
sub_directory = file_name + '_' + str(args['frames'])
new_path = os.path.join(directory, sub_directory)

print('Frames will be saved in following directory: {}'.format(new_path))

while True:
    ret, frame = vid.read()

    if ret is True:
        frame_tiny = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if count % numFrameToSave == 0:
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            else:
                pass
            count_4digits = '{0:04}'.format(count)
            filename = file_name + '_%s.jpg' % count_4digits
            cv2.imwrite(os.path.join(new_path, filename), frame)

        count += 1
        cv2.imshow('frame', frame_tiny)
        key = cv2.waitKey(1) & 0XFF
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
vid.release()
