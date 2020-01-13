#!/usr/bin/env python3

"""
Module for recursively listing all videos in a folder and extracting meta information associated to videos
"""

import os
import csv
from datetime import datetime
import subprocess
import json
import pprint
import argparse

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", help="Provide path to folder containing video(s)")
args = vars(ap.parse_args())

now = datetime.now()
now = now.strftime('%Y-%m-%d %H:%M')
file_list = open('videos.csv', 'w', newline='\n')

wr = csv.writer(file_list, delimiter=',')
wr.writerow(['File created at ' + str(now)])
wr.writerow(['file_name', 'file_directory', 'file_extension', 'error', 'file_path', 'file_size', 'file_created',
             'tag_major_brand', 'file_duration', 'duration_ts', 'nb_frames', 'fps', 'codec', 'codec_tag', 'aspect_ratio',
             'width', 'height'])

for root, dirs, files in os.walk(args['path']):
    for name in files:
        file_path = os.path.join(root, name)
        file_path = os.path.abspath(file_path)
        if name.endswith(('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI')):
            print(name)
            command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-sexagesimal',
                       '-show_streams', file_path]

            try:
                call_1 = subprocess.check_output(command).decode('utf-8')
                info = json.loads(call_1)
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(info)

                file_name_reading = info['format']['filename']
                format_long_name = info['format']['format_long_name']
                size = info['format']['size']
                created = info['format']['tags']['creation_time']
                tag_major_brand = info['format']['tags']['major_brand']
                duration = info['format']['duration']
                duration_ts = info['streams'][0]['duration_ts']
                nb_frames = info['streams'][0]['nb_frames']
                fps = info['streams'][0]['avg_frame_rate']
                codec = info['streams'][0]['codec_name']
                codec_tag = info['streams'][0]['codec_tag']
                aspect_ratio = info['streams'][0]['display_aspect_ratio']
                width = info['streams'][0]['width']
                height = info['streams'][0]['height']
                error = 'FALSE'

                name, name_ext = os.path.splitext(name)
                wr.writerow([name, root, name_ext, error, file_name_reading, size, created, tag_major_brand,
                             duration, duration_ts, nb_frames, fps, codec, codec_tag, aspect_ratio, width, height])
                pass

            except subprocess.CalledProcessError as exception:
                name, name_ext = os.path.splitext(name)
                error = 'True'
                wr.writerow([name, root, name_ext, error, exception])

            except KeyError as e:
                name, name_ext = os.path.splitext(name)
                error = 'True'
                wr.writerow([name, root, name_ext, error, "KeyError: {}".format(e)])
                pass
        else:
            # print(name)
            pass
file_list.close()
