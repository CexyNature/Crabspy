#!/usr/bin/env python3

import cv2
import os
import logging
import sys
# import numpy as np
# import math
# import time
# import csv


"""
Function that defines the end of tracking
"""


def read_video(video_path):
    """
    This function loads the video file, and it retrieves meta-information associated to the video been analysed.

    :param video_path: a relative path to the video to be analyzed.
        The video should be placed inside folder 'video/'.

    :return:

        vid: a VideoCapture object to pass to 'while' statement.
        vid_length: an integer representing the video length in number of frames
        vid_fps: a floating point number representing the frame rate per seconds
        vid_width: an integer representing the video width in pixel unit
        vid_height: an integer representing the video height in pixel unit
        vid_fourcc: a four (4) character code which identifies and define a video codec, color and compression format.
            More information about FOURCC codes can be found at https://www.fourcc.org/
    """
    logging.basicConfig(filename="results/Log/log_" + video_path + '.log',
                        level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("================ New event ================")
    logger.info("Creating 'Log' folder")

    try:
        os.mkdir("results/Log")
    except FileExistsError:
        logger.info("'Log' folder found")
        pass

    logger.info("Reading video " + video_path)
    try:
        vid = cv2.VideoCapture("video/" + video_path)
        if not vid.isOpened():
            logger.error("Video could not been read.")
            raise NameError("Video stream from video file could not been read. Check path and video file name.")

        else:
            logger.info("Video was read successfully")
            # Total length in frames
            vid_length = vid.get(cv2.CAP_PROP_FRAME_COUNT)
            # Frame per seconds rate
            vid_fps = vid.get(cv2.CAP_PROP_FPS)
            # Width in pixels
            vid_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            # Height in pixels
            vid_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # Return video format
            # vid.get(cv2.CAP_PROP_FORMAT)
            # Video fourcc
            vid_fourcc = vid.get(cv2.CAP_PROP_FOURCC)
            vid_fourcc = int(vid_fourcc)
            vid_fourcc = ''.join([chr((vid_fourcc >> 8 * i) & 0xFF) for i in range(4)])

            logger.info("The total number of frames in video is %d" % vid_length)
            logger.info("Frame rate per seconds is %f" % vid_fps)
            logger.info("Frame width is %d pixels, and frame height is %d pixels" % vid_width, vid_height)
            logger.info("Finish reading video")

            return vid, vid_length, vid_fps, vid_width, vid_height, vid_fourcc

    except cv2.error as e:
        print("cv2.error:", e)
        logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        print("Exception:", e)
        logger.exception(str(e))
        sys.exit(1)


def set_video_star(vid, seconds, fps):
    """
    This function calculates the frame where video stream should start.
    It is calculated based on the time in seconds provided by user and the frame rate per second from the video.

    :param vid: A video stream gotten using cv2.VideoCapture function.
    :param seconds: A time in seconds where user want to start video stream. Please observe default value is None.
    :param fps: A frame rate per second. It is used to multiply target time in seconds.

    :return:

        vid: a video stream set to start at the targeted time
        target_frame: an integer representing the frame number where video stream should start
    """

    if seconds is None:
        target_frame = 1
    else:
        target_frame = int(int(seconds) * fps)

    # # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
    vid.set(1, target_frame - 1)

    return vid, target_frame


#
# '''
# Function to display percentage of video analyzed based on total number of frames
# '''
#
#
# '''
# Function to sample size and color for each crab.
# It should return one value for size (blob maximum axis?), and three values for color (Blue, Green and Red)
# '''
#
#
# '''
# Function to return smoothed x and x position
# '''
#
# '''
# Function to process frame with high light contrast.
# It should return a frame
# '''
#
# '''
# Function to initialize all trackers
# '''
#
#
#
def draw_quadrat(frame, vertices):

    # This function draws a rectangle in the original frame
    # based on the four quadrat vertices

    cv2.polylines(frame, [vertices], True, (204, 204, 0), thickness=2)
    return frame

#
# def calc_proj(quadrat_pts):
#     orig_pts = np.float32([quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]])
#
#     # dist = math.hypot(x2-x1, y2-y1)
#     dist_a = math.sqrt((quadrat_pts[0][0] - quadrat_pts[1][0]) ** 2 + (quadrat_pts[0][1] - quadrat_pts[1][1]) ** 2)
#     dist_b = math.sqrt((quadrat_pts[1][0] - quadrat_pts[2][0]) ** 2 + (quadrat_pts[1][1] - quadrat_pts[2][1]) ** 2)
#     dist_c = math.sqrt((quadrat_pts[2][0] - quadrat_pts[3][0]) ** 2 + (quadrat_pts[2][1] - quadrat_pts[3][1]) ** 2)
#     dist_d = math.sqrt((quadrat_pts[0][0] - quadrat_pts[3][0]) ** 2 + (quadrat_pts[0][1] - quadrat_pts[3][1]) ** 2)
#
#     width = int(max(dist_a, dist_c) + 10)
#     height = int(max(dist_b, dist_d) + 10)
#
#     print(dist_a, dist_b, dist_c, dist_d)
#     print(width, height)
#
#     dest_pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
#     M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
#
#     return M, width, height
#
#
# def data_writer(video):
#
#     path = os.path.basename(video)
#     file_name, file_ext = os.path.splitext(path)
#
#     date_now = time.strftime('%d%m%Y')
#     time_now = time.strftime('%H%M')
#
#     name_resultFile = 'results/' + file_name + '_' + str(date_now) + '_' + str(time_now) + '.csv'
#
#     resultFile = open(name_resultFile, 'w', newline='\n')
#
#     wr = csv.writer(resultFile, delimiter=',')
#
#     wr.writerow(['file_name', 'processed_at_date', 'processed_at_time', 'length_video', 'fps_video',
#                  'target_frame_used', 'point_1', 'point_2', 'point_3', 'point_4'])
#
#     resultFile.close()
#
#
#
