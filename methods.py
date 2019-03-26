#!/usr/bin/env python3

import cv2
import os
import logging
import sys
import numpy as np
import math
import time
import csv
import datetime
import pickle

import constant

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

quadratpts = []
position = (0, 0)
posmouse = (0, 0)
videoname = ""

def read_video(video_path):
    """
    Loads the video file, and it retrieves meta-information associated to the video been analysed.
    Creates Log file.

    Parameters
    ----------
    video_path: str
        A relative path to the video to be analyzed. The video should be placed inside folder 'video/'.

    Returns
    -------
    vid:
        a VideoCapture object to pass to 'while' statement.
    vid_length:
        an integer representing the video length in number of frames
    vid_fps:
        a floating point number representing the frame rate per seconds
    vid_width:
        an integer representing the video width in pixel unit
    vid_height:
        an integer representing the video height in pixel unit
    vid_fourcc:
        a four (4) character code which identifies and define a video codec, color and compression format.
        More information about FOURCC codes can be found at https://www.fourcc.org/

    Notes
    -----
    I should add condition that prevent logging same information twice given that script dist.py uses
    this function two times.
    Probably of logging functions should be handle in its own method.
    """

    global videoname

    name = os.path.basename(video_path)
    videoname, _ = os.path.splitext(name)

    try:
        os.mkdir("results/Log")
    except FileExistsError:
        pass

    logging.basicConfig(filename="results/Log/log_" + video_path + '.log',
                        level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("================ New event ================")
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
            logger.info("The total number of frames in video is %d" % vid_length)
            # Frame per seconds rate
            vid_fps = vid.get(cv2.CAP_PROP_FPS)
            logger.info("Frame rate per seconds is %f" % vid_fps)
            # Width in pixels
            vid_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            # Height in pixels
            vid_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info("Frame width is %d pixels, and frame height is %d pixels" % (vid_width, vid_height))
            vid_duration = int(vid_length)/vid_fps
            logger.info("Video duration %d seconds" % vid_duration)
            # Return video format
            # vid.get(cv2.CAP_PROP_FORMAT)
            # Video fourcc
            vid_fourcc = vid.get(cv2.CAP_PROP_FOURCC)
            vid_fourcc = int(vid_fourcc)
            vid_fourcc = ''.join([chr((vid_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            logger.info("Finish reading video")

    except cv2.error as e:
        print("cv2.error:", e)
        logger.error(str(e))
        sys.exit(1)

    except Exception as e:
        print("Exception:", e)
        logger.exception(str(e))
        sys.exit(1)

    return vid, vid_length, vid_fps, vid_width, vid_height, vid_duration, vid_fourcc


def set_video_star(vid, seconds, fps):

    """
    Calculates the frame where video stream should start.
    It is calculated based on the time in seconds provided by user and the frame rate per second from the video.

    Parameters
    ----------
    vid:
        A video stream gotten using cv2.VideoCapture function.
    seconds:
        A time in seconds where user want to start video stream. Please observe default value is None.
    fps:
        A frame rate per second. It is used to multiply target time in seconds.

    Returns
    -------
    vid:
        a video stream set to start at the targeted time
    target_frame:
        an integer representing the frame number where video stream should start
    """

    if seconds is None:
        target_frame = 1
    else:
        target_frame = int(int(seconds) * fps)

    # # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
    vid.set(1, target_frame - 1)

    return vid, target_frame


def draw_quadrat(frame, vertices):

    """
    Draw a rectangle in the original frame based on four coordinates representing quadrat's vertices.

    Parameters
    ----------
    frame:
        image
    vertices:
        A Numpy array np.int32 containing the coordinates (x,y) of four points.

    Returns
    -------
    frame:
        new image with drawn quadrat
    """

    cv2.polylines(frame, [vertices], True, constant.COLOR_SET_0, thickness=2)
    return frame


def mouse_click(event, x, y, flags, param):

    """
    Define mouse click events and capture mouse position and mouse left click coordinate on image.

    Parameters
    ----------
    event:
        A mouse event such as move, left button click, etc.
    x:
        Mouse x coordinate position on image
    y:
        Mouse y coordinate position on image
    flags:

    param:

    Returns
    -------
    """

    global quadratpts, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        position = (x, y)
        quadratpts.append(position)
        # print(quadratpts)

    if event == cv2.EVENT_MOUSEMOVE:
        posmouse = (x, y)


def enable_point_capture(capture_vertices):

    """
    Resolve if the OpenCV mouse call-back function must be called to capture quadrat's vertices.

    Parameters
    ----------
    capture_vertices: bool

    Returns
    -------

    """

    global quadratpts

    if capture_vertices is False:
        quadratpts = constant.QUADRAT_POINTS
    else:
        cv2.setMouseCallback("Vertices selection", mouse_click)


def draw_points_mousepos(frame, list_points, pos_mouse):
    """
    Draw the current position of the mouse and the vertices selected by user on the image.

    Parameters
    ----------
    frame:
        An image.
    list_points:
        List of points containing vertices' coordinates
    pos_mouse:
        The coordinates of the current position of the mouse.

    Returns
    -------
    frame:
        new image with drawn points
    """

    for i, val in enumerate(list_points):
        cv2.circle(frame, val, 3, constant.COLOR_SET_0, 2)

    cv2.putText(frame, "Mouse position {}".format(pos_mouse), (50, 710),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, constant.COLOR_SET_0, 2)

    return frame


def calc_proj(quadrat_pts):

    """
    Calculate a perspective transform 'M', and its inverse 'IM' based on four coordinates.
    Destination coordinates are calculated in this function.

    Parameters
    ----------
    quadrat_pts:
        Represent the vertices of the quadrat. It must be provided in the following order:
        first select top left, second select top right, third select bottom left, and fourth
        select bottom right.

    Returns
    -------
    M:
        A 3x3 perspective transform matrix.
    side:
        An integer which defines the width and height in pixels of destination points (i.e. new image, or quadrat)
    vertices_draw:
        A Numpy array np.int32 containing the coordinates (x,y) of the four points.
        To be passed to function draw_quadrat.
    IM:
        The inverse 3x3 perspective transform of M.

    Notes
    -------
    I should add the functionality to handle cases when the quadrat does not have equal sides (e.g. rectangular)
    """

    vertices = np.array([quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]], np.float32)
    vertices_draw = np.array([quadrat_pts[0], quadrat_pts[1], quadrat_pts[3], quadrat_pts[2]], np.int32)
    orig_pts = vertices
    # print(vertices)

    # dist = math.hypot(x2-x1, y2-y1)
    dist_a = math.sqrt((vertices[0][0] - vertices[1][0]) ** 2 + (vertices[0][1] - vertices[1][1]) ** 2)
    dist_b = math.sqrt((vertices[0][0] - vertices[2][0]) ** 2 + (vertices[0][1] - vertices[2][1]) ** 2)
    dist_c = math.sqrt((vertices[2][0] - vertices[3][0]) ** 2 + (vertices[2][1] - vertices[3][1]) ** 2)
    dist_d = math.sqrt((vertices[3][0] - vertices[1][0]) ** 2 + (vertices[3][1] - vertices[1][1]) ** 2)

    width = int(max(dist_a, dist_c))
    # width_10 = int(max(dist_a, dist_c) + 10)
    height = int(max(dist_b, dist_d))
    # height_10 = int(max(dist_b, dist_d) + 10)

    # print(dist_a, dist_b, dist_c, dist_d)
    # print("This is the width ", width, "This is the height ", height)

    # Conversion factors from pixel to cm per each side
    # side_a_c = constant.DIM[0] / dist_a
    # side_b_c = constant.DIM[1] / dist_b
    # side_c_c = constant.DIM[2] / dist_c
    # side_d_c = constant.DIM[3] / dist_d

    # print("Conversion factor per side", side_a_c, " ", side_b_c, " ", side_c_c, " ", side_d_c)

    # Average conversion factors from pixel to cm for quadrat height and wide
    # q_w = float(side_a_c + side_c_c) / 2
    # q_h = float(side_b_c + side_d_c) / 2
    # area = q_w * q_h
    side = np.max([width, height], axis=0)
    conversion = constant.DIM[0] / side

    # print("Quadrat wide factor is ", q_w, "\nQuadrat height factor is ", q_h,
    #       "\nQuadrat area factor is ", area, "\nDistance coversion factor is ", conversion)

    # print("The selected side vertices is ", side)
    dest_pts = np.float32([[0, 0], [side, 0], [0, side], [side, side]])
    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    IM = cv2.getPerspectiveTransform(dest_pts, orig_pts)

    # mini = np.amin(vertices, axis=0)
    # maxi = np.amax(vertices, axis=0)
    # print(mini, "and ", maxi)

    return M, side, vertices_draw, IM, conversion


def get_file_creation(video_path):
    local_creation = time.strftime("%d%m%Y %H%M%S", time.localtime(os.path.getctime("video/" + video_path)))
    creation = time.strftime("%d%m%Y %H%M%S", time.localtime(os.path.getmtime("video/" + video_path)))
    return local_creation, creation


def frame_to_time(info_video):

    start = datetime.datetime.strptime(info_video["creation"], "%d%m%Y %H%M%S")
    vid_duration = info_video['vid_duration']
    end = start + datetime.timedelta(0, vid_duration)
    step = vid_duration / info_video['length_vid']

    if "Counter" in info_video:
        time_absolute = start + (datetime.timedelta(0, step * (info_video["Counter"]+1)))
        time_absolute = time_absolute.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
        time_since_start = step * (info_video["Counter"]+1)

    else:
        time_absolute = start.strftime('%Y-%m-%d %H:%M:%S.%f').rstrip('0')
        time_since_start = 0

    return start, end, step, time_absolute, time_since_start


def data_writer(video_path, info_video, head_true):

    # create file name with name
    name = os.path.basename(video_path)
    video_name, file_extension = os.path.splitext(name)
    name_result_file = "results/" + video_name + "_" + "track_id" + ".csv"

    if head_true:
        result_file = open(name_result_file, "w", newline="\n")
        wr = csv.writer(result_file, delimiter=",")
        date_now = time.strftime("%d%m%Y")
        time_now = time.strftime("%H%M")
        wr.writerow(["file_name", "processed_at_date", "processed_at_time", "length_video", "fps_video",
                     "target_frame_used", "vertice_1", "vertice_2", "vertice_3", "vertice_4",
                     "projected_q_side", "q_conversion_factor_distance", "tracker_method"])

        wr.writerow([name, date_now, time_now, info_video["length_vid"], info_video["fps"],
                    info_video["target_frame"], quadratpts[0], quadratpts[1],
                    quadratpts[2], quadratpts[3], info_video["side"], info_video["conversion"],
                    info_video["tracker"]])
        wr.writerow(["\n"])
        wr.writerow(["Frame_number", "Time_absolute", "Time_lapsed_since_start(secs)",
                     "Crab_ID", "Crab_Position_x", "Crab_position_y"])

    else:
        # save track_info to file
        result_file = open(name_result_file, "a+", newline="\n")
        wr = csv.writer(result_file, delimiter=",")

        wr.writerow([info_video["Frame"], info_video["Time_absolute"], info_video["Time_since_start"],
                     info_video["Crab_ID"], info_video["Crab_Position_x"], info_video["Crab_Position_y"]])

    return result_file


class CompileInformation(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value


class CrabNames(object):

    instances = []

    def __init__(self, crab_name, crab_start_position, crab_species, sex, crab_handedness):
        self.__class__.instances.append(self)
        self.crab_name = crab_name
        self.start_position = crab_start_position
        self.species = crab_species
        self.sex = sex
        self.handedness = crab_handedness

    def save_crab_names(self):
        filename = "results/" + "example"
        file = open(filename, "wb")
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()
        # with open("file", "wb") as f:
        #     pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def open_crab_names(path):
        file = open(path, "rb")
        temp_dict = pickle.load(file)
        for instance in temp_dict:
            # print("This is a ", instance.crab_name)
            __class__.instances.append(instance)
        file.close()

    # def print_crab_names(__class__.instances):
    #
    #     for k, v in vars(__class__.instances).items():
    #         print(k, v)




# '''
# Function to display percentage of video analyzed based on total number of frames
# '''
#
# '''
# Function to sample size and color for each crab.
# It should return one value for size (blob maximum axis?), and three values for color (Blue, Green and Red)
# '''
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
# """
# Function that defines the end of tracking
# """
