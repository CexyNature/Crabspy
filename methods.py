#!/usr/bin/env python3

import cv2
import os
import logging
import sys
import numpy as np
import math
import time
import csv


quadrat_pts = []
position = (0, 0)
posmouse = (0, 0)
dim = [50, 50, 50, 50]
color_set = (243, 28, 20)


def read_video(video_path):
    """
    Loads the video file, and it retrieves meta-information associated to the video been analysed.

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
    """

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

    return vid, vid_length, vid_fps, vid_width, vid_height, vid_fourcc


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

    cv2.polylines(frame, [vertices], True, (204, 204, 0), thickness=2)
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

    global quadrat_pts, position, posmouse

    if event == cv2.EVENT_LBUTTONDOWN:
        position = (x, y)
        quadrat_pts.append(position)
        # print(quadrat_pts)

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

    global quadrat_pts
    if capture_vertices is False:
        quadrat_pts = [(628, 105), (946, 302), (264, 393), (559, 698)]
    else:
        cv2.setMouseCallback("Vertices selection", mouse_click)


def draw_points_mousepos(frame, list_points, posmouse):

    """
    Draw the current position of the mouse and the vertices selected by user on the image.

    Parameters
    ----------
    frame:
        An image.
    list_object:
        List of points containing vertices' coordinates
    posmouse:
        The coordinates of the current position of the mouse.

    Returns
    -------
    frame:
        new image with drawn points
    """

    for i, val in enumerate(list_points):
        cv2.circle(frame, val, 3, color_set, 2)

    cv2.putText(frame, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_set, 2)

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
        A Numpy array np.int32 containing the coordinates (x,y) of the four points. To be passed to function draw_quadrat.
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
    width_10 = int(max(dist_a, dist_c) + 10)
    height = int(max(dist_b, dist_d))
    height_10 = int(max(dist_b, dist_d) + 10)

    # print(dist_a, dist_b, dist_c, dist_d)
    # print("This is the width ", width, "This is the height ", height)

    # Conversion factors from pixel to cm per each side
    side_a_c = dim[0] / dist_a
    side_b_c = dim[1] / dist_b
    side_c_c = dim[2] / dist_c
    side_d_c = dim[3] / dist_d

    # print("Conversion factor per side", side_a_c, " ", side_b_c, " ", side_c_c, " ", side_d_c)

    # Average conversion factors from pixel to cm for quadrat height and wide
    q_w = float(side_a_c + side_c_c) / 2
    q_h = float(side_b_c + side_d_c) / 2
    area = q_w * q_h
    side = np.max([width, height], axis=0)
    conversion = dim[0] / side

    # print("Quadrat wide factor is ", q_w, "\nQuadrat height factor is ", q_h,
    #       "\nQuadrat area factor is ", area, "\nDistance coversion factor is ", conversion)

    print("The selected side vertices is ", side)
    dest_pts = np.float32([[0, 0], [side, 0], [0, side], [side, side]])
    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    IM = cv2.getPerspectiveTransform(dest_pts, orig_pts)

    mini = np.amin(vertices, axis=0)
    maxi = np.amax(vertices, axis=0)
    # print(mini, "and ", maxi)

    return M, side, vertices_draw, IM



"""
Function that defines the end of tracking
"""

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

# def save_tracks(video_path):
#
#     try:
#         os.mkdir("results")
#     except FileExistsError:
#         pass
#
#     path = os.path.basename(video_path)
#     file_name, file_ext = os.path.splitext(path)
#     name_resultFile = "results/" + file_name + ".csv"
#
#     if os.path.exists(name_resultFile):
#         append_condition = 'a'
#     else:
#         append_condition = 'w'
#     resultFile = open(name_resultFile, append_condition, newline="\n")
#     wr = csv.writer(resultFile, delimiter=",")
#
#     date_now = time.strftime("%d%m%Y")
#     time_now = time.strftime("%H%M")
#     wr.writerow(["Video {}".format(path), "Processed at date {} time {}".format(date_now, time_now)])
#     wr.writerow(["file_name", "processed_at_date", "processed_at_time", "length_video", "fps_video",
#                  "target_frame_used", "vertice_1", "vertice_2", "vertice_3", "vertice_4",
#                  "projected_q_side", "q_factor_distance", "tracker_method"])
#     resultFile.close()



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