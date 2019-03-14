'''
Some methods used in dist.py and other files
'''


import cv2
import numpy as np
import math
import os
import time
import csv


'''
Function that defines the end of tracking
'''


'''
Function to retrieve video meta-information and save it in log file.
'''


'''
Function to display percentage of video analyzed based on total number of frames
'''


'''
Function to sample size and color for each crab.
It should return one value for size (blob maximum axis?), and three values for color (Blue, Green and Red)
'''


'''
Function to return smoothed x and x position
'''

'''
Function to process frame with high light contrast.
It should return a frame
'''

'''
Function to initialize all trackers 
'''



def draw_quadrat(frame, vertices):

    # This function draws a rectangle in the original frame
    # based on the four quadrat vertices

    cv2.polylines(frame, [vertices], True, (204, 204, 0), thickness=2)
    return frame


def calc_proj(quadrat_pts):
    orig_pts = np.float32([quadrat_pts[0], quadrat_pts[1], quadrat_pts[2], quadrat_pts[3]])

    # dist = math.hypot(x2-x1, y2-y1)
    dist_a = math.sqrt((quadrat_pts[0][0] - quadrat_pts[1][0]) ** 2 + (quadrat_pts[0][1] - quadrat_pts[1][1]) ** 2)
    dist_b = math.sqrt((quadrat_pts[1][0] - quadrat_pts[2][0]) ** 2 + (quadrat_pts[1][1] - quadrat_pts[2][1]) ** 2)
    dist_c = math.sqrt((quadrat_pts[2][0] - quadrat_pts[3][0]) ** 2 + (quadrat_pts[2][1] - quadrat_pts[3][1]) ** 2)
    dist_d = math.sqrt((quadrat_pts[0][0] - quadrat_pts[3][0]) ** 2 + (quadrat_pts[0][1] - quadrat_pts[3][1]) ** 2)

    width = int(max(dist_a, dist_c) + 10)
    height = int(max(dist_b, dist_d) + 10)

    print(dist_a, dist_b, dist_c, dist_d)
    print(width, height)

    dest_pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)

    return M, width, height


def data_writer(video):

    path = os.path.basename(video)
    file_name, file_ext = os.path.splitext(path)

    date_now = time.strftime('%d%m%Y')
    time_now = time.strftime('%H%M')

    name_resultFile = 'results/' + file_name + '_' + str(date_now) + '_' + str(time_now) + '.csv'

    resultFile = open(name_resultFile, 'w', newline='\n')

    wr = csv.writer(resultFile, delimiter=',')

    wr.writerow(['file_name', 'processed_at_date', 'processed_at_time', 'length_video', 'fps_video',
                 'target_frame_used', 'point_1', 'point_2', 'point_3', 'point_4'])

    resultFile.close()



