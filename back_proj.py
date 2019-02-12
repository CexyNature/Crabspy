'''
This is an incomplete code trying to back transform coordinates from warped image to original image.
'''


import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="GP010016.MP4", help='Provide path to video file')
args = vars(ap.parse_args())


vid = cv2.VideoCapture('video/' + args['video'])

# quadrat_pts = [(628, 105), (946, 302), (264, 393), (559, 698)]
# pts1 = np.float32([[600, 100], [950, 300], [300, 400], [600, 700]])
# pts1 = np.float32([[250, 100], [350,100], [200, 200], [300, 200]])
pts1 = np.float32([[250, 100], [400,100], [200, 200], [300, 200]])
print(pts1)
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

print(pts1[0])

M = cv2.getPerspectiveTransform(pts1, pts2)
IM = cv2.getPerspectiveTransform(pts2, pts1)

while True:
    _, frame = vid.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    _, img = vid.read()
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # print(frame.shape) #780 x 1280

    cv2.circle(img, (int(pts1[0][0]), int(pts1[0][1])), 5, (0,255,0), -1)
    cv2.circle(img, (int(pts1[1][0]), int(pts1[1][1])), 5, (255,0,0), -1)
    cv2.circle(img, (int(pts1[2][0]), int(pts1[2][1])), 5, (0,0,255), -1)
    cv2.circle(img, (int(pts1[3][0]), int(pts1[3][1])), 5, (255,255,255), -1)

    # pts1 = np.float32([[250,100], [350,100], [200, 200], [300, 200]])
    # pts2 = np.float32([[0,0], [300,0], [0, 300], [300,300]])
    #
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # IM = cv2.getPerspectiveTransform(pts2, pts1)

    warped = cv2.warpPerspective(frame, M, (300, 300))

    proj_pts = np.array([pts2], dtype='float32')
    # print('this is proj_pts before ', proj_pts)
    # proj_pts = np.array([proj_pts])
    # print('this is proj_pts after ', proj_pts)

    back_t = cv2.transform(proj_pts, IM)
    print('This is back t', back_t)

    cv2.circle(frame, (back_t[0][0][0], back_t[0][0][1]), 5, (0, 255, 0), -1)
    cv2.circle(frame, (back_t[0][1][0], back_t[0][1][1]), 5, (255, 0, 0), -1)
    cv2.circle(frame, (back_t[0][2][0], back_t[0][2][1]), 5, (0, 0, 255), -1)
    cv2.circle(frame, (back_t[0][3][0], back_t[0][3][1]), 5, (255, 255, 255), -1)

    cv2.imshow('Back_transformed', frame)
    cv2.imshow('Original', img)
    cv2.imshow('Warped', warped)


    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
