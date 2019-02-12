import cv2
import argparse
import methods
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default="GP010016.MP4", help='Provide path to video file')
ap.add_argument('-p', '--reproject', type=bool, default=False,  help='Should a warp transofrmation be done')
ap.add_argument('-d', '--dimension', default=[10, 10, 10, 10], nargs='+', type=int, help='Provide dimension each side')
ap.add_argument('-s', '--seconds', default=None,
                help='Provide time in seconds of target video section showing the key points')
args = vars(ap.parse_args())


if args['reproject'] is None:
    print('Error parsing reproject argument')

elif args['reproject']:
    print('Doing warp transformation', '\nPlease select quadrat vertices')

    def click(event, x, y, flags, param):
        global quadrat_pts, position, posmouse

        if event == cv2.EVENT_LBUTTONDOWN:
            position = (x, y)
            quadrat_pts.append(position)
            # print(quadrat_pts)

        if event == cv2.EVENT_MOUSEMOVE:
            posmouse = (x, y)


    # Create list for holding quadrat position
    quadrat_pts = []
    # Set initial default position of  first point and mouse
    position = (0, 0)
    posmouse = (0, 0)

    vid = cv2.VideoCapture('video/' + args['video'])
    dim = args['dimension']
    print(dim)

    # Total length of video in frames
    length_vid = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid.get(cv2.CAP_PROP_FPS)

    if args['seconds'] is None:
        target_frame = 1
    else:
        target_frame = int(int(args['seconds']) * fps)

    # print('video length is ', length_vid, '\nFPS is', fps,  '\ntarget frame is ', target_frame)

    # # First argument is: cv2.cv.CV_CAP_PROP_POS_FRAMES
    vid.set(1, target_frame - 1)

    while vid.isOpened():
        ret, frame = vid.read()

        cv2.namedWindow('Select vertices quadrat')
        cv2.setMouseCallback('Select vertices quadrat', click)
        for i, val in enumerate(quadrat_pts):
            cv2.circle(frame, val, 3, (204, 204, 0), 2)
        cv2.putText(frame, "Mouse position {}".format(posmouse), (50, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (204, 204, 0), 2)

        cv2.imshow('Select vertices quadrat', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # print("Q - key pressed. Window quit by user")
            break

    vid.release()
    cv2.destroyAllWindows()
    print(quadrat_pts)
    points = np.array(quadrat_pts, np.int32)
    print('Warp calculations done')



else:
    print('Warp transformation not selected')


vid = cv2.VideoCapture('video/' + args['video'])
methods.data_writer(args['video'])

while vid.isOpened():
    ret, img = vid.read()
    img = cv2.resize(img, (640, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('test', gray)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

vid.release()
cv2.destroyAllWindows()
