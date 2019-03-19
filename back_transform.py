import cv2
import numpy as np
import argparse

import methods

capture_vertices = True
capturing = True


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="Capture.PNG", help="Provide path to video file")
args = vars(ap.parse_args())

# pts = np.array([(384, 380), (586, 501), (788, 384), (586, 261)], dtype=np.float32)
while capturing is True:
    frame = cv2.imread("video/" + args["image"])
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    methods.enable_point_capture(capture_vertices)
    frame = methods.draw_points_mousepos(frame, methods.quadrat_pts, methods.posmouse)
    cv2.imshow("Vertices selection", frame)

    if len(methods.quadrat_pts) == 4:
        print("Vertices were captured")
        break


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # print("Q - key pressed. Window quit by user")
        break

cv2.destroyAllWindows()
#
M, side, vertices_draw, IM = methods.calc_proj(methods.quadrat_pts)
frame = cv2.imread("video/" + args["image"])
frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
frame = methods.draw_quadrat(frame, vertices_draw)
warped = cv2.warpPerspective(frame, M, (side, side))
cv2.circle(warped, (60,60), 3, (100,200,100), 2)

back_to_orig = cv2.perspectiveTransform(np.float32([[[60,60]]]), IM)
print(back_to_orig)
print(back_to_orig[0][0][0])
print(back_to_orig[0][0][1])
new = cv2.circle(frame, (back_to_orig[0][0][0], back_to_orig[0][0][1]), 3, (150,0,150), 2)

cv2.imshow("original image", frame)
cv2.imshow("warped image", warped)
cv2.imshow("Back to original", new)
cv2.waitKey(0)
cv2.destroyAllWindows()
