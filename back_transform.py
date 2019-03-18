import cv2

import methods

capture_vertices = True
capturing = True


while capturing is True:
    frame = cv2.imread("video/Capture.PNG")
    methods.enable_point_capture(capture_vertices)
    frame = methods.draw_points_mousepos(frame, methods.quadrat_pts, methods.posmouse, capture_vertices)
    cv2.imshow("Vertices selection", frame)

    if len(methods.quadrat_pts) == 4:
        print("Vertices were captured")
        break


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        # print("Q - key pressed. Window quit by user")
        break

cv2.destroyAllWindows()

M, side = methods.calc_proj(methods.quadrat_pts)
frame = cv2.imread("video/Capture.PNG")
warped = cv2.warpPerspective(frame, M, (side, side))
warped = cv2.resize(warped, None, fx=0.8, fy=0.8)
cv2.imshow("original image", frame)
cv2.imshow("warped image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
