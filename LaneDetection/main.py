from random import random

import cv2
import numpy as np
from cv2 import cvtColor

cam = cv2.VideoCapture('Lane Detection Test Video 01 1.mp4')

width = int(1920/3)
height = int(1080/4)

while True:
    ret, frame = cam.read()

    if ret is False:
        break
#2) Shrink the frame
    frame =  cv2.resize(frame, (640, 270))

#3) Grayscale
    new_frame = np.zeros((270,640), dtype = np.uint8)

    # for i in range(0, frame.shape[0]):
    #     for j in range(0, frame.shape[1]):
    #         B, G, R = frame[i][j]
    #         GRAYSCALE = R * 0.3 + G * 0.59 + B * 0.11
    #         new_frame[i][j] = GRAYSCALE

    frame = cvtColor(frame, cv2.COLOR_BGR2GRAY)

#4) Trapezoid

    trapezoid_frame = np.zeros((270, 640), dtype = np.uint8)

    upper_left = (int(width*0.45), int(height * 0.77))
    upper_right = (width - int(width*0.45), int(height * 0.77))
    lower_left = (0,height)
    lower_right = (width,height)

    trapezoidPoints = np.array([upper_right, upper_left, lower_left, lower_right], dtype = np.int32)

    cv2.fillConvexPoly(trapezoid_frame, trapezoidPoints, 1)
    # cv2.imshow('Trapezoid', black_frame * 255)

    frame = trapezoid_frame * frame

#5) Top-down view

    screen_upper_right = (width, 0)
    screen_upper_left = (0, 0)
    screen_lower_left = (0, height)
    screen_lower_right = (width, height)

    screenPoints = np.array([screen_upper_right, screen_upper_left, screen_lower_left, screen_lower_right],
                            dtype = np.float32)

    trapezoidPoints = np.float32(trapezoidPoints)

    stretch_matrix = cv2.getPerspectiveTransform(trapezoidPoints, screenPoints)

    frame = cv2.warpPerspective(frame, stretch_matrix, dsize=(640, 270))

    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()