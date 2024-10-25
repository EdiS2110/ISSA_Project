from random import random

import cv2
import numpy as np
from cv2 import cvtColor
from six import binary_type

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

    cv2.imshow('Original', frame)

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

    cv2.imshow('Cropped', frame)

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

    cv2.imshow('Top down view', frame)


#6) Blur

    frame = cv2.blur(frame, ksize=(5,5))

    cv2.imshow('Blurred', frame)

#7) Edge detection

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_vertical)

    frame = np.float32(frame)

    filter_matrix_vertical = cv2.filter2D(frame, -1, kernel=sobel_vertical)
    filter_matrix_horizontal = cv2.filter2D(frame, -1, kernel=sobel_horizontal)

    # filter_matrix_horizontal = cv2.filter2D(sobel_horizontal, -1, frame )

    final_matrix = np.sqrt((filter_matrix_vertical ** 2) + (filter_matrix_horizontal ** 2))

    frame = cv2.convertScaleAbs(final_matrix)

    cv2.imshow('Edge detection', frame)

#8) Binarize the frames

    # for rows in range(0,frame.shape[0]):
    #     for cols in range(0,frame.shape[1]):
    #         if(frame[rows][cols] < int(255/2)):
    #             frame[rows][cols] = 0
    #         else:
    #             frame[rows][cols] = 255

    def binarize(n):
        threshold = 80
        if n > threshold:
            return 255
        else:
            return 0

    vectorized_binarize = np.vectorize(binarize)

    # frame = vectorized_binarize(frame)
    #
    # frame = np.uint8(frame)

    _, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # AFISEAZA TOATE FERESTRELE UNA LANGA ALTA

cam.release()
cv2.destroyAllWindows()