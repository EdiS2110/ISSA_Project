from random import random

import cv2
import numpy as np
from cv2 import cvtColor
from six import binary_type

cam = cv2.VideoCapture('Lane Detection Test Video 01 1.mp4')

width = int(1920/3)
height = int(1080/4)

left_top = 0
right_top = 0
left_bottom = 0
right_bottom = 0

while True:
    ret, frame = cam.read()


    if ret is False:
        break
#2) Shrink the frame
    frame =  cv2.resize(frame, (640, 270))
    main_frame = np.copy(frame)


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
    #             frame[rows,cols] = 0
    #         else:
    #             frame[rows,cols] = 255

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

    _, frame = cv2.threshold(frame,  80, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', frame)

#9)  coordinates of street markings on each side of the road

    # frameCopy = np.copy(frame)


    frame[:,0:int(width * 0.05)] = 0
    frame[:,width - int(width * 0.05):] = 0

    cv2.imshow("Blackout first and last 5% of cols", frame)

    whiteDotsCoordsLeft = np.argwhere(frame[:,0:int(width/2)]) # return coords in form of (y,x)
    whiteDotsCoordsRight = np.argwhere(frame[:,int(width/2)+1:])

    x_coords_left = whiteDotsCoordsLeft[:, 1]
    y_coords_left = whiteDotsCoordsLeft[:, 0]

    x_coords_right = int(width/2) + whiteDotsCoordsRight[:, 1] # + width/2 because argwhere treats frameCopy[:,int(width/2)+1:] from index 0
    y_coords_right = whiteDotsCoordsRight[:, 0]

#10) Find the lines that detect the edges of the lane

    right_side = np.polynomial.polynomial.polyfit(x_coords_right, y_coords_right, deg=1)
    left_side = np.polynomial.polynomial.polyfit(x_coords_left, y_coords_left, deg=1)

    left_top_y = 0
    left_top_x = (left_top_y - left_side[0])/left_side[1]

    left_bottom_y = height-1
    left_bottom_x = (left_bottom_y - left_side[0])/left_side[1]

    right_top_y = 0
    right_top_x =  (right_top_y - right_side[0])/right_side[1]

    right_bottom_y = height-1
    right_bottom_x = (right_bottom_y - right_side[0])/right_side[1]

    if -(10**8) <= left_top_x <= 10**8:
        if -(10**8) <= left_bottom_x <= 10**8:
            if -(10**8) <= right_top_x <= 10**8:
                if -(10**8) <= right_bottom_x <= 10**8:
                    left_top = (int(left_top_x), int(left_top_y))
                    right_top = (int(right_top_x), int(right_top_y))
                    left_bottom = (int(left_bottom_x), int(left_bottom_y))
                    right_bottom = (int(right_bottom_x), int(right_bottom_y))

    frame = cv2.line(frame, left_top, left_bottom, (200, 0, 0), 5)
    frame = cv2.line(frame, right_top, right_bottom, (100, 0, 0), 5)


    cv2.imshow("Draw lines", frame)


#11) Final visualization

    empty_frame_left = np.zeros((270, 640), dtype=np.uint8)

    empty_frame_left = cv2.line(empty_frame_left, left_top, left_bottom, (255, 0, 0), 3)

    stretch_matrix = cv2.getPerspectiveTransform(screenPoints, trapezoidPoints)

    empty_frame_left = cv2.warpPerspective(empty_frame_left, stretch_matrix, dsize=(640, 270))

    left = np.argwhere(empty_frame_left)

    # cv2.imshow("Final visualization LEFT", empty_frame_left)

    empty_frame_right = np.zeros((270, 640), dtype=np.uint8)

    empty_frame_right = cv2.line(empty_frame_right, right_top, right_bottom, (255, 0, 0), 3)

    # stretch_matrix = cv2.getPerspectiveTransform(screenPoints, trapezoidPoints)

    empty_frame_right = cv2.warpPerspective(empty_frame_right, stretch_matrix, dsize=(640, 270))

    right = np.argwhere(empty_frame_right)

    # cv2.imshow("Final visualization RIGHT", empty_frame_right)

    main_frame[left[:,0], left[:,1]] = (50, 50, 250)
    main_frame[right[:, 0], right[:, 1]] = (50, 250, 50)

    cv2.imshow("Final visualization COLOR", main_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cam.release()
cv2.destroyAllWindows()