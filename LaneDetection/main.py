from random import random

import cv2
import numpy as np
from cv2 import cvtColor

cam = cv2.VideoCapture('Lane Detection Test Video 01 1.mp4')

width = 1920/3
height = 1080/4

while True:
    ret, frame = cam.read()

    if ret is False:
        break

    frame =  cv2.resize(frame, (640, 270))


    frame = cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

cam.release()
cv2.destroyAllWindows()