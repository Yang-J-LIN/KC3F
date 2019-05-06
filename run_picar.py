# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import image_processing
import camera_capturer

if __name__ == "__main__":
    cap = camera_capturer.CameraCapturer("front")
    while(True):
        frame = cap.get_frame()
        cv.imshow("frame", frame)
        cv.waitKey(1000)
