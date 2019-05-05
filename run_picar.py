# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import utils
import camera_capturer

if __name__ == "__main__":
    cap = camera_capturer.CameraCapturer("front")
    while(True):
        frame = cap.get_frame()
        frame_rectified = utils.front_distortion_rectify(frame)
        cv.imshow("frame_rectified", frame_rectified)
        cv.waitKey(1000)