# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import driver

import image_processing
import camera_capturer

DEBUG = False

PERIOD = 0.5  # the period of image caption, processing and sending signal


def cruise():
    """ Tracks the black line.

    Acquires images from front camera and uses it to do pure pursuit.
    Uses functions in driver.py to drive the pi car.

    There is a three-step process to reach the goal.
    Step 1.
        Employs CameraCapturer class to acquire images from front camera and
        rectify lens distortion.
    Step 2.
        Chooses the ROI and binarizes the it. Then uses morphology method to
        get the target point.
    Step 3.
        According to target point, applies pure pursuit algorithm and uses
        functions in driver.py to drive the car.

    Args:
        None

    Returns:
        None
    """

    # Initialize CameraCapturer and drive
    cap = camera_capturer.CameraCapturer("rear")
    d = driver.driver()
    d.setStatus(motor=0.1, mode="speed")
    last_time = time.time()
    while True:
        this_time = time.time()
        if this_time - last_time > PERIOD:
            last_time = this_time
            # --------------------------------------------------------------- #
            #                       Start your code here                      #
            frame = cap.get_frame()
            start = time.time()
            skel, _ = image_processing.image_process(frame)
            target_point, w, _ = image_processing.choose_target_point(skel)
            end = time.time()
            print("Time of image processing:", end - start)

            # If there is no target point found, set servo to 0; otherwise, set
            # servo to the uniformed bias.
            if target_point[0] == 0:
                d.setStatus(servo=0)
            else:
                # The code below is very inelegant. Remember to modify it.
                # 371 is the x value of actual middleline in frame.
                # int(cap.width / 5) is the edge cut off when extracting the
                # roi.
                # 5 is the scale factor.
                bias_uniformed = \
                    - 5*(target_point[0] - (371 - int(cap.width / 5)))/w
                d.setStatus(servo=bias_uniformed)
            print(- (target_point[0] - (371 - int(cap.width / 5)))/w)

            if DEBUG:
                cv.imshow("win", frame)
                cv.waitKey(300)

            # --------------------------------------------------------------- #
        else:
            time.sleep(0.01)


if __name__ == "__main__":
    cruise()
