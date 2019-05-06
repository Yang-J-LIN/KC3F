# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import driver

import image_processing
import camera_capturer


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
    cap = camera_capturer.CameraCapturer("front")
    d = driver.driver()

    while True:
        start_time = time.time()
        frame = cap.get_frame()
        skel = image_processing.image_process(frame)
        end_time = time.time()
        print("Time: %fs" % (end_time - start_time))
        time.sleep(1)


if __name__ == "__main__":
    cruise()
