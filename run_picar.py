# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import driver

import image_processing
import camera_capturer
import utils

DEBUG = True

PERIOD = 0.5  # the period of image caption, processing and sending signal

OFFSET = 371


def cruise_control(bias, k_p=1, k_i=0, k_d=1):
    """ Controls the picar on the mode of cruise

    """
    return 0


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

    target = OFFSET - int(cap.width / 5)

    # Parameters of PID controller
    kp = 3
    ki = 0
    kd = 0

    # Initialize error to 0 for PID controller
    error_i = 0
    error = 0

    while True:
        this_time = time.time()
        if this_time - last_time > PERIOD:  # Execute the code below every
                                            # PERIOD time
            last_time = this_time
            # --------------------------------------------------------------- #
            #                       Start your code here                      #
            
            # Image processing. Outputs a target_point.
            frame = cap.get_frame()
            start = time.time()
            skel, _ = image_processing.image_process(frame)
            target_point, width, _ = image_processing.choose_target_point(skel)
            end = time.time()
            print("Time required for image processing:", end - start)

            # Picar control

            # If there is no target point found, set servo to 0; otherwise, set
            # servo to the uniformed bias.
            if target_point[0] == 0:
                servo = 0
                d.setStatus(servo=0)
                pass
            else:
                # Update the PID error
                error_p = (target_point[0] - target)/width
                error_i += error_p
                error_d = error_p - error
                error = error_p

                # PID controller
                servo = utils.constrain(- kp*error_p
                	                    - ki*error_i
                	                    - kd*error_d,
                	                    1, -1)

                d.setStatus(servo=servo)

            print(servo, error_p, error_i, error_d)

            if DEBUG:
                cv.imshow("win", frame)
                cv.waitKey(300)

            # --------------------------------------------------------------- #
        else:
            time.sleep(0.01)


if __name__ == "__main__":
    cruise()
