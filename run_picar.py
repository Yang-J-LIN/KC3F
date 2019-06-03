# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import driver

import image_processing
import camera_capturer
import utils

DEBUG = False

PERIOD = 0  # the period of image caption, processing and sending signal

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
    cap = camera_capturer.CameraCapturer("front")
    d = driver.driver()
    d.setStatus(motor=0.2, servo=0, mode="speed")
    last_time = time.time()

    target = OFFSET #- int(cap.width / 5)

    # Parameters of PID controller
    kp = 1.2
    ki = 0
    kd = 0.1

    # Initialize error to 0 for PID controller
    error_p = 0
    error_i = 0
    error_d = 0
    error = 0
    
    servo = 0
    last_servo = 0
    last_angle = 0

    n = 0.2
    
    try:
        while True:
            this_time = time.time()
            if this_time - last_time > PERIOD:  # Execute the code below every
                                                # PERIOD time
                last_time = this_time


                # d.setStatus(motor=0, servo=n, mode="speed")
                # n = -n
                # continue

                # --------------------------------------------------------------- #
                #                       Start your code here                      #
                
                # Image processing. Outputs a target_point.
                frame = cap.get_frame()
                start = time.time()
                skel, img_bin_rev = image_processing.image_process(frame)

                white_rate = \
                    np.size(img_bin_rev[img_bin_rev == 255]) / img_bin_rev.size

                if white_rate > 0.3:
                    print("stay", white_rate)
                    d.setStatus(servo=last_servo)
                    continue

                target_point, width, _, img_DEBUG, angle = \
                    choose_target_point(skel, target)
                end = time.time()
                # print("Time required for image processing:", end - start)

                # Picar control

                # If there is no target point found, set servo to 0; otherwise, set
                # servo to the uniformed bias.

                # if target_point[0] == 0:
                #     servo = last_servo
                #     pass
                # else:
                #     # Update the PID error
                #     error_p = ((target_point[0] - target)/width)
                #     error_i += error_p
                #     error_d = error_p - error
                #     error = error_p

                #     # PID controller
                #     servo = utils.constrain(- kp*error_p
                #                             - ki*error_i
                #                             - kd*error_d,
                #                             1, -1)

                if angle == 0:
                    angle = last_angle
                    pass
                else:
                    # Update the PID error
                    error_p = angle #**(9/7)
                    error_i += error_p
                    error_d = error_p - error
                    error = error_p

                    # PID controller
                    servo = utils.constrain(- kp*error_p
                                            - ki*error_i
                                            - kd*error_d,
                                            1, -1)

                d.setStatus(servo=servo)
                last_servo = servo

                # print("servo: ", servo, "error_p: ", error_p)

                # img_DEBUG[:, target] = 255

                # if DEBUG:
                #     # cv.imshow("frame", frame)
                #     cv.imshow("img_bin_rev", img_bin_rev)
                #     cv.imshow("img_DEBUG", img_DEBUG)
                #     cv.waitKey(300)

                # --------------------------------------------------------------- #
            else:
                # time.sleep(0.01)
                pass
    except KeyboardInterrupt:
        d.setStatus(servo=0, motor=0)


def choose_target_point(skel, target):
    """ Selects a target poitn from skeleton for pure pursuit.

    Draws a ellipse and applies an and operation to the ellipse with the skel.
    Then returns a point that has least distance with the center of the
    ellipse.

    Args:
        skel: skeleton of trajectory.

    Returns:
        target_point: target point for pure pursuit.

    """
    width = skel.shape[1]
    height = skel.shape[0]

    img = np.zeros((height, width), dtype=np.uint8)
    
    ellipse_a = width // 2
    ellipse_b = height // 3
    ellipse = cv.ellipse(img,
                         center=(target, height),
                         axes=(ellipse_a, ellipse_b),
                         angle=0,
                         startAngle=180,
                         endAngle=360,
                         color=255,
                         thickness=1)

    img_points = np.bitwise_and(skel, ellipse)

    _, contours, _ = cv.findContours(img_points,
                                     mode=cv.RETR_EXTERNAL,
                                     method=cv.CHAIN_APPROX_NONE)

    discrete_points = []

    img_DEBUG = np.zeros((height, width, 3), dtype=np.uint8)

    img_DEBUG[:, :, 0] = skel
    img_DEBUG[:, :, 1] = img_points

    # cv.imshow("img_DEBUG", img_DEBUG)
    # cv.waitKey(200)

    for contour in contours:
        if contour.size == 2:
            discrete_points.append(np.squeeze(contour))
        else:
            pass

    # discrete_points = sorted(discrete_points,
    #                          key=lambda x: (x[0] - width // 2)**2 +
    #                                        (x[1] - height) ** 2)

    discrete_points = sorted(discrete_points,
                             key=lambda x: np.abs(x[0] - target))
    
    if len(discrete_points) != 0:
        px = discrete_points[0][0] - target
        angle = np.arctan2(px, ellipse_a)
        # angle = angle[0]
        print("angle:", angle)

        return discrete_points[0], width, height, img_DEBUG, angle
    else:
        return [0, 0], width, height, img_DEBUG, 0
        # return [target, 0], width, height, img_DEBUG


if __name__ == "__main__":
    cruise()
