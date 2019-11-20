# This is the main code to run the pi car.

import time

import numpy as np
from cv2 import cv2 as cv

import driver

import image_processing
import camera_capturer
import line_detection
import parking_control
import utils

DEBUG = False

PERIOD = 0  # the period of image caption, processing and sending signal

OFFSET = 371

LOT_LENGTH = 52
LOT_WIDTH = 26


class lotOutOfRangeError(Exception):
    pass


def coordinate_transform(result):
    line_1_slope = result[1][0]
    line_1_interception = result[1][1][1]

    y = 122.4 - 173.3*line_1_slope - 0.8235*line_1_interception

    line_3_slope = result[3][0]
    line_3_interception = result[3][1][1]

    x = 142.3 + 664.6*line_3_slope + 1.565*line_3_interception + \
        630.9*(line_3_slope**2) + 1.904*line_3_slope*line_3_interception

    return x, y


def get_frame():
    """ Get the frame from rear camera.

    To catch an image from the rear camera. It lacks of real time performance,
    and the VideoCapture instance will be released at once, which means the
    images catched from the camera will not buffer.

    Args:
        None

    Returns:
        img: the image catched from the camera, having a delay about 0.1
        seconds.
    """
    # cap = cv.VideoCapture(0 + cv.CAP_DSHOW)
    cap = cv.VideoCapture(0)
    img = None
    for i in range(11):
        _, img = cap.read()
        time.sleep(0.03)
    cap.release()
    print(img)
    return img


def parking():
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
    # cap = camera_capturer.CameraCapturer("rear")
    # d = driver.driver()
    # d.setStatus(motor=0.2, servo=0, mode="speed")

    # parking_state
    # 0: initial state
    # 1: on parking
    # 2: waiting for another parking instruction
    parking_state = 0
    target_lot = 3
    current_lot = 3
    # cap = camera_capturer.CameraCapturer("rear")
    # d = driver.driver()
    # d.setStatus(motor=0.4, servo=0, mode="speed")

    target_x = 0
    target_y = 0

    while True:
        if parking_state == 0:
            print("------------------------------------------------------")
            print("Which lot would you like to park? :-)")
            try:
                target_lot = int(input())
                if target_lot == -1:
                    break
                elif target_lot < 0 or target_lot > 4:
                    raise lotOutOfRangeError(
                        "The lot number must be 1, 2, 3 or 4."
                    )
            except:
                print(
                    "Invalid input! The lot number must be 1, 2, 3 or 4."
                )
                continue
            parking_state = 1
        if parking_state == 1:
            print("Start parking...")
            print("Target lot: %d" % (target_lot))
            # Start
            # img = cap.get_frame()

            img = get_frame()

            print("New image catched!")
            width = img.shape[1]
            height = img.shape[0]
            roi = img[int(height*2/5):height, 0:int(width//2), :]
            lines, linePoints = line_detection.line_detection(roi)
            result = line_detection.digit_detection(roi, lines, linePoints)
            result = result[1:]
            print("result", result)

            if result[0] in [1, 2, 3, 4]:
                current_lot = result[0]
                delta = current_lot - target_lot
                if delta == 0:
                    pass
                else:
                    parking_control.gostraight(delta*LOT_LENGTH)
                    img = get_frame()
                    print("New image catched!")
                    width = img.shape[1]
                    height = img.shape[0]
                    roi = img[int(height*2/5):height, 0:int(width//2), :]
                    lines, linePoints = line_detection.line_detection(roi)
                    result = line_detection.digit_detection(roi,
                                                            lines,
                                                            linePoints)
                    result = result[1:]
                    print("result", result)
            else:
                current_lot = 2
                delta = current_lot - target_lot
                if delta == 0:
                    pass
                else:
                    parking_control.gostraight(delta*LOT_LENGTH)
                    img = get_frame()
                    print("New image catched!")
                    width = img.shape[1]
                    height = img.shape[0]
                    roi = img[int(height*2/5):height, 0:int(width//2), :]
                    lines, linePoints = line_detection.line_detection(roi)
                    result = line_detection.digit_detection(roi,
                                                            lines,
                                                            linePoints)
                    result = result[1:]
                    print("result", result)

            x, y = coordinate_transform(result)
            print("x", x, "y", y)
            x = utils.constrain(x, 40, 20)
            y = utils.constrain(y, 82, 70)
            print("x", x, "y", y)
            target_x = x + LOT_WIDTH//2 + 5
            target_y = y - LOT_LENGTH//2 + 5
            print("target", target_x, target_y)

            parking_control.park(target_x, target_y, 0)
            parking_control.gostraight(-5)
            current_lot = target_lot
            print("Parking finished!")
            parking_state = 2
        if parking_state == 2:
            print("------------------------------------------------------")
            print("Which lot would you like to park? :-)")
            try:
                target_lot = int(input())
                if target_lot == -1:
                    break
                elif target_lot < 0 or target_lot > 4:
                    raise lotOutOfRangeError(
                        "The lot number must be 1, 2, 3 or 4."
                    )
            except:
                print(
                    "Invalid input! The lot number must be 1, 2, 3 or 4."
                )
                continue

            if current_lot == target_lot:
                print("Already in lot %d" % (current_lot))
            else:
                parking_control.gostraight(5)
                parking_control.reverse(target_x, target_y)
                # delta = current_lot - target_lot
                # print("distance", delta*LOT_LENGTH)
                # parking_control.gostraight(delta*LOT_LENGTH)
                parking_state = 1

if __name__ == "__main__":
    parking()
