# Used for image processing test

from cv2 import cv2 as cv
import numpy as np
import camera_capturer

import image_processing

if __name__ == "__main__":
    cap = camera_capturer.CameraCapturer("rear")
    while True:
        img = cap.get_frame()
        width = img.shape[1]
        height = img.shape[0]
        roi = img[int(height / 3 * 2):int(height / 10 * 9),
                  int(width / 5):int(width * 4 / 5)]
        cv.imshow("roi", roi)
        cv.waitKey(1)

    cv.waitKey()
