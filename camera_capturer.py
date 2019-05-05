# This module defines the CameraCapturer class and its methods

import logging

import numpy as np
from cv2 import cv2 as cv


class CameraCapturer(object):
    """ A class to acquire images from the camera.

    Attributes:
        cap: the VideoCapture instance of opencv
        width: the width of image captured
        height: the height of image captured
    """

    cap = None
    width = 0
    height = 0

    def __init__(self, camera):
        """ Initialize the instance by the selection of the camera.

        Args:
            camera: "front" standing for front camera or "rear" standing for
            rear camera.
        """
        if camera == "front":
            self.cap = cv.VideoCapture(1)
        elif camera == "rear":
            self.cap = cv.VideoCapture(0)
        else:
            self.cap = None

        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        """ Release the VideoCapture instance when deleted.

        Args:
            None

        Returns:
            None
        """
        if self.cap is not None:
            self.cap.release()
        else:
            pass

    def get_frame(self):
        """ Catch the frame captured by the camera.

        Args:
            None

        Returns:
            frame: the frame captured by the camera
        """
        _, frame = self.cap.read()
        return frame
