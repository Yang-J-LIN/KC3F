from cv2 import cv2 as cv
import numpy as np


def binarize(img):
    """Binarize a grayscale image.

    Binarize the input grayscale image by ostu threshold method.

    Args:
        img: an image. Grayscale image is preffered.

    Returns:
       img_binary: the binarized image
    """

    # Make sure that img_gray is a grayscale image.
    if len(img.shape) == 2:
        img_gray = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        print("Converting image failed:", img.shape)
        return None
    # Apply the threshold method. It can be improved by changing the arguments.
    _, img_binary = cv.threshold(
        img_gray, 150, 255, cv.THRESH_BINARY, cv.THRESH_OTSU)

    return img_binary


def image_process(img):
    """ Binarizes and skeletonizes the image.

    Args:
        img
    
    Returns:
        target_point
    """
    width = img.shape[1]
    height = img.shape[0]
    roi = img[int(height / 3 * 2):height, int(width / 5):int(width * 4 / 5)]
    img_bin = binarize(roi)

    ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_bin_rev = cv.morphologyEx(255 - img_bin, cv.MORPH_OPEN, ele)

    skel = cv.ximgproc.thinning(img_bin_rev)

    return skel  # for test
