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


def choose_target_point(skel):
    """ Selects a target poitn from skeleton for pure pursuit.

    Args:
        skel: skeleton of trajectory.

    Returns:
        target_point: target point for pure pursuit.

    """
    width = skel.shape[1]
    height = skel.shape[0]

    img = np.zeros((height, width), dtype=np.uint8)

    ellipse = cv.ellipse(img,
                         center=(width // 2, height),
                         axes=(width // 3, height // 2),
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

    for contour in contours:
        if contour.size == 2:
            discrete_points.append(np.squeeze(contour))
        else:
            pass

    discrete_points = sorted(discrete_points,
                             key=lambda x: (x[0] - width // 2)**2 +
                                           (x[1] - height) ** 2)

    if len(discrete_points) != 0:
        return discrete_points[0]
    else:
        return [0, 0]
