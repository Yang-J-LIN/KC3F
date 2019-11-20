from cv2 import cv2 as cv
import numpy as np
from skimage import morphology

DEBUG = False


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
        img_gray, 170, 255, cv.THRESH_OTSU)

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
    roi = img[int(height / 3 * 2):int(height / 10 * 9),
              0:width]

    if DEBUG:
        cv.imwrite("ROI.jpg", roi)

    img_bin = binarize(roi)

    ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_bin_rev = cv.morphologyEx(255 - img_bin, cv.MORPH_OPEN, ele)
    img_bin_rev = cv.medianBlur(img_bin_rev, 11)

    skel = morphology.skeletonize(img_bin_rev//255).astype(np.uint8)*255

    img_bin_rev[skel == 255] = 120

    if DEBUG:
        cv.imwrite("img_bin_rev.jpg", img_bin_rev)
        cv.imshow("skel", skel)
        cv.waitKey(10000)

    return skel, img_bin_rev  # for test


def choose_target_point(skel):
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

    img_DEBUG = np.zeros((height, width, 3), dtype=np.uint8)

    img_DEBUG[:, :, 0] = skel
    img_DEBUG[:, :, 1] = img_points

    if DEBUG:
        cv.imwrite("img_DEBUG.jpg", img_DEBUG)

    # cv.waitKey(200)

    for contour in contours:
        if contour.size == 2:
            discrete_points.append(np.squeeze(contour))
        else:
            pass

    discrete_points = sorted(discrete_points,
                             key=lambda x: (x[0] - width // 2)**2 +
                                           (x[1] - height) ** 2)

    if len(discrete_points) != 0:
        return discrete_points[0], width, height, img_DEBUG
    else:
        return [371, 0], width, height, img_DEBUG
