# Used for image processing test

from cv2 import cv2 as cv
import numpy as np

import image_processing

if __name__ == "__main__":
    img = cv.imread("image/binarize2.jpg", 0)
    width = img.shape[1]
    height = img.shape[0]
    roi = img[int(height / 3 * 2):height, int(width / 5):int(width * 4 / 5)]
    img_bin = image_processing.binarize(roi)

    ele = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_bin_rev = cv.morphologyEx(255 - img_bin, cv.MORPH_OPEN, ele)

    skel = cv.ximgproc.thinning(img_bin_rev)
    cv.imshow("img_bin_rev", img_bin_rev)
    cv.imshow("img_skel", skel)
    cv.waitKey()
