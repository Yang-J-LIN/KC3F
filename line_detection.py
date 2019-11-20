
# Used for image processing test

import cv2 as cv
import numpy as np
import camera_capturer
import image_processing
from sklearn.cluster import KMeans
import time
import pytesseract
from skimage import morphology, measure


def point_offset(point, flag, offset):
    """ Adjust the critical points in digit detection.

    Args:
        point:
        flag:
        offset:

    Returns:
        point
    """
    # 数字识别时取ROI的调整
    if('L' in flag):  # 左下
        point[0] -= offset
    if('R' in flag):
        point[0] += offset
    if('U' in flag):  # 上
        point[1] -= offset
    if('D' in flag):  # 下
        point[1] += offset
    point = tuple(point)
    return point


def line_detection(image, DEBUG=False):
    """ Detects the edge lines of the lot. HoG is utilized to detect straight
    lines. And K-means is used to cluster analysis to distinguish lines
    representing different edges.

    Args:
        image: image to be detected
        DEBUG: indicator of debug

    Returns:
        lines: line equations drawn from HoG and cluster analysis
        linePoints: points of the lines
    """
    imageDEBUG = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # gray = cv.GaussianBlur(gray,(3,3),0)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    edges[0:30, :] = 0
    edges[0:1, :] = 255
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    data = []
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   # theta是弧度
        b = np.sin(theta)
        x0 = a * rho    # 代表x = r * cos（theta）
        y0 = b * rho    # 代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标
        cv.line(imageDEBUG, (x1, y1), (x2, y2), (0, 0, 255), 2)
        data.append([theta, x0, y0])
    data = np.array(data)
    # 聚类
    class_num = 3
    slope_data = data[:, 0]
    slope_data = slope_data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=class_num, random_state=10).fit(slope_data)
    kmeans_idx = kmeans.labels_
    target = kmeans.cluster_centers_
    if(max(target) < 1.4):
        print('[warning]max target:', max(target))

    # 区分聚类获得的三条直线
    idx_hori = np.argmax(target)  # 离车最近的线
    idx_near = np.argmin(target)
    idx_far = list(set(range(class_num))-set([idx_near])-set([idx_hori]))[0]
    lines = []
    for i in [idx_hori, idx_far, idx_near]:
        temp = data[kmeans_idx == i]
        if(i == idx_hori):  # 水平
            idx = np.argmax(temp[:, 2])
            if(temp[idx][-1] < 25):
                print('[warning]hori offset', temp[idx]
                      [-1], '-> 50')  # 如果是0代表没有检测到横线
                temp[idx][-1] = 50
            elif(temp[idx][-1] > 100):
                print('[warning]hori offset', temp[idx]
                      [-1], '-> 50')  # 如果是0代表没有检测到横线
                temp[idx][-1] = 50
            lines.append(temp[idx])
        elif(i == idx_near):  # 近的线
            idx = np.argmin(abs(temp[:, 0]-target[i]))
            lines.append(temp[idx])
        elif(i == idx_far):
            idx = np.argmin(abs(temp[:, 0]-target[i]))
            lines.append(temp[idx])

    linePoints = []
    for i in lines:
        theta = i[0]
        x0 = i[1]
        y0 = i[2]
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标
        linePoints.append([x1, y1, x2, y2])
        cv.line(imageDEBUG, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv.imwrite('edges.jpg', edges)
    cv.imwrite('lineDetection.jpg', imageDEBUG)
    if(DEBUG):
        cv.imshow("edges", edges)
        cv.imshow("image-lines", imageDEBUG)
        cv.waitKey(0)

    return lines, linePoints


def digit_detection(image, lines, linePoints, DEBUG=False):
    ''' Detected the digit in the image with tesseract-ocr.

    Args:
        image: the roi image to be detected
        lines: line equations drawn from HoG and cluster analysis
        linePoints: points of the lines
        DEBUG: indication of debug

    Returns:
        Reuslt: a list contains line equations and detected digit
                result[0]是一个len=3的列表，表中是Kmeans聚类后最接近的三条线，
                按水平、远、近排列。用极角,x,y表示。
                result[1]是对应车库数字识别结果，返回的是数字，若识别不出则返回
                -1。
                result[2]是水平线的性质，是一个len=2的列表，第一项是斜率，第二
                项是与左边框的交点(0,y)
                result[3]是远线的性质，是一个len=2的列表，第一项是斜率，第二项
                是与左边框的交点(0,y)
                result[4]是近线的性质，是一个len=2的列表，第一项是斜率，第二项
                是与左边框的交点(0,y)

    Exampel of result:
        [[array([  1.2915436,  29.217564 , 101.89374  ], dtype=float32),
        array([  1.134464,  60.857025, 130.50833 ], dtype=float32),
        array([  0.80285144, 152.13019   , 157.53542   ], dtype=float32)],
        4,
        [-0.2861602497398543, (0, 110)],
        [-0.4663355408388521, (0, 158)],
        [-0.9659248956884562, (0, 304)]]
    '''
    digit = -1
    imageDEBUG = image.copy()
    # 求出直线性质
    para_hori = getLineParameters(linePoints[0])
    para_far = getLineParameters(linePoints[1])
    para_near = getLineParameters(linePoints[2])
    # 求出交点
    targetPoint1 = cross_point(linePoints[0], linePoints[2])  # 近的
    targetPoint2 = cross_point(linePoints[0], linePoints[1])  # 远的
    fringePoint1 = cross_point(linePoints[2], [0, 0, 0, 1])
    fringePoint2 = cross_point(linePoints[1], [0, 0, 0, 1])
    cv.circle(imageDEBUG, tuple(targetPoint1), 5, (255, 0, 0), 5)
    cv.circle(imageDEBUG, tuple(targetPoint2), 5, (255, 0, 0), 5)
    cv.circle(imageDEBUG, tuple(fringePoint1), 5, (0, 255, 0), 5)
    cv.circle(imageDEBUG, tuple(fringePoint2), 5, (0, 255, 0), 5)
    targetPoint1 = point_offset(targetPoint1, 'LD', 5)  # 右上的点
    targetPoint2 = point_offset(targetPoint2, 'RD', 5)  # 左上的点
    fringePoint1 = point_offset(fringePoint1, 'U', 20)  # 右下的点
    fringePoint2 = point_offset(fringePoint2, 'D', 20)  # 左下的点
    cv.circle(imageDEBUG, targetPoint1, 5, (128, 0, 0), 5)
    cv.circle(imageDEBUG, targetPoint2, 5, (128, 0, 0), 5)
    cv.circle(imageDEBUG, fringePoint1, 5, (0, 128, 0), 5)
    cv.circle(imageDEBUG, fringePoint2, 5, (0, 128, 0), 5)

    # 求出该区域
    mask = np.zeros_like(image)
    pts = np.array([targetPoint1, targetPoint2, fringePoint2, fringePoint1])
    mask = cv.polylines(mask, [pts], True, (255, 255, 255), 2)
    mask2 = cv.fillPoly(mask.copy(), [pts], (255, 255, 255))
    ROI = cv.bitwise_and(mask2, image)
    # 四点透视变换
    PerspectMatrix = cv.getPerspectiveTransform(np.float32(
        [targetPoint2, targetPoint1, fringePoint2, fringePoint1]),
        np.float32([(0, 0), (80, 0), (0, 80), (80, 80)]))
    PerspectImg = cv.warpPerspective(
        ROI, PerspectMatrix, (image.shape[1], image.shape[0]))
    PerspectImg = image_processing.binarize(PerspectImg)
    PerspectImg_rev = 255 - PerspectImg
    PerspectImg_rev = cv.dilate(
        PerspectImg_rev, np.ones((3, 3)))  # 使数字变粗 识别效果变好
    target_label, num = measure.label(
        PerspectImg_rev, connectivity=2, return_num=True)
    props = measure.regionprops(target_label)
    temp = []
    for i in range(num):
        if(props[i].area > 100):
            temp.append(props[i].area)
        else:
            print('[warning]digit area too small', props[i].area)
            temp.append(float('inf'))
    bbox = props[np.argmin(temp)].bbox
    PerspectImg_ROI = 255 - \
        PerspectImg_rev[max(0, bbox[0]-15):bbox[2]+15,
                        max(bbox[1]-15, 0):bbox[3]+15]
    # 三点仿射
    AffineMatrix = cv.getAffineTransform(np.float32(
        [targetPoint2, targetPoint1, fringePoint2]),
         np.float32([(0, 0), (80, 0), (0, 80)]))
    AffineImg = cv.warpAffine(
        ROI, AffineMatrix, (image.shape[1], image.shape[0]))
    AffineImg = image_processing.binarize(AffineImg)
    AffineImg_rev = 255 - AffineImg
    AffineImg_rev = cv.dilate(AffineImg_rev, np.ones((3, 3)))
    target_label, num = measure.label(
        AffineImg_rev, connectivity=2, return_num=True)
    props = measure.regionprops(target_label)
    temp = []
    for i in range(num):
        if(props[i].area > 100):
            temp.append(props[i].area)
        else:
            print('[warning]digit area too small', props[i].area)
            temp.append(float('inf'))
    bbox = props[np.argmin(temp)].bbox
    AffineImg_ROI = 255 - \
        AffineImg_rev[max(0, bbox[0]-15):bbox[2]+15,
                      max(bbox[1]-15, 0):bbox[3]+15]
    if(DEBUG):
        for i in lines:
            theta = i[0]
            x0 = i[1]
            y0 = i[2]
            a = np.cos(theta)  # theta是弧度
            b = np.sin(theta)
            x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
            y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
            x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
            y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标
            linePoints.append([x1, y1, x2, y2])
            cv.line(imageDEBUG, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv.imshow('digit_detection_DEBUG', imageDEBUG)
#        cv.imshow("digit_detection_ROI",ROI)
        cv.imshow("After warpAffine", AffineImg)
        cv.imshow("warpAffine ROI", AffineImg_ROI)
        cv.imshow("After warpPerspect", PerspectImg)
        cv.imshow("warpPerspect ROI", PerspectImg_ROI)
        cv.waitKey()
    cv.imwrite('Affine_ROI.jpg', AffineImg_ROI)
    cv.imwrite('Perspect_ROI.jpg', PerspectImg_ROI)
    cv.imwrite('AffineImg.jpg', AffineImg)
    cv.imwrite('PerspectImg.jpg', PerspectImg)
    cv.imwrite('digit_detection_DEBUG.jpg', imageDEBUG)
    cv.imwrite('digit_detection_ROI.jpg', ROI)
    # 识别数字
    # tessedit_char_whitelist=1234'tessedit_char_whitelist="1234" digits
    digit = pytesseract.image_to_string(
        AffineImg_ROI, lang='eng', config='--psm 10')
    print('Affine', digit)
    # tessedit_char_whitelist=1234'tessedit_char_whitelist="1234" digits
    digit2 = pytesseract.image_to_string(
        AffineImg_ROI, lang='eng', config='--psm 10')
    print('Perspect', digit2)
    digit = digit_correct(digit)
    result = [lines, digit, para_hori, para_far, para_near]
    return result


def digit_correct(digit):
    """ Corrects the digit when OCR doesn't work well.

    Args:
        digit: result of OCR

    Returns:
        result: result of correction
    """
    result = -1
    if len(digit) == 1:
        if digit in ['4', 'A', 'q']:
            result = 4
        elif digit in ['3', 's', 'S', 'a']:
            result = 3
        elif digit in ['2']:
            result = 2
        elif digit in ['1', 'l', "\\"]:
            result = 1
        elif digit not in ['1', '2', '3', '4']:
            result = -1
    else:
        for d in digit:
            if d in ['4', 'A', 'q']:
                result = 4
            elif d in ['3', 's', 'S', 'a']:
                result = 3
            elif d in ['2']:
                result = 2
            elif d in ['1', 'l', "\\"]:
                result = 1
            elif d not in ['1', '2', '3', '4']:
                result = -1
            if result != -1:
                break
    return result


def cross_point(line1, line2):  # 计算交点函数
    """ Calculated the intersection of two line.

    Args:
        line1
        line2

    Returns:
        point = intersection of two lines
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    k1 = (y2-y1) / (x2-x1)
    b1 = y1 - x1*k1
    if x4 == x3:  # L2直线斜率不存在
        k2 = None
        b2 = 0
    else:
        k2 = (y4-y3)/(x4-x3)
        b2 = y3 - x3*k2
    if k2 is None:
        x = x3
    else:
        x = (b2-b1) / (k1-k2)
    y = k1*x + b1
    point = [int(x), int(y)]
    return point


def getLineParameters(line1):
    """ Calculates the slope and interception of lines.

    Args:
        line1

    Returns:
        equ: parameters for a slope-interception style lien equation
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = 0, 0, 0, 1
    k1 = (y2-y1) / (x2-x1)
    b1 = y1 - x1*k1
    if (x4 == x3):
        k2 = None
        b2 = 0
    else:
        k2 = (y4-y3)/(x4-x3)
        b2 = y3 - x3*k2
    if k2 is None:
        x = x3
    else:
        x = (b2-b1) / (k1-k2)
    y = k1*x + b1
    print('line', line1, k1)
    equ = [k1, (int(x), int(y))]
    return equ
