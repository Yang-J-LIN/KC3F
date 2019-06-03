# Used for image processing test

from cv2 import cv2 as cv
import numpy as np
import camera_capturer
import func
import image_processing
from sklearn.cluster import KMeans
import time
import os

def find_nearest(target, data):
    #data n*3 第一列角度 第二列x 第三列y
    #target 3*1 是一列的矩阵 三个元素
#    results = []
#    for i in range(len(target)):
#        data = sorted(data, key=lambda x: np.abs(x[0] - target[i]))
#        results.append(data[0, :])
    results = []
    for i in range(len(target)):
        temp = data[:,0] - target[i]
        idx = np.argmin(abs(temp))
        results.append(data[idx])
    return results


def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)
    cv.waitKey(1)


def cluster_analysis(data, class_num=3):
    kmeans = KMeans(n_clusters=class_num, random_state=10).fit(data)
#    print(kmeans.cluster_centers_)
    return kmeans.cluster_centers_


def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    data = []
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标    
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        data.append([theta, x0, y0]) 
#        print(theta)
    data = np.array(data)
    slope_data = data[:, 0]
    slope_data = slope_data.reshape(-1, 1)
    kmeans = cluster_analysis(slope_data)

    results = find_nearest(kmeans, data)
    print(results)

    for i in results:
        theta = i[0]
        x0 = i[1]
        y0 = i[2]
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x1 = int(x0 + 1000 * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + 1000 * a)     # 计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - 1000 * a)     # 计算直线终点纵坐标    
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)         
    # return kmeans, data
    cv.imshow("image-lines", image)
    cv.waitKey(5000)
    # cv.waitKey()


if __name__ == "__main__":
    # cap = camera_capturer.CameraCapturer("rear")
    # img = cv.imread("hog.jpg")
    name = os.listdir("test_image/")
    # while True:
        # img = cap.get_frame()
        # cv.imwrite(str(time.time())[-10:] + ".jpg", img)
        # time.sleep(0.5)
        # break   
    for i in name:
        img = cv.imread("test_image/" + i)
        width = img.shape[1]
        height = img.shape[0]
        roi = img[int(height*2/6):height, 0:int(width//2), :]
        start = time.time()
        line_detection(roi)
        end = time.time()
        print(end - start)


        
       
        # break

        # # roi = img
        # roi_bin = image_processing.binarize(roi)
        # roi_bin_rev = 255 - roi_bin

        # img_edge = cv.Canny(roi_bin_rev,50,100)
        # cv.imshow("edge",img_edge)
        # skel = np.float32(img_edge)

        # temp = np.zeros_like(skel, dtype=np.uint8)

        # lines = cv.HoughLinesP(img_edge, 1, np.pi / 180, 80,
        #                 minLineLength=50, maxLineGap=50)
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(temp, (x1, y1), (x2, y2), 120, 1, lineType=cv.LINE_AA)
        # cv.imshow("skel",temp)
        # # out = func.getCentriod(roi_bin, True)

        # cv.imshow("roi_bin", roi_bin)
        # cv.waitKey(1)

    # cv.waitKey()
