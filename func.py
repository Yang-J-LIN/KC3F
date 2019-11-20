import cv2 as cv
import numpy as np
from skimage import morphology,measure

# Binarize image
# img = cv.imread("D:/KC3F/test.JPG")
# img_gray = cv.imread("D:/KC3F/test2.jpg",0)
# img_gray = cv.resize(img_gray,(1080,604))
# _, img_bin = cv.threshold(img_gray, 170, 255, cv.THRESH_OTSU)
# img_bin_rev = cv.medianBlur(255 - img_bin, 5)

#cv.imshow("origin",img_bin_rev)

def getCentriod(img_bin_rev, DEBUG = False):
## 寻找轮廓 只会返回完整的车位！
    temp_CCOMP = np.zeros_like(img_bin_rev)
    temp_EXTERNAL = np.zeros_like(img_bin_rev)
    target = np.zeros_like(img_bin_rev)
    _,contours_CCOMP, _ = cv.findContours(img_bin_rev, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    _,contours_EXTERNAL, _ = cv.findContours(img_bin_rev, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(temp_CCOMP, contours_CCOMP,-1,255,3)
    cv.drawContours(temp_EXTERNAL, contours_EXTERNAL,-1,255,3)
    contours_xor = cv.bitwise_xor(temp_CCOMP, temp_EXTERNAL) # 去除最外的一圈 内轮廓
    _,contours, _ = cv.findContours(contours_xor, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # 四个车位的外轮廓
    cv.drawContours(target, contours, -1, 255, 3)
    
    if(DEBUG):
        cv.imshow('temp_CCOMP', temp_CCOMP)
        cv.imshow('temp_EXTERNAL', temp_EXTERNAL)
        cv.imshow('contours_xor', contours_xor) 
        #cv.imshow('target',target) # 去除最外的一圈 内轮廓
        cv.waitKey(0)
    
    # 连通域
    centroid = []
    target_label, num = measure.label(target, connectivity = 2, return_num = True)
    props = measure.regionprops(target_label)
    for i in range(num):
        centroid += [props[i].centroid]
        if(DEBUG): print(props[i].centroid)
    return centroid
    

if __name__ == "__main__":
    centroid = getCentriod(img_bin_rev)




'''
    # 填充
    #target_fill = target.copy()
    #area = []
    #for i in range(len(contours)):
    #    cv.fillConvexPoly(target_fill, contours[i], 255)
    #cv.imshow('target_fill',target_fill) 
    # 角点检测
    #dst = cv.cornerHarris(target,5,7,0.04)
    #dst = cv.dilate(dst,None)
    #target[dst > 0.01 * dst.max()] = 255 
    #cv.imshow('dst',target)
    
### 骨架
#skel = morphology.skeletonize(img_bin_rev//255).astype(np.uint8)*255
## 霍夫变换
lines = cv.HoughLinesP(img_edge, 1, np.pi / 180, 80,
                        minLineLength=50, maxLineGap=50)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(temp, (x1, y1), (x2, y2), 120, 1, lineType=cv.LINE_AA)
cv.imshow("skel",temp)
#cv.imshow("test",skel)

## 边缘检测Canny
img_edge = cv.Canny(img_bin_rev,50,100)
cv.imshow("edge",img_edge)
skel = np.float32(img_edge)


cv.waitKey(0)
'''