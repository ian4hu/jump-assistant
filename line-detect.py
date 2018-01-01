# -*- coding: utf-8 -*-

import cv2
import numpy as np

#两个回调函数
def HoughLinesP(minLineLength):
    global minLINELENGTH
    minLINELENGTH = minLineLength + 1
    print "minLINELENGTH:",minLineLength + 1
    tempIamge = edges.copy()
    lines = cv2.HoughLinesP( edges, 1, np.pi/180, minLINELENGTH, 10 )
    print(lines.shape)
    for l in lines:
        for x1,y1,x2,y2 in l:
            x = abs(x1-x2)
            y = abs(y1-y2)
            if False and y == 0:
                continue
            if False and x == 0:
                continue
            if False and y != 0 and x != 0:
                r = np.arctan2(y, x) / np.pi * 180
                if r > 30.5 or r < 29.5:
                    continue
            #ls.append(l)
            cv2.line(tempIamge,(x1,y1),(x2,y2),255,3)
    cv2.imshow(window_name,tempIamge)

#临时变量
minLineLength = 88

#全局变量
minLINELENGTH = 88
max_value = 800
window_name = "HoughLines Demo"
trackbar_value = "minLineLength"

#读入图片，模式为灰度图，创建窗口
scr = cv2.imread("b.png")
gray = cv2.cvtColor(scr,cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(gray,(9,9),0)
#img = gray
edges = cv2.Canny(img, 6, 18, apertureSize = 3)
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 600, 800)

#创建滑动条
cv2.createTrackbar( trackbar_value, window_name, minLineLength, max_value, HoughLinesP)

#初始化
HoughLinesP(minLineLength)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()