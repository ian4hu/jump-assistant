# coding=utf-8
import cv2
import numpy as np
import math


## 750 1200
pt = (273, 1264)

img = cv2.imread("e.png", 0)

canny = cv2.Canny(img, 15, 30, apertureSize=3)

pick = np.zeros(canny.shape, np.uint8)

leftpt = (0, pt[1] - int(pt[0] * math.tan(math.pi / 6)))
rightpt = (img.shape[0], pt[1] - int((img.shape[0] - pt[0]) * math.tan(math.pi / 6)))
cv2.line(pick, pt, leftpt, 255, 2)
cv2.line(pick, pt, rightpt, 255, 2)

cv2.line(pick, (pt[0], pt[1] + 30), (leftpt[0], leftpt[1]+30), 255, 2)
cv2.line(pick, (pt[0], pt[1]+30), (rightpt[0], rightpt[1]+30), 255, 2)
cv2.line(pick, (pt[0], pt[1] - 30), (leftpt[0], leftpt[1]-30), 255, 2)
cv2.line(pick, (pt[0], pt[1]-30), (rightpt[0], rightpt[1]-30), 255, 2)


cv2.imwrite('e-e.png', cv2.bitwise_and(pick, canny))
#cv2.imshow('Canny', canny)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#import cv2
#import numpy as np


#def CannyThreshold(lowThreshold):
#    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
#    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
#    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
#    cv2.imshow('canny demo', dst)


#lowThreshold = 0
#max_lowThreshold = 100
#ratio = 3
#kernel_size = 3

#img = cv2.imread('c.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.namedWindow('canny demo')

#cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

#CannyThreshold(0)  # initialization
#if cv2.waitKey(0) == 27:
#    cv2.destroyAllWindows()