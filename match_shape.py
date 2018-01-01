import cv2
import numpy as np
img1 = cv2.cvtColor(cv2.imread('e.png'), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('features/5-bottle-22.png'),cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
_,contours,hierarchy = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
_,contours,hierarchy = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]
ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print ret