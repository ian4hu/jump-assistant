import numpy as np
import cv2

img = cv2.imread('e.png')

rows, cols = img.shape[:2]
dst = img.copy()
a=1.2
b=-20
pixels = np.reshape(dst, rows*cols*3)
#pixels = pixels * a + b
pixels[pixels * a + b > 255] = 255
pixels[pixels * a + b < 0] = 0
dst = np.reshape(pixels, [rows, cols, 3])
#for i in range(rows):
#    for j in range(cols):
#        for c in range(3):
#            color=img[i,j][c]*a+b
#            if color>255:
#                dst[i,j][c]=255
#            elif color<0:
#                dst[i,j][c]=0
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',dst)
cv2.resizeWindow('dst', 600, 800)
cv2.waitKey()