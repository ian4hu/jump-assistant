import cv2
import numpy as np

filename = 'e.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3 ,0.06)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.00000001*dst.max()]=[0,0,255]

cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
cv2.imshow('dst',img)
cv2.resizeWindow("dst", 600, 800)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()