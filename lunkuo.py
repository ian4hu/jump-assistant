import cv2
import numpy as np

img = cv2.imread('e.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
av = np.mean(gray)
fa = 60
fb = 240

k = 255.0 / (fb - fa)
w, h = gray.shape[:2]
pixels = np.reshape(gray, w*h)
pixels[pixels.any()] = (0 if pixels < fa else (255 if pixels > fb else pixels * k - fa))
gray = np.reshape(pixels.astype(int), [w, h])


ret, binary = cv2.threshold(gray, av, 255, cv2.THRESH_BINARY)

_, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray, contours, -1, 1, 3)

cv2.imwrite("luankuo-b.png", gray)
#cv2.imshow("img", img)
#cv2.waitKey(0)