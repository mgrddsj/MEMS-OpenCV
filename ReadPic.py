import cv2
import numpy as np

file_path = "img/triangle.png"

img = cv2.imread(file_path)
# resized = cv2.resize(img, (1280,720))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 150, apertureSize = 3)
# cv2.imshow("Photo", edge)
# cv2.waitKey(0)

im = cv2.imread(file_path)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret, imthresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(imthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(im.shape, np.uint8)
img = cv2.drawContours(blank, contours, -1, (255,255,255), 3)
# cv2.imshow("轮廓", img)
# cv2.waitKey(0)

# 霍夫变换
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blank2 = np.zeros(im.shape, np.uint8)
cv2.imshow("霍夫变换前", img)
cv2.waitKey(0)
lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(blank2,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow("霍夫变换", blank2)
cv2.waitKey(0)