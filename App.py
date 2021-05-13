import cv2
import streamlit as st
import numpy as np
import glob
from PIL import Image
from streamlit_cropper import st_cropper

st.set_page_config(layout="centered")
st.header("")

file_list = glob.glob("img/*")
file_path = st.sidebar.selectbox("Image:", file_list, index=8)

image = cv2.imread(file_path)
cropped = st_cropper(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2GRAY)
st.image(gray, caption="灰度图")

# 最大滤波处理
kernel = np.ones((3, 3), np.uint8)
dilate_iteration = st.sidebar.slider(
    "最大滤波（膨涨）次数", min_value=1, max_value=50, value=3)
dilate = cv2.dilate(gray, kernel, iterations=dilate_iteration)
st.image(dilate, caption="最大滤波处理")

# 最小滤波处理
kernel = np.ones((3, 3), np.uint8)
# iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系且只能是整数
erosion_iteration = st.sidebar.slider(
    "最小滤波（腐蚀）次数", min_value=1, max_value=50, value=3)
erosion = cv2.erode(dilate, kernel, iterations=erosion_iteration)
st.image(erosion, caption="最小滤波处理")

# 阈值处理
threshhold_value = st.sidebar.slider(
    "阈值", min_value=50, max_value=255, value=100)
ret, thresh = cv2.threshold(erosion, threshhold_value, 255, cv2.THRESH_BINARY)
thresh = thresh.astype(np.uint8)
st.image(thresh, caption="阈值处理")

# 轮廓
# contours, hierarchy = cv2.findContours(
#     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# blank = np.zeros(thresh.shape, np.uint8)
# img = cv2.drawContours(blank, contours, -1, (255, 255, 255), 1)
img = cv2.Canny(thresh, 100, 200)
st.image(img, caption="轮廓")
# cv2.imwrite("contour.png", img)

# 霍夫变换
blank2 = np.zeros(img.shape, np.uint8)
blank2 = cv2.cvtColor(blank2, cv2.COLOR_GRAY2BGR)
houghRho = st.sidebar.slider("霍夫变换 rho 值（搜索步长）", min_value=1, max_value=10, value=1)
houghThreshhold = st.sidebar.slider(
    "霍夫变换阈值", min_value=1, max_value=1000, value=30)
houghMinLineLength = st.sidebar.slider(
    "霍夫最短线段长度", min_value=1, max_value=500, value=1)
houghMaxLineGap = st.sidebar.slider("霍夫最长间隙", min_value=1, max_value=200, value=50)
lines = cv2.HoughLinesP(img, houghRho, np.pi/360, houghThreshhold,
                        minLineLength=houghMinLineLength, maxLineGap=houghMaxLineGap)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(blank2, (x1, y1), (x2, y2), (0, 255, 0), 1)
st.image(blank2, caption="霍夫变换")
st.text("lines detected: {}".format(len(lines)))

line_num = st.number_input("line number", 0, len(lines)-1, 0)
blank3 = np.zeros(img.shape, np.uint8)
blank3 = cv2.cvtColor(blank3, cv2.COLOR_GRAY2BGR)
x1, y1, x2, y2 = lines[line_num][0]
cv2.line(blank3, (x1, y1), (x2, y2), (0, 255, 0), 1)
st.image(blank3, caption="selected line")

# Shi-Tomasi 角点检测
corners_s = cv2.goodFeaturesToTrack(
    img, maxCorners=4, qualityLevel=0.01, minDistance=10)
corners_s = np.int0(corners_s)
shitomasi_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in corners_s:
    x, y = i.ravel()
    cv2.circle(shitomasi_img, (x, y), 5, (0, 255, 0), -1)
st.image(shitomasi_img, caption="Shi-Tomasi 角点检测")
st.text("Corners detected: {}".format(len(corners_s)))