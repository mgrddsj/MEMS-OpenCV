import cv2
import streamlit as st
import numpy as np
import glob
from PIL import Image

st.set_page_config(layout="centered")
st.header("")

file_list = glob.glob("camcalib/*.jpg")


CHESSBOARD = (7, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001

# Creating vector to store vectors of 3D points for each CHESSBOARD image
objpoints = []
# Creating vector to store vectors of 2D points for each CHESSBOARD image
imgpoints = [] 

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

for file_path in file_list:
    image = cv2.imread(file_path)  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # st.image(image, use_column_width=True, caption="原图", channels="BGR")

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        corners_img = cv2.drawChessboardCorners(image, CHESSBOARD, corners2,ret)
        # st.image(corners_img, use_column_width=True, caption="棋盘检测结果", channels="BGR")
        st.write("{} corners detected".format(file_path))

# 相机校准
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
st.write("相机内参矩阵 mtx:")
st.write(mtx)
st.write("透镜畸变系数 dist:")
st.write(dist)
st.write("旋转向量 rvecs:")
st.write(rvecs[0])
st.write("位移向量 tvecs:")
st.write(tvecs[0])

image = cv2.imread("camcalib/1.jpg")
undistorted = cv2.undistort(image, mtx, dist)
# cv2.imwrite("undistorted.jpg", undistorted)
st.image(undistorted, use_column_width=True, caption="校正后的图像", channels="BGR")