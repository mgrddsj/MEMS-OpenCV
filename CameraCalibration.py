from functools import partial
import cv2
import streamlit as st
import numpy as np
import glob
import multiprocessing
import time
import stqdm
from PIL import Image

def processImages(file_path, CHESSBOARD, criteria, objpoints, imgpoints, objp):
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
        # corners_img = cv2.drawChessboardCorners(image, CHESSBOARD, corners2,ret)

if __name__ == '__main__':    
    st.set_page_config(layout="centered")
    st.header("")

    file_list = glob.glob("camcalib/*.jpg")
    CHESSBOARD = (9, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001

    manager = multiprocessing.Manager()
    # Creating vector to store vectors of 3D points for each CHESSBOARD image
    objpoints = manager.list()
    # Creating vector to store vectors of 2D points for each CHESSBOARD image
    imgpoints = manager.list()

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    # Multiprocess
    start_time = time.time()
    pool = multiprocessing.Pool()
    func = partial(processImages, CHESSBOARD=CHESSBOARD, criteria=criteria, objpoints=objpoints, imgpoints=imgpoints, objp=objp)
    for _ in stqdm.stqdm(pool.imap_unordered(func, file_list), total=len(file_list), unit="photo"):
        pass
        
    pool.close()
    pool.join()
    
    st.write("objpoints length:", len(objpoints))
    st.write("imgpoints length:", len(imgpoints))
    st.write("Time used:", time.time()-start_time)

    # 相机校准
    image = cv2.imread("camcalib/1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    st.write("相机内参矩阵 mtx:")
    st.write(mtx)
    st.write("透镜畸变系数 dist:")
    st.write(dist)
    st.write("旋转向量 rvecs:")
    st.write(rvecs[0])
    st.write("位移向量 tvecs:")
    st.write(tvecs[0])

    undistorted = cv2.undistort(image, mtx, dist)
    # cv2.imwrite("undistorted.jpg", undistorted)
    st.image(undistorted, use_column_width=True, caption="校正后的图像", channels="BGR")