import cv2
import streamlit as st
import numpy as np
import glob
from PIL import Image

st.set_page_config(layout="centered")
st.header("")

file_list = glob.glob("img/*")
file_path = st.sidebar.selectbox("Image:", file_list, index=8)

image = cv2.imread(file_path, 0)  # Read as grayscale image
st.image(image, use_column_width=True, caption="灰度图")