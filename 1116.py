# 1116.py
#https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html
import cv2
import numpy as np
import matplotlib.pyplot as plt

#1 
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)    
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  # for cv2==4.13


#2
fig = plt.figure(figsize=(6, 6))
nx = 2
ny = 2
for i in range(nx*ny):
    ax = fig.add_subplot(ny,nx, i+1)
    # img = cv2.aruco.drawMarker(aruco_dict, i, 600)
    img = cv2.aruco.generateImageMarker(aruco_dict, i, 600) # for cv2==4.13
    plt.imshow(img, cmap = "gray", interpolation = "nearest")
    ax.axis("off")
#3
plt.savefig("./data/aruco_5x5.png")
plt.show()


