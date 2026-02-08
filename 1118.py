# 1118.py
import cv2
import numpy as np

#1 
img = cv2.imread("./data/aruco_6x6.png") # "./data/aruco_5x5.png"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#2
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) #cv2.aruco.DICT_5X5_250
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # for cv2==4.13
#param  = cv2.aruco.DetectorParameters_create()

corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)#, parameters=param)
print("corners=", corners)
print("ids=", ids)

#3
img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
##img_markers = img.copy()
##for i in range(len(corners)):
##    pts    = np.int32(corners[i])
##    cv2.polylines(img_markers,[pts],True, (0,255,255), 2)
cv2.imshow('img_markers', img_markers)

#4
rejected_img = cv2.aruco.drawDetectedMarkers(img.copy(), rejected, borderColor=(0, 255, 255))

##rejected_img = img.copy()
##for i in range(len(rejected)):
##    pts    = np.int32(rejected[i])
##    cv2.polylines(rejected_img,[pts],True, (0,255,255), 2)
cv2.imshow('rejected', rejected_img) 

cv2.waitKey(0)    
cv2.destroyAllWindows()
