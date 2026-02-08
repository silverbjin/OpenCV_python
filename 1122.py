# 1122.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1
select = 1 # 2
if select == 1:
    nx, ny  = 3, 3
    img = cv2.imread("./data/charuco_6x6_250.png")
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250) 
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)      # for cv2==4.13
else:
    nx, ny  = 4, 7
    img = cv2.imread("./data/charuco_5x5_1000.png")
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)     # for cv2==4.13
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#2: detect markers
#2-1
#param  = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)#, parameters=param)
#print("corners=", corners)
#print("ids=", ids)

#2-2
img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

##img_markers = img.copy()
##for i in range(len(corners)):
##    pts    = np.int32(corners[i])
##    cv2.polylines(img_markers,[pts],True, (0,255,255), 2)

#3: detect board corners
#3-1
# board = cv2.aruco.CharucoBoard_create(squaresX=nx, squaresY=ny, 
                #   squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)
board = cv2.aruco.CharucoBoard(     # for cv2==4.13
    size=(nx, ny),
    squareLength=0.04, 
    markerLength=0.02, 
    dictionary=aruco_dict    
)
corners, ids, _, _=cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)

#3-2
ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
cv2.aruco.drawDetectedCornersCharuco(img_markers, charucoCorners)

#3-3
hull = cv2.convexHull(charucoCorners)
pts    = np.int32(hull)
cv2.polylines(img_markers, [pts],True, (0,255,255), 2)
cv2.imshow('img_markers', img_markers)

cv2.waitKey(0)    
cv2.destroyAllWindows()
