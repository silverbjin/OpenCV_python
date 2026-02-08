# 1124.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/charuco6x6_250.mp4')
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
imageSize = width, height
    
#2: load camera matrix K
with np.load('./data/calib_1123.npz') as X: #  'calib_1119.npz', 'calib_1104.npz'
    K = X['K']
    dists = X['dists']  
#dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)
    
#3
nx = 3 
ny = 3   
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)     # for cv2==4.13
# board = cv2.aruco.CharucoBoard_create(squaresX=nx, squaresY=ny, 
#                                       squareLength=0.04, markerLength=0.02, 
#                                       dictionary=aruco_dict)
board = cv2.aruco.CharucoBoard(         # for cv2==4.13
    size=(nx, ny),
    squareLength=0.04, 
    markerLength=0.02, 
    dictionary=aruco_dict    
)
#4: 
t = 0 # frame counter
while True:
#4-1
     ret, frame = cap.read()
     if not ret:
        break
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)#, parameters=param)
     corners, ids, _, _=cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)    
     
#4-2
     if ids is None:
          cv2.imshow('frame',frame)
          key = cv2.waitKey(20)
          if key == 27:  break
          continue
     
#4-3: draw and interpolate     	
     cv2.aruco.drawDetectedMarkers(frame, corners)                                                                                                      
     ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)

#4-4: board pos: rvec, tvec
     if  ret >= 4: # ret == len(charucoCorners)
          pts    = np.int32(charucoCorners)
          hull = cv2.convexHull(pts)
          cv2.polylines(frame, [hull],True, (0,255,255), 2)
          
     
          cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners)
          ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds,
                                                               board, K, dists, None, None)
          # cv2.aruco.drawAxis(frame, K, dists, rvec, tvec, 0.1)
          cv2.drawFrameAxes(frame, K, dists, rvec, tvec, 0.1)
          
     else:
          print("charucoCorners are not found enough!!!")
 
#4-5: markers' pos: rvecs, tvecs
     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, K, dists)        
     for i in range(rvecs.shape[0]):
          #   cv2.aruco.drawAxis(frame, K, dists, rvecs[i, :, :], tvecs[i, :, :], 0.03)      
            cv2.drawFrameAxes(frame, K, dists, rvecs[i, :, :], tvecs[i, :, :], 0.03)
#4-6
     print("t=", t)
     t += 1
     cv2.imshow('frame',frame)
          
     key = cv2.waitKey(20)     
     if key == 27:
        break
#5
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





