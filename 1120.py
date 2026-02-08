# 1120.py
import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

#1: open video capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/aruco6x6_250.mp4')
if (not cap.isOpened()): 
     print('Error opening video')
     import sys
     sys.exit()
     
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
imageSize = width, height
    
#2: load camera matrix K
with np.load('./data/calib_1119.npz') as X: #'./data/calib_1104.npz'
    K = X['K']
    dists = X['dists']  
#dists = np.zeros(5)
print("K=\n", K)
print("dists=", dists)
    
#3
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
nx = 2 
ny = 2   
# board = cv2.aruco.GridBoard_create(nx, ny,
#                                    markerLength=1.0,
#                                    markerSeparation= 0.1,
#                                    dictionary = aruco_dict, firstMarker=0)
board = cv2.aruco.GridBoard(
    size=(nx, ny),
    markerLength=1.0,
    markerSeparation=0.1,
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
     
#4-3: board pos 
     ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, dists, None, None)

     # cv2.aruco.drawAxis(frame, K, dists, rvec, tvec, 1.0) 
     cv2.drawFrameAxes(frame, K, dists, rvec, tvec, 1.0)    # for cv2==4.13
     cv2.aruco.drawDetectedMarkers(frame, corners)

#4-4: markers' pose
     rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, K, dists)        
     for i in range(rvecs.shape[0]):
          #   cv2.aruco.drawAxis(frame, K, dists, rvecs[i, :, :], tvecs[i, :, :], 0.03)
            cv2.drawFrameAxes(frame, K, dists, rvecs[i, :, :], tvecs[i, :, :], 0.03)      # for cv2==4.13
            
#4-5
     #print("t=", t)
     t += 1
     cv2.imshow('frame',frame)  
     key = cv2.waitKey(20)
          
     if key == 27:
        break
#5
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
 





