# 1123.py
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
    
   
#2
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)     # for cv2==4.13
nx = 3 
ny = 3   
# board = cv2.aruco.CharucoBoard_create(squaresX=nx, squaresY=ny, 
#                  squareLength=0.04, markerLength=0.02, dictionary=aruco_dict)
board = cv2.aruco.CharucoBoard(         # for cv2==4.13
    size=(nx, ny),
    squareLength=0.04, 
    markerLength=0.02, 
    dictionary=aruco_dict    
)

#3: calibrate K, dists using calibrateCameraCharuco()
t = 0
count = 0
N_FRAMES = 20

marker_counter = [] # number of markers in a frame
all_corners = []
all_ids = []

while True:
#3-1
     ret, frame = cap.read()
     if not ret:
        print(ret, t)
        break
     if t%10 != 0: # sample
          t += 1
          continue
          
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)#, parameters=param)
     corners, ids, _, _=cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
                                                        
#3-2
     if ids is None:
          cv2.imshow('frame',frame)
          key = cv2.waitKey(20)
          if key == 27:  break
          continue
#3-3
     cv2.aruco.drawDetectedMarkers(frame, corners)                                                                                                      
     ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
     if ret >= 4: # number of charucoCorners
          all_corners.append(charucoCorners)
          all_ids.append(charucoIds)
#3-4 
     count += 1
     if count >= N_FRAMES:
          break
     
     #print("t=", t)    
     t += 1
     cv2.imshow('frame',frame)
     key = cv2.waitKey(20)
     if key == 27:  break
          
#3-5
errs, K, dists, rvecs, tvecs =     cv2.aruco.calibrateCameraCharuco(all_corners, all_ids,
                                                                 board, imageSize, None, None)
                                                                                               
np.savez('./data/calib_1123.npz', K=K, dists= dists)
print("K=\n", K)
print("dists=", dists)                                                                  

#4: 
if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
